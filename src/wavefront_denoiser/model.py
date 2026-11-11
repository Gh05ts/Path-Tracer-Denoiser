"""Uniform-SPP denoiser following NTASAD Figure 3 and sections 2.1–2.2.

Frame inputs are batched linear HDR capture buffers (NCHW). Sequence inputs
are NTCHW, as produced by a DataLoader over SequenceDataset. Reference buffers
are never read. History is explicit, so independent calls cannot leak state.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .ntasad_radiance import MAX_ENCODED, decode_radiance, encode_radiance
from .temporal import apply_kernel, combine_scales, reproject_history


INPUT_CHANNELS = {"radiance": 3, "albedo": 3, "normal": 3, "depth": 1,
                  "guide_valid": 1, "sample_count": 1, "motion": 2, "history_valid": 1}
FEATURE_CHANNELS = 15
FEATURE_VERSION = 2
ARCHITECTURE = "ntasad_spatiotemporal_unet_v1"


class NTASADDenoiser(nn.Module):
    """Three spatial 5x5 kernels, one temporal 5x5 kernel, and learned blends.

    Returns nonnegative linear HDR RGB. No tone mapping or exposure is applied.
    All spatial filtering, temporal filtering, and blending share one U-Net.
    """

    def __init__(self, base_channels: int = 48):
        super().__init__()
        if type(base_channels) is not int or not 4 <= base_channels <= 128:
            raise ValueError("base_channels must be an integer between 4 and 128")
        self.base_channels = base_channels
        b = base_channels
        # Figure 3: 48 throughout the encoder, 96 in coarse decoders, 64 at full size.
        # Width overrides retain the topology for small tests/ablation studies.
        fine_width = max(1, round(b * 64 / 48))
        self.encoder0 = nn.Sequential(nn.Conv2d(FEATURE_CHANNELS, b, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
                                   for _ in range(5)])
        self.decoders = nn.ModuleList()
        inputs = b
        for level in range(4, -1, -1):
            outputs = fine_width if level == 0 else 2*b
            self.decoders.append(nn.Sequential(
                nn.Conv2d(inputs + b, outputs, 3, padding=1), nn.ReLU(),
                nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU()))
            inputs = outputs
        self.coarse_head = nn.Conv2d(2*b, 26, 3, padding=1)
        self.middle_head = nn.Conv2d(2*b, 26, 3, padding=1)
        self.fine_head = nn.Conv2d(fine_width, 51, 3, padding=1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _predict_kernels(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        levels = [self.encoder0(features)]
        for block in self.down:
            levels.append(block(F.avg_pool2d(levels[-1], 2)))
        decoded = levels[-1]
        heads = {}
        for level, block in zip(range(4, -1, -1), self.decoders):
            decoded = block(torch.cat((F.interpolate(decoded, scale_factor=2, mode="nearest"),
                                       levels[level]), dim=1))
            if level == 2:
                heads[2] = self.coarse_head(decoded)
            elif level == 1:
                heads[1] = self.middle_head(decoded)
            elif level == 0:
                heads[0] = self.fine_head(decoded)
        return heads[0], heads[1], heads[2]

    @staticmethod
    def _validate(buffers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = INPUT_CHANNELS.keys() - buffers.keys()
        if missing:
            raise ValueError(f"Missing recurrent buffers: {sorted(missing)}")
        rgb = buffers["radiance"]
        if rgb.ndim != 4 or min(rgb.shape) < 1 or not rgb.is_floating_point():
            raise ValueError("radiance must be a nonempty floating N3HW tensor")
        n, _, h, w = rgb.shape
        for name, channels in INPUT_CHANNELS.items():
            value = buffers[name]
            if value.shape != (n, channels, h, w) or value.device != rgb.device:
                raise ValueError(f"Invalid shape or device for recurrent buffer {name}")
            if name != "sample_count" and value.dtype != rgb.dtype:
                raise ValueError(f"Recurrent buffer {name} must share radiance dtype")
            # Invalid motion can be nonfinite; reprojection explicitly masks it.
            if name != "motion" and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite recurrent buffer {name}")
        if (rgb < 0).any() or (buffers["depth"] < 0).any():
            raise ValueError("Radiance and depth must be nonnegative")
        count = buffers["sample_count"]
        if (count < 1).any() or (count != count.round()).any():
            raise ValueError("Uniform-SPP recurrence requires positive integer sample counts")
        for name in ("guide_valid", "history_valid"):
            if not torch.all((buffers[name] == 0) | (buffers[name] == 1)):
                raise ValueError(f"{name} must be binary")
        return rgb

    def _step(self, buffers: Mapping[str, torch.Tensor],
              history: torch.Tensor | None, *, initialize: bool = True) -> torch.Tensor:
        """Internal recurrence and reconstruction both live in gamma/log space."""
        rgb = self._validate(buffers)
        current = encode_radiance(rgb)
        if history is None and initialize:
            # Section 2.1: the first frame initializes history from uniform noisy RGB.
            return current
        if history is None:
            warped = torch.zeros_like(current)
            valid = torch.zeros_like(current[:, :1], dtype=torch.bool)
        else:
            if history.shape != rgb.shape or history.dtype != rgb.dtype or history.device != rgb.device:
                raise ValueError("History must match current RGB shape, dtype and device; reset on resize")
            warped, valid = reproject_history(history, buffers["motion"], buffers["history_valid"])
        motion = torch.where(torch.isfinite(buffers["motion"]), buffers["motion"], 0)
        # Section 2.1 lists normals, depth, motion, albedo, history, and noisy RGB.
        # Visibility and sample counts are contract checks, not extra learned features.
        features = torch.cat((current, buffers["albedo"], buffers["normal"],
                              buffers["depth"], motion, warped), dim=1)
        h, w = rgb.shape[-2:]
        pad = (0, (-w) % 32, 0, (-h) % 32)
        fine, middle, coarse = self._predict_kernels(F.pad(features, pad, mode="replicate"))
        image0 = F.pad(current, pad, mode="replicate")
        image1 = F.avg_pool2d(image0, 2)
        image2 = F.avg_pool2d(image1, 2)
        filtered2 = apply_kernel(image2, coarse[:, :25])
        filtered1 = combine_scales(apply_kernel(image1, middle[:, :25]), filtered2,
                                   F.interpolate(coarse[:, 25:], scale_factor=2, mode="nearest"))
        spatial = combine_scales(apply_kernel(image0, fine[:, :25]), filtered1,
                                 F.interpolate(middle[:, 25:], scale_factor=2, mode="nearest"))
        temporal = apply_kernel(warped, fine[:, 25:50, :h, :w], valid=valid)
        alpha = torch.sigmoid(fine[:, 50:, :h, :w]) * valid.to(rgb.dtype)
        return ((1 - alpha) * spatial[..., :h, :w] + alpha * temporal).clamp(0, MAX_ENCODED)

    def forward(self, buffers: Mapping[str, torch.Tensor],
                history: torch.Tensor | None = None) -> torch.Tensor:
        """Return linear HDR; None initializes from noisy RGB, without a network pass.

        Explicit history is linear HDR, encoded before warping to match training.
        Internally kernels and recurrence operate in the paper's gamma/log space.
        """
        if history is not None:
            if not torch.isfinite(history).all() or (history < 0).any():
                raise ValueError("History must be finite nonnegative linear HDR")
            history = encode_radiance(history)
        return decode_radiance(self._step(buffers, history))

    def forward_spatial(self, buffers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Denoise a still with this same U-Net and its three spatial kernels.

        No temporal buffers are needed. This is an inference fallback using the
        jointly trained weights; sequence training uses noisy initialization as
        specified by the paper, followed by four learned recurrent steps.
        """
        if "radiance" not in buffers or buffers["radiance"].ndim != 4:
            raise ValueError("Spatial fallback requires N3HW radiance")
        rgb = buffers["radiance"]
        frame = dict(buffers, motion=torch.zeros_like(rgb[:, :2]),
                     history_valid=torch.zeros_like(rgb[:, :1]))
        return decode_radiance(self._step(frame, None, initialize=False))

    def forward_sequence(self, buffers: Mapping[str, torch.Tensor], *,
                         reset: torch.Tensor | None = None,
                         transformed: bool = False) -> torch.Tensor:
        """Run NTCHW buffers with fresh history; return NT3HW linear HDR.

        Optional binary reset[N,T] starts independent sequences within a batch.
        The first frame initializes from noisy RGB. Set transformed=True for the
        training loss, avoiding an unnecessary decode/encode round trip.
        No history is stored or detached; losses on later outputs can
        back-propagate into earlier frames.
        """
        missing = INPUT_CHANNELS.keys() - buffers.keys()
        if missing:
            raise ValueError(f"Missing recurrent buffers: {sorted(missing)}")
        rgb = buffers["radiance"]
        if rgb.ndim != 5 or min(rgb.shape) < 1:
            raise ValueError("Expected nonempty NT3HW sequence radiance")
        n, t, _, h, w = rgb.shape
        for name, c in INPUT_CHANNELS.items():
            if buffers[name].shape != (n, t, c, h, w):
                raise ValueError(f"Invalid NTCHW sequence buffer {name}")
        if reset is not None:
            if reset.shape != (n, t) or reset.device != rgb.device:
                raise ValueError("reset must have matching NT shape and device")
            if not torch.all((reset == 0) | (reset == 1)):
                raise ValueError("reset must be binary")
        outputs = []
        history = None
        for index in range(t):
            frame = {name: buffers[name][:, index] for name in INPUT_CHANNELS}
            if reset is not None:
                keep = ~reset[:, index, None, None, None].bool()
                frame["history_valid"] = frame["history_valid"] * keep
            history = self._step(frame, history)
            if reset is not None:
                history = torch.where(reset[:, index, None, None, None].bool(),
                                      encode_radiance(frame["radiance"]), history)
            outputs.append(history if transformed else decode_radiance(history))
        return torch.stack(outputs, dim=1)
