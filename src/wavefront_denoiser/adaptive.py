"""NTASAD sampler and equation (2), independent of the renderer.

The continuous allocation carries normalization gradients. Integer counts are
for forward rendering only; this module does not invent a renderer gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ntasad_radiance import encode_radiance
from .temporal import reproject_available_history, reproject_history, reproject_history_gather


GUIDE_CHANNELS = {"albedo": 3, "normal": 3, "depth": 1,
                  "motion": 2, "history_valid": 1}
FEATURE_CHANNELS = 8
ARCHITECTURE = "ntasad_sampler_unet_v1"


@dataclass(frozen=True)
class SampleMap:
    """N1HW expected allocation and rounded int64 path counts.

    Expected allocations sum to average_spp * H * W per image. Independent
    rounding follows equation (2); integer totals need not equal that budget.
    """

    expected: torch.Tensor
    counts: torch.Tensor


def normalize_sample_map(logits: torch.Tensor, average_spp: float) -> SampleMap:
    """Spatial softmax per image, followed by round-to-nearest (ties to even).

    No minimum, cap, straight-through derivative, or budget repair is imposed.
    Normalize the original image, after cropping network padding. Float16 and
    bfloat16 inputs accumulate in float32; doubles retain double precision.
    """
    if (logits.ndim != 4 or min(logits.shape) < 1 or logits.shape[1] != 1
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("logits must be finite nonempty floating N1HW")
    if (isinstance(average_spp, bool) or not isinstance(average_spp, (int, float))
            or not math.isfinite(average_spp) or average_spp <= 0):
        raise ValueError("average_spp must be finite and positive")
    pixels = logits.shape[-2] * logits.shape[-1]
    budget = average_spp * pixels
    if not math.isfinite(budget) or budget > 0xFFFFFFFF:
        raise ValueError("Allocation exceeds the uint32 per-pixel renderer range")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    expected = (torch.softmax(logits.flatten(1).to(dtype), dim=1) * budget).reshape(logits.shape)
    return SampleMap(expected, expected.round().to(torch.int64))


def _grayscale(rgb: torch.Tensor) -> torch.Tensor:
    # Section 2.2 gives these exact CCIR 601 coefficients (sum = 0.9999).
    return (rgb * rgb.new_tensor([.2989, .587, .114])[None, :, None, None]).sum(1, keepdim=True)


class AdaptiveSampler(nn.Module):
    """Figure 3 U-Net with grayscale albedo/history and one scalar output.

    Geometry is NCHW. History is previous denoised *linear* HDR RGB, encoded
    before the bilinear warp to match NTASADDenoiser's convention.
    Current noisy RGB and reference RGB are never consumed. None initializes
    with uniform sampling, without executing the network. The topology follows
    the existing denoiser reconstruction; unresolved figure details remain
    documented in the audit, rather than claiming author-code equivalence.
    """

    def __init__(self, base_channels: int = 48):
        super().__init__()
        if type(base_channels) is not int or not 4 <= base_channels <= 128:
            raise ValueError("base_channels must be an integer between 4 and 128")
        self.base_channels = base_channels
        b = base_channels
        self.encoder0 = nn.Sequential(nn.Conv2d(FEATURE_CHANNELS, b, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
                                   for _ in range(5)])
        self.decoders = nn.ModuleList()
        inputs = b
        for level in range(4, -1, -1):
            outputs = max(1, round(b * 64 / 48)) if level == 0 else 2*b
            self.decoders.append(nn.Sequential(
                nn.Conv2d(inputs + b, outputs, 3, padding=1), nn.ReLU(),
                nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU()))
            inputs = outputs
        self.head = nn.Conv2d(inputs, 1, 3, padding=1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _validate(guides: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = GUIDE_CHANNELS.keys() - guides.keys()
        if missing:
            raise ValueError(f"Missing sampler guides: {sorted(missing)}")
        albedo = guides["albedo"]
        if albedo.ndim != 4 or min(albedo.shape) < 1 or not albedo.is_floating_point():
            raise ValueError("albedo must be nonempty floating N3HW")
        n, _, h, w = albedo.shape
        for name, channels in GUIDE_CHANNELS.items():
            value = guides[name]
            if (value.shape != (n, channels, h, w) or value.device != albedo.device
                    or value.dtype != albedo.dtype):
                raise ValueError(f"Invalid shape, dtype or device for sampler guide {name}")
            if name != "motion" and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite sampler guide {name}")
        if (albedo < 0).any() or (guides["depth"] < 0).any():
            raise ValueError("Albedo and depth must be nonnegative")
        if not torch.all((guides["history_valid"] == 0) | (guides["history_valid"] == 1)):
            raise ValueError("history_valid must be binary")
        return albedo

    def features(self, guides: Mapping[str, torch.Tensor], history: torch.Tensor, *,
                 history_available: torch.Tensor | None = None, history_backend="grid_sample") -> torch.Tensor:
        """Build eight channels, preserving history gradients across frames."""
        albedo = self._validate(guides)
        n, _, h, w = albedo.shape
        if (history.shape != (n, 3, h, w) or history.device != albedo.device
                or history.dtype != albedo.dtype or not torch.isfinite(history).all()
                or (history < 0).any()):
            raise ValueError("History must be finite nonnegative linear N3HW matching guides")
        if history_available is None:
            if history_backend not in ("grid_sample","gather"):
                raise ValueError("Unknown sampler history backend")
            warp = reproject_history if history_backend == "grid_sample" else reproject_history_gather
            warped, _ = warp(encode_radiance(history), guides["motion"], guides["history_valid"])
        else:
            warped, _ = reproject_available_history(encode_radiance(history), guides["motion"],
                                                    guides["history_valid"], history_available,backend=history_backend)
        motion = torch.where(torch.isfinite(guides["motion"]), guides["motion"], 0)
        return torch.cat((_grayscale(albedo), guides["normal"], guides["depth"],
                          motion, _grayscale(warped)), dim=1)

    def predict_logits(self, guides: Mapping[str, torch.Tensor], history: torch.Tensor, *,
                       history_available: torch.Tensor | None = None, history_backend="grid_sample") -> torch.Tensor:
        features = self.features(guides, history, history_available=history_available,history_backend=history_backend)
        h, w = features.shape[-2:]
        levels = [self.encoder0(F.pad(features, (0, (-w) % 32, 0, (-h) % 32), mode="replicate"))]
        for block in self.down:
            levels.append(block(F.avg_pool2d(levels[-1], 2)))
        decoded = levels[-1]
        for level, block in zip(range(4, -1, -1), self.decoders):
            decoded = block(torch.cat((F.interpolate(decoded, scale_factor=2, mode="nearest"),
                                       levels[level]), dim=1))
        return self.head(decoded)[..., :h, :w]

    def forward(self, guides: Mapping[str, torch.Tensor], history: torch.Tensor | None = None,
                *, average_spp: float = 4., history_available: torch.Tensor | None = None,
                history_backend="grid_sample") -> SampleMap:
        if history is None:
            if history_available is not None:
                raise ValueError("history_available requires explicit history RGB")
            albedo = self._validate(guides)
            logits = torch.zeros_like(albedo[:, :1])
        else:
            logits = self.predict_logits(guides, history, history_available=history_available,history_backend=history_backend)
        return normalize_sample_map(logits, average_spp)
