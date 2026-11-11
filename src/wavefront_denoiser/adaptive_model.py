"""Count-aware NTASAD reconstruction with explicit availability in history.

Uses the existing unified U-Net as its backbone. Missing-sample masking,
available-pixel pyramid construction and fallback rules are documented reconstruction
choices; they are not a claim to reproduce unpublished author code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import INPUT_CHANNELS, NTASADDenoiser
from .ntasad_radiance import MAX_ENCODED, decode_radiance, encode_radiance
from .temporal import apply_kernel, reproject_available_history


ARCHITECTURE = "ntasad_adaptive_spatiotemporal_unet_v2"
LEGACY_ARCHITECTURE = "ntasad_adaptive_spatiotemporal_unet_v1"
FEATURE_VERSION = 3
RECONSTRUCTION_VERSIONS = ("sample_mass_v1", "pixel_mean_v2")
DEFAULT_RECONSTRUCTION = "pixel_mean_v2"


@dataclass(frozen=True)
class AdaptiveReconstruction:
    """Linear RGB and boolean availability at the public streaming API.

    Sequence tensors have NT3HW/NT1HW shapes. In the transformed training API,
    radiance is gamma/log encoded. Missing outputs are zero with valid=False;
    these cannot be reused as measured black history.
    """

    radiance: torch.Tensor
    valid: torch.Tensor


def _weighted_downsample(image: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = F.avg_pool2d(weight, 2)
    numerator = F.avg_pool2d(image * weight, 2)
    safe = torch.where(mass > 0, mass, torch.ones_like(mass))
    return numerator / safe, mass


def _filtered(image, logits, valid):
    value = apply_kernel(image, logits, valid=valid)
    support = F.max_pool2d(valid.to(image.dtype), 5, stride=1, padding=2) > 0
    return value, support


def _combine(fine, fine_valid, coarse, coarse_valid, alpha_logits):
    """Equation (3) where supported; normalized low frequencies and explicit fallbacks."""
    low, _ = _weighted_downsample(fine, fine_valid.to(fine.dtype))
    low = F.interpolate(low, scale_factor=2, mode="nearest")
    up = F.interpolate(coarse, scale_factor=2, mode="nearest")
    up_valid = F.interpolate(coarse_valid.to(fine.dtype), scale_factor=2, mode="nearest") > 0
    alpha = torch.sigmoid(alpha_logits) * up_valid.to(fine.dtype)
    combined = fine + alpha * (up - low)
    # A coarse-only estimate must not be attenuated by a blend against missing black.
    value = torch.where(fine_valid, combined, torch.where(up_valid, up, 0))
    return value, fine_valid | up_valid


class AdaptiveNTASADDenoiser(nn.Module):
    """Existing 15-feature U-Net with sparse spatial/temporal reconstruction.

    Explicit history includes availability. None initializes from dense uniform
    noisy RGB, as in the paper. forward_spatial supports sparse inputs without
    history. No reference pixels are read. Uniform checkpoints remain separate;
    warm-start explicitly via model.network.load_state_dict(uniform.state_dict()).
    """

    feature_version = FEATURE_VERSION

    def __init__(self, base_channels: int = 48, *, history_backend="grid_sample",
                 reconstruction_version=DEFAULT_RECONSTRUCTION):
        super().__init__()
        if reconstruction_version not in RECONSTRUCTION_VERSIONS:
            raise ValueError("Unknown adaptive reconstruction_version")
        self.network = NTASADDenoiser(base_channels)
        self.base_channels = base_channels
        if history_backend not in ("grid_sample","gather"):
            raise ValueError("Unknown adaptive history backend")
        self.history_backend = history_backend
        self.reconstruction_version = reconstruction_version

    @property
    def architecture(self):
        return LEGACY_ARCHITECTURE if self.reconstruction_version == "sample_mass_v1" else ARCHITECTURE

    @staticmethod
    def _validate(buffers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "sample_count" not in buffers:
            raise ValueError("Missing adaptive sample_count")
        count = buffers["sample_count"]
        if count.dtype not in (torch.int32, torch.int64) or (count < 0).any() or (count > 0xFFFFFFFF).any():
            raise ValueError("Adaptive sample_count must be nonnegative integer counts in uint32 range")
        # Retain the uniform guide/radiance checks without changing its API or
        # accepting fractional original counts through a clamp-to-one operation.
        return NTASADDenoiser._validate(dict(buffers, sample_count=count.clamp_min(1)))

    @staticmethod
    def _initialization(current, counts, selected=None):
        if selected is None:
            selected = torch.ones(current.shape[0], dtype=torch.bool, device=current.device)
        values = counts[selected]
        if values.numel() and ((values < 1).any() or (values != values[..., :1, :1]).any()):
            raise ValueError("Initialization/reset requires dense uniform samples; use forward_spatial for sparse stills")
        return AdaptiveReconstruction(current, torch.ones_like(counts, dtype=torch.bool))

    @staticmethod
    def _validate_history(history: AdaptiveReconstruction, rgb):
        value, available = history.radiance, history.valid
        if (value.shape != rgb.shape or value.dtype != rgb.dtype or value.device != rgb.device
                or not torch.isfinite(value).all() or (value < 0).any()
                or available.shape != rgb[:, :1].shape or available.dtype != torch.bool
                or available.device != rgb.device):
            raise ValueError("Adaptive history must have matching finite RGB and boolean N1HW availability")

    @staticmethod
    def _warp(history: AdaptiveReconstruction, frame, rgb, *, backend="grid_sample"):
        AdaptiveNTASADDenoiser._validate_history(history, rgb)
        value, available = history.radiance, history.valid
        return reproject_available_history(value, frame["motion"], frame["history_valid"], available,backend=backend)

    def _step(self, buffers, history, *, initialize=True):
        rgb = self._validate(buffers)
        counts = buffers["sample_count"]
        current = encode_radiance(torch.where(counts > 0, rgb, 0))
        if history is None and initialize:
            return self._initialization(current, counts)
        if history is None:
            warped = torch.zeros_like(current)
            valid = torch.zeros_like(counts, dtype=torch.bool)
        else:
            warped, valid = self._warp(history, buffers, rgb, backend=self.history_backend)
        motion = torch.where(torch.isfinite(buffers["motion"]), buffers["motion"], 0)
        features = torch.cat((current, buffers["albedo"], buffers["normal"], buffers["depth"], motion, warped), dim=1)
        h, w = rgb.shape[-2:]
        pad = (0, (-w) % 32, 0, (-h) % 32)
        fine, middle, coarse = self.network._predict_kernels(F.pad(features, pad, mode="replicate"))
        image0 = F.pad(current, pad, mode="replicate")
        # Section 2.1 filters downscaled versions of the pixel-mean image.
        # Sample-count weighting changes those images even without any holes.
        # At missing pixels normalize over available pixels; coverage, rather
        # than sample mass, propagates to keep original pixels equally weighted.
        weight = counts if self.reconstruction_version == "sample_mass_v1" else counts > 0
        mass0 = F.pad(weight.to(rgb.dtype), pad, mode="replicate")
        image1, mass1 = _weighted_downsample(image0, mass0)
        image2, mass2 = _weighted_downsample(image1, mass1)
        filtered2, support2 = _filtered(image2, coarse[:, :25], mass2 > 0)
        filtered1, support1 = _filtered(image1, middle[:, :25], mass1 > 0)
        filtered1, support1 = _combine(filtered1, support1, filtered2, support2,
            F.interpolate(coarse[:, 25:], scale_factor=2, mode="nearest"))
        spatial, support0 = _filtered(image0, fine[:, :25], mass0 > 0)
        spatial, support0 = _combine(spatial, support0, filtered1, support1,
            F.interpolate(middle[:, 25:], scale_factor=2, mode="nearest"))
        spatial, support0 = spatial[..., :h, :w], support0[..., :h, :w]
        temporal = apply_kernel(warped, fine[:, 25:50, :h, :w], valid=valid)
        alpha = torch.sigmoid(fine[:, 50:, :h, :w]) * valid.to(rgb.dtype)
        both = (1-alpha) * spatial + alpha * temporal
        result = torch.where(support0, both, torch.where(valid, temporal, 0))
        return AdaptiveReconstruction(result.clamp(0, MAX_ENCODED), support0 | valid)

    def forward(self, buffers: Mapping[str, torch.Tensor],
                history: AdaptiveReconstruction | None = None) -> AdaptiveReconstruction:
        if history is not None:
            if not isinstance(history, AdaptiveReconstruction):
                raise ValueError("Adaptive history must include RGB and availability; reset on resize")
            self._validate_history(history, self._validate(buffers))
            history = AdaptiveReconstruction(encode_radiance(history.radiance), history.valid)
        result = self._step(buffers, history)
        return AdaptiveReconstruction(decode_radiance(result.radiance), result.valid)

    def forward_spatial(self, buffers: Mapping[str, torch.Tensor]) -> AdaptiveReconstruction:
        result = self._step(buffers, None, initialize=False)
        return AdaptiveReconstruction(decode_radiance(result.radiance), result.valid)

    def forward_sequence(self, buffers: Mapping[str, torch.Tensor], *,
                         reset: torch.Tensor | None = None, transformed=False) -> AdaptiveReconstruction:
        missing = INPUT_CHANNELS.keys() - buffers.keys()
        if missing:
            raise ValueError(f"Missing adaptive sequence buffers: {sorted(missing)}")
        rgb = buffers["radiance"]
        if rgb.ndim != 5 or min(rgb.shape) < 1:
            raise ValueError("Expected nonempty NT3HW adaptive sequence")
        n, t, _, h, w = rgb.shape
        for key, channels in INPUT_CHANNELS.items():
            if buffers[key].shape != (n, t, channels, h, w):
                raise ValueError(f"Invalid NTCHW adaptive buffer {key}")
        if reset is not None and (reset.shape != (n,t) or reset.device != rgb.device
                                  or not torch.all((reset == 0) | (reset == 1))):
            raise ValueError("reset must be binary with matching NT shape and device")
        history = None
        values, masks = [], []
        for index in range(t):
            frame = {name: buffers[name][:, index] for name in INPUT_CHANNELS}
            history = self._step(frame, history)
            if reset is not None:
                selected = reset[:, index].bool()
                initial = self._initialization(encode_radiance(frame["radiance"]), frame["sample_count"], selected)
                history = AdaptiveReconstruction(
                    torch.where(selected[:,None,None,None], initial.radiance, history.radiance),
                    torch.where(selected[:,None,None,None], initial.valid, history.valid))
            values.append(history.radiance if transformed else decode_radiance(history.radiance))
            masks.append(history.valid)
        return AdaptiveReconstruction(torch.stack(values, 1), torch.stack(masks, 1))
