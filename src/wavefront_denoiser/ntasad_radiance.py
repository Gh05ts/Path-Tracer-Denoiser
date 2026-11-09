"""NTASAD section 2.2 radiance space; guides retain signed/native units."""

import math

import torch


MAX_RADIANCE = 65535.
GAMMA = 2.2
MAX_ENCODED = math.log1p(MAX_RADIANCE) ** (1 / GAMMA)


def encode_radiance(radiance: torch.Tensor) -> torch.Tensor:
    """x' = log(1 + clamp(x, 0, 65535)) ** (1/2.2), with finite black gradients."""
    value = torch.log1p(radiance.clamp(0, MAX_RADIANCE))
    # pow has an infinite derivative at zero. Choose a zero subgradient there;
    # positive values retain the exact paper transform, without an epsilon bias.
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.pow(1 / GAMMA), torch.zeros_like(value))


def decode_radiance(value: torch.Tensor) -> torch.Tensor:
    """Decode to finite nonnegative linear HDR, bounded by the input clamp range.

    Output bounding is a documented implementation choice: equation (3) can
    overshoot, and the publication does not specify output overshoot handling.
    """
    return torch.expm1(value.clamp(0, MAX_ENCODED).pow(GAMMA)).clamp_max(MAX_RADIANCE)
