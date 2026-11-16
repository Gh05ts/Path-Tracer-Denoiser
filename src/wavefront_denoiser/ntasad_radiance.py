"""NTASAD section 2.2 radiance space; guides retain signed/native units."""

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch


MAX_RADIANCE = 65535.
GAMMA = 2.2
MAX_ENCODED = math.log1p(MAX_RADIANCE) ** (1 / GAMMA)
_GRADIENT_FLOOR = ContextVar("ntasad_radiance_gradient_floor", default=0.)


@contextmanager
def radiance_gradient_policy(floor=0.):
    """Opt-in backward stabilization; forward radiance values are unchanged.

    A positive linear-radiance floor bounds the gamma derivative below that
    value. This is an explicit gradient surrogate, not the exact derivative
    in that interval. Zero retains the original positive-radiance derivative.
    """
    if (isinstance(floor, bool) or not isinstance(floor, (int,float)) or not math.isfinite(floor)
            or not (floor == 0 or 1e-12 <= floor <= 1)):
        raise ValueError("radiance_gradient_floor must be zero or finite in [1e-12,1]")
    token = _GRADIENT_FLOOR.set(float(floor))
    try:
        yield
    finally:
        _GRADIENT_FLOOR.reset(token)


class _BoundedGammaGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, floor):
        positive = value > 0
        ctx.save_for_backward(value, positive)
        ctx.floor = math.log1p(floor)
        safe = torch.where(positive, value, torch.ones_like(value))
        return torch.where(positive, safe.pow(1/GAMMA), torch.zeros_like(value))

    @staticmethod
    def backward(ctx, gradient):
        value, positive = ctx.saved_tensors
        derivative = (1/GAMMA)*value.clamp_min(ctx.floor).pow(1/GAMMA-1)
        # Exact black keeps the existing zero subgradient. Positive near-black
        # values use the finite slope at the explicit floor, without changing
        # the forward transform, renderer samples, allocation, or loss values.
        return torch.where(positive, gradient*derivative, 0), None


def encode_radiance(radiance: torch.Tensor) -> torch.Tensor:
    """x' = log(1 + clamp(x, 0, 65535)) ** (1/2.2), with finite black gradients."""
    value = torch.log1p(radiance.clamp(0, MAX_RADIANCE))
    floor = _GRADIENT_FLOOR.get()
    if floor:
        return _BoundedGammaGradient.apply(value, floor)
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
