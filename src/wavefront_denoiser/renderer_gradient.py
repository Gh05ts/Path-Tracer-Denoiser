"""Training-only DASR equation (8) surrogate for discrete sample-layer replay.

The forward pass is exact integer replay in linear radiance. The backward is
an estimator, NOT the derivative of that discontinuous integer-valued renderer.
See Kuznetsov et al., EGSR 2018, section 4.2, equations (3)–(8).
"""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from .sample_layers import replay_sample_layers


ZERO_POLICIES = ("unit_denominator", "zero")
GRADIENT_VERSION = "dasr_equation8_linear_v1"


class _RendererGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, expected, reference, counts, zero_policy):
        ctx.zero_policy = zero_policy
        ctx.save_for_backward(image.detach(), reference.detach(), counts)
        # The reference never contributes a pixel to the forward image.
        return image.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        image, reference, counts = ctx.saved_tensors
        grad_expected = None
        if ctx.needs_input_grad[1]:
            # Equation (8) uses actual positive sample count s, not 2*s or s+1.
            derivative = (reference - image) / counts.clamp_min(1).to(image.dtype)
            if ctx.zero_policy == "zero":
                derivative = torch.where(counts > 0, derivative, 0)
            grad_expected = (grad_output * derivative).sum(dim=1, keepdim=True)
        # Preserve ordinary layer-value gradients; no gradients into targets or
        # integer counts. Higher-order derivatives of this estimator are unsupported.
        return grad_output, grad_expected, None, None, None


@dataclass(frozen=True)
class DifferentiableRender:
    radiance: torch.Tensor
    counts: torch.Tensor
    zero_policy: str


def differentiable_replay(layers: torch.Tensor, expected: torch.Tensor,
                          reference: torch.Tensor, *, zero_policy: str) -> DifferentiableRender:
    """Replay rounded expected allocations and attach the DASR count surrogate.

    layers: N,6,3,H,W independent linear RGB means, ordered 1/2/4/8/16/32.
    expected: differentiable floating N1HW allocations BEFORE rounding.
    reference: independent linear N3HW target, used only by backward.

    Rounded counts must be in 0..63; no cap, floor-to-one or budget repair is
    silently added. The caller must choose a zero policy explicitly, since the
    published (reference-image)/s expression is undefined at s=0:

    * unit_denominator: use denominator 1 at zero, keeping a finite surrogate;
    * zero: suppress the local count gradient at zero.

    Both are engineering extensions, identical to equation (8) for s>0. Hard
    visibility/tap masks in reconstruction are not differentiated by this rule.
    Inference uses replay_sample_layers directly and requires no reference.
    """
    if zero_policy not in ZERO_POLICIES:
        raise ValueError(f"zero_policy must be one of {ZERO_POLICIES}")
    if (expected.ndim != 4 or expected.shape[1] != 1 or min(expected.shape) < 1
            or not expected.is_floating_point() or not torch.isfinite(expected).all()
            or (expected < 0).any()):
        raise ValueError("expected must be finite nonnegative floating N1HW")
    counts = expected.round().to(torch.int64)
    image = replay_sample_layers(layers, counts)
    if expected.device != image.device or expected.dtype != image.dtype:
        raise ValueError("Expected allocations must match replay output device and floating dtype")
    if (reference.shape != image.shape or reference.device != image.device or reference.dtype != image.dtype
            or not torch.isfinite(reference).all() or (reference < 0).any()):
        raise ValueError("Reference must be finite nonnegative linear N3HW matching replay output")
    result = _RendererGradient.apply(image, expected, reference, counts, zero_policy)
    return DifferentiableRender(result, counts, zero_policy)
