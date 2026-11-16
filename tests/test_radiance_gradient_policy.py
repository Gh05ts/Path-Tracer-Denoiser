import math
import pytest
import torch

from wavefront_denoiser.ntasad_radiance import encode_radiance, radiance_gradient_policy, GAMMA
from wavefront_denoiser.adaptive import normalize_sample_map
from wavefront_denoiser.renderer_gradient import differentiable_replay


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_policy_keeps_exact_forward_and_bounds_only_near_black_derivatives(dtype):
    x = torch.tensor([0., 1e-40, 1e-12, 1e-6, 1., 65535., 1e6], dtype=dtype, requires_grad=True)
    baseline = encode_radiance(x).detach()
    with radiance_gradient_policy(1e-6):
        y = encode_radiance(x)
        torch.testing.assert_close(y, baseline, rtol=0, atol=0)
        y.sum().backward()
    oracle = (1/GAMMA)*torch.log1p(x.detach().clamp(0,65535)).clamp_min(math.log1p(1e-6)).pow(1/GAMMA-1)/(1+x.detach().clamp(0,65535))
    oracle[0] = oracle[-1] = 0
    torch.testing.assert_close(x.grad, oracle)
    assert torch.isfinite(x.grad).all()


def test_policy_restores_after_exception_and_does_not_change_inference():
    x = torch.tensor([1e-40], requires_grad=True)
    original = torch.autograd.grad(encode_radiance(x).sum(), x)[0]
    with pytest.raises(RuntimeError):
        with radiance_gradient_policy(1e-6):
            derivative = torch.autograd.grad(encode_radiance(x).sum(), x)[0]
            assert derivative < original
            raise RuntimeError("fixture")
    restored = torch.autograd.grad(encode_radiance(x).sum(), x)[0]
    torch.testing.assert_close(restored, original, rtol=0, atol=0)


@pytest.mark.parametrize("floor", [-1,2,1e-100,float("nan"),float("inf"),True,"1e-6"])
def test_policy_rejects_invalid_floor(floor):
    with pytest.raises(ValueError):
        with radiance_gradient_policy(floor):
            pass


def test_near_black_recurrent_renderer_estimator_stays_finite_with_policy():
    def run(floor):
        logits = torch.zeros(1,1,2,3, requires_grad=True)
        layers = torch.full((1,6,3,2,3), 1e-40)
        reference = torch.arange(1.,7.).reshape(1,1,2,3).expand(1,3,2,3)
        state = logits.expand(1,3,2,3)*0
        with radiance_gradient_policy(floor):
            for _ in range(4):
                allocation = normalize_sample_map(logits+state.mean(1,keepdim=True),4)
                render = differentiable_replay(layers, allocation.expected, reference, zero_policy="unit_denominator")
                state = encode_radiance(render.radiance)
            (state*reference).mean().backward()
        return logits.grad
    assert not torch.isfinite(run(0.)).all()
    stabilized = run(1e-6)
    assert torch.isfinite(stabilized).all() and stabilized.abs().sum() > 0
