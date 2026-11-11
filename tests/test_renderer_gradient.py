"""Independent analytic oracles for the DASR estimator, not integer gradcheck."""

import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.renderer_gradient import differentiable_replay
from wavefront_denoiser.sample_layers import replay_sample_layers
from wavefront_denoiser.adaptive import normalize_sample_map
from wavefront_denoiser.ntasad_radiance import encode_radiance


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("zero_policy", ["unit_denominator", "zero"])
def test_exact_forward_independent_of_reference_and_no_target_gradients(device, zero_policy):
    torch.manual_seed(4)
    layers = torch.rand(1,6,3,1,64, device=device, dtype=torch.double, requires_grad=True)
    expected = torch.arange(64., device=device, dtype=torch.double).reshape(1,1,1,64).requires_grad_()
    reference = torch.full((1,3,1,64), 2., device=device, dtype=torch.double, requires_grad=True)
    output = differentiable_replay(layers, expected, reference, zero_policy=zero_policy)
    baseline = replay_sample_layers(layers, expected.detach().long())
    torch.testing.assert_close(output.radiance, baseline, rtol=0, atol=0)
    different = differentiable_replay(layers, expected, reference*100, zero_policy=zero_policy)
    torch.testing.assert_close(different.radiance, baseline, rtol=0, atol=0)
    output.radiance.sum().backward()
    assert reference.grad is None
    assert torch.isfinite(expected.grad).all() and torch.isfinite(layers.grad).all()
    assert output.counts.dtype == torch.int64 and not output.counts.requires_grad


@pytest.mark.parametrize("zero_policy", ["unit_denominator", "zero"])
def test_equation8_vjp_and_actual_rounded_count_denominator(device, zero_policy):
    layers = torch.arange(1.,7.,device=device,dtype=torch.double).reshape(1,6,1,1,1).expand(1,6,3,1,5).clone()
    expected = torch.tensor([[[[0.,1.2,4.3,13.2,63.]]]], device=device,dtype=torch.double,requires_grad=True)
    reference = torch.tensor([2.,5.,8.],device=device,dtype=torch.double)[None,:,None,None].expand(1,3,1,5)
    weights = torch.tensor([1.,-2.,3.],device=device,dtype=torch.double)[None,:,None,None]
    result = differentiable_replay(layers, expected, reference, zero_policy=zero_policy)
    (result.radiance*weights).sum().backward()
    expected_grad = torch.empty_like(expected)
    # Independent scalar construction of each binary mean and the RGB dot product.
    for x, count in enumerate([0,1,4,13,63]):
        total = sum(spp*(i+1) for i,spp in enumerate([1,2,4,8,16,32]) if count & spp)
        image = total/count if count else 0.
        derivative = sum(w*(target-image) for w,target in zip([1.,-2.,3.],[2.,5.,8.])) / max(count,1)
        expected_grad[0,0,0,x] = 0 if count == 0 and zero_policy == "zero" else derivative
    torch.testing.assert_close(expected.grad, expected_grad)


def test_positive_count_derivative_matches_independent_append_sample_limit(device):
    layers = torch.full((1,6,3,1,3), 2., device=device,dtype=torch.double)
    expected = torch.tensor([[[[1.,13.,63.]]]],device=device,dtype=torch.double,requires_grad=True)
    reference = torch.full((1,3,1,3),8.,device=device,dtype=torch.double)
    result = differentiable_replay(layers,expected,reference,zero_policy="unit_denominator")
    result.radiance.sum().backward()
    for x, count in enumerate([1.,13.,63.]):
        # Eq. (3): append an independent batch whose ensemble mean is the target.
        h = 1e-6
        extended = (count*2 + h*8)/(count+h)
        finite_difference = 3*(extended-2)/h
        assert float(expected.grad[0,0,0,x]) == pytest.approx(finite_difference, rel=2e-6)


def test_exact_layer_value_gradients_are_preserved(device):
    layers = torch.rand(1,6,3,2,3,device=device,dtype=torch.double,requires_grad=True)
    ordinary = layers.detach().clone().requires_grad_()
    expected = torch.tensor([[[[0.,1.,2.],[4.,13.,63.]]]],device=device,dtype=torch.double,requires_grad=True)
    reference = torch.ones(1,3,2,3,device=device,dtype=torch.double)
    weights = torch.rand_like(reference)
    (differentiable_replay(layers,expected,reference,zero_policy="zero").radiance*weights).sum().backward()
    (replay_sample_layers(ordinary,expected.detach().long())*weights).sum().backward()
    torch.testing.assert_close(layers.grad,ordinary.grad,rtol=0,atol=0)


def test_chain_rule_through_softmax_and_radiance_transform(device):
    logits = torch.tensor([[[[.1,-.1,.2]]]],device=device,dtype=torch.double,requires_grad=True)
    allocation = normalize_sample_map(logits,4)
    layers = torch.full((1,6,3,1,3),2.,device=device,dtype=torch.double)
    reference = torch.tensor([[[[1.,5.,9.]]]],device=device,dtype=torch.double).expand(1,3,1,3)
    result = differentiable_replay(layers,allocation.expected,reference,zero_policy="unit_denominator")
    encode_radiance(result.radiance).sum().backward()
    power = 1/2.2
    transform_derivative = power * torch.log(torch.tensor(3.,dtype=torch.double,device=device))**(power-1) / 3
    count_grad = 3 * transform_derivative * (reference[:, :1]-2) / result.counts
    probabilities = allocation.expected.detach() / 12
    oracle = 12*probabilities*(count_grad-(probabilities*count_grad).sum())
    torch.testing.assert_close(logits.grad,oracle)


@pytest.mark.parametrize("count", [-.1,64.,float("nan"),float("inf")])
def test_invalid_allocations_rejected_without_clipping(device,count):
    with pytest.raises(ValueError):
        differentiable_replay(torch.ones(1,6,3,1,1,device=device),torch.full((1,1,1,1),count,device=device),
                              torch.ones(1,3,1,1,device=device),zero_policy="zero")


def test_invalid_policy_reference_and_shape_rejected(device):
    layers = torch.ones(1,6,3,1,1,device=device)
    expected = torch.ones(1,1,1,1,device=device)
    reference = torch.ones(1,3,1,1,device=device)
    with pytest.raises(ValueError,match="zero_policy"):
        differentiable_replay(layers,expected,reference,zero_policy="implicit")
    with pytest.raises(ValueError,match="Reference"):
        differentiable_replay(layers,expected,reference*float("nan"),zero_policy="zero")
    with pytest.raises(ValueError):
        differentiable_replay(layers,expected.expand(1,1,1,2),reference,zero_policy="zero")


def test_zero_match_and_hdr_black_gradients_are_finite(device):
    layers = torch.zeros(1,6,3,1,3,device=device,requires_grad=True)
    layers.data[...,2] = 65535.
    expected = torch.tensor([[[[0.,4.,63.]]]],device=device,requires_grad=True)
    reference = torch.zeros(1,3,1,3,device=device)
    reference[...,0] = 100
    reference[...,2] = 65535.
    result = differentiable_replay(layers,expected,reference,zero_policy="unit_denominator")
    result.radiance.sum().backward()
    assert torch.isfinite(expected.grad).all() and torch.isfinite(layers.grad).all()
    assert expected.grad[0,0,0,0] == 300
    assert (expected.grad[...,1:] == 0).all()
