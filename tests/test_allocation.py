import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.allocation import allocate_samples
from wavefront_denoiser.adaptive import normalize_sample_map


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_capped_proportional_budget_and_independent_rows(device):
    scores = torch.tensor([[[[4.,1.,0.,-.3]]],[[[0.,0.,0.,0.]]]],device=device,dtype=torch.double,requires_grad=True)
    result = allocate_samples(scores,2,policy="capped_softmax_v1",max_spp=3)
    remaining = torch.softmax(torch.tensor([0.,-.3],dtype=torch.double,device=device),0)*2
    oracle = torch.stack((torch.cat((remaining.new_tensor([3.,3.]),remaining)),remaining.new_full((4,),2.)))
    torch.testing.assert_close(result.expected[:,0,0],oracle)
    torch.testing.assert_close(result.expected.sum((1,2,3)),scores.new_full((2,),8.))
    assert (result.counts <= 3).all() and result.counts.min() >= 0
    assert torch.autograd.gradcheck(lambda z:allocate_samples(z,2,policy="capped_softmax_v1",max_spp=3).expected,(scores,))
    (result.expected*torch.arange(4,device=device)).sum().backward()
    assert (scores.grad[0,0,0,:2] == 0).all() and scores.grad[0,0,0,2:].abs().sum() > 0


def test_bounded_matches_paper_without_overflow_and_reports_rounding(device):
    scores = torch.tensor([[[[.1,-.1,.2,-.2]]]],device=device)
    expected = normalize_sample_map(scores,2)
    actual = allocate_samples(scores,2,policy="capped_softmax_v1",max_spp=5)
    torch.testing.assert_close(actual.expected,expected.expected)
    drift = allocate_samples(torch.zeros(1,1,1,2,device=device),1.5,policy="capped_softmax_v1",max_spp=3)
    assert drift.expected.sum() == 3 and drift.counts.sum() == 4


def test_extreme_logits_redistribute_mass_and_capacity_boundary(device):
    scores = torch.tensor([[[[1e20,-1e20,-1e20,-1e20]]]],device=device,requires_grad=True)
    result = allocate_samples(scores,2,policy="capped_softmax_v1",max_spp=3)
    torch.testing.assert_close(result.expected,scores.new_tensor([[[[3.,5/3,5/3,5/3]]]]))
    full = allocate_samples(scores,3,policy="capped_softmax_v1",max_spp=3)
    full.expected.sum().backward()
    assert (scores.grad == 0).all() and (full.counts == 3).all()
    with pytest.raises(ValueError,match="capacity"):
        allocate_samples(scores.detach(),2,policy="paper_softmax_v1",max_spp=3)


@pytest.mark.parametrize("changes", [{"policy":"silent_clip"},{"max_spp":64},{"max_spp":True},
    {"average_spp":0},{"average_spp":float("nan")},{"average_spp":64}])
def test_invalid_policy_and_capacity_rejected(changes):
    options = dict(average_spp=4,policy="capped_softmax_v1",max_spp=63)
    options.update(changes)
    with pytest.raises(ValueError):
        allocate_samples(torch.zeros(1,1,2,2),**options)
