"""Sparse reconstruction and availability tests against independent oracles."""

import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.adaptive_model import (
    AdaptiveNTASADDenoiser, AdaptiveReconstruction, _combine, _weighted_downsample,
)
from wavefront_denoiser.checkpoint import save_checkpoint
from wavefront_denoiser.model import NTASADDenoiser


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def frame(rgb, count=4):
    mask = torch.ones_like(rgb[:, :1])
    return {"radiance":rgb,"albedo":torch.full_like(rgb,.5),"normal":torch.zeros_like(rgb),
            "depth":mask,"guide_valid":mask.clone(),"history_valid":mask.clone(),
            "motion":torch.zeros_like(rgb[:,:2]),"sample_count":torch.full_like(mask,count,dtype=torch.int64)}


@pytest.mark.parametrize("shape", [(1,1),(1,7),(7,1),(9,13),(32,64)])
def test_uniform_parity_with_shared_backbone_weights(device,shape):
    torch.manual_seed(2)
    uniform = NTASADDenoiser(4).to(device)
    adaptive = AdaptiveNTASADDenoiser(4).to(device)
    adaptive.network.load_state_dict(uniform.state_dict())
    rgb = torch.rand(2,3,*shape,device=device)*20+1
    buffers = frame(rgb)
    history = torch.rand_like(rgb)*20+1
    state = AdaptiveReconstruction(history,torch.ones_like(rgb[:,:1],dtype=torch.bool))
    torch.testing.assert_close(adaptive(buffers,state).radiance,uniform(buffers,history),rtol=5e-6,atol=1e-5)
    torch.testing.assert_close(adaptive.forward_spatial(buffers).radiance,uniform.forward_spatial(buffers),rtol=5e-6,atol=1e-5)
    initialized = adaptive(buffers)
    torch.testing.assert_close(initialized.radiance,uniform(buffers))
    assert initialized.valid.all()


def test_dense_nonuniform_counts_match_paper_image_pyramid(device):
    # Section 2.1 downsamples the pixel-mean image, not a new sample estimator.
    # Compare the complete reconstruction to the independent dense box-pyramid
    # implementation, with the same learned kernels and strongly unequal SPP.
    torch.manual_seed(53)
    uniform = NTASADDenoiser(4).to(device).double()
    model = AdaptiveNTASADDenoiser(4).to(device).double()
    legacy = AdaptiveNTASADDenoiser(4,reconstruction_version="sample_mass_v1").to(device).double()
    model.network.load_state_dict(uniform.state_dict())
    legacy.network.load_state_dict(uniform.state_dict())
    rgb = torch.rand(1,3,9,13,device=device,dtype=torch.double)*20+.01
    buffers = frame(rgb)
    buffers["sample_count"][...,::2,::2] = 63
    buffers["sample_count"][...,1::2,1::2] = 1
    expected = uniform.forward_spatial(buffers)
    torch.testing.assert_close(model.forward_spatial(buffers).radiance,expected,rtol=1e-12,atol=1e-12)
    assert (legacy.forward_spatial(buffers).radiance-expected).abs().max() > 1e-3


def test_sparse_pyramid_weights_pixels_equally_regardless_of_positive_spp(device):
    torch.manual_seed(29)
    model = AdaptiveNTASADDenoiser(4).to(device)
    rgb = torch.rand(1,3,7,9,device=device)*100+.1
    buffers = frame(rgb)
    buffers["sample_count"][...,::2,::2] = 0
    first = model.forward_spatial(buffers)
    buffers["sample_count"][...,1::2,:] = 63
    second = model.forward_spatial(buffers)
    torch.testing.assert_close(first.radiance,second.radiance,rtol=0,atol=0)
    assert torch.equal(first.valid,second.valid)
    # Unequal block coverage must carry through a second downsample: three
    # available original pixels [2, 4, 18] have mean 8, not mean-of-blocks 10.5.
    image = torch.zeros(1,1,4,4,device=device,dtype=torch.double)
    image[...,0,0],image[...,0,1],image[...,2,2] = 2,4,18
    weight = (image > 0).double()
    first_image,coverage = _weighted_downsample(image,weight)
    value,coverage = _weighted_downsample(first_image,coverage)
    assert value.item() == 8 and coverage.item() == 3/16


def test_unknown_reconstruction_semantics_are_rejected():
    with pytest.raises(ValueError,match="reconstruction_version"):
        AdaptiveNTASADDenoiser(4,reconstruction_version="unknown")


def test_weighted_pyramid_ignores_missing_values_and_preserves_sample_mass(device):
    image = torch.tensor([[[[2.,100000.],[10.,200000.]]]],device=device,dtype=torch.double)
    mass = torch.tensor([[[[1.,0.],[3.,0.]]]],device=device,dtype=torch.double)
    value, weight = _weighted_downsample(image,mass)
    assert value.item() == 8 and weight.item() == 1
    value, weight = _weighted_downsample(image,torch.zeros_like(mass))
    assert value.item() == 0 and weight.item() == 0
    # An average mass below one is still real support, not a denominator of one.
    value, weight = _weighted_downsample(image,torch.tensor([[[[1.,0.],[0.,0.]]]],device=device,dtype=torch.double))
    assert value.item() == 2 and weight.item() == .25


def test_scale_fallbacks_do_not_blend_against_missing_black(device):
    fine = torch.tensor([[[[2.,999.],[6.,999.]]]],device=device)
    valid = torch.tensor([[[[True,False],[True,False]]]],device=device)
    coarse = torch.tensor([[[[10.]]]],device=device)
    output,support = _combine(fine,valid,coarse,torch.ones_like(coarse,dtype=torch.bool),torch.zeros_like(fine))
    torch.testing.assert_close(output,torch.tensor([[[[5.,10.],[9.,10.]]]],device=device))
    assert support.all()
    output,support = _combine(fine,valid,coarse,torch.zeros_like(coarse,dtype=torch.bool),torch.zeros_like(fine))
    torch.testing.assert_close(output,torch.tensor([[[[2.,0.],[6.,0.]]]],device=device))
    assert torch.equal(support,valid)


@pytest.mark.parametrize("value", [0.,100.,65535.])
def test_sparse_constants_preserved_at_holes_and_no_placeholder_gradients(device,value):
    torch.manual_seed(6)
    model = AdaptiveNTASADDenoiser(4).to(device)
    rgb = torch.full((1,3,9,13),value,device=device,requires_grad=True)
    buffers = frame(rgb)
    buffers["sample_count"][...,::2,::2] = 0
    output = model.forward_spatial(buffers)
    assert output.valid.all()
    torch.testing.assert_close(output.radiance,rgb,rtol=1e-5,atol=1e-4)
    output.radiance.mean().backward()
    assert torch.isfinite(rgb.grad).all()
    assert (rgb.grad[...,::2,::2] == 0).all()
    changed = frame(rgb.detach().clone())
    changed["sample_count"] = buffers["sample_count"]
    changed["radiance"][...,::2,::2] = 12345
    torch.testing.assert_close(model.forward_spatial(changed).radiance,output.radiance)


def test_empty_current_frame_reuses_valid_history_without_attenuation(device):
    model = AdaptiveNTASADDenoiser(4).to(device)
    rgb = torch.full((1,3,7,9),100.,device=device)
    buffers = frame(torch.zeros_like(rgb),count=0)
    state = AdaptiveReconstruction(rgb,torch.ones_like(rgb[:,:1],dtype=torch.bool))
    result = model(buffers,state)
    assert result.valid.all()
    torch.testing.assert_close(result.radiance,rgb,rtol=5e-6,atol=1e-5)
    buffers["history_valid"].zero_()
    result = model(buffers,state)
    assert not result.valid.any() and (result.radiance == 0).all()


def test_empty_still_and_unavailable_history_remain_unavailable(device):
    model = AdaptiveNTASADDenoiser(4).to(device)
    rgb = torch.full((1,3,7,9),5000.,device=device,requires_grad=True)
    buffers = frame(rgb,count=0)
    result = model.forward_spatial(buffers)
    assert not result.valid.any() and (result.radiance == 0).all()
    result.radiance.sum().backward()
    assert (rgb.grad == 0).all()
    state = AdaptiveReconstruction(rgb.detach(),result.valid)
    result = model(buffers,state)
    assert not result.valid.any() and (result.radiance == 0).all()


def test_fractional_history_warp_normalizes_only_available_taps(device):
    rgb = torch.zeros(1,3,1,3,device=device,dtype=torch.double)
    buffers = frame(rgb,count=0)
    buffers["motion"][:,0] = .5
    history = AdaptiveReconstruction(
        torch.tensor([[[[2.,10000.,8.]]]],device=device,dtype=torch.double).expand(1,3,1,3),
        torch.tensor([[[[True,False,True]]]],device=device))
    warped,valid = AdaptiveNTASADDenoiser._warp(history,buffers,rgb)
    torch.testing.assert_close(warped,torch.tensor([[[[2.,8.,0.]]]],device=device,dtype=torch.double).expand_as(rgb))
    assert valid.flatten().tolist() == [True,True,False]


def test_available_history_warp_gradients_through_rgb_and_motion(device):
    from wavefront_denoiser.temporal import reproject_available_history
    history = torch.rand(1,3,3,4,device=device,dtype=torch.double,requires_grad=True)
    motion = torch.full((1,2,3,4),.17,device=device,dtype=torch.double,requires_grad=True)
    available = torch.tensor([[[[True,False,True,False],[False,True,False,True],
                                [True,False,True,False]]]],device=device)
    mask = torch.ones(1,1,3,4,device=device,dtype=torch.double)
    assert torch.autograd.gradcheck(
        lambda h,m: reproject_available_history(h,m,mask,available)[0], (history,motion),
        nondet_tol=1e-6 if device.type == "cuda" else 0.)


def test_sparse_initialization_reset_and_bad_history_rejected(device):
    model = AdaptiveNTASADDenoiser(4).to(device)
    buffers = frame(torch.ones(1,3,5,7,device=device),count=0)
    with pytest.raises(ValueError,match="dense uniform"):
        model(buffers)
    buffers["sample_count"].fill_(4)
    with pytest.raises(ValueError,match="availability"):
        model(buffers,torch.zeros_like(buffers["radiance"]))
    bad = AdaptiveReconstruction(torch.zeros_like(buffers["radiance"]),torch.ones_like(buffers["depth"]))
    with pytest.raises(ValueError,match="boolean"):
        model(buffers,bad)
    buffers["sample_count"] = buffers["sample_count"].float()
    with pytest.raises(ValueError,match="integer"):
        model.forward_spatial(buffers)


def test_adaptive_model_cannot_be_mislabeled_as_uniform_checkpoint(tmp_path):
    with pytest.raises(ValueError,match="NTASADDenoiser"):
        save_checkpoint(tmp_path / "wrong.pt",AdaptiveNTASADDenoiser(4))
    assert not (tmp_path / "wrong.pt").exists()


def test_sequence_streaming_parity_resets_and_gradient_boundaries(device):
    torch.manual_seed(3)
    model = AdaptiveNTASADDenoiser(4).to(device)
    rgbs = (torch.rand(2,5,3,7,9,device=device)+1).requires_grad_()
    frames = [frame(rgbs[:,i]) for i in range(5)]
    frames[1]["sample_count"][...,::2,::2] = 0
    buffers = {key:torch.stack([f[key] for f in frames],1) for key in frames[0]}
    reset = torch.zeros(2,5,device=device,dtype=torch.bool)
    reset[0,2] = True
    result = model.forward_sequence(buffers,reset=reset)
    history = None
    for index,current in enumerate(frames):
        actual = model(current,history)
        if reset[:,index].any():
            initialized = model(current)
            selected = reset[:,index,None,None,None]
            actual = AdaptiveReconstruction(torch.where(selected,initialized.radiance,actual.radiance),
                                             torch.where(selected,initialized.valid,actual.valid))
        torch.testing.assert_close(result.radiance[:,index],actual.radiance,rtol=5e-6,atol=1e-5)
        assert torch.equal(result.valid[:,index],actual.valid)
        history = actual
    result.radiance[:,-1].mean().backward()
    assert torch.isfinite(rgbs.grad).all()
    assert (rgbs.grad[0,:2] == 0).all() and rgbs.grad[1,0].abs().sum() > 0
    assert model.network.fine_head.weight.grad.abs().sum() > 0
    buffers["sample_count"][0,2,0,0,0] = 0
    with pytest.raises(ValueError,match="dense uniform"):
        model.forward_sequence(buffers,reset=reset)


def test_sampler_uses_history_availability_without_changing_feature_width(device):
    from wavefront_denoiser.adaptive import AdaptiveSampler
    model = AdaptiveSampler(4).to(device)
    rgb = torch.full((1,3,1,3),9.,device=device,requires_grad=True)
    buffers = frame(rgb)
    available = torch.zeros_like(rgb[:,:1],dtype=torch.bool)
    features = model.features(buffers,rgb,history_available=available)
    assert features.shape[1] == 8 and (features[:,-1] == 0).all()
    features.sum().backward()
    assert (rgb.grad == 0).all()
    with pytest.raises(ValueError,match="requires explicit"):
        model(buffers,history_available=available)


def test_final_sequence_loss_updates_sampler_and_denoiser_through_replay(device):
    from wavefront_denoiser.adaptive import AdaptiveSampler
    from wavefront_denoiser.renderer_gradient import differentiable_replay
    from wavefront_denoiser.sample_layers import replay_sample_layers
    from wavefront_denoiser.loss import sequence_loss

    torch.manual_seed(17)
    model = AdaptiveNTASADDenoiser(4).to(device)
    sampler = AdaptiveSampler(4).to(device)
    optimizer = torch.optim.Adam(list(model.parameters())+list(sampler.parameters()),lr=.001)
    layers = torch.rand(1,5,6,3,7,9,device=device)*3+.2
    reference = torch.rand(1,5,3,7,9,device=device)*2+1
    initial_counts = torch.full((1,1,7,9),4,device=device,dtype=torch.int64)
    state = model(frame(replay_sample_layers(layers[:,0],initial_counts)))
    outputs = [state.radiance]
    allocations = []
    for index in range(1,5):
        guides = frame(torch.zeros_like(state.radiance))
        allocation = sampler(guides,state.radiance,history_available=state.valid)
        allocation.expected.retain_grad()
        allocations.append(allocation.expected)
        render = differentiable_replay(layers[:,index],allocation.expected,reference[:,index],
                                       zero_policy="unit_denominator")
        current = dict(guides,radiance=render.radiance,sample_count=render.counts)
        state = model(current,state)
        outputs.append(state.radiance)
    loss = sequence_loss(torch.stack(outputs,1),reference)["loss"]
    loss.backward()
    for value in allocations:
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    before = [sampler.head.weight.detach().clone(), model.network.fine_head.weight.detach().clone()]
    for weight in (sampler.head.weight,model.network.fine_head.weight):
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before[0],sampler.head.weight)
    assert not torch.equal(before[1],model.network.fine_head.weight)
