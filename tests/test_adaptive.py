"""Paper equations and independent allocation/replay oracles; no captured data."""

import math

import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.adaptive import AdaptiveSampler, FEATURE_CHANNELS, normalize_sample_map
from wavefront_denoiser.ntasad_radiance import encode_radiance
from wavefront_denoiser.sample_layers import LAYER_SPP, replay_sample_layers


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def guides(h=5, w=7, n=1, dtype=torch.float32):
    return {"albedo": torch.full((n, 3, h, w), .5, dtype=dtype),
            "normal": torch.full((n, 3, h, w), -.3, dtype=dtype),
            "depth": torch.ones(n, 1, h, w, dtype=dtype),
            "motion": torch.zeros(n, 2, h, w, dtype=dtype),
            "history_valid": torch.ones(n, 1, h, w, dtype=dtype)}


def test_equation2_spatial_softmax_and_independent_batch_normalization():
    logits = torch.tensor([[[[0., math.log(2), math.log(3)]]],
                           [[[100., 100., 100.]]]], dtype=torch.double)
    allocation = normalize_sample_map(logits, 4)
    torch.testing.assert_close(allocation.expected,
        torch.tensor([[[[2., 4., 6.]]], [[[4., 4., 4.]]]], dtype=torch.double))
    assert allocation.counts.tolist() == [[[[2, 4, 6]]], [[[4, 4, 4]]]]
    assert allocation.counts.dtype == torch.int64 and not allocation.counts.requires_grad
    torch.testing.assert_close(normalize_sample_map(logits + 1000, 4).expected, allocation.expected)


def test_equation2_allows_zero_and_exposes_rounding_drift():
    # The expected total is 3, but separate rounding yields 4. No budget repair.
    allocation = normalize_sample_map(torch.zeros(1, 1, 1, 2), 1.5)
    assert allocation.expected.sum() == 3 and allocation.counts.sum() == 4
    sparse = normalize_sample_map(torch.tensor([[[[1000., -1000., -1000.]]]]), 4)
    assert sparse.counts.tolist() == [[[[12, 0, 0]]]]


def test_normalization_gradients_match_softmax_jacobian():
    logits = torch.tensor([[[[.2, -.7, .5]]]], dtype=torch.double, requires_grad=True)
    result = normalize_sample_map(logits, 4).expected
    weights = torch.tensor([[[[1., 3., -2.]]]], dtype=torch.double)
    (result * weights).sum().backward()
    probabilities = result.detach() / 12
    expected = 12 * probabilities * (weights - (weights * probabilities).sum())
    torch.testing.assert_close(logits.grad, expected)
    assert abs(float(logits.grad.sum())) < 1e-12
    assert torch.autograd.gradcheck(lambda x: normalize_sample_map(x, 4).expected, (logits,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_logits_and_accumulation_dtype(dtype):
    allocation = normalize_sample_map(torch.tensor([[[[10000., 10001., -10000.]]]], dtype=dtype), 4)
    assert torch.isfinite(allocation.expected).all()
    torch.testing.assert_close(allocation.expected.sum(), allocation.expected.new_tensor(12))
    assert allocation.expected.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "4", 2**33])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        normalize_sample_map(torch.zeros(1, 1, 2, 3), budget)


@pytest.mark.parametrize("logits", [torch.zeros(1, 2, 2, 2), torch.zeros(1, 1, 0, 2),
    torch.ones(1, 1, 2, 2, dtype=torch.int64), torch.full((1, 1, 1, 1), float("nan"))])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError):
        normalize_sample_map(logits, 4)


def test_sampler_features_grayscale_radiance_space_and_backward_warp():
    model = AdaptiveSampler(4).double()
    g = guides(1, 3, dtype=torch.double)
    g["motion"][:, 0] = .5
    history = torch.tensor([[[[0., 2., 8.]], [[2., 4., 16.]], [[4., 8., 32.]]]], dtype=torch.double)
    features = model.features(g, history)
    assert FEATURE_CHANNELS == 8 and features.shape == (1, 8, 1, 3)
    torch.testing.assert_close(features[:, 0], torch.full((1, 1, 3), .5 * .9999, dtype=torch.double))
    torch.testing.assert_close(features[:, 1:4], g["normal"])
    torch.testing.assert_close(features[:, 4:5], g["depth"])
    torch.testing.assert_close(features[:, 5:7], g["motion"])
    color = encode_radiance(history)
    weights = torch.tensor([.2989, .587, .114], dtype=torch.double)
    expected = ((color[0, :, 0, :2] + color[0, :, 0, 1:]) * .5 * weights[:, None]).sum(0)
    torch.testing.assert_close(features[0, -1, 0, :2], expected)
    assert features[0, -1, 0, -1] == 0  # History outside frame.


def test_invalid_history_and_motion_cannot_reach_features():
    model = AdaptiveSampler(4)
    g = guides()
    g["history_valid"].zero_()
    g["motion"].fill_(float("nan"))
    history = torch.rand(1, 3, 5, 7, requires_grad=True)
    features = model.features(g, history)
    assert torch.isfinite(features).all() and (features[:, 5:] == 0).all()
    features.sum().backward()
    assert (history.grad == 0).all()


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (32, 64)])
def test_sampler_initialization_shapes_padding_and_no_target_leakage(shape):
    h, w = shape
    model = AdaptiveSampler(4)
    g = guides(h, w, n=2)
    # Deliberately unusable extras must never be read by a pre-render sampler.
    g.update(radiance=object(), reference=object(), sample_count=object())
    initialized = model(g, average_spp=4)
    assert (initialized.counts == 4).all()
    history = torch.rand(2, 3, h, w)
    result = model(g, history, average_spp=4)
    assert result.expected.shape == (2, 1, h, w)
    torch.testing.assert_close(result.expected.sum((-2, -1)), torch.full((2, 1), float(4*h*w)))
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            assert layer.kernel_size == (3, 3) and (layer.bias == 0).all()


def test_default_sampler_topology_and_history_gradients():
    large = AdaptiveSampler()
    assert large.encoder0[0].in_channels == 8 and large.base_channels == 48
    assert len(large.down) == 5 and large.head.out_channels == 1
    assert [block[0].out_channels for block in large.decoders] == [96, 96, 96, 96, 64]
    torch.manual_seed(17)
    model = AdaptiveSampler(4)
    history = torch.rand(1, 3, 7, 9, requires_grad=True)
    output = model(guides(7, 9), history)
    (output.expected * torch.arange(63.).reshape(1, 1, 7, 9)).sum().backward()
    assert torch.isfinite(history.grad).all() and history.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0


def test_sampler_rejects_incompatible_history_and_guides():
    model = AdaptiveSampler(4)
    g = guides()
    with pytest.raises(ValueError, match="Missing"):
        model({})
    with pytest.raises(ValueError, match="History"):
        model(g, torch.zeros(1, 3, 2, 2))
    g["history_valid"].fill_(.5)
    with pytest.raises(ValueError, match="binary"):
        model(g)


def test_sampler_accepts_unified_denoiser_history_and_preserves_gradients():
    # V4 renamed/unified the model and added a still-image spatial entry point.
    # Exercise the real interface, including gradients into both networks.
    from wavefront_denoiser.model import NTASADDenoiser

    torch.manual_seed(17)
    denoiser = NTASADDenoiser(4)
    sampler = AdaptiveSampler(4)
    g = guides(7, 9)
    frame = dict(g, radiance=torch.rand(1, 3, 7, 9) + 1,
                 guide_valid=torch.ones(1, 1, 7, 9),
                 sample_count=torch.full((1, 1, 7, 9), 4, dtype=torch.int64))
    history = denoiser.forward_spatial(frame)
    allocation = sampler(g, history)
    torch.testing.assert_close(allocation.expected.sum(), torch.tensor(4. * 7 * 9))
    (allocation.expected * torch.arange(63.).reshape(1, 1, 7, 9)).sum().backward()
    for weight in (sampler.head.weight, denoiser.fine_head.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0


def test_replay_all_representable_counts_against_explicit_sample_pool():
    # Build independent samples, then aggregate the six DISJOINT chunks.
    torch.manual_seed(9)
    samples = torch.rand(1, 63, 3, 1, 64, dtype=torch.double)
    chunks = samples.split(LAYER_SPP, dim=1)
    layers = torch.stack([chunk.mean(1) for chunk in chunks], dim=1)
    counts = torch.arange(64).reshape(1, 1, 1, 64)
    result = replay_sample_layers(layers, counts)
    for count in range(64):
        chosen = [chunk[..., count] for spp, chunk in zip(LAYER_SPP, chunks) if count & spp]
        expected = torch.cat(chosen, dim=1).mean(1) if chosen else torch.zeros(1, 3, 1, dtype=torch.double)
        torch.testing.assert_close(result[..., count], expected)


def test_replay_13_spp_weighted_means_zero_and_gradients():
    layers = torch.arange(1., 7., dtype=torch.double).reshape(1, 6, 1, 1, 1).expand(1, 6, 3, 1, 2).clone()
    layers.requires_grad_()
    result = replay_sample_layers(layers, torch.tensor([[[[13, 0]]]]))
    torch.testing.assert_close(result[..., 0], torch.full((1, 3, 1), (1 + 3*4 + 4*8)/13, dtype=torch.double))
    assert (result[..., 1] == 0).all()
    result.sum().backward()
    expected = torch.tensor([1/13, 0, 4/13, 8/13, 0, 0], dtype=torch.double)
    torch.testing.assert_close(layers.grad[0, :, 0, 0, 0], expected)
    assert (layers.grad[..., 1] == 0).all()


@pytest.mark.parametrize("count", [-1, 64, 2.5])
def test_replay_rejects_unsupported_counts_instead_of_clipping(count):
    with pytest.raises(ValueError, match="counts"):
        replay_sample_layers(torch.ones(1, 6, 3, 1, 1), torch.tensor([[[[count]]]]))
