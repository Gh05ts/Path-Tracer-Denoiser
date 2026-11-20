import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.temporal import apply_kernel, combine_scales, reproject_history


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5)])
def test_identity_warp_including_singleton_axes(shape, device):
    image = torch.rand(2, 3, *shape, dtype=torch.double, device=device)
    warped, valid = reproject_history(image, torch.zeros(2, 2, *shape, dtype=torch.double, device=device),
                                      torch.ones(2, 1, *shape, device=device))
    torch.testing.assert_close(warped, image)
    assert valid.all()


def test_backward_motion_sign_fractional_filter_and_bounds(device):
    image = torch.tensor([[[[0., 2., 4., 6.]]]], device=device)
    motion = torch.zeros(1, 2, 1, 4, device=device)
    motion[:, 0] = .5
    warped, valid = reproject_history(image, motion, torch.ones(1, 1, 1, 4, device=device))
    torch.testing.assert_close(warped, torch.tensor([[[[1., 3., 5., 0.]]]], device=device))
    assert valid.flatten().tolist() == [True, True, True, False]


def test_reset_and_nonfinite_invalid_motion(device):
    image = torch.ones(1, 3, 2, 2, requires_grad=True, device=device)
    motion = torch.full((1, 2, 2, 2), float("nan"), device=device)
    warped, valid = reproject_history(image, motion, torch.zeros(1, 1, 2, 2, device=device))
    assert not valid.any()
    assert torch.equal(warped, torch.zeros_like(warped))
    warped.sum().backward()
    assert torch.equal(image.grad, torch.zeros_like(image))


def test_kernel_preserves_hdr_constants_at_borders(device):
    image = torch.full((2, 3, 3, 4), 400., device=device)
    output = apply_kernel(image, torch.randn(2, 25, 3, 4, device=device))
    torch.testing.assert_close(output, image)


def test_kernel_offset_order_and_masked_renormalization(device):
    image = torch.arange(9., device=device).reshape(1, 1, 3, 3)
    logits = torch.full((1, 9, 3, 3), -1000., device=device)
    logits[:, 5] = 0  # One pixel right of center.
    assert apply_kernel(image, logits)[0, 0, 1, 1] == 5
    valid = torch.zeros_like(image, dtype=torch.bool)
    valid[..., 1, 1] = True
    torch.testing.assert_close(apply_kernel(image, torch.zeros_like(logits), valid=valid),
                               torch.full_like(image, 4.))
    valid.zero_()
    assert torch.equal(apply_kernel(image, logits, valid=valid), torch.zeros_like(image))


def test_scale_combination_preserves_detail_and_replaces_low_frequency(device):
    fine = torch.tensor([[[[1., 3.], [5., 7.]]]], device=device)
    coarse = torch.tensor([[[[10.]]]], device=device)
    torch.testing.assert_close(combine_scales(fine, coarse, torch.zeros(1, 1, 2, 2, device=device)), fine + 3)


def test_gradients_through_reprojection_and_predicted_kernels(device):
    history = torch.rand(1, 1, 2, 3, dtype=torch.double, requires_grad=True, device=device)
    motion = torch.full((1, 2, 2, 3), .1, dtype=torch.double, requires_grad=True, device=device)
    logits = torch.randn(1, 9, 2, 3, dtype=torch.double, requires_grad=True, device=device)
    def operation(h, m, k):
        warped, valid = reproject_history(h, m, torch.ones(1, 1, 2, 3, device=device))
        return apply_kernel(warped, k, valid=valid)
    assert torch.autograd.gradcheck(
        operation, (history, motion, logits),
        nondet_tol=1e-6 if device.type == "cuda" else 0.,
    )


def test_invalid_inputs_rejected(device):
    image = torch.ones(1, 3, 2, 2, device=device)
    with pytest.raises(ValueError, match="odd square"):
        apply_kernel(image, torch.zeros(1, 4, 2, 2, device=device))
    with pytest.raises(ValueError, match="binary"):
        reproject_history(image, torch.zeros(1, 2, 2, 2, device=device), torch.full((1, 1, 2, 2), .5, device=device))
    with pytest.raises(ValueError, match="2:1"):
        combine_scales(image, image, torch.zeros(1, 1, 2, 2, device=device))


@pytest.mark.parametrize('chunk_rows', [1, 3, 64])
@pytest.mark.parametrize('kernel_size', [1, 3, 5])
def test_striped_kernel_matches_whole_frame_and_bounds_unfold(device, monkeypatch, chunk_rows, kernel_size):
    torch.manual_seed(73)
    image = torch.rand(2, 3, 13, 7, device=device, dtype=torch.double, requires_grad=True)
    logits = torch.randn(2, kernel_size**2, 13, 7, device=device, dtype=torch.double, requires_grad=True)
    valid = torch.rand(2, 1, 13, 7, device=device) > .3
    valid[..., 3:10, :] = False  # Includes entirely unsupported neighborhoods.
    expected = apply_kernel(image, logits, valid=valid)
    unfold = torch.nn.functional.unfold
    heights = []
    def checked_unfold(value, *args, **kwargs):
        heights.append(value.shape[-2])
        return unfold(value, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'unfold', checked_unfold)
    actual = apply_kernel(image, logits, valid=valid, chunk_rows=chunk_rows)
    assert max(heights) <= min(13, chunk_rows + kernel_size - 1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_grad = torch.autograd.grad(expected.square().sum(), (image, logits))
    actual_grad = torch.autograd.grad(actual.square().sum(), (image, logits))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('chunk_rows', [-1, True, 1.5])
def test_invalid_kernel_chunk_size_rejected(chunk_rows, device):
    with pytest.raises(ValueError, match='nonnegative integer'):
        apply_kernel(torch.ones(1, 3, 2, 2, device=device),
                     torch.zeros(1, 25, 2, 2, device=device), chunk_rows=chunk_rows)
