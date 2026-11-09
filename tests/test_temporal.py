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
