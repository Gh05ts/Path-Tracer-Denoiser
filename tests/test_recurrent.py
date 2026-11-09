import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.recurrent import TemporalKernelDenoiser
from wavefront_denoiser.ntasad_radiance import encode_radiance, decode_radiance
from wavefront_denoiser.sequences import SequenceDataset


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def frame(rgb):
    mask = torch.ones_like(rgb[:, :1])
    normal = torch.zeros_like(rgb)
    normal[:, 2] = 1
    return {"radiance": rgb, "albedo": torch.full_like(rgb, .5), "normal": normal,
            "depth": mask, "guide_valid": mask.clone(), "history_valid": mask.clone(),
            "motion": torch.zeros_like(rgb[:, :2]),
            "sample_count": torch.full_like(mask, 4, dtype=torch.int64)}


def sequence(rgb):
    frames = [frame(rgb[:, t]) for t in range(rgb.shape[1])]
    return {name: torch.stack([f[name] for f in frames], dim=1) for name in frames[0]}


def center_kernels(model):
    """Choose center taps to isolate temporal blending and reset behavior."""
    with torch.no_grad():
        for head in (model.coarse_head, model.middle_head, model.fine_head):
            head.weight.zero_()
            head.bias.fill_(-40)
            head.bias[12] = 40
            head.bias[-1] = 0
        model.fine_head.bias[37] = 40  # Center of temporal kernel.
    return model


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (9, 13), (16, 24)])
def test_full_model_shapes_hdr_and_backward(shape, device):
    model = TemporalKernelDenoiser(4).to(device)
    rgb = torch.full((2, 3, *shape), 100., device=device, requires_grad=True)
    output = model(frame(rgb), history=rgb)
    assert output.shape == rgb.shape
    assert torch.isfinite(output).all() and (output >= 0).all()
    # Gamma/log -> filtering -> inverse amplifies float32 rounding slightly.
    torch.testing.assert_close(output, rgb, rtol=5e-6, atol=1e-5)
    output.mean().backward()
    assert torch.isfinite(rgb.grad).all()


def test_first_frame_ignores_unavailable_history_and_extra_reference(device):
    model = center_kernels(TemporalKernelDenoiser(4).to(device))
    rgb = torch.rand(1, 3, 9, 11, device=device) * 20
    buffers = frame(rgb)
    buffers["reference"] = torch.full_like(rgb, float("nan"))
    buffers["motion"].fill_(float("nan"))
    torch.testing.assert_close(model(buffers), rgb)
    buffers["history_valid"].zero_()
    torch.testing.assert_close(model(buffers, torch.full_like(rgb, 1000)), rgb)


def test_temporal_blend_motion_and_disoccluded_centers(device):
    model = center_kernels(TemporalKernelDenoiser(4).to(device))
    rgb = torch.full((1, 3, 5, 7), 2., device=device)
    history = torch.arange(7., device=device).expand_as(rgb).contiguous()
    buffers = frame(rgb)
    buffers["motion"][:, 0] = 1
    buffers["history_valid"][..., 2, 2] = 0
    expected = decode_radiance((encode_radiance(rgb) + encode_radiance(history + 1)) / 2)
    expected[..., -1] = rgb[..., -1]  # History coordinate outside original image.
    expected[..., 2, 2] = rgb[..., 2, 2]  # Valid neighboring taps cannot override this.
    torch.testing.assert_close(model(buffers, history), expected)


def test_all_invalid_history_matches_spatial_path_with_random_kernels(device):
    model = TemporalKernelDenoiser(4).to(device)
    buffers = frame(torch.rand(2, 3, 7, 9, device=device))
    buffers["history_valid"].zero_()
    torch.testing.assert_close(model(buffers, torch.rand_like(buffers["radiance"]) * 100),
                               model(buffers, torch.zeros_like(buffers["radiance"])))


def test_sequence_matches_streaming_resets_without_mutation(device):
    model = TemporalKernelDenoiser(4).to(device)
    buffers = sequence(torch.rand(2, 4, 3, 9, 11, device=device))
    original_mask = buffers["history_valid"].clone()
    reset = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], device=device)
    actual = model.forward_sequence(buffers, reset=reset)
    history = None
    expected = []
    for t in range(4):
        current = {name: value[:, t] for name, value in buffers.items()}
        current["history_valid"] = current["history_valid"] * (1 - reset[:, t, None, None, None])
        history = model(current, history)
        history = torch.where(reset[:, t, None, None, None].bool(), current["radiance"], history)
        expected.append(history)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
    torch.testing.assert_close(buffers["history_valid"], original_mask)
    # Independent sequence calls cannot retain state from earlier calls.
    torch.testing.assert_close(model.forward_sequence(buffers, reset=reset), actual)


def test_later_frame_gradients_reach_earlier_frames_except_across_reset(device):
    model = TemporalKernelDenoiser(4).to(device)
    rgb = (torch.rand(2, 3, 3, 8, 8, device=device) + 1).requires_grad_()
    buffers = sequence(rgb)
    reset = torch.zeros(2, 3, device=device, dtype=torch.bool)
    reset[0, 1] = True
    model.forward_sequence(buffers, reset=reset)[:, -1].mean().backward()
    assert torch.isfinite(rgb.grad).all()
    assert rgb.grad[0, 0].abs().sum() == 0
    assert rgb.grad[1, 0].abs().sum() > 0
    assert rgb.grad[:, 1:].abs().sum() > 0
    for head in (model.coarse_head, model.middle_head, model.fine_head):
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    assert model.encoder0[0].weight.grad.abs().sum() > 0


def test_sequence_dataset_dataloader_contract(make_frame, tmp_path, device):
    for t in range(3):
        make_frame(f"frame_{t}", index=t, previous=t > 0, width=9, height=7)
    dataset = SequenceDataset(tmp_path, length=3)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    buffers = {name: value.to(device) for name, value in batch["buffers"].items()}
    model = TemporalKernelDenoiser(4).to(device)
    result = model.forward_sequence(buffers)
    assert result.shape == (1, 3, 3, 7, 9)
    assert torch.isfinite(result).all()


def test_bad_history_inputs_and_sequence_resets_rejected(device):
    model = TemporalKernelDenoiser(4).to(device)
    rgb = torch.ones(1, 3, 5, 7, device=device)
    with pytest.raises(ValueError, match="reset on resize"):
        model(frame(rgb), rgb[..., :1, :1])
    buffers = frame(rgb)
    buffers["sample_count"].zero_()
    with pytest.raises(ValueError, match="positive integer"):
        model(buffers)
    with pytest.raises(ValueError, match="Missing recurrent"):
        model({"radiance": rgb})
    with pytest.raises(ValueError, match="binary"):
        model.forward_sequence(sequence(rgb[:, None]), reset=torch.full((1, 1), .5, device=device))
    with pytest.raises(ValueError, match="NT shape"):
        model.forward_sequence(sequence(rgb[:, None]), reset=torch.zeros(2, 1, device=device))
