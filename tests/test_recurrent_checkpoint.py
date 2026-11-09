import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.recurrent import TemporalKernelDenoiser
from wavefront_denoiser.recurrent_checkpoint import (
    load_recurrent_checkpoint, restore_recurrent_training, save_recurrent_checkpoint,
)
from wavefront_denoiser.spatial_checkpoint import load_spatial_checkpoint


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def buffers(device, dtype):
    rgb = torch.rand(1, 3, 7, 9, device=device, dtype=dtype)
    mask = torch.ones_like(rgb[:, :1])
    return {"radiance": rgb, "albedo": torch.ones_like(rgb), "normal": torch.zeros_like(rgb),
            "depth": mask, "guide_valid": mask, "history_valid": mask,
            "sample_count": torch.full_like(mask, 4, dtype=torch.int64),
            "motion": torch.zeros_like(rgb[:, :2])}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inference_roundtrip_portable_cpu_weights(tmp_path, device, dtype):
    model = TemporalKernelDenoiser(4).to(device=device, dtype=dtype).eval()
    frame = buffers(device, dtype)
    with torch.no_grad():
        expected = model(frame)
    path = tmp_path / "model.pt"
    save_recurrent_checkpoint(path, model, epoch=3, global_step=12,
                              config={"patch_size": 64}, dataset_split={"train": ["scene_a"]})
    loaded, payload = load_recurrent_checkpoint(path, device)
    assert not loaded.training
    assert next(loaded.parameters()).dtype == dtype
    assert payload["epoch"] == 3 and payload["global_step"] == 12
    assert payload["config"] == {"patch_size": 64}
    assert payload["dataset_split"] == {"train": ["scene_a"]}
    assert payload["training"] is None
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())
    with torch.no_grad():
        torch.testing.assert_close(loaded(frame), expected)
    cpu_model, _ = load_recurrent_checkpoint(path, torch.device("cpu"))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(cpu_model.state_dict()[name], value.cpu(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unsupported spatial"):
        load_spatial_checkpoint(path, torch.device("cpu"))


def step(model, optimizer, multiplier):
    optimizer.zero_grad(set_to_none=True)
    loss = multiplier * sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def random_values(device, generator):
    return (random.random(), np.random.random(4), torch.rand(4),
            torch.rand(4, device=device), torch.randperm(10, generator=generator))


def test_resume_reproduces_rng_optimizer_and_next_update(tmp_path, device):
    model = TemporalKernelDenoiser(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(17)
    step(model, optimizer, 2.)
    path = tmp_path / "last.pt"
    save_recurrent_checkpoint(path, model, optimizer=optimizer, epoch=2, global_step=7,
                              generator=generator)
    expected_random = random_values(device, generator)
    step(model, optimizer, expected_random[0])
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    loaded, payload = load_recurrent_checkpoint(path, device)
    resumed_optimizer = torch.optim.Adam(loaded.parameters(), lr=.8)
    resumed_generator = torch.Generator().manual_seed(999)
    progress = restore_recurrent_training(loaded, resumed_optimizer, payload, generator=resumed_generator)
    assert loaded.training
    assert progress == {"epoch": 2, "global_step": 7}
    assert resumed_optimizer.param_groups[0]["lr"] == .003
    actual_random = random_values(device, resumed_generator)
    assert actual_random[0] == expected_random[0]
    np.testing.assert_array_equal(actual_random[1], expected_random[1])
    for actual, wanted in zip(actual_random[2:], expected_random[2:]):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    step(loaded, resumed_optimizer, actual_random[0])
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    model = TemporalKernelDenoiser(4)
    save_recurrent_checkpoint(path, model)
    previous = path.read_bytes()
    def interrupted_save(payload, stream):
        stream.write(b"interrupted checkpoint")
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        save_recurrent_checkpoint(path, model)
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.partial")) == []


@pytest.mark.parametrize("change, message", [
    ({"format": "wavefront-spatial-checkpoint"}, "Unsupported recurrent"),
    ({"version": 9}, "Unsupported recurrent"),
    ({"architecture": "other"}, "architecture"),
    ({"feature_version": 99}, "preprocessing"),
    ({"parameter_dtype": "float16"}, "dtype"),
    ({"epoch": -1}, "epoch"),
    ({"state_dict": {}}, "weights"),
])
def test_incompatible_checkpoints_rejected(tmp_path, change, message):
    path = tmp_path / "model.pt"
    save_recurrent_checkpoint(path, TemporalKernelDenoiser(4))
    payload = torch.load(path, weights_only=True)
    payload.update(change)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        load_recurrent_checkpoint(path, torch.device("cpu"))


def test_invalid_weights_and_truncated_file_rejected(tmp_path):
    path = tmp_path / "model.pt"
    save_recurrent_checkpoint(path, TemporalKernelDenoiser(4))
    payload = torch.load(path, weights_only=True)
    next(iter(payload["state_dict"].values())).fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_recurrent_checkpoint(path, torch.device("cpu"))
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Invalid recurrent"):
        load_recurrent_checkpoint(path, torch.device("cpu"))


def test_resume_checks_optimizer_groups_and_generator(tmp_path):
    model = TemporalKernelDenoiser(4)
    path = tmp_path / "last.pt"
    save_recurrent_checkpoint(path, model, optimizer=torch.optim.Adam(model.parameters()),
                              generator=torch.Generator())
    loaded, payload = load_recurrent_checkpoint(path, torch.device("cpu"))
    with pytest.raises(ValueError, match="Optimizer type"):
        restore_recurrent_training(loaded, torch.optim.SGD(loaded.parameters(), lr=.01), payload)
    with pytest.raises(ValueError, match="ordering"):
        restore_recurrent_training(loaded, torch.optim.Adam(list(loaded.parameters())[::-1]), payload)
    optimizer = torch.optim.Adam(loaded.parameters())
    with pytest.raises(ValueError, match="generator setup"):
        restore_recurrent_training(loaded, optimizer, payload)
    restore_recurrent_training(loaded, optimizer, payload, restore_rng=False)
    save_recurrent_checkpoint(path, model)
    _, inference = load_recurrent_checkpoint(path, torch.device("cpu"))
    with pytest.raises(ValueError, match="no resumable"):
        restore_recurrent_training(loaded, optimizer, inference)


def test_save_rejects_invalid_progress_metadata_and_foreign_optimizer(tmp_path):
    path = tmp_path / "model.pt"
    model = TemporalKernelDenoiser(4)
    with pytest.raises(ValueError, match="nonnegative"):
        save_recurrent_checkpoint(path, model, epoch=True)
    with pytest.raises(ValueError, match="finite JSON"):
        save_recurrent_checkpoint(path, model, config={"loss": float("nan")})
    other = TemporalKernelDenoiser(4)
    with pytest.raises(ValueError, match="outside this model"):
        save_recurrent_checkpoint(path, model, optimizer=torch.optim.Adam(other.parameters()))
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        save_recurrent_checkpoint(path, model)
