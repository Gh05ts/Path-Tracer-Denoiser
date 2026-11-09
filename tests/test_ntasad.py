import json
import math

import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.cli import main
from wavefront_denoiser.ntasad_radiance import MAX_ENCODED, encode_radiance, decode_radiance
from wavefront_denoiser.recurrent import ARCHITECTURE, FEATURE_CHANNELS, TemporalKernelDenoiser
from wavefront_denoiser.recurrent_checkpoint import load_recurrent_checkpoint, save_recurrent_checkpoint
from wavefront_denoiser.recurrent_data import RecurrentPatchDataset, transform_window
from wavefront_denoiser.recurrent_loss import sequence_loss
from wavefront_denoiser.recurrent_training import train_recurrent


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_paper_radiance_transform_and_black_gradients(device):
    values = torch.tensor([-1., 0., 1., 10., 65535., 100000.], device=device,
                           dtype=torch.double, requires_grad=True)
    encoded = encode_radiance(values)
    expected = torch.tensor([0, 0, math.log(2)**(1/2.2), math.log(11)**(1/2.2),
                             math.log(65536)**(1/2.2), math.log(65536)**(1/2.2)],
                            device=device, dtype=torch.double)
    torch.testing.assert_close(encoded, expected)
    torch.testing.assert_close(decode_radiance(encoded), values.clamp(0, 65535))
    encoded.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[1] == 0
    assert torch.isfinite(decode_radiance(torch.tensor([1e9], device=device))).all()


def test_figure3_depth_widths_head_shapes_and_xavier_biases(device):
    model = TemporalKernelDenoiser().to(device)
    assert model.base_channels == 48 and FEATURE_CHANNELS == 15
    assert len(model.down) == 5
    assert all(block[0].out_channels == 48 for block in model.down)
    assert [block[0].out_channels for block in model.decoders] == [96, 96, 96, 96, 64]
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            assert layer.kernel_size == (3, 3)
            assert (layer.bias == 0).all()
    with torch.no_grad():
        fine, middle, coarse = model._predict_kernels(torch.randn(1, 15, 32, 64, device=device))
    assert fine.shape == (1, 51, 32, 64)
    assert middle.shape == (1, 26, 16, 32)
    assert coarse.shape == (1, 26, 8, 16)


def test_loss_uses_only_final_frame_and_final_difference(device):
    prediction = torch.zeros(1, 5, 3, 2, 2, device=device)
    prediction[:, :3] = 50  # Earlier frames must not affect the loss directly.
    prediction[:, -2] = 1
    prediction[:, -1] = 2
    prediction.requires_grad_()
    reference = torch.zeros_like(prediction)
    loss = sequence_loss(prediction, reference, transformed=True)
    assert loss["spatial_l1"] == 2 and loss["temporal_l1"] == 1 and loss["loss"] == 3
    loss["loss"].backward()
    assert (prediction.grad[:, :3] == 0).all()
    assert prediction.grad[:, -2:].abs().sum() > 0


def test_loss_transforms_reference_and_vanishes_for_exact_sequence(device):
    reference = torch.rand(2, 5, 3, 4, 4, device=device) * 50
    assert sequence_loss(reference, reference)["loss"] == 0
    assert sequence_loss(encode_radiance(reference), reference, transformed=True)["loss"] == 0
    with pytest.raises(ValueError, match="at least three"):
        sequence_loss(reference[:, :2], reference[:, :2])


@pytest.mark.parametrize("flip_x,flip_y,rotations", [(False, False, 0), (True, False, 0),
    (False, True, 0), (False, False, 1), (False, False, 2), (False, False, 3), (True, True, 1)])
def test_augmentation_preserves_motion_correspondence_and_world_normals(device, flip_x, flip_y, rotations):
    # An integer correspondence is an independent oracle for transformed vector signs.
    image = torch.arange(49., device=device).reshape(1, 1, 7, 7).repeat(3, 1, 1, 1)
    motion = torch.zeros(3, 2, 7, 7, device=device)
    motion[:, 0] = 1
    motion[:, 1] = -1
    normals = torch.tensor([.2, -.3, .9], device=device).reshape(1, 3, 1, 1).expand(3, 3, 7, 7)
    buffers = {"radiance": image, "reference": image.clone(), "normal": normals,
               "motion": motion, "history_valid": torch.ones(3, 1, 7, 7, device=device)}
    result = transform_window(buffers, top=1, left=1, size=5, flip_x=flip_x, flip_y=flip_y, rotations=rotations)
    for y in range(5):
        for x in range(5):
            if result["history_valid"][1, 0, y, x]:
                dx, dy = result["motion"][1, :, y, x].long().tolist()
                # Source ID changes by dx_original + width*dy_original = 1-7.
                assert result["radiance"][1, 0, y+dy, x+dx] == result["radiance"][1, 0, y, x] - 6
    torch.testing.assert_close(result["normal"], normals[..., :5, :5])
    assert (result["history_valid"][0] == 0).all()
    assert (buffers["history_valid"] == 1).all()
    torch.testing.assert_close(result["radiance"], result["reference"])


def test_previous_checkpoint_architecture_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    save_recurrent_checkpoint(path, TemporalKernelDenoiser(4))
    payload = torch.load(path, weights_only=True)
    payload.update(architecture="temporal_kernel_unet_v1", feature_version=1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="architecture"):
        load_recurrent_checkpoint(path, torch.device("cpu"))


def create_training_data(make_frame, tmp_path):
    for group in ("train", "validation"):
        for t in range(5):
            make_frame(f"{group}/frame_{t}", index=t, previous=t > 0, sequence=group, width=9, height=9)
    config = {"data_root": str(tmp_path), "train_sequences": ["train"],
              "validation_sequences": ["validation"], "epochs": 2, "patch_size": 8,
              "base_channels": 4, "batch_size": 1, "cpu_threads": 1}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config))
    return config, path


def test_real_training_resume_matches_uninterrupted_run(make_frame, tmp_path, device):
    config, path = create_training_data(make_frame, tmp_path)
    complete = train_recurrent(path, tmp_path / "complete", device.type)
    config["epochs"] = 1
    path.write_text(json.dumps(config))
    assert main(["train-recurrent", str(path), str(tmp_path / "first"), "--device", device.type]) == 0
    config["epochs"] = 2
    path.write_text(json.dumps(config))
    resumed = train_recurrent(path, tmp_path / "resumed", device.type, resume=tmp_path / "first/last.pt")
    assert complete["global_step"] == resumed["global_step"] == 2
    _, expected = load_recurrent_checkpoint(tmp_path / "complete/last.pt", torch.device("cpu"))
    _, actual = load_recurrent_checkpoint(tmp_path / "resumed/last.pt", torch.device("cpu"))
    for name, value in actual["state_dict"].items():
        torch.testing.assert_close(value, expected["state_dict"][name], rtol=0, atol=0)
    assert actual["architecture"] == ARCHITECTURE
    config["learning_rate"] = .002
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration"):
        train_recurrent(path, tmp_path / "invalid", device.type, resume=tmp_path / "first/last.pt")
    assert not (tmp_path / "invalid").exists()


def test_patch_dataset_keeps_guides_reference_aligned(make_frame, tmp_path):
    create_training_data(make_frame, tmp_path)
    dataset = RecurrentPatchDataset(tmp_path, sequence_ids=["train"], patch_size=8)
    sample = dataset[0]
    assert sample["buffers"]["radiance"].shape == (5, 3, 8, 8)
    assert (sample["buffers"]["history_valid"][0] == 0).all()


def test_deterministic_validation_crops_keep_alignment_and_rng(make_frame, tmp_path, monkeypatch):
    create_training_data(make_frame, tmp_path)
    dataset = RecurrentPatchDataset(tmp_path, sequence_ids=["validation"], patch_size=8,
                                    patches_per_window=5, augment=False)
    # Crop loading must not call the full-frame tensor-stacking path.
    monkeypatch.setattr(type(dataset.sequences), "__getitem__", lambda *args: pytest.fail("full window allocated"))
    rng = torch.get_rng_state().clone()
    first = dataset[0]
    assert len(dataset) == 5
    for i in range(5):
        sample = dataset[i]
        assert sample["buffers"]["radiance"].shape == (5, 3, 8, 8)
        assert sample["buffers"]["reference"].shape == (5, 3, 8, 8)
        assert (sample["buffers"]["history_valid"][0] == 0).all()
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in first["buffers"].items():
        torch.testing.assert_close(dataset[0]["buffers"][key], value, rtol=0, atol=0)


def test_cropped_validation_training_resume(make_frame, tmp_path, device):
    config, path = create_training_data(make_frame, tmp_path)
    config.update(validation_patch_size=8, validation_crops_per_window=2, epochs=1)
    path.write_text(json.dumps(config))
    first = train_recurrent(path, tmp_path/"cropped", device.type)
    config["epochs"] = 2
    path.write_text(json.dumps(config))
    second = train_recurrent(path, tmp_path/"cropped-resumed", device.type,
                             resume=tmp_path/"cropped/last.pt")
    assert first["global_step"] == 1 and second["global_step"] == 2


def test_legacy_full_frame_validation_config_can_still_resume(make_frame, tmp_path, device):
    config, path = create_training_data(make_frame, tmp_path)
    config["epochs"] = 1
    path.write_text(json.dumps(config))
    train_recurrent(path, tmp_path/"legacy-first", device.type)
    payload = torch.load(tmp_path/"legacy-first/last.pt", weights_only=True)
    payload["config"].pop("validation_patch_size")
    payload["config"].pop("validation_crops_per_window")
    torch.save(payload, tmp_path/"legacy.pt")
    config["epochs"] = 2
    path.write_text(json.dumps(config))
    result = train_recurrent(path, tmp_path/"legacy-resumed", device.type, resume=tmp_path/"legacy.pt")
    assert result["global_step"] == 2
