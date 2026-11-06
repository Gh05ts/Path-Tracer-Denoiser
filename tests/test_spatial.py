import json
from pathlib import Path

import numpy as np
import OpenEXR
import Imath
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from wavefront_denoiser.frames import CHANNELS, load_frame
from wavefront_denoiser.spatial import (SpatialDenoiser, SpatialPatchDataset, decode_radiance,
                                       discover_spatial_frames, make_features)
from wavefront_denoiser.spatial_checkpoint import (ARCHITECTURE, CHECKPOINT_FORMAT,
                                                  load_spatial_checkpoint)
from wavefront_denoiser.spatial_io import load_spatial_input, write_rgb_exr
from wavefront_denoiser.spatial_metrics import image_metrics
from wavefront_denoiser.spatial_runtime import denoise_image, evaluate_spatial
from wavefront_denoiser.spatial_training import train_spatial, training_config
from wavefront_denoiser.cli import main


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def save_identity(path):
    model = SpatialDenoiser(4)
    torch.save({"format": CHECKPOINT_FORMAT, "version": 1, "architecture": ARCHITECTURE,
                "feature_version": 1, "base_channels": 4, "state_dict": model.state_dict()}, path)


def test_features_use_no_reference_or_history(make_frame):
    frame = load_frame(make_frame())
    expected = make_features(frame.buffers)
    frame.buffers["reference"].fill(900)
    frame.buffers["motion"].fill(700)
    frame.buffers["history_valid"].fill(1)
    torch.testing.assert_close(make_features(frame.buffers), expected)
    assert expected.shape == (12, 2, 3)
    assert expected[8, 0, 0] == -1  # Signed world-space normal.
    assert expected[0, 0, 0] > 1  # HDR was compressed without clipping to one.


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (17, 19), (16, 16)])
def test_identity_initialization_handles_odd_and_tiny_images(shape):
    model = SpatialDenoiser(4)
    features = torch.rand(2, 12, *shape)
    torch.testing.assert_close(model(features), features[:, :3], rtol=0, atol=0)
    expected = torch.expm1(features[:, :3])
    torch.testing.assert_close(decode_radiance(model(features)), expected)


def test_black_noisy_pixels_can_receive_training_gradients():
    model = SpatialDenoiser(4)
    loss = torch.nn.functional.l1_loss(model(torch.zeros(1, 12, 4, 4)), torch.ones(1, 3, 4, 4))
    loss.backward()
    assert model.head.bias.grad.abs().sum() > 0


def test_nonfinite_model_output_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        decode_radiance(torch.full((1, 3, 1, 1), 1000.0))


def test_zero_sample_input_rejected(make_frame):
    frame = load_frame(make_frame())
    frame.buffers["sample_count"].fill(0)
    with pytest.raises(ValueError, match="at least one"):
        make_features(frame.buffers)


def test_spatial_discovery_does_not_require_frame_adjacency(make_frame, tmp_path):
    make_frame("first", index=0)
    make_frame("later", index=9, previous=True)
    assert len(discover_spatial_frames(tmp_path)["fixture"]) == 2


def test_aligned_crops_and_source_buffers_are_independent(make_frame):
    path = make_frame(width=8, height=8)
    dataset = SpatialPatchDataset([path / "manifest.json"], 4, 2)
    sample = dataset[0]
    assert sample["features"].shape == (12, 4, 4)
    assert sample["target"].shape == (3, 4, 4)
    sample["features"].zero_()
    assert dataset[0]["features"].abs().sum() > 0


def input_only_exr(source_directory, output):
    source = OpenEXR.InputFile(str(source_directory / "frame.exr"))
    try:
        header = source.header()
        required = [channel for name in ("radiance", "albedo", "normal", "depth", "guide_valid", "sample_count")
                    for channel in CHANNELS[name]]
        header["channels"] = {key: value for key, value in header["channels"].items() if key in required}
        writer = OpenEXR.OutputFile(str(output), header)
        writer.writePixels({key: source.channel(key) for key in required})
        writer.close()
    finally:
        source.close()


def test_inference_needs_no_reference_or_temporal_buffers(make_frame, tmp_path):
    checkpoint = tmp_path / "identity.pt"
    save_identity(checkpoint)
    original = make_frame()
    image = tmp_path / "input-only.exr"
    input_only_exr(original, image)
    loaded = load_spatial_input(image)
    assert "reference" not in loaded.buffers and "motion" not in loaded.buffers
    output = tmp_path / "denoised"
    report = denoise_image(checkpoint, image, output, "cpu")
    assert "metrics" not in report
    assert (output / "comparison.png").exists()
    source = OpenEXR.InputFile(str(output / "denoised.exr"))
    values = np.frombuffer(source.channel("G", Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
    source.close()
    np.testing.assert_allclose(values[0], 4, rtol=1e-6)
    with pytest.raises(FileExistsError):
        denoise_image(checkpoint, image, output, "cpu")


def test_unknown_checkpoint_preprocessing_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    save_identity(path)
    payload = torch.load(path, weights_only=True)
    payload["feature_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_spatial_checkpoint(path, torch.device("cpu"))


def test_metrics_preserve_hdr_error():
    reference = np.full((2, 3, 3), 4, np.float32)
    identity = image_metrics(reference, reference)
    clipped = image_metrics(np.ones_like(reference), reference)
    assert identity["log_l1"] == 0 and identity["linear_rmse"] == 0
    assert clipped["linear_rmse"] == 3
    assert identity["display_psnr_db"] > clipped["display_psnr_db"]


def test_train_evaluate_and_infer_pipeline(make_frame, tmp_path):
    make_frame("data/train/frame_0", sequence="train", width=8, height=8)
    validation = make_frame("data/validation/frame_0", sequence="validation", width=8, height=8)
    config = {"data_root": "data", "train_sequences": ["train"], "validation_sequences": ["validation"],
              "epochs": 4, "patch_size": 4, "patches_per_frame": 4, "batch_size": 2,
              "base_channels": 4, "learning_rate": 0.01, "cpu_threads": 1}
    filename = tmp_path / "training.json"
    filename.write_text(json.dumps(config))
    output = tmp_path / "run"
    summary = train_spatial(filename, output, "cpu")
    assert (output / "model.pt").exists()
    assert summary["best_epoch"] > 0
    assert summary["best_validation"]["denoised_mean"]["log_l1"] < summary["input_baseline"]["log_l1"]
    evaluation = evaluate_spatial(output / "model.pt", tmp_path / "data", ["validation"], "cpu")
    assert not evaluation["overlaps_training_groups"]
    assert len(evaluation["images"]) == 1
    report = denoise_image(output / "model.pt", validation, tmp_path / "prediction", "cpu")
    assert report["metrics"]["denoised"]["log_l1"] < report["metrics"]["input"]["log_l1"]
    with pytest.raises(FileExistsError):
        train_spatial(filename, output, "cpu")


def test_training_rejects_split_leakage_and_typos(tmp_path):
    config = {"data_root": "data", "train_sequences": ["same"], "validation_sequences": ["same"]}
    filename = tmp_path / "training.json"
    filename.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="disjoint"):
        training_config(filename)
    config.update(train_sequences=["train"], epochs_typo=3)
    filename.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unknown"):
        training_config(filename)


def test_cli_bad_training_config_returns_error(tmp_path, capsys):
    filename = tmp_path / "config.json"
    filename.write_text("{}")
    assert main(["train", str(filename), str(tmp_path / "output"), "--device", "cpu"]) == 1
    assert "data_root" in capsys.readouterr().err


def test_checkpoint_selection_rejects_hdr_regression(make_frame, tmp_path, monkeypatch):
    import wavefront_denoiser.spatial_training as training

    make_frame("data/train/frame", sequence="train", width=8, height=8)
    make_frame("data/val/frame", sequence="val", width=8, height=8)
    config = {"data_root": "data", "train_sequences": ["train"], "validation_sequences": ["val"],
              "epochs": 1, "patch_size": 4, "patches_per_frame": 1, "base_channels": 4, "cpu_threads": 1}
    filename = tmp_path / "config.json"
    filename.write_text(json.dumps(config))
    calls = 0

    def measured(*args):
        nonlocal calls
        calls += 1
        baseline = {"log_l1": 0.2, "linear_rmse": 1.0, "display_psnr_db": 20.0, "relative_mse": 0.1}
        candidate = dict(baseline) if calls == 1 else dict(baseline, log_l1=0.1, linear_rmse=2.0)
        return {"input_mean": baseline, "denoised_mean": candidate, "images": []}

    monkeypatch.setattr(training, "evaluate_paths", measured)
    output = tmp_path / "guarded-run"
    summary = training.train_spatial(filename, output, "cpu")
    assert summary["best_epoch"] == 0
    model, payload = load_spatial_checkpoint(output / "model.pt", torch.device("cpu"))
    assert payload["epoch"] == 0
    assert not model.head.weight.count_nonzero()
