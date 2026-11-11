import json

import Imath
import numpy as np
import OpenEXR
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from wavefront_denoiser.checkpoint import load_checkpoint, save_checkpoint
from wavefront_denoiser.cli import main
from wavefront_denoiser.frames import CHANNELS, load_frame
from wavefront_denoiser.image_io import load_input
from wavefront_denoiser.model import NTASADDenoiser
from wavefront_denoiser.runtime import denoise_image, denoise_sequence, evaluate_sequences
from wavefront_denoiser.training import train_denoiser


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def input_only_exr(source_directory, output, *, temporal=False):
    source = OpenEXR.InputFile(str(source_directory / "frame.exr"))
    try:
        header = source.header()
        names = ["radiance", "albedo", "normal", "depth", "guide_valid", "sample_count"]
        if temporal:
            names += ["motion", "history_valid"]
        required = [channel for name in names for channel in CHANNELS[name]]
        header["channels"] = {key: value for key, value in header["channels"].items() if key in required}
        writer = OpenEXR.OutputFile(str(output), header)
        try:
            writer.writePixels({key: source.channel(key) for key in required})
        finally:
            writer.close()
    finally:
        source.close()


def rgb_exr(path):
    source = OpenEXR.InputFile(str(path))
    try:
        window = source.header()["dataWindow"]
        shape = (window.max.y + 1, window.max.x + 1)
        return np.stack([np.frombuffer(source.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
                         .reshape(shape) for c in "RGB"], axis=-1)
    finally:
        source.close()


def test_one_checkpoint_trains_and_runs_still_sequence_and_evaluation(make_frame, tmp_path):
    paths = {}
    for group in ("train", "validation"):
        for index in range(5):
            paths[group, index] = make_frame(f"data/{group}/frame_{index}", index=index,
                sequence=group, previous=index > 0, width=8, height=8,
                mutate=lambda c, g=group: c["radiance.R"].fill(4 if g == "train" else 14))
    config = {"data_root": "data", "train_sequences": ["train"],
              "validation_sequences": ["validation"], "epochs": 1, "patch_size": 8,
              "base_channels": 4, "batch_size": 1, "cpu_threads": 1}
    filename = tmp_path / "training.json"
    filename.write_text(json.dumps(config))
    summary = train_denoiser(filename, tmp_path / "run", "cpu")
    assert summary["global_step"] == 1
    checkpoint = tmp_path / "run/last.pt"
    model, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert payload["format"] == "wavefront-denoiser-checkpoint"
    source = paths["validation", 0]
    still = tmp_path / "still.exr"
    input_only_exr(source, still)
    report = denoise_image(checkpoint, still, tmp_path / "still-output", "cpu")
    assert report["mode"] == "spatial_fallback" and "metrics" not in report
    image = load_frame(source)
    buffers = {name: value.unsqueeze(0) for name, value in image.as_tensors().items()}
    with torch.no_grad():
        expected = model.forward_spatial(buffers)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(rgb_exr(tmp_path / "still-output/denoised.exr"), expected, rtol=1e-6)
    assert main(["denoise-sequence", str(checkpoint), str(tmp_path / "data"),
                 str(tmp_path / "sequence-output"), "--device", "cpu"]) == 0
    # Both independent clips must initialize from their own noisy first frames.
    for group in ("train", "validation"):
        predicted = rgb_exr(tmp_path / f"sequence-output/{group}/frame_000000/denoised.exr")
        own_noisy = load_frame(paths[group, 0]).buffers["radiance"]
        np.testing.assert_allclose(predicted, own_noisy, rtol=1e-6)
        record = json.loads((tmp_path / f"sequence-output/{group}/frame_000000/report.json").read_text())
        assert record["initialization"] is True
    evaluation = evaluate_sequences(checkpoint, tmp_path / "data", ["validation"], "cpu")
    assert len(evaluation["images"]) == 4
    assert evaluation["initialization_frames_excluded"] == 1
    assert evaluation["temporal_l1"]["denoised"] >= 0
    assert not evaluation["overlaps_training_groups"]
    assert evaluation["architecture"] == report["architecture"]
    with pytest.raises(FileExistsError):
        denoise_image(checkpoint, still, tmp_path / "still-output", "cpu")


def test_streaming_inference_accepts_input_only_captures_without_reference(make_frame, tmp_path):
    # Capture metadata permits an input-only EXR; training's full frame loader does not.
    from wavefront_denoiser.image_io import file_sha256
    model = NTASADDenoiser(4)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model)
    for index in range(3):
        original = make_frame(f"data/clip/frame_{index}", index=index, previous=index > 0, sequence="clip")
        export = original / "input.exr"
        input_only_exr(original, export, temporal=True)
        metadata = json.loads((original / "manifest.json").read_text())
        export.replace(original / "frame.exr")
        metadata.update(image_sha256=file_sha256(original / "frame.exr"))
        (original / "manifest.json").write_text(json.dumps(metadata))
    report = denoise_sequence(checkpoint, tmp_path / "data", tmp_path / "output", device_name="cpu")
    assert len(report["sequences"]["clip"]) == 3
    assert all("metrics" not in row for row in report["sequences"]["clip"])


def test_sequence_failure_cleans_staging_and_output_names_are_safe(make_frame, tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, NTASADDenoiser(4))
    original = make_frame("data/clip/frame_0", sequence="clip")
    # Corruption is detected before writing any completed output.
    (original / "frame.exr").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        denoise_sequence(checkpoint, tmp_path / "data", tmp_path / "output", device_name="cpu")
    assert not (tmp_path / "output").exists() and not (tmp_path / "output.partial").exists()
    metadata = json.loads((original / "manifest.json").read_text())
    metadata["sequence_id"] = "../escape"
    (original / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsafe"):
        denoise_sequence(checkpoint, tmp_path / "data", tmp_path / "output", device_name="cpu")
