"""Exercise actual serialized ONNX against both checkpoint families."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from wavefront_denoiser.checkpoint import save_checkpoint
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import save_joint_checkpoint
from wavefront_denoiser.model import NTASADDenoiser
from wavefront_denoiser.onnx_export import (
    INPUT_NAME, OUTPUT_NAME, example_frame, export_spatial, load_spatial, sha256)


@pytest.mark.parametrize("family,height,width", [("uniform", 17, 19), ("joint", 33, 35),
                                                  ("legacy_joint", 1, 7)])
def test_serialized_export_matches_checkpoint(tmp_path, family, height, width):
    torch.set_num_threads(1)
    torch.manual_seed(16)
    checkpoint = tmp_path / "weights.pt"
    if family == "uniform":
        save_checkpoint(checkpoint, NTASADDenoiser(4))
    else:
        model = JointNTASAD(denoiser_channels=4, sampler_channels=4,
                           reconstruction_version="sample_mass_v1" if family == "legacy_joint" else "pixel_mean_v2")
        save_joint_checkpoint(checkpoint, model)
    output = tmp_path / "export"
    report = export_spatial(checkpoint, output, width=width, height=height)
    assert report == json.loads((output / "manifest.json").read_text())
    assert report["checkpoint_sha256"] == sha256(checkpoint)
    assert report["onnx_sha256"] == sha256(output / "denoiser.onnx")
    assert report["input"]["shape"] == [1, 10, height, width]
    assert not report["requirements"]["history"]
    session = ort.InferenceSession(str(output / "denoiser.onnx"), providers=["CPUExecutionProvider"])
    assert [v.name for v in session.get_inputs()] == [INPUT_NAME]
    value = example_frame(height, width, seed=120)
    model, _ = load_spatial(checkpoint)
    with torch.inference_mode():
        expected = model.reference(value).numpy()
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: value.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
    with pytest.raises(FileExistsError):
        export_spatial(checkpoint, output, width=width, height=height)


def test_rejects_invalid_export_before_publication(tmp_path):
    with pytest.raises(ValueError, match="dimensions"):
        export_spatial(tmp_path / "missing.pt", tmp_path / "out", width=0, height=8)
    assert not (tmp_path / "out").exists()
    bad = tmp_path / "bad.pt"
    torch.save({"format": "unknown"}, bad)
    with pytest.raises(ValueError, match="supported"):
        export_spatial(bad, tmp_path / "out", width=8, height=8)
    assert not (tmp_path / "out").exists()
