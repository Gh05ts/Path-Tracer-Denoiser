import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wavefront_denoiser.dataset_generation import contact_sheet

spec = importlib.util.spec_from_file_location(
    "generate_images", Path(__file__).resolve().parents[1] / "tools/generate_images.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_explicit_exposure_is_shared_and_hdr_unchanged(make_frame, tmp_path):
    frame = make_frame(metadata_mutate=lambda m: m.update(scene="bistro"))
    source = frame / "frame.exr"
    before = hashlib.sha256(source.read_bytes()).digest()
    exposure = 0.005789725053854193
    paths = exporter.generate_images(source, tmp_path / "previews", exposure=exposure)
    for path, value in zip(paths, (4, 5)):
        linear = value * exposure
        expected = int((linear / (1 + linear)) ** (1 / 2.2) * 255)
        assert np.all(np.asarray(Image.open(path))[0, 0] == expected)
    assert hashlib.sha256(source.read_bytes()).digest() == before
    assert exporter.main([str(source), "-o", str(tmp_path / "override"), "--exposure", "1"]) == 0
    expected = int((5 / 6) ** (1 / 2.2) * 255)
    assert np.all(np.asarray(Image.open(tmp_path / "override/frame-reference.png"))[0, 0] == expected)
    (frame / "manifest.json").unlink()
    exporter.generate_images(source, tmp_path / "no-manifest")
    assert np.all(np.asarray(Image.open(tmp_path / "no-manifest/frame-reference.png"))[0, 0] == expected)


@pytest.mark.parametrize("exposure", [0, -1, float("inf"), float("nan")])
def test_invalid_exposure_writes_no_previews(make_frame, tmp_path, exposure):
    frame = make_frame()
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Exposure"):
        exporter.generate_images(frame / "frame.exr", destination, exposure=exposure)
    assert not destination.exists()


def test_dataset_contact_sheet_uses_the_same_bistro_exposure(make_frame, tmp_path):
    frame = make_frame(metadata_mutate=lambda m: m.update(scene="bistro"))
    group = tmp_path / "bistro"
    group.mkdir()
    source = frame.rename(group / frame.name) / "frame.exr"
    standalone = exporter.generate_images(source, tmp_path / "standalone")
    contact_sheet(tmp_path, {"bistro": {"frames": [{}]}}, ["bistro"])
    preview = tmp_path / "previews/bistro/frame_000000-reference-16spp.png"
    assert np.array_equal(np.asarray(Image.open(standalone[1])), np.asarray(Image.open(preview)))
