import json
import os
from pathlib import Path

import numpy as np
import pytest

from wavefront_denoiser import FrameFormatError, load_frame


def test_hdr_channels_and_layout(make_frame):
    frame = load_frame(make_frame())
    assert frame.buffers["radiance"].shape == (2, 3, 3)
    assert frame.buffers["radiance"].dtype == np.float32
    assert frame.buffers["radiance"][0, 0, 1] == 4
    assert frame.buffers["radiance"][-1, -1, 1] == 2
    assert frame.buffers["normal"][0, 0, 2] == -1
    assert frame.buffers["sample_count"].dtype == np.uint32


def test_checksum_rejects_replaced_image(make_frame):
    path = make_frame()
    with (path / "frame.exr").open("ab") as image:
        image.write(b"changed")
    with pytest.raises(FrameFormatError, match="SHA-256"):
        load_frame(path)


@pytest.mark.parametrize("kwargs, message", [
    ({"missing": "normal.X"}, "Missing EXR channel"),
    ({"half": "radiance.R"}, "precision"),
    ({"mutate": lambda c: c["radiance.R"].fill(np.nan)}, "Non-finite"),
    ({"mutate": lambda c: c["sample_count.N"].fill(0)}, "counts"),
    ({"mutate": lambda c: c["guide_valid.Y"].fill(0.5)}, "binary"),
    ({"mutate": lambda c: c["history_valid.Y"].fill(1)}, "First frame"),
    ({"mutate": lambda c: c["normal.Z"].fill(0)}, "unit length"),
    ({"mutate": lambda c: c["depth.Z"].fill(-1)}, "positive"),
    ({"metadata_mutate": lambda m: m.update(version=2)}, "schema"),
    ({"metadata_mutate": lambda m: m.update(width=True)}, "width"),
    ({"metadata_mutate": lambda m: m.update(image="../frame.exr")}, "v1 image"),
    ({"metadata_mutate": lambda m: m["conventions"].update(motion="forward")}, "conventions"),
    ({"metadata_mutate": lambda m: m["render"].update(reference_sample_start=2)}, "overlap"),
    ({"metadata_mutate": lambda m: m["camera"].update(right=[1, 0, 0])}, "handedness"),
    ({"metadata_mutate": lambda m: m.pop("previous_frame_index")}, "previous-frame metadata"),
])
def test_invalid_capture_rejected(make_frame, kwargs, message):
    with pytest.raises(FrameFormatError, match=message):
        load_frame(make_frame(**kwargs))


def test_valid_history_must_project_inside_image(make_frame):
    path = make_frame(index=1, previous=True, mutate=lambda c: c["motion.X"].fill(100))
    with pytest.raises(FrameFormatError, match="outside"):
        load_frame(path)


def test_tensors_preserve_signed_hdr_data_and_do_not_alias(make_frame):
    torch = pytest.importorskip("torch")
    frame = load_frame(make_frame())
    tensors = frame.as_tensors()
    assert tuple(tensors["radiance"].shape) == (3, 2, 3)
    assert tensors["radiance"][1, 0, 0] == 4
    assert tensors["sample_count"].dtype == torch.int64
    tensors["depth"].zero_()
    assert frame.buffers["depth"][0, 0, 0] == 2


def test_cpp_export_fixture_if_available():
    path = os.environ.get("WAVEFRONT_CPP_FIXTURE")
    if not path:
        pytest.skip("Set WAVEFRONT_CPP_FIXTURE to a frame_export_tests output directory")
    frame = load_frame(Path(path))
    assert frame.buffers["radiance"][0, 0].tolist() == [0.125, 4.0, 32.0]
    assert frame.buffers["radiance"][-1, -1].tolist() == [1.0, 2.0, 3.0]
