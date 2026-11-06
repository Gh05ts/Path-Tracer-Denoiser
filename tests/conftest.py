from copy import deepcopy
import hashlib
import json

import Imath
import numpy as np
import OpenEXR
import pytest

from wavefront_denoiser.frames import CHANNELS, CONVENTIONS, COUNT_BUFFERS


@pytest.fixture
def make_frame(tmp_path):
    def make(name="frame_000000", *, index=0, sequence="fixture", previous=False,
             missing=None, half=None, mutate=None, metadata_mutate=None, width=3, height=2):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        camera = {"position": [0, 0, 0], "forward": [0, 0, 1], "right": [-1, 0, 0],
                  "up": [0, 1, 0], "vertical_fov_radians": np.pi / 2, "aspect_ratio": width / height}
        metadata = {
            "schema": "cuda-wavefront-frame", "version": 1,
            "width": width, "height": height, "image": "frame.exr",
            "sequence_id": sequence, "frame_index": index,
            "previous_frame_index": index - 1 if previous else None,
            "camera": camera, "previous_camera": deepcopy(camera) if previous else None,
            "scene": "fixture", "render_fingerprint": "fixture",
            "conventions": deepcopy(CONVENTIONS),
            "render": {"rng": "pcg32", "rng_key": "pixel_sample_v1",
                       "sample_start": index * 20, "samples": 4,
                       "reference_sample_start": index * 20 + 4, "reference_samples": 16,
                       "reference_estimator": "matched_renderer_settings"},
        }
        channels = {}
        for key, names in CHANNELS.items():
            for channel in names:
                value = 0
                if key == "radiance": value = 4
                if key == "reference": value = 5
                if key == "albedo": value = 0.5
                if channel in ("normal.Z", "geometric_normal.Z"): value = -1
                if key == "depth": value = 2
                if key == "guide_valid": value = 1
                if key == "history_valid": value = int(previous)
                if key == "sample_count": value = 4
                if key == "reference_count": value = 16
                dtype = np.uint32 if key in COUNT_BUFFERS else np.float32
                channels[channel] = np.full((height, width), value, dtype=dtype)
        channels["radiance.G"][-1, -1] = 2
        if mutate: mutate(channels)
        if missing: del channels[missing]
        header = OpenEXR.Header(width, height)
        header["channels"] = {
            key: Imath.Channel(Imath.PixelType(
                Imath.PixelType.HALF if key == half else
                Imath.PixelType.UINT if values.dtype == np.uint32 else Imath.PixelType.FLOAT))
            for key, values in channels.items()
        }
        output = OpenEXR.OutputFile(str(directory / "frame.exr"), header)
        output.writePixels({key: values.astype(np.float16).tobytes() if key == half else values.tobytes()
                            for key, values in channels.items()})
        output.close()
        metadata["image_sha256"] = hashlib.sha256((directory / "frame.exr").read_bytes()).hexdigest()
        if metadata_mutate: metadata_mutate(metadata)
        (directory / "manifest.json").write_text(json.dumps(metadata))
        return directory
    return make
