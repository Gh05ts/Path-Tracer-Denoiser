"""The v1 capture contract. Arrays are HWC, float32 (counts are uint32)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class FrameFormatError(ValueError):
    """A capture is incomplete, incompatible, or internally inconsistent."""


CONVENTIONS = {
    "radiance": "linear_scene_referred_rgb",
    "radiance_normalization": "mean",
    "exposure": 1.0,
    "image_origin": "top_left",
    "normal_space": "world",
    "depth": "camera_forward_distance",
    "motion": "previous_pixel_minus_current_pixel",
    "motion_units": "pixels",
    "pixel_coordinates": "integer_centers",
    "guides": "unjittered_pixel_centers",
    "radiance_sampling": "independent_uniform_subpixel_jitter",
    "scene_motion": "static",
    "history_valid": "previous_frustum_and_visibility",
}

CHANNELS = {
    "radiance": ("radiance.R", "radiance.G", "radiance.B"),
    "reference": ("reference.R", "reference.G", "reference.B"),
    "albedo": ("albedo.R", "albedo.G", "albedo.B"),
    "normal": ("normal.X", "normal.Y", "normal.Z"),
    "geometric_normal": ("geometric_normal.X", "geometric_normal.Y", "geometric_normal.Z"),
    "depth": ("depth.Z",),
    "motion": ("motion.X", "motion.Y"),
    "guide_valid": ("guide_valid.Y",),
    "history_valid": ("history_valid.Y",),
    "sample_count": ("sample_count.N",),
    "reference_count": ("reference_count.N",),
}
COUNT_BUFFERS = {"sample_count", "reference_count"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrameFormatError(message)


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and minimum <= value <= 0xFFFFFFFF,
             f"{field} must be an integer in [{minimum}, 4294967295]")
    return value


def _camera(camera: Any, width: int, height: int, field: str) -> None:
    _require(isinstance(camera, dict), f"{field} must be an object")
    vectors = {}
    for key in ("position", "forward", "right", "up"):
        value = camera.get(key)
        _require(isinstance(value, list) and len(value) == 3 and
                 all(type(x) in (int, float) for x in value), f"Invalid {field}.{key}")
        vectors[key] = np.asarray(value, dtype=np.float64)
        _require(bool(np.isfinite(vectors[key]).all()), f"Non-finite {field}.{key}")
    basis = np.stack([vectors[k] for k in ("forward", "right", "up")])
    _require(bool(np.allclose(basis @ basis.T, np.eye(3), atol=1e-4)),
             f"{field} basis must be orthonormal")
    _require(bool(np.allclose(np.cross(vectors["right"], vectors["forward"]), vectors["up"], atol=1e-4)),
             f"{field} basis has the wrong handedness")
    fov = camera.get("vertical_fov_radians")
    aspect = camera.get("aspect_ratio")
    _require(type(fov) in (int, float) and np.isfinite(fov) and 0 < fov < np.pi,
             f"Invalid {field} vertical FOV")
    _require(type(aspect) in (int, float) and np.isfinite(aspect) and
             bool(np.isclose(aspect, width / height)), f"Invalid {field} aspect ratio")


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Validate metadata without importing OpenEXR or allocating image arrays."""
    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    try:
        def reject_constant(value: str) -> None:
            raise FrameFormatError(f"Non-finite JSON number: {value}")

        metadata = json.loads(path.read_text(), parse_constant=reject_constant)
        _require(isinstance(metadata, dict), "Manifest must be an object")
        _require(metadata.get("schema") == "cuda-wavefront-frame" and
                 type(metadata.get("version")) is int and metadata["version"] == 1,
                 "Unsupported frame schema or version")
        width = _integer(metadata.get("width"), "width", minimum=1)
        height = _integer(metadata.get("height"), "height", minimum=1)
        _require(width * height <= 0xFFFFFFFF, "Image exceeds renderer index range")
        _require(metadata.get("image") == "frame.exr", "v1 image must be frame.exr in the frame directory")
        digest = metadata.get("image_sha256")
        _require(isinstance(digest, str) and len(digest) == 64 and
                 all(c in "0123456789abcdef" for c in digest), "Invalid image SHA-256")
        _require(metadata.get("conventions") == CONVENTIONS, "Unsupported frame conventions")
        _require(isinstance(metadata.get("sequence_id"), str) and bool(metadata["sequence_id"]),
                 "sequence_id must be nonempty")
        _require(isinstance(metadata.get("scene"), str) and bool(metadata["scene"]), "scene must be nonempty")
        _require(isinstance(metadata.get("render_fingerprint"), str) and bool(metadata["render_fingerprint"]),
                 "Missing renderer fingerprint")
        _require("previous_frame_index" in metadata and "previous_camera" in metadata,
                 "Missing explicit previous-frame metadata")
        index = _integer(metadata.get("frame_index"), "frame_index")
        previous = metadata.get("previous_frame_index")
        if previous is not None:
            _integer(previous, "previous_frame_index")
            _require(previous < index, "Previous frame must precede the current frame")
            _camera(metadata.get("previous_camera"), width, height, "previous_camera")
        else:
            _require(metadata.get("previous_camera") is None, "Unexpected previous camera")
        _camera(metadata.get("camera"), width, height, "camera")
        render = metadata.get("render")
        _require(isinstance(render, dict), "Missing render settings")
        _require(render.get("rng") == "pcg32" and render.get("rng_key") == "pixel_sample_v1",
                 "Unsupported RNG contract")
        start = _integer(render.get("sample_start"), "sample_start")
        samples = _integer(render.get("samples"), "samples", minimum=1)
        reference_start = _integer(render.get("reference_sample_start"), "reference_sample_start")
        reference_samples = _integer(render.get("reference_samples"), "reference_samples", minimum=1)
        _require(start + samples <= 0xFFFFFFFF and reference_start + reference_samples <= 0xFFFFFFFF,
                 "Sample range exceeds renderer index range")
        _require(start + samples <= reference_start or reference_start + reference_samples <= start,
                 "Input and reference sample ranges overlap")
        _require(render.get("reference_estimator") == "matched_renderer_settings",
                 "Unsupported reference estimator")
        return metadata
    except FrameFormatError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise FrameFormatError(f"Cannot read manifest {path}: {error}") from error


@dataclass(frozen=True)
class Frame:
    manifest_path: Path
    metadata: dict[str, Any]
    buffers: dict[str, np.ndarray]

    def as_tensors(self) -> dict[str, Any]:
        """Return independent CHW tensors; promote counts to int64 for PyTorch."""
        import torch

        return {name: torch.from_numpy(np.array(value.transpose(2, 0, 1),
                    dtype=np.int64 if name in COUNT_BUFFERS else np.float32, copy=True, order="C"))
                for name, value in self.buffers.items()}


def _validate_buffers(metadata: dict[str, Any], buffers: dict[str, np.ndarray]) -> None:
    valid = buffers["guide_valid"][..., 0]
    history = buffers["history_valid"][..., 0]
    for name, values in buffers.items():
        _require(bool(np.isfinite(values).all()), f"Non-finite {name} buffer")
    for name in ("guide_valid", "history_valid"):
        _require(bool(np.isin(buffers[name], (0, 1)).all()), f"{name} must be binary")
    _require(bool((history <= valid).all()), "History is valid without a surface guide")
    _require(bool((buffers["depth"][..., 0][valid == 1] > 0).all()), "Surface depths must be positive")
    for name in ("albedo", "normal", "geometric_normal", "depth", "motion"):
        _require(bool((buffers[name][valid == 0] == 0).all()), f"Background {name} must be zero")
    for name in ("normal", "geometric_normal"):
        lengths = np.linalg.norm(buffers[name][valid == 1], axis=-1)
        _require(bool(np.allclose(lengths, 1.0, atol=1e-3)), f"Surface {name} must have unit length")
    render = metadata["render"]
    _require(bool((buffers["sample_count"] == render["samples"]).all()), "Input sample counts disagree with metadata")
    _require(bool((buffers["reference_count"] == render["reference_samples"]).all()),
             "Reference sample counts disagree with metadata")
    if metadata["previous_frame_index"] is None:
        _require(bool((history == 0).all() and (buffers["motion"] == 0).all()),
                 "First frame must have zero motion and invalid history")
    else:
        y, x = np.indices(valid.shape)
        previous_x = x + buffers["motion"][..., 0]
        previous_y = y + buffers["motion"][..., 1]
        inside = ((previous_x >= -1e-4) & (previous_y >= -1e-4) &
                  (previous_x <= metadata["width"] - 1 + 1e-4) &
                  (previous_y <= metadata["height"] - 1 + 1e-4))
        _require(bool(inside[history == 1].all()), "Valid history projects outside the previous image")


def load_frame(path: str | Path, *, verify_digest: bool = True) -> Frame:
    """Load float32 HDR channels without tone mapping, normalization, or clipping."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    metadata = read_manifest(manifest_path)
    image = manifest_path.parent / metadata["image"]
    try:
        if verify_digest:
            digest = hashlib.sha256()
            with image.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            _require(digest.hexdigest() == metadata["image_sha256"], "Image SHA-256 does not match manifest")
        import OpenEXR
        import Imath

        source = OpenEXR.InputFile(str(image))
        try:
            header = source.header()
            window = header["dataWindow"]
            display = header["displayWindow"]
            width, height = metadata["width"], metadata["height"]
            _require(window.min.x == 0 and window.min.y == 0 and window.max.x == width - 1 and
                     window.max.y == height - 1, "EXR data window disagrees with manifest")
            _require(display.min.x == 0 and display.min.y == 0 and display.max.x == width - 1 and
                     display.max.y == height - 1, "Unsupported EXR display window")
            for name, channels in CHANNELS.items():
                kind = Imath.PixelType.UINT if name in COUNT_BUFFERS else Imath.PixelType.FLOAT
                for channel in channels:
                    info = header["channels"].get(channel)
                    _require(info is not None, f"Missing EXR channel {channel}")
                    _require(info.type.v == kind and info.xSampling == 1 and info.ySampling == 1,
                             f"Unsupported precision or subsampling for {channel}")
            buffers = {}
            # Batch by pixel type so ZIP blocks are not decoded once per channel.
            # Keep counts as UINT to preserve integers above float32's exact range.
            for counts in (False, True):
                names = [name for name in CHANNELS if (name in COUNT_BUFFERS) == counts]
                channels = [channel for name in names for channel in CHANNELS[name]]
                kind = Imath.PixelType.UINT if counts else Imath.PixelType.FLOAT
                dtype = np.uint32 if counts else np.float32
                planes = dict(zip(channels, source.channels(channels, Imath.PixelType(kind))))
                for name in names:
                    buffers[name] = np.stack([
                        np.frombuffer(planes[channel], dtype=dtype).reshape(height, width)
                        for channel in CHANNELS[name]
                    ], axis=-1)
                del planes
        finally:
            source.close()
        _validate_buffers(metadata, buffers)
        return Frame(manifest_path, metadata, buffers)
    except FrameFormatError:
        raise
    except ImportError as error:
        raise RuntimeError("Install the project dependencies: pip install -e .") from error
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise FrameFormatError(f"Cannot read frame {image}: {error}") from error
