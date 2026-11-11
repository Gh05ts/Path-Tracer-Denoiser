"""Named EXR inputs and linear HDR outputs for the unified denoiser."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import OpenEXR
import Imath

from .frames import CHANNELS, FrameFormatError, read_manifest


@dataclass(frozen=True)
class FrameInput:
    source: Path
    buffers: dict[str, np.ndarray]
    sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_input(path: str | Path, *, temporal: bool = False) -> FrameInput:
    """Accept a capture directory/manifest or an EXR with spatial input channels.

    A standalone EXR needs no reference, geometric normal, motion, or history.
    Temporal inference additionally requires backward motion and history masks.
    References are optional and are used only for reporting metrics.
    """
    path = Path(path)
    metadata = None
    if path.is_dir() or path.suffix.lower() == ".json":
        metadata = read_manifest(path)
        path = (path if path.is_dir() else path.parent) / metadata["image"]
    digest = file_sha256(path)
    if metadata and digest != metadata["image_sha256"]:
        raise FrameFormatError("Image SHA-256 does not match manifest")
    source = OpenEXR.InputFile(str(path))
    try:
        header = source.header()
        window = header["dataWindow"]
        width, height = window.max.x + 1, window.max.y + 1
        if window.min.x != 0 or window.min.y != 0 or width <= 0 or height <= 0:
            raise FrameFormatError("Expected a zero-origin EXR data window")
        display = header["displayWindow"]
        if (display.min.x, display.min.y, display.max.x, display.max.y) != (0, 0, width - 1, height - 1):
            raise FrameFormatError("Unsupported EXR display window")
        if metadata and (metadata["width"], metadata["height"]) != (width, height):
            raise FrameFormatError("EXR dimensions disagree with manifest")
        names = ["radiance", "albedo", "normal", "depth", "guide_valid", "sample_count"]
        if temporal:
            names += ["motion", "history_valid"]
        reference_present = [name in header["channels"] for name in CHANNELS["reference"]]
        if any(reference_present):
            if not all(reference_present):
                raise FrameFormatError("Incomplete reference RGB channels")
            names.append("reference")
        buffers = {}
        for name in names:
            kind = Imath.PixelType.UINT if name == "sample_count" else Imath.PixelType.FLOAT
            dtype = np.uint32 if name == "sample_count" else np.float32
            planes = []
            for channel in CHANNELS[name]:
                info = header["channels"].get(channel)
                if info is None or info.type.v != kind or info.xSampling != 1 or info.ySampling != 1:
                    raise FrameFormatError(f"Missing channel or unsupported precision: {channel}")
                planes.append(np.frombuffer(source.channel(channel, Imath.PixelType(kind)), dtype=dtype)
                              .reshape(height, width))
            buffers[name] = np.stack(planes, axis=-1)
        if metadata and not (buffers["sample_count"] == metadata["render"]["samples"]).all():
            raise FrameFormatError("Sample counts disagree with manifest")
        return FrameInput(path, buffers, digest)
    finally:
        source.close()


def write_rgb_exr(path: str | Path, pixels: np.ndarray) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    if pixels.ndim != 3 or pixels.shape[2] != 3 or not np.isfinite(pixels).all():
        raise ValueError("Expected finite HWC RGB output")
    height, width = pixels.shape[:2]
    header = OpenEXR.Header(width, height)
    header["channels"] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in "RGB"}
    output = OpenEXR.OutputFile(str(path), header)
    try:
        output.writePixels({name: np.ascontiguousarray(pixels[..., index], dtype=np.float32).tobytes()
                            for index, name in enumerate("RGB")})
    finally:
        output.close()
