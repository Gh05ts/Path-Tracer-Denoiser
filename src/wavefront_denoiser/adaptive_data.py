"""Validated v5 companion layers; v4 frames retain their original v1 format."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .dataset_generation import digest, verify_frame
from .frames import CONVENTIONS, Frame, FrameFormatError, load_frame, read_manifest


LAYER_SPP = (1, 2, 4, 8, 16, 32)
NEW_LAYER_SPP = (1, 2, 8, 16, 32)
RANGE_FIELDS = {"samples", "sample_start", "reference_samples", "reference_sample_start"}


def read_json(path: Path) -> dict:
    def reject(value):
        raise FrameFormatError(f"Non-finite JSON constant: {value}")
    value = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise FrameFormatError(f"Expected JSON object: {path}")
    return value


def layer_layout(config: dict, report_reference_samples: int, *, spps=NEW_LAYER_SPP) -> list[dict]:
    """Reserve noisy and both reference budgets, then allocate disjoint blocks."""
    for name in ("samples", "reference_samples", "sample_stride", "sample_start"):
        if type(config.get(name)) is not int or config[name] < (0 if name == "sample_start" else 1):
            raise ValueError(f"Invalid source sample allocation: {name}")
    if config["samples"] != 4 or type(report_reference_samples) is not int or report_reference_samples < config["reference_samples"]:
        raise ValueError("v5 requires a 4-SPP source and a reserved report budget >= training reference SPP")
    offset = 4 + report_reference_samples
    result = []
    if (not spps or len(set(spps)) != len(spps)
            or any(type(n) is not int or n < 1 or n > 32 or n & (n-1) for n in spps)):
        raise ValueError("Sample layers require unique powers of two up to 32")
    for spp in spps:
        result.append({"samples": spp, "offset": offset})
        offset += spp
    if offset > config["sample_stride"]:
        raise ValueError("Source sample_stride has no room for independent v5 layers")
    if config["sample_start"] + len(config["frames"]) * config["sample_stride"] > 0xFFFFFFFF:
        raise ValueError("Source sample allocation exceeds uint32")
    return result


def read_layer_capture(directory: Path, base: dict, layout: list[dict], *, pixels=True, verify_digest=True) -> tuple[dict, dict]:
    """Check layer provenance against a completed source frame, then read RGB means."""
    directory = Path(directory)
    metadata = read_json(directory / "manifest.json")
    if (metadata.get("schema") != "cuda-wavefront-sample-layers" or type(metadata.get("version")) is not int
            or metadata["version"] != 1 or metadata.get("image") != "layers.exr"
            or metadata.get("conventions") != CONVENTIONS):
        raise FrameFormatError("Unsupported companion-layer format or conventions")
    for key in ("width", "height", "sequence_id", "scene", "frame_index", "previous_frame_index",
                "render_fingerprint", "scene_fingerprint"):
        if key not in base or metadata.get(key) != base[key]:
            raise FrameFormatError(f"Companion layers disagree with source: {key}")
    for key in ("camera", "previous_camera"):
        left, right = metadata.get(key), base[key]
        if left is None or right is None:
            if left != right:
                raise FrameFormatError(f"Companion camera disagrees with source: {key}")
        elif left.keys() != right.keys() or any(not np.allclose(left[k], right[k], rtol=2e-5, atol=2e-5) for k in right):
            raise FrameFormatError(f"Companion camera disagrees with source: {key}")
    render = {key: value for key, value in base["render"].items() if key not in RANGE_FIELDS}
    if metadata.get("render") != render:
        raise FrameFormatError("Companion integrator/render settings disagree with source")
    expected = [{"samples": item["samples"], "sample_start": base["render"]["sample_start"] + item["offset"],
                 "channels": [f"spp_{item['samples']}.{c}" for c in "RGB"]} for item in layout]
    if metadata.get("layers") != expected:
        raise FrameFormatError("Companion sample ranges or channel declarations disagree with plan")
    if verify_digest and digest(directory / "layers.exr") != metadata.get("image_sha256"):
        raise FrameFormatError("Companion EXR checksum mismatch")
    if not pixels:
        return metadata, {}
    import OpenEXR
    import Imath

    source = OpenEXR.InputFile(str(directory / "layers.exr"))
    try:
        header = source.header()
        w, h = base["width"], base["height"]
        for name in ("dataWindow", "displayWindow"):
            window = header[name]
            if (window.min.x, window.min.y, window.max.x, window.max.y) != (0, 0, w-1, h-1):
                raise FrameFormatError("Companion EXR windows disagree with source")
        names = [channel for item in expected for channel in item["channels"]]
        if set(header["channels"]) != set(names):
            raise FrameFormatError("Companion EXR has missing or unexpected channels")
        for channel in names:
            info = header["channels"][channel]
            if info.type.v != Imath.PixelType.FLOAT or info.xSampling != 1 or info.ySampling != 1:
                raise FrameFormatError("Companion RGB must be full-resolution FLOAT")
        # Read every layer together to decompress shared ZIP blocks only once.
        planes = dict(zip(names, source.channels(names, Imath.PixelType(Imath.PixelType.FLOAT))))
        values = {}
        for item in expected:
            image = np.stack([
                np.frombuffer(planes[channel], dtype=np.float32).reshape(h, w)
                for channel in item["channels"]
            ], axis=-1)
            if not np.isfinite(image).all() or (image < 0).any():
                raise FrameFormatError("Companion RGB must be finite and nonnegative")
            values[item["samples"]] = image
        return metadata, values
    finally:
        source.close()


def read_v5_dataset(root: str | Path) -> dict:
    root = Path(root)
    if not (root / "dataset.json").exists() and (root / "decoded_cache_lock.json").exists():
        from .decoded_cache import decoded_dataset_descriptor
        value = decoded_dataset_descriptor(root)
    else:
        value = read_json(root / "dataset.json")
    if value.get("format") != "wavefront-ntasad-v5-dataset" or value.get("version") != 1:
        raise FrameFormatError("Unsupported v5 dataset")
    if set(value.get("partitions", {})) != {"train", "validation", "test"}:
        raise FrameFormatError("v5 dataset must declare train/validation/test partitions")
    groups = value.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise FrameFormatError("Missing v5 groups")
    membership = sum(value["partitions"].values(), [])
    if len(membership) != len(set(membership)) or set(membership) != set(groups):
        raise FrameFormatError("Overlapping or missing v5 partition groups")
    families = {}
    for partition, names in value["partitions"].items():
        for name in names:
            if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
                raise FrameFormatError("Invalid v5 group identifier")
            scene = groups[name]["scene"]
            if scene in families and families[scene] != partition:
                raise FrameFormatError("v5 scene occurs in multiple partitions")
            families[scene] = partition
            info = groups[name]
            config = info["source_config"]
            if (config["sequence_id"] != name or config["scene"] != scene
                    or type(info["frame_count"]) is not int or info["frame_count"] != len(config["frames"])
                    or info["frame_count"] < 1
                    or info["sample_layers"] != layer_layout(config, info["reserved_reference_samples"])):
                raise FrameFormatError("v5 group/config/sample allocation mismatch")
    return value


def verify_v5_record(root: Path, group: str, index: int, *, verify_images=True) -> tuple[dict, dict]:
    """Verify a published frame's copied source and companion bytes without decoding."""
    dataset = read_v5_dataset(root)
    if group not in dataset["groups"] or type(index) is not int or not 0 <= index < dataset["groups"][group]["frame_count"]:
        raise FrameFormatError("Unknown v5 frame")
    if dataset.get("storage") == "decoded":
        from .decoded_cache import verify_decoded_frame
        base, _ = verify_decoded_frame(root, group, index, verify_hashes=verify_images,
                                       allow_missing=not verify_images)
        return dataset, base
    relative = Path(group) / f"frame_{index:06d}"
    record = read_json(root / "frames" / relative / "manifest.json")
    if (record.get("format") != "wavefront-ntasad-v5-frame" or record.get("version") != 1
            or record.get("sequence_id") != group or record.get("frame_index") != index):
        raise FrameFormatError("Invalid v5 completion record")
    base_path, layer_path = root / "base" / relative, root / "layers" / relative
    for key, path in (("source_manifest_sha256", base_path / "manifest.json"),
                      ("layer_manifest_sha256", layer_path / "manifest.json")):
        if digest(path) != record.get(key):
            raise FrameFormatError(f"v5 manifest checksum mismatch: {key}")
    base = read_manifest(base_path)
    if base["sequence_id"] != group or base["frame_index"] != index:
        raise FrameFormatError("v5 source frame identity mismatch")
    config = dataset["groups"][group]["source_config"]
    if verify_images:
        verify_frame(base_path, config, index)
    layout = dataset["groups"][group]["sample_layers"]
    read_layer_capture(layer_path, base, layout, pixels=False, verify_digest=verify_images)
    return dataset, base


@dataclass(frozen=True)
class AdaptiveFrame:
    base: Frame
    layers: np.ndarray  # 6,H,W,3 linear RGB means; ordered 1/2/4/8/16/32.


def load_v5_frame(root: str | Path, group: str, index: int) -> AdaptiveFrame:
    root = Path(root)
    dataset, _ = verify_v5_record(root, group, index)
    if dataset.get("storage") == "decoded":
        from .decoded_cache import load_decoded_crop
        base, layers = load_decoded_crop(root, group, index)
        return AdaptiveFrame(base, layers)
    relative = Path(group) / f"frame_{index:06d}"
    base = load_frame(root / "base" / relative)
    _, images = read_layer_capture(root / "layers" / relative, base.metadata,
                                   dataset["groups"][group]["sample_layers"])
    images[4] = base.buffers["radiance"]
    return AdaptiveFrame(base, np.stack([images[spp] for spp in LAYER_SPP]))


class V5SequenceDataset:
    """Explicit-partition lazy windows; discovery never reads reserved test pixels.

    NumPy layers are T,6,H,W,3. Tensor layers are T,6,3,H,W. Buffers use the
    existing TCHW/THWC layouts. Full-resolution windows can be large; crop-aware
    augmentation for joint training is a later stage.
    """

    def __init__(self, root: str | Path, *, partition: str, length: int = 5, tensors=True,
                 sequence_ids: list[str] | None = None):
        self.root = Path(root)
        self.tensors = tensors
        dataset = read_v5_dataset(root)
        if partition not in dataset["partitions"] or type(length) is not int or length < 1:
            raise ValueError("Need an explicit partition and positive window length")
        selected = dataset["partitions"][partition] if sequence_ids is None else sequence_ids
        if (not isinstance(selected,list) or not selected or any(not isinstance(g,str) for g in selected)
                or len(set(selected)) != len(selected) or set(selected)-set(dataset["partitions"][partition])):
            raise ValueError("Selected v5 groups must belong to the declared partition")
        self.windows = []
        for group in selected:
            count = dataset["groups"][group]["frame_count"]
            previous = None
            for index in range(count):
                _, metadata = verify_v5_record(self.root, group, index, verify_images=False)
                if (metadata["previous_frame_index"] != (index-1 if index else None) or
                        (previous is not None and (metadata["previous_camera"] != previous["camera"] or
                         metadata["render_fingerprint"] != previous["render_fingerprint"]))):
                    raise FrameFormatError("Broken v5 sequence history")
                previous = metadata
            self.windows.extend((group, start, length) for start in range(count-length+1))
        if not self.windows:
            raise ValueError("No complete v5 sequences of the requested length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        group, start, length = self.windows[index]
        frames = [load_v5_frame(self.root, group, i) for i in range(start, start+length)]
        if self.tensors:
            import torch
            values = [frame.base.as_tensors() for frame in frames]
            buffers = {key: torch.stack([value[key] for value in values]) for key in values[0]}
            layers = torch.from_numpy(np.stack([frame.layers for frame in frames])).permute(0, 1, 4, 2, 3)
        else:
            buffers = {key: np.stack([frame.base.buffers[key] for frame in frames]) for key in frames[0].base.buffers}
            layers = np.stack([frame.layers for frame in frames])
        buffers["history_valid"][0] = 0
        return {"buffers": buffers, "layers": layers, "sequence_id": group,
                "frame_indices": list(range(start, start+length))}
