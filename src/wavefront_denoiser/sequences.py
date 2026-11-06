"""Sequence discovery and lazy, consecutive training windows."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .frames import FrameFormatError, load_frame, read_manifest


def discover_sequences(root: str | Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for path in sorted(Path(root).rglob("manifest.json")):
        # Interrupted exports are intentionally invisible until directory rename.
        if any(part.endswith(".partial") for part in path.parts):
            continue
        metadata = read_manifest(path)
        grouped[metadata["sequence_id"]].append((metadata, path))
    if not grouped:
        raise FrameFormatError(f"No completed frame manifests found in {root}")
    sequences = {}
    for name, entries in grouped.items():
        entries.sort(key=lambda item: item[0]["frame_index"])
        first = entries[0][0]
        previous = None
        for metadata, path in entries:
            if (metadata["width"], metadata["height"], metadata["scene"], metadata["render_fingerprint"]) != (
                    first["width"], first["height"], first["scene"], first["render_fingerprint"]):
                raise FrameFormatError(f"Inconsistent scene, settings, or dimensions in sequence {name}")
            if previous is None:
                if metadata["previous_frame_index"] is not None:
                    raise FrameFormatError(f"Sequence {name} starts with missing history")
            else:
                if (metadata["frame_index"] != previous["frame_index"] + 1 or
                        metadata["previous_frame_index"] != previous["frame_index"] or
                        metadata["previous_camera"] != previous["camera"]):
                    raise FrameFormatError(f"Broken frame/camera history in sequence {name}: {path}")
            previous = metadata
        sequences[name] = [path for _, path in entries]
    return sequences


class SequenceDataset:
    """PyTorch DataLoader-compatible windows that never cross sequence boundaries.

    Select train/validation sequence IDs explicitly. Splitting adjacent windows
    randomly leaks scene and trajectory content between training and evaluation.
    """

    def __init__(self, root: str | Path, length: int = 5, *,
                 sequence_ids: list[str] | None = None, tensors: bool = True):
        if type(length) is not int or length < 1:
            raise ValueError("Window length must be a positive integer")
        sequences = discover_sequences(root)
        if sequence_ids is not None:
            missing = set(sequence_ids) - sequences.keys()
            if missing:
                raise ValueError(f"Unknown sequence IDs: {sorted(missing)}")
            sequences = {key: sequences[key] for key in sequence_ids}
        self.windows = [paths[start:start + length] for paths in sequences.values()
                        for start in range(len(paths) - length + 1)]
        if not self.windows:
            raise ValueError("No sequences are long enough for the requested window")
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frames = [load_frame(path) for path in self.windows[index]]
        if self.tensors:
            import torch

            values = [frame.as_tensors() for frame in frames]
            buffers = {key: torch.stack([value[key] for value in values]) for key in values[0]}
        else:
            import numpy as np

            buffers = {key: np.stack([frame.buffers[key] for frame in frames]) for key in frames[0].buffers}
        # History before a window is unavailable to its recurrent initialization.
        # Masks are copied by stacking; cached/on-disk frames are never modified.
        buffers["history_valid"][0] = 0
        return {"buffers": buffers, "sequence_id": frames[0].metadata["sequence_id"],
                "frame_indices": [frame.metadata["frame_index"] for frame in frames]}
