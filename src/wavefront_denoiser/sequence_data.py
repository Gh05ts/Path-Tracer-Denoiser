"""Coherent sequence crops and spatial augmentation for NTASAD training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from .frames import Frame, load_frame, read_manifest
from .sequences import SequenceDataset


def transform_window(buffers: Mapping[str, torch.Tensor], *, top: int, left: int,
                     size: int, flip_x: bool = False, flip_y: bool = False,
                     rotations: int = 0) -> dict[str, torch.Tensor]:
    """Transform TCHW buffers together; motion is in pixel units, normals world-space.

    Shared crop origins cancel out of backward motion vectors. Rotating/reflection
    of the image changes their components but not world-space normal components.
    Returned values own storage; the source sequence is never modified.
    """
    if "radiance" not in buffers or buffers["radiance"].ndim != 4:
        raise ValueError("Expected TCHW window buffers")
    t, _, h, w = buffers["radiance"].shape
    if (any(type(value) is not int for value in (top, left, size, rotations))
            or min(top, left) < 0 or size < 1 or top + size > h or left + size > w):
        raise ValueError("Invalid aligned sequence crop")
    result = {}
    for name, value in buffers.items():
        if value.ndim != 4 or value.shape[0] != t or value.shape[-2:] != (h, w):
            raise ValueError(f"Inconsistent TCHW buffer {name}")
        value = value[..., top:top+size, left:left+size]
        if flip_x:
            value = value.flip(-1)
        if flip_y:
            value = value.flip(-2)
        result[name] = torch.rot90(value, rotations % 4, dims=(-2, -1)).clone()
    motion = result["motion"]
    if motion.shape[1] != 2 or result["history_valid"].shape[1] != 1:
        raise ValueError("Expected two motion channels and one history mask channel")
    if flip_x:
        motion[:, 0].neg_()
    if flip_y:
        motion[:, 1].neg_()
    for _ in range(rotations % 4):
        # torch.rot90 is counterclockwise: (dx,dy) -> (dy,-dx) in image coordinates.
        motion = torch.stack((motion[:, 1], -motion[:, 0]), dim=1)
    result["motion"] = motion
    y, x = torch.meshgrid(torch.arange(size, device=motion.device),
                          torch.arange(size, device=motion.device), indexing="ij")
    px, py = x + motion[:, 0], y + motion[:, 1]
    valid = torch.isfinite(px) & torch.isfinite(py) & (px >= 0) & (px <= size-1) & (py >= 0) & (py <= size-1)
    result["history_valid"] *= valid[:, None]
    result["history_valid"][0] = 0
    return result


class SequencePatchDataset(torch.utils.data.Dataset):
    """Five-frame windows by default; a fresh shared crop/augmentation per access."""

    def __init__(self, root, *, sequence_ids: list[str], length: int = 5,
                 patch_size: int = 128, patches_per_window: int = 1, augment: bool = True):
        if (type(patch_size) is not int or patch_size < 1 or type(patches_per_window) is not int
                or patches_per_window < 1 or type(length) is not int or length < 3):
            raise ValueError("Need positive patch size/count and at least three frames")
        if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("Need unique explicit sequence IDs")
        if type(augment) is not bool or (not augment and patches_per_window > 5):
            raise ValueError("Deterministic validation supports one to five crops per window")
        self.sequences = SequenceDataset(root, length=length, sequence_ids=sequence_ids)
        # Validate dimensions without reading all image pixels.
        for window in self.sequences.windows:
            metadata = read_manifest(window[0])
            if min(metadata["width"], metadata["height"]) < patch_size:
                raise ValueError("Patch size exceeds sequence dimensions")
        self.patch_size = patch_size
        self.patches_per_window = patches_per_window
        self.augment = augment

    def __len__(self):
        return len(self.sequences) * self.patches_per_window

    def __getitem__(self, index):
        paths = self.sequences.windows[index // self.patches_per_window]
        metadata = read_manifest(paths[0])
        h, w, size = metadata["height"], metadata["width"], self.patch_size
        if self.augment:
            top = int(torch.randint(h-size+1, (1,)))
            left = int(torch.randint(w-size+1, (1,)))
        else:
            origins = [((h-size)//2, (w-size)//2), (0, 0),
                       (0, w-size), (h-size, 0), (h-size, w-size)]
            top, left = origins[index % self.patches_per_window]
        # Decode one full frame at a time, crop before tensor conversion/stacking.
        # A 1080p five-frame window must not allocate five full sets of tensors.
        values = []
        for path in paths:
            frame = load_frame(path)
            cropped = {key: value[top:top+size, left:left+size] for key, value in frame.buffers.items()}
            values.append(Frame(frame.manifest_path, frame.metadata, cropped).as_tensors())
            del cropped, frame
        buffers = {key: torch.stack([value[key] for value in values]) for key in values[0]}
        buffers = transform_window(buffers, top=0, left=0, size=size,
            flip_x=bool(torch.randint(2, (1,))) if self.augment else False,
            flip_y=bool(torch.randint(2, (1,))) if self.augment else False,
            rotations=int(torch.randint(4, (1,))) if self.augment else 0)
        return {"buffers": buffers, "sequence_id": metadata["sequence_id"],
                "frame_indices": [read_manifest(path)["frame_index"] for path in paths]}
