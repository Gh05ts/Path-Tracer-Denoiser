"""Single-image features and a small residual U-Net. No history is consumed."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .frames import load_frame, read_manifest


FEATURE_VERSION = 1
FEATURE_CHANNELS = 12
INPUT_BUFFERS = ("radiance", "albedo", "normal", "depth", "guide_valid", "sample_count")


def make_features(buffers: dict[str, np.ndarray]) -> torch.Tensor:
    """HWC linear HDR buffers -> 12-channel CHW float32; references are unused."""
    shape = buffers["radiance"].shape[:2]
    sizes = {"radiance": 3, "albedo": 3, "normal": 3, "depth": 1,
             "guide_valid": 1, "sample_count": 1}
    for name, channels in sizes.items():
        value = buffers[name]
        if value.shape != (*shape, channels) or not np.isfinite(value).all():
            raise ValueError(f"Invalid spatial input {name}")
    if (buffers["radiance"] < 0).any() or (buffers["depth"] < 0).any():
        raise ValueError("Spatial radiance and depth must be nonnegative")
    if (buffers["sample_count"] < 1).any():
        raise ValueError("The spatial baseline requires at least one sample per pixel")
    if not np.isin(buffers["guide_valid"], (0, 1)).all():
        raise ValueError("guide_valid must be binary")
    valid = buffers["guide_valid"][..., 0] == 1
    if not np.allclose(np.linalg.norm(buffers["normal"][valid], axis=-1), 1, atol=1e-3):
        raise ValueError("Valid surface normals must have unit length")
    if (buffers["depth"][..., 0][valid] <= 0).any():
        raise ValueError("Valid surface depth must be positive")
    for name in ("albedo", "normal", "depth"):
        if (buffers[name][~valid] != 0).any():
            raise ValueError(f"Background {name} must be zero")
    depth = buffers["depth"].astype(np.float32)
    features = np.concatenate([
        np.log1p(buffers["radiance"]), buffers["albedo"], buffers["normal"],
        depth / (1 + depth), buffers["guide_valid"],
        1 / np.sqrt(buffers["sample_count"].astype(np.float32)),
    ], axis=-1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(features.transpose(2, 0, 1)))


class ConvBlock(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class SpatialDenoiser(nn.Module):
    """Predict a correction to log1p radiance; zero head starts as identity."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        if type(base_channels) is not int or not 4 <= base_channels <= 128:
            raise ValueError("base_channels must be an integer between 4 and 128")
        self.base_channels = base_channels
        self.encoder0 = ConvBlock(FEATURE_CHANNELS, base_channels)
        self.encoder1 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder1 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder0 = ConvBlock(base_channels * 3, base_channels)
        self.head = nn.Conv2d(base_channels, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != FEATURE_CHANNELS or min(features.shape[-2:]) < 1:
            raise ValueError("Expected N x 12 x H x W spatial features")
        height, width = features.shape[-2:]
        padded = F.pad(features, (0, (-width) % 4, 0, (-height) % 4), mode="replicate")
        level0 = self.encoder0(padded)
        level1 = self.encoder1(F.avg_pool2d(level0, 2))
        coarse = self.bottleneck(F.avg_pool2d(level1, 2))
        up1 = F.interpolate(coarse, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        fine1 = self.decoder1(torch.cat((up1, level1), dim=1))
        up0 = F.interpolate(fine1, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        fine0 = self.decoder0(torch.cat((up0, level0), dim=1))
        return features[:, :3] + self.head(fine0)[..., :height, :width]


def decode_radiance(log_radiance: torch.Tensor) -> torch.Tensor:
    radiance = torch.expm1(log_radiance).clamp_min(0)
    if not torch.isfinite(radiance).all():
        raise ValueError("Model produced non-finite HDR radiance")
    return radiance


def discover_spatial_frames(root: str | Path) -> dict[str, list[Path]]:
    """Group independent images; no frame adjacency or history is required."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(Path(root).rglob("manifest.json")):
        if any(part.endswith(".partial") for part in path.parts):
            continue
        metadata = read_manifest(path)
        groups.setdefault(metadata["sequence_id"], []).append(path)
    if not groups:
        raise ValueError(f"No completed spatial captures found in {root}")
    return groups


class SpatialPatchDataset(torch.utils.data.Dataset):
    """Random aligned input/reference crops with a bounded per-worker frame cache."""

    def __init__(self, paths: list[Path], patch_size: int, patches_per_frame: int = 8):
        if not paths or type(patch_size) is not int or patch_size < 4 or patches_per_frame < 1:
            raise ValueError("Need frames, patch_size >= 4, and positive patches_per_frame")
        self.paths = paths
        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.cache: OrderedDict[Path, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        for path in paths:
            metadata = read_manifest(path)
            if min(metadata["width"], metadata["height"]) < patch_size:
                raise ValueError(f"Patch size exceeds frame dimensions: {path}")

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_frame

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index // self.patches_per_frame]
        if path not in self.cache:
            frame = load_frame(path)
            if (frame.buffers["reference"] < 0).any():
                raise ValueError("Reference radiance must be nonnegative")
            target = torch.from_numpy(np.ascontiguousarray(np.log1p(frame.buffers["reference"]).transpose(2, 0, 1)))
            self.cache[path] = make_features(frame.buffers), target
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        features, target = self.cache[path]
        height, width = features.shape[-2:]
        y = int(torch.randint(height - self.patch_size + 1, (1,)))
        x = int(torch.randint(width - self.patch_size + 1, (1,)))
        crop = (..., slice(y, y + self.patch_size), slice(x, x + self.patch_size))
        return {"features": features[crop].clone(), "target": target[crop].clone()}
