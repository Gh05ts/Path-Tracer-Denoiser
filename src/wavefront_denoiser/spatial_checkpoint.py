"""Versioned spatial weights and device selection."""

from pathlib import Path
import pickle

import torch

from .spatial import FEATURE_VERSION, SpatialDenoiser


CHECKPOINT_FORMAT = "wavefront-spatial-checkpoint"
ARCHITECTURE = "residual_unet_v1"


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_spatial_checkpoint(path: str | Path, device: torch.device) -> tuple[SpatialDenoiser, dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"Invalid spatial checkpoint: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT or payload.get("version") != 1:
        raise ValueError("Unsupported spatial checkpoint format")
    if payload.get("feature_version") != FEATURE_VERSION or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint preprocessing or architecture is incompatible")
    if "base_channels" not in payload or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint is missing its architecture settings or weights")
    model = SpatialDenoiser(payload["base_channels"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
