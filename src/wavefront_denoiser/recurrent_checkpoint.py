"""Versioned recurrent weights and epoch-boundary training resume state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np
import torch

from .recurrent import ARCHITECTURE, FEATURE_VERSION, TemporalKernelDenoiser


CHECKPOINT_FORMAT = "wavefront-recurrent-checkpoint"
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _json_metadata(value: dict | None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint metadata must contain finite JSON values") from error


def _optimizer_parameters(model, optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    seen = []
    for group in optimizer.param_groups:
        ids = [id(parameter) for parameter in group["params"]]
        if any(identity not in names for identity in ids):
            raise ValueError("Optimizer contains parameters outside this model")
        groups.append([names[identity] for identity in ids])
        seen.extend(ids)
    if len(seen) != len(names) or set(seen) != names.keys():
        raise ValueError("Optimizer must contain every model parameter exactly once")
    return groups


def _optimizer_type(optimizer) -> str:
    cls = type(optimizer)
    return f"{cls.__module__}.{cls.__qualname__}"


def _rng_state(generator: torch.Generator | None) -> dict:
    numpy_state = np.random.get_state()
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("The DataLoader generator must use CPU")
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0],
                      "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "generator": generator.get_state() if generator is not None else None}


def save_recurrent_checkpoint(path: str | Path, model: TemporalKernelDenoiser, *,
                              optimizer: torch.optim.Optimizer | None = None,
                              epoch: int = 0, global_step: int = 0,
                              config: dict | None = None, dataset_split: dict | None = None,
                              generator: torch.Generator | None = None) -> None:
    """Atomically save portable weights and, optionally, a complete optimizer.

    epoch counts completed epochs; global_step counts completed optimizer steps.
    Save after an optimizer step at an epoch boundary, with no pending gradient
    accumulation. Recurrent images and DataLoader worker state are not saved.
    """
    if not isinstance(model, TemporalKernelDenoiser):
        raise ValueError("Expected a TemporalKernelDenoiser")
    for value in (epoch, global_step):
        if type(value) is not int or value < 0:
            raise ValueError("epoch and global_step must be nonnegative integers")
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    if dtype not in DTYPES or any(parameter.dtype != DTYPES[dtype] for parameter in model.parameters()):
        raise ValueError("Checkpoint weights must all use float32 or float64")
    if generator is not None and optimizer is None:
        raise ValueError("A training generator requires an optimizer checkpoint")
    weights = _cpu_tree(model.state_dict())
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Checkpoint weights must be finite")
    training = None
    if optimizer is not None:
        training = {"optimizer_type": _optimizer_type(optimizer),
                    "parameter_groups": _optimizer_parameters(model, optimizer),
                    "optimizer_state": _cpu_tree(optimizer.state_dict()),
                    "rng": _cpu_tree(_rng_state(generator))}
    payload = {"format": CHECKPOINT_FORMAT, "version": 1, "architecture": ARCHITECTURE,
               "feature_version": FEATURE_VERSION, "base_channels": model.base_channels,
               "parameter_dtype": dtype, "state_dict": weights,
               "epoch": epoch, "global_step": global_step,
               "config": _json_metadata(config), "dataset_split": _json_metadata(dataset_split),
               "training": training, "torch_version": str(torch.__version__)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_recurrent_checkpoint(path: str | Path, device: torch.device) -> tuple[TemporalKernelDenoiser, dict]:
    """Load with restricted deserialization on CPU, then place the model in eval mode."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, IndexError) as error:
        raise ValueError(f"Invalid recurrent checkpoint: {error}") from error
    if (not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT
            or type(payload.get("version")) is not int or payload["version"] != 1):
        raise ValueError("Unsupported recurrent checkpoint format")
    if (payload.get("architecture") != ARCHITECTURE
            or type(payload.get("feature_version")) is not int or payload["feature_version"] != FEATURE_VERSION):
        raise ValueError("Checkpoint preprocessing or architecture is incompatible")
    if (not isinstance(payload.get("parameter_dtype"), str) or payload["parameter_dtype"] not in DTYPES
            or not isinstance(payload.get("state_dict"), dict)):
        raise ValueError("Checkpoint is missing compatible dtype or weights")
    for key in ("epoch", "global_step"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"Invalid checkpoint {key}")
    for key in ("config", "dataset_split"):
        _json_metadata(payload.get(key))
    model = TemporalKernelDenoiser(payload.get("base_channels")).to(dtype=DTYPES[payload["parameter_dtype"]])
    if any(not isinstance(value, torch.Tensor) or value.dtype != DTYPES[payload["parameter_dtype"]]
           or not torch.isfinite(value).all() for value in payload["state_dict"].values()):
        raise ValueError("Checkpoint weights must be finite tensors with the declared dtype")
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"Incompatible recurrent checkpoint weights: {error}") from error
    return model.to(device).eval(), payload


def restore_recurrent_training(model: TemporalKernelDenoiser, optimizer: torch.optim.Optimizer,
                               payload: dict, *, generator: torch.Generator | None = None,
                               restore_rng: bool = True) -> dict[str, int]:
    """Restore optimizer/RNG after loading weights and constructing its optimizer.

    The optimizer must have the same type and named parameter-group ordering.
    Restoring RNG requires the same CUDA device count if CUDA states were saved.
    Use restore_rng=False for a deliberate device migration without exact replay.
    """
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("Checkpoint has no resumable training state")
    if training.get("optimizer_type") != _optimizer_type(optimizer):
        raise ValueError("Optimizer type differs from the training checkpoint")
    if training.get("parameter_groups") != _optimizer_parameters(model, optimizer):
        raise ValueError("Optimizer parameter-group ordering differs from the checkpoint")
    rng = training.get("rng")
    if restore_rng:
        if not isinstance(rng, dict):
            raise ValueError("Training checkpoint is missing RNG state")
        if (rng.get("generator") is None) != (generator is None):
            raise ValueError("Resume must provide the same DataLoader generator setup")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("The DataLoader generator must use CPU")
        if rng["cuda"] and (not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count()):
            raise ValueError("CUDA RNG topology changed; use restore_rng=False for device migration")
    optimizer.load_state_dict(training["optimizer_state"])
    if restore_rng:
        random.setstate(rng["python"])
        state = rng["numpy"]
        np.random.set_state((state["algorithm"], state["keys"].numpy().astype(np.uint32),
                             state["position"], state["has_gauss"], state["cached_gaussian"]))
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if generator is not None:
            generator.set_state(rng["generator"])
    model.train()
    return {"epoch": payload["epoch"], "global_step": payload["global_step"]}
