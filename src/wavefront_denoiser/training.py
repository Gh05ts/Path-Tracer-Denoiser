"""Uniform-SPP NTASAD denoiser training; adaptive sampling remains separate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .dataset_splits import audit_dataset_split, read_split_manifest
from .frames import read_manifest
from .model import INPUT_CHANNELS, NTASADDenoiser
from .checkpoint import load_checkpoint, restore_training, save_checkpoint
from .sequence_data import SequencePatchDataset
from .loss import sequence_loss
from .sequences import SequenceDataset
from .device import select_device


def training_config(path: str | Path) -> dict:
    path = Path(path)
    root = json.loads(path.read_text())
    defaults = {"epochs": 1000, "batch_size": 4, "patch_size": 128,
                "patches_per_window": 1, "window_length": 5, "learning_rate": .001,
                "seed": 7, "base_channels": 48, "num_workers": 0, "cpu_threads": 4,
                "validation_patch_size": None, "validation_crops_per_window": 5}
    if not isinstance(root, dict) or root.keys() - (defaults.keys() | {
            "data_root", "train_sequences", "validation_sequences", "split_manifest"}):
        raise ValueError("Invalid or unknown recurrent training options")
    config = dict(defaults, **root)
    if "split_manifest" in config:
        spec = read_split_manifest(path.parent / config["split_manifest"])
        for key, partition in (("train_sequences", "train"), ("validation_sequences", "validation")):
            if key in root and root[key] != spec["partitions"][partition]:
                raise ValueError("Training groups cannot override the dataset split")
            config[key] = spec["partitions"][partition]
        if "data_root" in root and (path.parent / root["data_root"]).resolve() != Path(spec["data_root"]):
            raise ValueError("Training data_root cannot override the dataset split")
        config.update(data_root=spec["data_root"], split_manifest=spec["manifest_path"])
    if not isinstance(config.get("data_root"), str) or not config["data_root"]:
        raise ValueError("Need a data_root")
    config["data_root"] = str((path.parent / config["data_root"]).resolve())
    for key in ("epochs", "batch_size", "patch_size", "patches_per_window", "window_length", "base_channels", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["window_length"] < 3 or not 4 <= config["base_channels"] <= 128:
        raise ValueError("Need at least three frames and 4..128 base channels")
    if (config["validation_patch_size"] is not None and
            (type(config["validation_patch_size"]) is not int or config["validation_patch_size"] < 1)):
        raise ValueError("validation_patch_size must be null or a positive integer")
    if type(config["validation_crops_per_window"]) is not int or not 1 <= config["validation_crops_per_window"] <= 5:
        raise ValueError("validation_crops_per_window must be between one and five")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if (type(config["learning_rate"]) not in (int, float)
            or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("learning_rate must be finite and positive")
    for key in ("train_sequences", "validation_sequences"):
        value = config.get(key)
        if (not isinstance(value, list) or not value or not all(isinstance(name, str) and name for name in value)
                or len(set(value)) != len(value)):
            raise ValueError(f"Need unique explicit {key}")
    if set(config["train_sequences"]) & set(config["validation_sequences"]):
        raise ValueError("Train and validation sequences must be disjoint")
    return config


def _inventory(dataset: SequenceDataset) -> list[dict]:
    paths = sorted({path for window in dataset.windows for path in window})
    return [{"manifest": str(path.resolve()), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "image_sha256": read_manifest(path)["image_sha256"]} for path in paths]


def _batch(batch, device):
    buffers = batch["buffers"]
    return {name: buffers[name].to(device) for name in INPUT_CHANNELS}, buffers["reference"].to(device)


@torch.inference_mode()
def evaluate_windows(model, dataset, device) -> dict:
    """Full-frame or deterministic cropped windows, each with fresh initialization."""
    model.eval()
    totals = {"loss": 0., "spatial_l1": 0., "temporal_l1": 0.}
    for batch in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             generator=torch.Generator().manual_seed(0)):
        buffers, reference = _batch(batch, device)
        metrics = sequence_loss(model.forward_sequence(buffers, transformed=True), reference, transformed=True)
        for key in totals:
            totals[key] += float(metrics[key])
    return {key: value / len(dataset) for key, value in totals.items()}


def train_denoiser(config_path: str | Path, output: str | Path, device_name: str = "auto", *,
                     resume: str | Path | None = None) -> dict:
    """Train to total configured epochs; resuming writes a new run directory."""
    config = training_config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    audit = audit_dataset_split(config["split_manifest"], allow_missing_test=True) if "split_manifest" in config else None
    train = SequencePatchDataset(config["data_root"], sequence_ids=config["train_sequences"],
        length=config["window_length"], patch_size=config["patch_size"], patches_per_window=config["patches_per_window"])
    validation = SequenceDataset(config["data_root"], length=config["window_length"],
                                 sequence_ids=config["validation_sequences"])
    provenance = {"train": _inventory(train.sequences), "validation": _inventory(validation), "split_audit": audit}
    if config["validation_patch_size"] is not None:
        validation = SequencePatchDataset(config["data_root"], sequence_ids=config["validation_sequences"],
            length=config["window_length"], patch_size=config["validation_patch_size"],
            patches_per_window=config["validation_crops_per_window"], augment=False)
    device = select_device(device_name)
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config["seed"])
    start_epoch, global_step = 0, 0
    if resume is None:
        model = NTASADDenoiser(config["base_channels"]).to(device)
        payload = None
    else:
        model, payload = load_checkpoint(resume, device)
        comparable = lambda values: {key: value for key, value in
            {"validation_patch_size": None, "validation_crops_per_window": 5, **values}.items() if key != "epochs"}
        if comparable(payload["config"]) != comparable(config) or payload["dataset_split"] != provenance:
            raise ValueError("Resume configuration or training/validation data changed")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if payload is not None:
        progress = restore_training(model, optimizer, payload, generator=generator)
        start_epoch, global_step = progress["epoch"], progress["global_step"]
        if start_epoch >= config["epochs"]:
            raise ValueError("Resume needs epochs greater than the completed checkpoint epoch")
    loader = torch.utils.data.DataLoader(train, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], generator=generator)
    initial = evaluate_windows(model, validation, device)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    (output / "split.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    best_loss, best_epoch, best_validation = initial["loss"], start_epoch, initial
    def save(name, epoch):
        save_checkpoint(output / name, model, optimizer=optimizer, epoch=epoch,
            global_step=global_step, config=config, dataset_split=provenance, generator=generator)
    save("model.pt", start_epoch)
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        model.train()
        totals = {"loss": 0., "spatial_l1": 0., "temporal_l1": 0.}
        windows = 0
        for batch in loader:
            buffers, reference = _batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            metrics = sequence_loss(model.forward_sequence(buffers, transformed=True), reference, transformed=True)
            metrics["loss"].backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError("Non-finite recurrent training gradient")
            optimizer.step()
            global_step += 1
            count = reference.shape[0]
            windows += count
            for key in totals:
                totals[key] += float(metrics[key].detach()) * count
        measured = evaluate_windows(model, validation, device)
        record = {"epoch": epoch, "global_step": global_step,
                  "train": {key: value/windows for key, value in totals.items()}, "validation": measured}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if measured["loss"] < best_loss:
            best_loss, best_epoch, best_validation = measured["loss"], epoch, measured
            save("model.pt", epoch)
        save("last.pt", epoch)
    summary = {"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
               "completed_epochs": config["epochs"], "global_step": global_step,
               "best_epoch": best_epoch, "best_validation": best_validation,
               "checkpoint": "model.pt", "resume_checkpoint": "last.pt", "resumed_from": str(resume) if resume else None,
               "selection_scope": "this_run_including_initial_model", "config": config}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
