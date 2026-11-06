"""Supervised single-frame training with explicit capture-group splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .frames import load_frame
from .dataset_splits import audit_dataset_split, read_split_manifest
from .spatial import (FEATURE_VERSION, SpatialDenoiser, SpatialPatchDataset,
                      decode_radiance, discover_spatial_frames, make_features)
from .spatial_checkpoint import ARCHITECTURE, CHECKPOINT_FORMAT, select_device
from .spatial_metrics import image_metrics, mean_metrics


def training_config(path: str | Path) -> dict:
    path = Path(path)
    root = json.loads(path.read_text())
    if not isinstance(root, dict):
        raise ValueError("Training configuration must be an object")
    defaults = {"epochs": 20, "batch_size": 4, "patch_size": 64, "patches_per_frame": 8,
                "learning_rate": 0.001, "seed": 7, "base_channels": 16,
                "num_workers": 0, "cpu_threads": 4, "linear_loss_weight": 0.1}
    allowed = defaults.keys() | {"data_root", "train_sequences", "validation_sequences", "split_manifest", "require_reference_policy"}
    if root.keys() - allowed:
        raise ValueError(f"Unknown training options: {sorted(root.keys() - allowed)}")
    config = dict(defaults, **root)
    if "split_manifest" in config:
        if not isinstance(config["split_manifest"], str) or not config["split_manifest"]:
            raise ValueError("split_manifest must be a nonempty path")
        spec = read_split_manifest(path.parent / config["split_manifest"])
        for key, expected in (("train_sequences", spec["partitions"]["train"]),
                              ("validation_sequences", spec["partitions"]["validation"])):
            if key in config and config[key] != expected:
                raise ValueError("Training groups cannot override the declared dataset split")
            config[key] = expected
        if "data_root" in config and (path.parent / config["data_root"]).resolve() != Path(spec["data_root"]):
            raise ValueError("data_root cannot override the declared dataset split")
        config["data_root"] = spec["data_root"]
        config["split_manifest"] = spec["manifest_path"]
        config.setdefault("require_reference_policy", True)
    if "require_reference_policy" in config and type(config["require_reference_policy"]) is not bool:
        raise ValueError("require_reference_policy must be boolean")
    for key in ("epochs", "batch_size", "patch_size", "patches_per_frame", "base_channels", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    rate = config["learning_rate"]
    if type(rate) not in (int, float) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    weight = config["linear_loss_weight"]
    if type(weight) not in (int, float) or not np.isfinite(weight) or weight < 0:
        raise ValueError("linear_loss_weight must be finite and nonnegative")
    if not isinstance(config.get("data_root"), str) or not config["data_root"]:
        raise ValueError("data_root must be a nonempty path")
    config["data_root"] = str((path.parent / config["data_root"]).resolve())
    for key in ("train_sequences", "validation_sequences"):
        values = config.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
            raise ValueError(f"{key} must contain explicit capture group IDs")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate capture IDs in {key}")
    if set(config["train_sequences"]) & set(config["validation_sequences"]):
        raise ValueError("Training and validation capture groups must be disjoint")
    return config


@torch.inference_mode()
def predict_radiance(model: SpatialDenoiser, buffers: dict, device: torch.device) -> np.ndarray:
    features = make_features(buffers).unsqueeze(0).to(device)
    result = decode_radiance(model(features))[0].permute(1, 2, 0).cpu().numpy()
    return result


def evaluate_paths(model: SpatialDenoiser, paths: list[Path], device: torch.device) -> dict:
    model.eval()
    rows = []
    for path in paths:
        frame = load_frame(path)
        prediction = predict_radiance(model, frame.buffers, device)
        rows.append({"manifest": str(path), "sequence_id": frame.metadata["sequence_id"],
                     "input": image_metrics(frame.buffers["radiance"], frame.buffers["reference"]),
                     "denoised": image_metrics(prediction, frame.buffers["reference"])})
    return {"images": rows,
            "input_mean": mean_metrics([row["input"] for row in rows]),
            "denoised_mean": mean_metrics([row["denoised"] for row in rows])}


def train_spatial(config_path: str | Path, output: str | Path, device_name: str = "auto") -> dict:
    config = training_config(config_path)
    split_audit = audit_dataset_split(config["split_manifest"], allow_missing_test=True) if "split_manifest" in config else None
    if split_audit is not None and config["require_reference_policy"] and split_audit["reference_samples"] is None:
        raise ValueError("Training under this split requires a qualified reference policy")
    device = select_device(device_name)
    groups = discover_spatial_frames(config["data_root"])
    selected = config["train_sequences"] + config["validation_sequences"]
    missing = set(selected) - groups.keys()
    if missing:
        raise ValueError(f"Missing capture groups: {sorted(missing)}")
    train_paths = [path for name in config["train_sequences"] for path in groups[name]]
    validation_paths = [path for name in config["validation_sequences"] for path in groups[name]]
    dataset = SpatialPatchDataset(train_paths, config["patch_size"], config["patches_per_frame"])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], generator=torch.Generator().manual_seed(config["seed"]))
    model = SpatialDenoiser(config["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # This pass validates the held-out files and records the noisy-input baseline.
    initial = evaluate_paths(model, validation_paths, device)
    output.mkdir(parents=True, exist_ok=False)
    split = {"train": [str(path) for path in train_paths], "validation": [str(path) for path in validation_paths]}
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    if split_audit is not None:
        (output / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n")
    # Identity is a real candidate: a failed training run must not quietly
    # select a checkpoint worse than the noisy baseline on the chosen metric.
    best_loss = initial["denoised_mean"]["log_l1"]
    best_epoch = 0
    best_report = initial
    def save_best(epoch: int, validation: dict) -> None:
        payload = {"format": CHECKPOINT_FORMAT, "version": 1, "architecture": ARCHITECTURE,
                   "feature_version": FEATURE_VERSION, "base_channels": config["base_channels"],
                   "state_dict": model.state_dict(), "epoch": epoch, "config": config,
                   "validation": validation, "selection": "log_l1_with_input_rmse_guard",
                   "torch_version": str(torch.__version__)}
        if split_audit is not None:
            payload["dataset_split"] = split_audit
        temporary = output / "model.pt.partial"
        torch.save(payload, temporary)
        temporary.replace(output / "model.pt")

    save_best(0, initial)
    history_path = output / "history.jsonl"
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum, log_loss_sum, images = 0.0, 0.0, 0
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            log_loss = F.l1_loss(prediction, target)
            target_linear = torch.expm1(target)
            linear_loss = (((torch.expm1(prediction) - target_linear) / (1 + target_linear)) ** 2).mean()
            loss = log_loss + config["linear_loss_weight"] * linear_loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * features.shape[0]
            log_loss_sum += float(log_loss.detach()) * features.shape[0]
            images += features.shape[0]
        validation = evaluate_paths(model, validation_paths, device)
        validation_loss = validation["denoised_mean"]["log_l1"]
        hdr_eligible = validation["denoised_mean"]["linear_rmse"] <= initial["input_mean"]["linear_rmse"] + 1e-7
        record = {"epoch": epoch, "train_loss": loss_sum / images, "train_log_l1": log_loss_sum / images,
                  "validation_log_l1": validation_loss,
                  "validation_input_log_l1": validation["input_mean"]["log_l1"],
                  "validation_linear_rmse": validation["denoised_mean"]["linear_rmse"],
                  "validation_display_psnr_db": validation["denoised_mean"]["display_psnr_db"],
                  "hdr_eligible": hdr_eligible}
        with history_path.open("a") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if hdr_eligible and validation_loss < best_loss:
            best_loss, best_epoch, best_report = validation_loss, epoch, validation
            save_best(epoch, validation)
    summary = {"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
               "train_images": len(train_paths), "validation_images": len(validation_paths),
               "best_epoch": best_epoch, "checkpoint": "model.pt", "config": config,
               "input_baseline": initial["input_mean"], "best_validation": best_report}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
