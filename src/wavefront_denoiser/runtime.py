"""Still/sequence inference and held-out evaluation using one NTASAD checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw
import torch

from .checkpoint import load_checkpoint
from .dataset_splits import audit_dataset_split, read_split_manifest
from .device import select_device
from .image_io import file_sha256, load_input, write_rgb_exr
from .metrics import display_rgb, image_metrics, mean_metrics
from .model import INPUT_CHANNELS
from .ntasad_radiance import encode_radiance
from .sequences import discover_sequences


def comparison_preview(path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    height, width = images[0][1].shape[:2]
    canvas = Image.new("RGB", (width * len(images), height + 20), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(images):
        rgb = np.clip(np.rint(display_rgb(pixels) * 255), 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (index * width, 20))
        draw.text((index * width + 3, 3), label, fill=(255, 255, 255))
    canvas.save(path)


def _tensors(image, model, *, temporal):
    parameter = next(model.parameters())
    names = INPUT_CHANNELS if temporal else (INPUT_CHANNELS.keys() - {"motion", "history_valid"})
    return {name: torch.from_numpy(np.ascontiguousarray(image.buffers[name].transpose(2, 0, 1)))
            .unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype) for name in names}


def _pixels(prediction):
    return prediction[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _report(checkpoint, payload, device):
    return {"format": "wavefront-denoiser-result", "version": 1,
            "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(checkpoint),
            "architecture": payload["architecture"], "model_epoch": payload["epoch"],
            "device": str(device), "color_space": "linear_scene_referred_rgb"}


def _output_paths(output):
    output = Path(output)
    staging = output.with_name(output.name + ".partial")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Inference output or staging directory already exists: {output}")
    return output, staging


def _write_frame(output, image, prediction, *, initialization=False):
    output.mkdir(parents=True, exist_ok=False)
    write_rgb_exr(output / "denoised.exr", prediction)
    previews = [("Noisy input", image.buffers["radiance"]),
                ("Noisy initialization" if initialization else "Denoised", prediction)]
    record = {"source": str(image.source.resolve()), "source_sha256": image.sha256,
              "width": prediction.shape[1], "height": prediction.shape[0],
              "initialization": initialization, "output": "denoised.exr"}
    if "reference" in image.buffers:
        reference = image.buffers["reference"]
        record["metrics"] = {"input": image_metrics(image.buffers["radiance"], reference),
                             "denoised": image_metrics(prediction, reference)}
        previews.append(("Reference", reference))
    comparison_preview(output / "comparison.png", previews)
    (output / "report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


@torch.inference_mode()
def denoise_image(checkpoint, source, output, device_name="auto") -> dict:
    """Use the learned spatial branch of the same jointly trained U-Net."""
    output, staging = _output_paths(output)
    device = select_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    image = load_input(source)
    prediction = _pixels(model.forward_spatial(_tensors(image, model, temporal=False)))
    report = dict(_report(checkpoint, payload, device), mode="spatial_fallback")
    try:
        report.update(_write_frame(staging, image, prediction))
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return report


def _selected_groups(root, sequence_ids):
    groups = discover_sequences(root)
    selected = list(groups) if sequence_ids is None else sequence_ids
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Need unique explicit sequence IDs")
    if set(selected) - groups.keys():
        raise ValueError(f"Missing sequences: {sorted(set(selected) - groups.keys())}")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..") for name in selected):
        raise ValueError("Unsafe sequence output name")
    return {name: groups[name] for name in selected}


@torch.inference_mode()
def denoise_sequence(checkpoint, root, output, sequence_ids=None, device_name="auto") -> dict:
    """Stream complete clips with fresh noisy initialization at each boundary.

    Only one frame plus history is resident. References are optional and never
    enter the model. Frame zero is the paper's noisy initialization, explicitly
    labelled in previews; subsequent frames use spatial and temporal filtering.
    """
    output, staging = _output_paths(output)
    groups = _selected_groups(root, sequence_ids)
    device = select_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    report = dict(_report(checkpoint, payload, device), mode="spatiotemporal", sequences={})
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for group, paths in groups.items():
            history = None
            rows = []
            for index, path in enumerate(paths):
                image = load_input(path, temporal=True)
                history = model(_tensors(image, model, temporal=True), history)
                destination = staging / group / f"frame_{index:06d}"
                rows.append(_write_frame(destination, image, _pixels(history), initialization=index == 0))
            report["sequences"][group] = rows
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


@torch.inference_mode()
def _evaluate_groups(model, groups):
    rows, temporal_errors = [], []
    for group, paths in groups.items():
        history = previous_prediction = previous_reference = previous_noisy = None
        for index, path in enumerate(paths):
            image = load_input(path, temporal=True)
            if "reference" not in image.buffers:
                raise ValueError("Evaluation needs reference RGB")
            buffers = _tensors(image, model, temporal=True)
            history = model(buffers, history)
            prediction = _pixels(history)
            reference = image.buffers["reference"]
            # Evaluate learned outputs, excluding each clip's noisy initialization.
            if index:
                rows.append({"sequence_id": group, "manifest": str(path), "frame_offset": index,
                             "input": image_metrics(image.buffers["radiance"], reference),
                             "denoised": image_metrics(prediction, reference)})
            encoded_reference = encode_radiance(torch.from_numpy(reference))
            encoded_prediction = encode_radiance(history[0].permute(1, 2, 0).cpu())
            encoded_noisy = encode_radiance(torch.from_numpy(image.buffers["radiance"]))
            if index > 1:
                target_difference = encoded_reference - previous_reference
                temporal_errors.append({
                    "input": float(((encoded_noisy - previous_noisy) - target_difference).abs().mean()),
                    "denoised": float(((encoded_prediction - previous_prediction) - target_difference).abs().mean())})
            previous_prediction, previous_reference, previous_noisy = encoded_prediction, encoded_reference, encoded_noisy
    if not rows:
        raise ValueError("Evaluation needs at least two frames per selected clip")
    return {"images": rows, "input_mean": mean_metrics([row["input"] for row in rows]),
            "denoised_mean": mean_metrics([row["denoised"] for row in rows]),
            "temporal_l1": mean_metrics(temporal_errors) if temporal_errors else None,
            "temporal_domain": "ntasad_gamma_log_finite_difference_without_motion_warp",
            "initialization_frames_excluded": len(groups)}


def evaluate_sequences(checkpoint, root, sequence_ids, device_name="auto") -> dict:
    if not sequence_ids:
        raise ValueError("Evaluation requires explicit sequence IDs")
    device = select_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    audit = payload["dataset_split"].get("split_audit") or {}
    if set(sequence_ids) & set(audit.get("partitions", {}).get("test", [])):
        raise ValueError("Reserved test groups require --split-manifest and --partition test")
    report = _evaluate_groups(model, _selected_groups(root, sequence_ids))
    report.update(_report(checkpoint, payload, device), sequence_ids=sequence_ids,
                  overlaps_training_groups=sorted(set(sequence_ids) & set(payload["config"].get("train_sequences", []))))
    return report


def evaluate_partition(checkpoint, split_path, partition, device_name="auto") -> dict:
    if partition not in ("validation", "test"):
        raise ValueError("Evaluation partition must be validation or test")
    spec = read_split_manifest(split_path)
    audit = audit_dataset_split(split_path, allow_missing_test=partition != "test")
    device = select_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    frozen = payload["dataset_split"].get("split_audit") or {}
    if frozen.get("split_manifest_sha256") != spec["manifest_sha256"]:
        raise ValueError("Checkpoint was not trained under this frozen dataset split")
    for name in spec["partitions"]["train"] + spec["partitions"]["validation"]:
        if frozen.get("groups", {}).get(name) != audit["groups"][name]:
            raise ValueError("Training/validation data changed since checkpoint selection")
    groups = _selected_groups(spec["data_root"], spec["partitions"][partition])
    report = _evaluate_groups(model, groups)
    report.update(_report(checkpoint, payload, device), partition=partition,
                  sequence_ids=list(groups), split_manifest_sha256=spec["manifest_sha256"],
                  test_used_for_checkpoint_selection=False)
    return report
