"""Single-image HDR inference and explicit held-out evaluation."""

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .spatial import discover_spatial_frames
from .dataset_splits import audit_dataset_split, read_split_manifest
from .spatial_checkpoint import load_spatial_checkpoint, select_device
from .spatial_io import file_sha256, load_spatial_input, write_rgb_exr
from .spatial_metrics import display_rgb, image_metrics
from .spatial_training import evaluate_paths, predict_radiance


def comparison_preview(path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    height, width = images[0][1].shape[:2]
    canvas = Image.new("RGB", (width * len(images), height + 20), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(images):
        rgb = np.clip(np.rint(display_rgb(pixels) * 255), 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (index * width, 20))
        draw.text((index * width + 3, 3), label, fill=(255, 255, 255))
    canvas.save(path)


def denoise_image(checkpoint: str | Path, source: str | Path, output: str | Path,
                  device_name: str = "auto") -> dict:
    output = Path(output)
    staging = output.with_name(output.name + ".partial")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Inference output or staging directory already exists: {output}")
    device = select_device(device_name)
    model, payload = load_spatial_checkpoint(checkpoint, device)
    image = load_spatial_input(source)
    prediction = predict_radiance(model, image.buffers, device)
    report = {"format": "wavefront-spatial-result", "version": 1,
              "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(checkpoint),
              "source": str(image.source.resolve()), "source_sha256": image.sha256,
              "device": str(device), "model_epoch": payload.get("epoch"),
              "width": prediction.shape[1], "height": prediction.shape[0],
              "minimum_radiance": float(prediction.min()), "maximum_radiance": float(prediction.max()),
              "output": "denoised.exr", "color_space": "linear_scene_referred_rgb"}
    previews = [("Noisy input", image.buffers["radiance"]), ("Denoised", prediction)]
    if "reference" in image.buffers:
        reference = image.buffers["reference"]
        report["metrics"] = {"input": image_metrics(image.buffers["radiance"], reference),
                             "denoised": image_metrics(prediction, reference)}
        previews.append(("Reference", reference))
    staging.mkdir(parents=True, exist_ok=False)
    try:
        write_rgb_exr(staging / "denoised.exr", prediction)
        comparison_preview(staging / "comparison.png", previews)
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def evaluate_spatial(checkpoint: str | Path, root: str | Path, sequence_ids: list[str],
                     device_name: str = "auto") -> dict:
    if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("Evaluation requires unique explicit capture group IDs")
    device = select_device(device_name)
    model, payload = load_spatial_checkpoint(checkpoint, device)
    held_out = payload.get("dataset_split", {}).get("partitions", {}).get("test", [])
    if set(sequence_ids) & set(held_out):
        raise ValueError("Reserved test groups require explicit --split-manifest and --partition test evaluation")
    groups = discover_spatial_frames(root)
    missing = set(sequence_ids) - groups.keys()
    if missing:
        raise ValueError(f"Missing evaluation groups: {sorted(missing)}")
    paths = [path for name in sequence_ids for path in groups[name]]
    report = evaluate_paths(model, paths, device)
    report.update(checkpoint_sha256=file_sha256(checkpoint), device=str(device),
                  sequence_ids=sequence_ids,
                  overlaps_training_groups=sorted(set(sequence_ids) & set(payload.get("config", {}).get("train_sequences", []))))
    return report


def evaluate_partition(checkpoint: str | Path, split_path: str | Path, partition: str,
                       device_name: str = "auto") -> dict:
    if partition not in ("validation", "test"):
        raise ValueError("Evaluation partition must be validation or test")
    spec = read_split_manifest(split_path)
    audit = audit_dataset_split(split_path, allow_missing_test=partition != "test")
    device = select_device(device_name)
    model, payload = load_spatial_checkpoint(checkpoint, device)
    provenance = payload.get("dataset_split", {})
    if provenance.get("split_manifest_sha256") != spec["manifest_sha256"]:
        raise ValueError("Checkpoint was not trained under this frozen dataset split")
    # A changed train/validation inventory invalidates the provenance claim.
    for name in spec["partitions"]["train"] + spec["partitions"]["validation"]:
        if provenance.get("groups", {}).get(name) != audit["groups"][name]:
            raise ValueError("Training/validation data changed since checkpoint selection")
    groups = discover_spatial_frames(spec["data_root"])
    selected = spec["partitions"][partition]
    paths = [path for group in selected for path in groups[group]]
    report = evaluate_paths(model, paths, device)
    report.update(partition=partition, sequence_ids=selected, checkpoint_sha256=file_sha256(checkpoint),
                  checkpoint_epoch=payload.get("epoch"), split_manifest_sha256=spec["manifest_sha256"],
                  device=str(device), test_used_for_checkpoint_selection=False)
    return report
