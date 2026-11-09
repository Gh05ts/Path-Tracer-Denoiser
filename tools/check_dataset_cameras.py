#!/usr/bin/env python3
"""Check every planned camera cheaply, pooled per scene; save representative sheets."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wavefront_denoiser.dataset_generation import (
    DEFAULT_DATASET_ROOT, DEFAULT_PLAN, build_configs, capture_lock, prepare, read_plan, run_captures, write_json,
)
from wavefront_denoiser.frames import load_frame


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan_path, executable = args.plan.resolve(), args.renderer.resolve()
    plan, original = read_plan(plan_path), build_configs(plan_path)
    output = (args.output or DEFAULT_DATASET_ROOT / (plan["dataset_id"]+"-cameras")).resolve()
    pooled, mapping = {}, {}
    for scene in plan["scenes"]:
        groups = [g for g, c in original.items() if c["scene"] == scene["id"]]
        config = deepcopy(original[groups[0]])
        config.update(sequence_id=scene["id"]+"_camera_checks", width=160, height=90,
                      samples=1, reference_samples=4, max_depth=4, frames=[])
        rows = []
        for group in groups:
            indices = []
            for i in range(len(original[group]["frames"])):
                indices.append(len(config["frames"]))
                config["frames"].append(original[group]["frames"][i])
            rows.append({"group": group, "indices": indices})
        pooled[config["sequence_id"]] = config
        mapping[config["sequence_id"]] = rows
    with capture_lock(output):
        prepare(plan_path, output, pooled, executable)
        run_captures(pooled, list(pooled), output, executable, here.parent.parent/"Wavefront-Path-Tracer")
        stats, flagged, motion_summaries = [], [], []
        animation_dir = output / "motion-previews"
        animation_dir.mkdir(exist_ok=True)
        for group, rows in mapping.items():
            sheet = Image.new("RGB", (160*3, 110*len(rows)+28))
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 6), f"{pooled[group]['scene']} | CAMERA CHECK | reference 4 SPP", fill="white")
            first = load_frame(output/group/"frame_000000")
            exposure = min(1., 1./max(float(np.quantile(first.buffers["reference"], .99)), 1e-8))
            for row, entry in enumerate(rows):
                animation, clip_stats = [], []
                for planned_frame, index in enumerate(entry["indices"]):
                    frame = load_frame(output/group/f"frame_{index:06d}")
                    rgb = frame.buffers["reference"].clip(0)*exposure
                    display = np.power(rgb/(1+rgb), 1/2.2)
                    if planned_frame in (0, 4, 7):
                        col = (0, 4, 7).index(planned_frame)
                        sheet.paste(Image.fromarray(np.uint8(display.clip(0, 1)*255)), (160*col, 110*row+28))
                        label = entry["group"].removeprefix(pooled[group]["scene"]+"_")
                        draw.text((160*col+2, 110*row+119), label, fill="white")
                    depth = frame.buffers["depth"][frame.buffers["guide_valid"] == 1]
                    near = float(np.mean(depth < .10)) if depth.size else 0.
                    record = {"group": entry["group"], "planned_frame": planned_frame,
                              "median_depth": float(np.median(depth)) if depth.size else None,
                              "near_surface_fraction": near, "preview_exposure": exposure}
                    valid = frame.buffers["history_valid"][..., 0] == 1
                    motion = np.linalg.norm(frame.buffers["motion"], axis=-1)[valid]
                    scale = plan["width"]/frame.metadata["width"]
                    record.update(history_valid_fraction=float(valid.mean()) if planned_frame else None,
                        motion_p50_pixels_at_dataset_resolution=float(np.median(motion)*scale) if planned_frame and motion.size else None,
                        motion_p95_pixels_at_dataset_resolution=float(np.quantile(motion, .95)*scale) if planned_frame and motion.size else None,
                        reference_samples=frame.metadata["render"]["reference_samples"])
                    stats.append(record)
                    clip_stats.append(record)
                    preview = Image.new("RGB", (320, 114))
                    preview.paste(Image.fromarray(np.uint8(display.clip(0, 1)*255)), (0, 24))
                    albedo = np.power(frame.buffers["albedo"].clip(0, 1), 1/2.2)
                    preview.paste(Image.fromarray(np.uint8(albedo*255)), (160, 24))
                    labels = ImageDraw.Draw(preview)
                    labels.text((3, 5), f"Reference 4 SPP | frame {planned_frame}", fill="white")
                    labels.text((163, 5), "Albedo guide (no noise)", fill="white")
                    animation.append(preview)
                    if near > .25:
                        flagged.append(record)
                animation[0].save(animation_dir/(entry["group"]+".gif"), save_all=True,
                    append_images=animation[1:], duration=[150]*7+[600], loop=0, disposal=2)
                values = [r["motion_p95_pixels_at_dataset_resolution"] for r in clip_stats
                          if r["motion_p95_pixels_at_dataset_resolution"] is not None]
                motion_summaries.append({"group": entry["group"],
                    "median_frame_p95_pixels_at_dataset_resolution": float(np.median(values)) if values else None,
                    "animation": str((animation_dir/(entry["group"]+".gif")).relative_to(output))})
            sheet.save(output/f"{group}.png")
        write_json(output/"camera-checks.json", {"checked_cameras": len(stats), "flagged": flagged,
                   "diagnostic_reference_samples": 4, "motion_resolution": [plan["width"], plan["height"]],
                   "clips": motion_summaries, "cameras": stats})
        print(json.dumps({"checked_cameras": len(stats), "flagged": flagged, "sheets": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
