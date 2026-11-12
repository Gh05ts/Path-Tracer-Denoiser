#!/usr/bin/env python3
"""Render cheap candidate views, then write an 18-clip training scene plan.

Existing captures are pinned. BMW's completed 36 clips are retained. The source
plan and all production images are untouched; this writes a new plan only.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
import numpy as np

from wavefront_denoiser.dataset_generation import (
    PROJECT_ROOT, build_configs, capture_lock, digest, read_plan, trajectory, write_json,
)
from wavefront_denoiser.frames import load_frame
from wavefront_denoiser.viewpoint_selection import candidate_anchors, descriptor, select_diverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=PROJECT_ROOT/"dataset_generation/plan-v4.json")
    parser.add_argument("--plan-output", type=Path, default=PROJECT_ROOT/"dataset_generation/plan-v4-18.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT.parent/"dataset/denoiser-v4-candidates")
    parser.add_argument("--renderer", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_plan.resolve()
    plan = read_plan(source)
    original = build_configs(source)
    output = args.output.resolve()
    report = {"source_plan_sha256": digest(source), "renderer_sha256": digest(args.renderer),
              "dimensions": [160, 90], "samples": 1, "reference_samples": 1,
              "selection": "material/normal/depth layout and camera pose; greedy diversity heuristic",
              "scenes": {}}
    with capture_lock(output):
        for scene in plan["scenes"]:
            if scene["partition"] != "train" or scene["id"] == "bmw":
                continue
            anchors = candidate_anchors(scene)
            config = deepcopy(next(c for c in original.values() if c["scene"] == scene["id"]))
            config.update(sequence_id=scene["id"]+"_candidates", width=160, height=90,
                          samples=1, reference_samples=1, max_depth=4,
                          frames=[trajectory(dict(scene, motion_scale=0), a, "pan_tilt")[0] for a in anchors])
            config_path = output/(scene["id"]+".json")
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError("Candidate inputs changed; use a fresh diagnostic output")
            write_json(config_path, config)
            directory = output/config["sequence_id"]
            complete = all((directory/f"frame_{i:06d}/manifest.json").exists() for i in range(len(anchors)))
            if not complete:
                with (output/(scene["id"]+".log")).open("a") as log:
                    subprocess.run([str(args.renderer.resolve()), str(config_path), str(directory)],
                                   cwd=PROJECT_ROOT.parent/"Wavefront-Path-Tracer", stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            features, records, eligible = [], [], []
            thumbnails = []
            for i, anchor in enumerate(anchors):
                frame = load_frame(directory/f"frame_{i:06d}")
                feature, metrics = descriptor(frame)
                records.append(dict(name=anchor["name"], **metrics))
                if metrics["eligible"]:
                    eligible.append(anchor)
                    features.append(feature)
                thumbnails.append(Image.fromarray(np.uint8(np.power(frame.buffers["albedo"].clip(0, 1), 1/2.2)*255)))
            pinned = ["view_00"] if scene["id"] == "bistro" else []
            indices, metrics = select_diverse(eligible, features, scene, pinned=pinned)
            selected = [deepcopy(eligible[i]) for i in indices]
            for a, anchor in enumerate(selected):
                # Alternate two disjoint sets of three motions; shift speed by
                # anchor index. Every motion x speed appears exactly once.
                selected_motions = range(3) if a % 2 == 0 else range(3, 6)
                anchor["motion_speeds"] = [[plan["motions"][m], ("slow", "medium", "fast")[(a+m) % 3]]
                                           for m in selected_motions]
            scene["anchors"] = selected
            selected_names = [a["name"] for a in selected]
            sheet = Image.new("RGB", (640, 114*6))
            draw = ImageDraw.Draw(sheet)
            for i, (anchor, thumbnail) in enumerate(zip(anchors, thumbnails)):
                x, y = (i % 4)*160, (i//4)*114
                sheet.paste(thumbnail, (x, y))
                label = ("KEEP " if anchor["name"] in selected_names else "")+anchor["name"]
                draw.text((x+2, y+92), label, fill="white")
            sheet.save(output/(scene["id"]+".png"))
            report["scenes"][scene["id"]] = dict(candidates=records, selected=selected_names,
                                                   pinned=pinned, **metrics)
            print(f"{scene['id']}: {len(anchors)} candidates -> {selected_names}", flush=True)
        # Scene manifests remain relative to the source plan's directory.
        for scene in plan["scenes"]:
            scene["scene_file"] = str((source.parent/scene["scene_file"]).resolve())
        # Assign original sample slots whenever a group exists; allocate new
        # groups in a separate region beyond the 1e9 convergence-replica region.
        plan["sample_starts"] = {g: c["sample_start"] for g, c in original.items()}
        next_slot = 1_200_000_000
        for scene in plan["scenes"]:
            for anchor in scene["anchors"]:
                for motion, speed in anchor.get("motion_speeds", []):
                    group = f"{scene['id']}_{anchor['name']}_{motion}_{speed}"
                    if group not in plan["sample_starts"]:
                        plan["sample_starts"][group] = next_slot
                        next_slot += plan["clip_sample_stride"]
        if next_slot >= 0xffffffff:
            raise ValueError("New groups exceed uint32 sample allocation")
        write_json(args.plan_output, plan)
        configs = build_configs(args.plan_output)
        report.update(plan_path=str(args.plan_output.resolve()), plan_sha256=digest(args.plan_output),
                      clips=len(configs), frames=sum(len(c["frames"]) for c in configs.values()))
        write_json(output/"selection.json", report)
        print(json.dumps({"clips": report["clips"], "frames": report["frames"]}))


if __name__ == "__main__":
    main()
