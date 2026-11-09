#!/usr/bin/env python3
"""Start fresh spatial and recurrent runs against the same independent scene split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavefront_denoiser.dataset_generation import DEFAULT_DATASET_ROOT, DEFAULT_PLAN, capture_lock, read_plan, verify_frame
from wavefront_denoiser.dataset_splits import audit_dataset_split, read_split_manifest


def main():
    here = Path(__file__).resolve().parent
    dataset_id = read_plan(DEFAULT_PLAN)["dataset_id"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT / dataset_id)
    parser.add_argument("--output", type=Path, default=here.parent / "runs" / (dataset_id+"-models"))
    parser.add_argument("--models", nargs="+", choices=("spatial", "recurrent"), default=["spatial", "recurrent"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recurrent-resume", type=Path, help="Optional explicit resume; otherwise both models start fresh")
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not (dataset / "capture_lock.json").is_file():
        parser.error("Generate the locked dataset before training")
    for model in args.models:
        if (output / model).exists():
            parser.error(f"Run directory exists; choose a new --output: {output / model}")
    if args.recurrent_resume and "recurrent" not in args.models:
        parser.error("--recurrent-resume requires the recurrent model")
    with capture_lock(dataset):
        spec = read_split_manifest(dataset / "split.json")
        audit_dataset_split(dataset / "split.json", allow_missing_test=True)
        locked = json.loads((dataset / "capture_lock.json").read_text())
        # Training verifies selected EXR bytes; test images are never opened.
        for partition in ("train", "validation"):
            for group in spec["partitions"][partition]:
                config = locked["configs"][group]
                for index in range(len(config["frames"])):
                    verify_frame(dataset / group / f"frame_{index:06d}", config, index)
        print("Train/validation inventory verified. Test pixels were not read.", flush=True)
        if "spatial" in args.models:
            from wavefront_denoiser.spatial_training import train_spatial
            train_spatial(dataset / "train-spatial.json", output / "spatial", args.device)
        if "recurrent" in args.models:
            from wavefront_denoiser.recurrent_training import train_recurrent
            train_recurrent(dataset / "train-recurrent.json", output / "recurrent", args.device,
                            resume=args.recurrent_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
