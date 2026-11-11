#!/usr/bin/env python3
"""Train the single spatial/temporal NTASAD denoiser on the independent split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavefront_denoiser.dataset_generation import (
    DEFAULT_DATASET_ROOT, DEFAULT_PLAN, capture_lock, read_plan, verify_frame, write_json,
)
from wavefront_denoiser.dataset_splits import audit_dataset_split, read_split_manifest
from wavefront_denoiser.training_defaults import dataset_training_config


def main(argv=None):
    here = Path(__file__).resolve().parent
    dataset_id = read_plan(DEFAULT_PLAN)["dataset_id"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT / dataset_id)
    parser.add_argument("--output", type=Path, default=here.parent / "runs" / (dataset_id+"-denoiser"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, help="Explicit checkpoint resume; otherwise initialize fresh weights")
    args = parser.parse_args(argv)
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not (dataset / "capture_lock.json").is_file():
        parser.error("Generate the locked dataset before training")
    if output.exists():
        parser.error(f"Run directory exists; choose a new --output: {output}")
    with capture_lock(dataset):
        spec = read_split_manifest(dataset / "split.json")
        audit_dataset_split(dataset / "split.json", allow_missing_test=True)
        locked = json.loads((dataset / "capture_lock.json").read_text())
        # Verify train/validation EXR bytes; never open reserved test pixels.
        for partition in ("train", "validation"):
            for group in spec["partitions"][partition]:
                config = locked["configs"][group]
                for index in range(len(config["frames"])):
                    verify_frame(dataset / group / f"frame_{index:06d}", config, index)
        # Captures started before unification have two obsolete training sidecars.
        # Create the one current config without changing capture provenance/images.
        config_path = dataset / "train.json"
        if not config_path.exists():
            write_json(config_path, dataset_training_config())
        print("Train/validation inventory verified. Test pixels were not read.", flush=True)
        from wavefront_denoiser.training import train_denoiser
        train_denoiser(config_path, output, args.device, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
