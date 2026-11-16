"""Initialize and run two-machine deferred v4 backfills."""

import argparse
import json
from pathlib import Path

from ..dataset_generation import capture_lock, write_json
from .backfill import (BackfillCoordinator, generation_ready, install_metadata, prepare_plan,
                       run_worker, validate_backfill_machine)
from .cli import settings
from .plan import identifier, prefix
from .worker import connection


def renderer_argument(value):
    try:
        version, path = value.split("=", 1)
        version = int(version)
        if version < 1 or not path:
            raise ValueError
        return version, Path(path).resolve()
    except ValueError:
        raise argparse.ArgumentTypeError("Use VERSION=PATH, for example 11=../dataset/bin/dataset_capture-base-integrator11")


def main(argv=None):
    workspace = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ("init", "check", "worker", "status", "release"):
        cmd = sub.add_parser(name)
        commands[name] = cmd
        cmd.add_argument("--destination", required=True, metavar="r2://BUCKET/PREFIX")
        cmd.add_argument("--settings-file", type=Path)
        cmd.add_argument("--run-id", default="denoiser-v4-backfill")
        cmd.add_argument("--output", type=Path, default=workspace / "dataset/denoiser-v4")
    commands["init"].add_argument("--generation-run-id", default="denoiser-v4")
    for name in ("worker", "check"):
        commands[name].add_argument("--workspace", type=Path, default=workspace)
        commands[name].add_argument("--renderer", action="append", type=renderer_argument, default=[],
                                    metavar="VERSION=PATH", help="Repeat for each required source integrator")
        commands[name].add_argument("--scenes", nargs="+", help="Optional scene subset, e.g. bmw or bistro")
    commands["worker"].add_argument("--worker-id", required=True)
    commands["worker"].add_argument("--max-clips", type=int)
    commands["release"].add_argument("--group", required=True)
    commands["release"].add_argument("--claim-token", required=True)
    args = parser.parse_args(argv)
    identifier(args.run_id)
    if args.command == "worker" and args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    settings(args.settings_file)
    archiver, store = connection(args.output, args.destination)
    plan_key = f"_backfill/{args.run_id}/plan.json"
    existing = store.get(plan_key)
    if args.command == "init":
        generation_record = store.get(f"{prefix(args.generation_run_id)}/plan.json")
        if generation_record is None:
            parser.error("Completed distributed generation plan is missing")
        generation = generation_record[0]
        generation_ready(store, generation)
        with capture_lock(args.output):
            install_metadata(store, generation, args.output)
            # An interrupted init reuses the frozen plan even after some scene
            # receipts have already been upgraded by completed backfill tasks.
            if existing:
                plan = existing[0]
                if (plan["generation_run_id"] != generation["run_id"]
                        or plan["capture_lock_sha256"] != generation["capture_lock_sha256"]
                        or plan["destination"] != archiver.destination or plan["endpoint"] != archiver.native.endpoint):
                    raise ValueError("Existing backfill belongs to another generation/destination")
            else:
                plan = prepare_plan(store, generation, args.output, run_id=args.run_id,
                                    destination=archiver.destination, endpoint=archiver.native.endpoint)
            BackfillCoordinator(store, plan).initialize()
            write_json(args.output / "backfill/plan.json", plan)
        print(json.dumps({"backfill_clips": len(plan["tasks"]),
                          "layer_clips": sum(t["new_layers"] for t in plan["tasks"].values()),
                          "frames": sum(t["frame_count"] for t in plan["tasks"].values()),
                          "renderers": {str(t["integrator_version"]): t["renderer_sha256"]
                                        for t in plan["tasks"].values() if t["new_layers"]}}, indent=2))
        return 0
    if existing is None:
        parser.error("Initialize backfills first: distributed/backfill.py init ...")
    plan = existing[0]
    if plan["destination"] != archiver.destination or plan["endpoint"] != archiver.native.endpoint:
        raise ValueError("Backfill destination/account changed")
    coordinator = BackfillCoordinator(store, plan)
    coordinator.require_ready()
    if args.command == "status":
        print(json.dumps(coordinator.status(), indent=2))
    elif args.command == "release":
        coordinator.release(args.group, args.claim_token)
        print("Backfill claim released; stop its old worker before starting a replacement.")
    else:
        versions = [version for version, _ in args.renderer]
        if len(versions) != len(set(versions)):
            parser.error("Specify each renderer integrator only once")
        scenes = set(args.scenes) if args.scenes else None
        known = {t["scene"] for t in plan["tasks"].values()}
        if scenes is not None and scenes - known:
            parser.error(f"Unknown backfill scenes: {sorted(scenes-known)}")
        with capture_lock(args.output):
            generation = store.get(f"{prefix(plan['generation_run_id'])}/plan.json")[0]
            install_metadata(store, generation, args.output)
            write_json(args.output / "backfill/plan.json", plan)
            validate_backfill_machine(plan, args.workspace, dict(args.renderer), scenes=scenes)
        if args.command == "check":
            print("Backfill source renderers/assets verified; no task claimed or rendered.")
        else:
            print(json.dumps(run_worker(plan, args.output, args.workspace, dict(args.renderer),
                                        args.destination, args.worker_id, scenes=scenes, max_clips=args.max_clips), indent=2))
    return 0
