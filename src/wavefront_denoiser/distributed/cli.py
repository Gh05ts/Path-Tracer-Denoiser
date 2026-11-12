"""Prepare, initialize, inspect and run the R2-backed clip coordinator."""

import argparse
import ast
import json
import os
from pathlib import Path

from ..b2_archive import file_hashes
from ..dataset_generation import capture_lock, write_json
from .coordinator import Coordinator
from .plan import prepare, install_bundle, prefix
from .worker import connection, run_worker, validate_machine


def settings(path):
    """Read only literal local credential constants; never execute the launcher."""
    if path is None:
        return
    names = {"ACCOUNT_ID": "R2_ACCOUNT_ID", "ACCESS_KEY_ID": "R2_ACCESS_KEY_ID",
             "SECRET_ACCESS_KEY": "R2_SECRET_ACCESS_KEY", "ENDPOINT_URL": "R2_ENDPOINT_URL"}
    for statement in ast.parse(Path(path).read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in names and names[target.id] not in os.environ:
                    if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                        if statement.value.value:
                            os.environ[names[target.id]] = statement.value.value


def main(argv=None):
    project = Path(__file__).resolve().parents[3]
    workspace = project.parent
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("prepare", help="Freeze the current dataset locally; no uploads or rendering")
    local.add_argument("--output", type=Path, default=workspace / "dataset/denoiser-v4")
    local.add_argument("--workspace", type=Path, default=workspace)
    local.add_argument("--prepared", type=Path, required=True)
    local.add_argument("--primary-worker", default="3090-a")
    local.add_argument("--run-id", default="denoiser-v4")
    commands = {}
    for command in ("init", "worker", "check", "status", "release"):
        cmd = sub.add_parser(command)
        commands[command] = cmd
        cmd.add_argument("--destination", required=True, metavar="r2://BUCKET/PREFIX")
        cmd.add_argument("--settings-file", type=Path, help="Optional local R2 launcher settings; environment takes precedence")
        cmd.add_argument("--run-id", default="denoiser-v4")
        cmd.add_argument("--output", type=Path, default=workspace / "dataset/denoiser-v4")
    commands["init"].add_argument("--prepared", type=Path, required=True)
    for command in ("worker", "check"):
        commands[command].add_argument("--workspace", type=Path, default=workspace)
        commands[command].add_argument("--renderer", type=Path, required=True)
    commands["worker"].add_argument("--worker-id", required=True)
    commands["worker"].add_argument("--max-clips", type=int)
    commands["release"].add_argument("--group", required=True)
    commands["release"].add_argument("--claim-token", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare(args.output, args.workspace, args.prepared,
                       primary_worker=args.primary_worker, run_id=args.run_id)
        print(json.dumps({"prepared": str(args.prepared.resolve()), "clips": len(plan["tasks"]),
            "already_archived": sum(t["seed"] is not None for t in plan["tasks"].values()),
            "primary_local_progress": sum(t["affinity"] is not None for t in plan["tasks"].values()),
            "capture_lock_sha256": plan["capture_lock_sha256"]}, indent=2))
        return 0
    settings(args.settings_file)
    archiver, store = connection(args.output, args.destination)
    if args.command == "init":
        plan = json.loads((args.prepared / "plan.json").read_text())
        if plan["run_id"] != args.run_id:
            parser.error("Run ID disagrees with the prepared plan")
        with capture_lock(args.output):
            for name, expected in plan["metadata_files"].items():
                if file_hashes(args.output / name) != expected:
                    raise ValueError("Dataset metadata changed after preparation")
            Coordinator(store, plan).initialize(args.prepared)
            write_json(args.output / "distributed/active.json", {
                "run_id": plan["run_id"], "capture_lock_sha256": plan["capture_lock_sha256"]})
        print("Distributed coordinator initialized; no worker or renderer started.")
        return 0
    record = store.get(f"{prefix(args.run_id)}/plan.json")
    if record is None:
        parser.error("Initialize the distributed coordinator on the primary machine first")
    plan = record[0]
    coordinator = Coordinator(store, plan)
    coordinator.require_ready()
    if args.command == "status":
        print(json.dumps(coordinator.status(), indent=2))
    elif args.command == "release":
        coordinator.release(args.group, args.claim_token)
        print("Claim released. Start a worker to resume the task.")
    elif args.command in ("worker", "check"):
        if args.command == "worker" and args.max_clips is not None and args.max_clips < 1:
            parser.error("--max-clips must be positive")
        with capture_lock(args.output):
            bundle = args.output / "distributed/metadata.tar"
            if not bundle.exists() or file_hashes(bundle) != plan["bundle"]:
                store.download(f"{prefix(args.run_id)}/metadata.tar", bundle, plan["bundle"])
            install_bundle(bundle, args.output, plan)
            write_json(args.output / "distributed/plan.json", plan)
        if args.command == "check":
            validate_machine(plan, args.workspace, args.renderer)
            print("Locked renderer and all scene/texture/geometry bytes verified; no task claimed or rendered.")
        else:
            print(json.dumps(run_worker(plan, args.output, args.workspace, args.renderer,
                args.destination, args.worker_id, max_clips=args.max_clips), indent=2))
    return 0
