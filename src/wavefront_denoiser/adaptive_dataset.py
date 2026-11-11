"""Plan and capture v5 companion layers without modifying the source v4 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from .adaptive_data import (NEW_LAYER_SPP, layer_layout, read_json, read_layer_capture,
                            read_v5_dataset, verify_v5_record)
from .dataset_generation import (DEFAULT_DATASET_ROOT, PROJECT_ROOT, asset_files, build_configs,
    capture_lock, digest, missing_ranges, read_plan, split_spec, verify_frame, write_json)
from .training_defaults import dataset_training_config, joint_training_config


DEFAULT_V5_PLAN = PROJECT_ROOT / "dataset_generation" / "plan-v5.json"


def v5_plan(path: Path) -> tuple[dict, Path, dict, dict]:
    path = Path(path).resolve()
    plan = read_json(path)
    if (plan.get("format") != "wavefront-ntasad-companion-plan" or plan.get("version") != 1
            or plan.get("new_layer_spp") != list(NEW_LAYER_SPP) or plan.get("reuse_spp") != 4
            or set(plan) != {"format", "version", "dataset_id", "source_plan", "new_layer_spp", "reuse_spp"}):
        raise ValueError("Unsupported v5 companion plan")
    name = plan["dataset_id"]
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Invalid v5 dataset ID")
    source_path = (path.parent / plan["source_plan"]).resolve()
    source_plan = read_plan(source_path)
    configs = build_configs(source_path)
    for config in configs.values():
        layer_layout(config, source_plan["report_reference_samples"])
    return plan, source_path, source_plan, configs


def _checked_write(path: Path, value) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Stored v5 metadata/config was changed: {path}")
    else:
        write_json(path, value)


def prepare_v5(plan_path: Path, source: Path, output: Path, *, renderer: Path | None = None,
               groups: list[str] | None = None) -> dict:
    """Plan without assets/source pixels; bind source lock and asset bytes before capture."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("v5 output and source datasets must be separate directories")
    plan, _, source_plan, configs = v5_plan(plan_path)
    selected = list(configs) if groups is None else groups
    if not selected or len(selected) != len(set(selected)) or set(selected) - configs.keys():
        raise ValueError("Need unique known capture groups")
    membership = {scene["id"]: scene["partition"] for scene in source_plan["scenes"]}
    lock = {"format": "wavefront-ntasad-v5-lock", "version": 1, "plan": plan,
            "source_root": str(source), "source_plan": source_plan, "configs": configs}
    # Planning is metadata-only and works while v4 is still being generated.
    # A separate binding can be created once its provenance lock is available.
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "v5_lock.json").exists() and any((output / name).exists() for name in ("base", "layers", "frames")):
        raise ValueError("Refusing to adopt v5 pixels without a provenance lock")
    _checked_write(output / "v5_lock.json", lock)
    dataset = {"format": "wavefront-ntasad-v5-dataset", "version": 1, "dataset_id": plan["dataset_id"],
        "reference_status": "fixed_budget_not_convergence_qualified",
        "partitions": {p: [g for g, c in configs.items() if membership[c["scene"]] == p]
                       for p in ("train", "validation", "test")},
        "groups": {g: {"scene": c["scene"], "frame_count": len(c["frames"]), "source_config": c,
                        "reserved_reference_samples": source_plan["report_reference_samples"],
                        "sample_layers": layer_layout(c, source_plan["report_reference_samples"])} for g, c in configs.items()}}
    _checked_write(output / "dataset.json", dataset)
    for group, config in configs.items():
        _checked_write(output / "source_configs" / f"{group}.json", config)
        companion = dict(config, sample_layers=dataset["groups"][group]["sample_layers"],
                         reserved_reference_samples=source_plan["report_reference_samples"])
        _checked_write(output / "capture_configs" / f"{group}.json", companion)
    split = split_spec(source_plan, configs, output)
    split["data_root"] = "base"
    for group in split["groups"]:
        split["groups"][group]["capture_config"] = f"source_configs/{group}.json"
    _checked_write(output / "split.json", split)
    if not (output / "train-uniform.json").exists():
        write_json(output / "train-uniform.json", dataset_training_config())
    if not (output / "train-joint.json").exists():
        write_json(output / "train-joint.json", joint_training_config())
    _checked_write(output / "train-joint-plan.json", {
        "format": "wavefront-ntasad-joint-training-plan", "version": 1,
        "status": "planned_requires_adaptive_trainer", "dataset": "dataset.json", "window_length": 5,
        "initialization": "uniform_4_spp", "epochs": 1000, "learning_rate": .001,
        "spatial_loss": "final_frame_L1", "temporal_loss": "last_pair_difference_L1",
        "loss_weights": [1, 1], "sampler_target_loss": None,
        "allocation_capacity": 63, "capacity_policy": "must_be_defined_before_joint_training",
        "sampler_evaluation": ["uniform", "seeded_random_equal_budget", "learned_fixed_denoiser", "joint"]})
    first = next(iter(configs.values()))
    _checked_write(output / "plan-summary.json", {
        "dataset_id": plan["dataset_id"], "source_dataset_id": source_plan["dataset_id"],
        "clips": len(configs), "frames": sum(len(c["frames"]) for c in configs.values()),
        "width": first["width"], "height": first["height"],
        "additional_samples_per_pixel": sum(NEW_LAYER_SPP), "reused_samples_per_pixel": 4,
        "additional_rgb_bytes_per_frame": first["width"] * first["height"] * 3 * 4 * len(NEW_LAYER_SPP),
        "copies_source_frame_exr": True, "reference_status": dataset["reference_status"]})
    if renderer is not None:
        renderer = Path(renderer).resolve()
        source_lock_path = source / "capture_lock.json"
        source_lock = read_json(source_lock_path)
        if (source_lock.get("format") != "wavefront-capture-lock" or source_lock.get("version") != 1
                or source_lock.get("plan") != source_plan or source_lock.get("configs") != configs
                or not source_lock.get("renderer_sha256")):
            raise ValueError("Source capture lock disagrees with the inherited plan/configs or has no renderer binding")
        if read_json(source / "split.json") != split_spec(source_plan, configs, source):
            raise ValueError("Source partition manifest disagrees with the inherited split")
        assets = source_lock.get("assets", {})
        for scene_file in sorted({configs[g]["scene_file"] for g in selected}):
            for path in asset_files(Path(scene_file)):
                if assets.get(str(path)) != digest(path):
                    raise ValueError(f"Source asset bytes changed or missing from its provenance lock: {path}")
        _checked_write(output / "capture_binding.json", {"source_lock_sha256": digest(source_lock_path),
            "source_renderer_sha256": source_lock["renderer_sha256"], "renderer_sha256": digest(renderer)})
    return dataset


def _recover(path: Path, output: Path, category: str, group: str) -> None:
    if path.exists():
        parent = output / "recovery.partial" / category / group
        parent.mkdir(parents=True, exist_ok=True)
        path.rename(parent / f"{path.name}-{time.time_ns()}")


def _copy_source(source: Path, target: Path, config: dict, index: int, output: Path, group: str) -> dict:
    if target.exists():
        metadata = verify_frame(target, config, index)
        if source.exists() and digest(source / "manifest.json") != digest(target / "manifest.json"):
            raise ValueError("Source manifest changed after v5 copy")
        return metadata
    metadata = verify_frame(source, config, index, pixels=True)
    staging = target.with_name(target.name + ".partial")
    _recover(staging, output, "base", group)
    staging.mkdir(parents=True)
    for filename in ("manifest.json", "frame.exr"):
        shutil.copy2(source / filename, staging / filename)
    copied = verify_frame(staging, config, index, pixels=True)
    if copied != metadata or digest(source / "manifest.json") != digest(staging / "manifest.json"):
        raise ValueError("Source changed while copying the completed frame")
    staging.rename(target)
    return copied


def capture_v5(output: Path, source: Path, renderer: Path, renderer_root: Path, groups: list[str], *,
               first_frame=0, end_frame=None) -> dict:
    """Capture ready source frames; missing source frames stay pending until the next invocation."""
    output, source, renderer = Path(output).resolve(), Path(source).resolve(), Path(renderer).resolve()
    dataset = read_v5_dataset(output)
    lock = read_json(output / "v5_lock.json")
    binding = read_json(output / "capture_binding.json")
    if lock["source_root"] != str(source):
        raise ValueError("Source dataset changed")
    if digest(source / "capture_lock.json") != binding["source_lock_sha256"]:
        raise ValueError("Source capture provenance lock changed")
    captured = skipped = pending = 0
    for group in groups:
        if group not in dataset["groups"]:
            raise ValueError(f"Unknown v5 group: {group}")
        config = lock["configs"][group]
        layer_config = dict(config, sample_layers=dataset["groups"][group]["sample_layers"],
                            reserved_reference_samples=dataset["groups"][group]["reserved_reference_samples"])
        if read_json(output / "capture_configs" / f"{group}.json") != layer_config:
            raise ValueError("Companion capture configuration changed")
        stop = len(config["frames"]) if end_frame is None else end_frame
        if not 0 <= first_frame < stop <= len(config["frames"]):
            raise ValueError("Frame range lies outside the source clip")
        ready, missing = [], []
        for index in range(first_frame, stop):
            relative = Path(group) / f"frame_{index:06d}"
            record = output / "frames" / relative
            if record.exists():
                verify_v5_record(output, group, index)
                skipped += 1
                continue
            source_frame, target = source / relative, output / "base" / relative
            if not source_frame.exists() and not target.exists():
                pending += 1
                continue
            base = _copy_source(source_frame, target, config, index, output, group)
            layer_path = output / "layers" / relative
            if layer_path.exists():
                read_layer_capture(layer_path, base, layer_config["sample_layers"], pixels=True)
            else:
                _recover(layer_path.with_name(layer_path.name + ".partial"), output, "layers", group)
                missing.append(index)
            ready.append(index)
        for first, end in missing_ranges(missing):
            if digest(renderer) != binding["renderer_sha256"]:
                raise ValueError("Companion renderer binary changed")
            logs = output / "logs"
            logs.mkdir(exist_ok=True)
            command = [str(renderer), str(output / "capture_configs" / f"{group}.json"),
                       str(output / "layers" / group), "--start-frame", str(first), "--end-frame", str(end)]
            with (logs / f"{group}.log").open("a") as log:
                log.write(json.dumps({"command": command, "started_unix": time.time()}) + "\n")
                log.flush()
                subprocess.run(command, cwd=renderer_root, stdout=log, stderr=subprocess.STDOUT, check=True)
            if digest(renderer) != binding["renderer_sha256"]:
                raise ValueError("Companion renderer binary changed during capture")
        for index in ready:
            relative = Path(group) / f"frame_{index:06d}"
            base_path, layer_path = output / "base" / relative, output / "layers" / relative
            base = verify_frame(base_path, config, index)
            read_layer_capture(layer_path, base, layer_config["sample_layers"], pixels=True)
            record_path = output / "frames" / relative
            staging = record_path.with_name(record_path.name + ".partial")
            _recover(staging, output, "records", group)
            staging.mkdir(parents=True)
            write_json(staging / "manifest.json", {"format": "wavefront-ntasad-v5-frame", "version": 1,
                "sequence_id": group, "frame_index": index,
                "source_manifest_sha256": digest(base_path / "manifest.json"),
                "layer_manifest_sha256": digest(layer_path / "manifest.json")})
            staging.rename(record_path)
            captured += 1
        print(f"{group}: {len(ready)} v5 frames published", flush=True)
    result = {"captured_frames": captured, "skipped_frames": skipped, "pending_source_frames": pending,
              "groups": groups, "reference_status": dataset["reference_status"]}
    write_json(output / "last-capture.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_V5_PLAN)
    parser.add_argument("--source", type=Path, required=True, help="Original locked v4 dataset (read only)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("plan", "capture"), default="plan")
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--renderer-root", type=Path)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--partitions", nargs="+", choices=("train", "validation", "test"), default=["train", "validation"])
    parser.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"))
    args = parser.parse_args(argv)
    plan, _, source_plan, configs = v5_plan(args.plan)
    output = (args.output or DEFAULT_DATASET_ROOT / plan["dataset_id"]).resolve()
    if args.stage == "capture" and (args.renderer is None or args.renderer_root is None):
        parser.error("Capture requires explicit --renderer and --renderer-root")
    membership = {s["id"]: s["partition"] for s in source_plan["scenes"]}
    if args.groups and (set(args.groups) - configs.keys() or len(args.groups) != len(set(args.groups))):
        parser.error("--groups must be unique known source groups")
    groups = args.groups or [g for g, c in configs.items() if membership[c["scene"]] in args.partitions]
    # Check source/output separation before capture_lock can create directories.
    source = args.source.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("v5 output and source datasets must be separate directories")
    with capture_lock(output):
        prepare_v5(args.plan, source, output, renderer=args.renderer if args.stage == "capture" else None, groups=groups)
        if args.stage == "capture":
            first, end = args.frame_range or (0, None)
            result = capture_v5(output, source, args.renderer, args.renderer_root.resolve(), groups,
                                first_frame=first, end_frame=end)
            print(json.dumps(result, indent=2))
        else:
            print(f"Planned {len(configs)} inherited clips in {output}; no frames rendered or copied.")
    return 0
