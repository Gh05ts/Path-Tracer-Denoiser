"""Matched-budget within-scene lighting ablation; never an independent scene test."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from .frames import load_frame, read_manifest
from .image_io import file_sha256


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def experiment_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("format") != "wavefront-lighting-experiment" or config.get("version") != 1:
        raise ValueError("Unsupported lighting experiment")
    allowed = {"format", "version", "width", "height", "samples", "max_depth", "reference_counts",
               "replicates", "max_noise_fraction", "seeds", "lighting", "control_condition", "cameras",
               "training", "baseline_checkpoint", "observed_benchmark", "extra_reference_counts",
               "minimum_reference_samples", "reuse_reference_captures"}
    if config.keys() - allowed:
        raise ValueError("Unknown lighting experiment option")
    for field in ("width", "height", "samples", "max_depth", "replicates"):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError(f"{field} must be positive integer")
    if config["replicates"] < 2:
        raise ValueError("Reference calibration requires independent replicas")
    counts = config.get("reference_counts")
    if (not isinstance(counts, list) or len(counts) < 2 or
        any(type(n) is not int or n < 1 for n in counts) or counts != sorted(set(counts))):
        raise ValueError("Reference counts must be positive and strictly increasing")
    config.setdefault("extra_reference_counts", {})
    config.setdefault("minimum_reference_samples", counts[0])
    config.setdefault("reuse_reference_captures", None)
    if type(config["minimum_reference_samples"]) is not int or config["minimum_reference_samples"] < 1:
        raise ValueError("Invalid minimum reference budget")
    if config["reuse_reference_captures"] is not None and not isinstance(config["reuse_reference_captures"], str):
        raise ValueError("Invalid reusable reference location")
    if not isinstance(config["extra_reference_counts"], dict):
        raise ValueError("Invalid extra reference counts")
    for extra in config["extra_reference_counts"].values():
        if not isinstance(extra, list) or not extra or extra != sorted(set(extra)) or any(type(n) is not int or n <= counts[-1] for n in extra):
            raise ValueError("Extra reference counts must extend the original ladder")
    fraction = config.get("max_noise_fraction")
    if type(fraction) not in (float, int) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Invalid reference noise threshold")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or
        any(type(n) is not int or not 0 <= n <= 0xFFFFFFFF for n in seeds)):
        raise ValueError("Need unique nonnegative training seeds")
    lighting = config.get("lighting")
    if not isinstance(lighting, dict) or len(lighting) < 2 or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not isinstance(value, dict)
            for name, value in lighting.items()):
        raise ValueError("Need named lighting interventions")
    if set(config["extra_reference_counts"]) - lighting.keys():
        raise ValueError("Extra reference budget for unknown lighting condition")
    if config.get("control_condition") not in lighting:
        raise ValueError("Control condition is missing")
    cameras = config.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {"train", "validation", "probe"}:
        raise ValueError("Need train, validation and probe camera lists")
    seen = set()
    for partition, frames in cameras.items():
        if not isinstance(frames, list) or not frames:
            raise ValueError("Every camera partition must be nonempty")
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) != {"position", "target"}:
                raise ValueError("Cameras require explicit position and target")
            for value in frame.values():
                if (not isinstance(value, list) or len(value) != 3 or
                    any(type(x) not in (int, float) or not np.isfinite(x) for x in value)):
                    raise ValueError("Camera vectors must be finite")
            if np.linalg.norm(np.subtract(frame["position"], frame["target"])) < 1e-6:
                raise ValueError("Degenerate camera")
            key = json.dumps(frame, sort_keys=True)
            if key in seen:
                raise ValueError("Camera duplicated within/across partitions")
            seen.add(key)
    training = config.get("training")
    if not isinstance(training, dict) or set(training) != {"epochs", "batch_size", "patch_size", "patches_per_frame",
                                                        "learning_rate", "linear_loss_weight", "base_channels", "cpu_threads"}:
        raise ValueError("Training settings must be explicit and shared by both arms")
    for field in ("epochs", "batch_size", "patch_size", "patches_per_frame", "base_channels", "cpu_threads"):
        if type(training[field]) is not int or training[field] < 1:
            raise ValueError("Invalid shared training budget")
    if not 4 <= training["patch_size"] <= min(config["width"], config["height"]):
        raise ValueError("Patch size is outside the image")
    for field in ("learning_rate", "linear_loss_weight"):
        value = training[field]
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0 or (field == "learning_rate" and value == 0):
            raise ValueError("Invalid shared loss/learning rate")
    if not 4 <= training["base_channels"] <= 128:
        raise ValueError("Invalid architecture size")
    if config["baseline_checkpoint"] is not None and not isinstance(config["baseline_checkpoint"], str):
        raise ValueError("Invalid baseline checkpoint")
    if config["observed_benchmark"] is not None and not isinstance(config["observed_benchmark"], str):
        raise ValueError("Invalid observed benchmark")
    return config


def training_groups(config: dict) -> dict[str, list[str]]:
    conditions = list(config["lighting"])
    control = config["control_condition"]
    # One common group provides exactly the same sky input/reference pixels.
    common = "train_shared_" + control
    return {"control": [common] + [f"train_control_repeat_{i}" for i in range(1, len(conditions))],
            "diverse": [common] + ["train_diverse_" + name for name in conditions if name != control]}


def capture_config(config: dict, condition: str, cameras: list[dict]) -> dict:
    return {"scene": "demo", "width": config["width"], "height": config["height"],
            "tile_width": config["width"], "tile_height": config["height"],
            "samples": config["samples"], "max_depth": config["max_depth"], "frames": cameras,
            "capture_lighting": config["lighting"][condition]}


def assert_fixed_guides(paths_by_condition: list[list[Path]]) -> None:
    """Reject accidental changes to visible non-light geometry/materials."""
    baseline = [load_frame(path) for path in paths_by_condition[0]]
    for paths in paths_by_condition[1:]:
        if len(paths) != len(baseline):
            raise ValueError("Lighting variants disagree on camera count")
        for original, path in zip(baseline, paths):
            frame = load_frame(path)
            if frame.metadata["camera"] != original.metadata["camera"]:
                raise ValueError("Lighting experiment changed a camera")
            if frame.metadata["geometry_fingerprint"] != original.metadata["geometry_fingerprint"]:
                raise ValueError("Lighting experiment changed base scene geometry")
            for key in ("albedo", "normal", "geometric_normal", "depth", "guide_valid"):
                if not np.array_equal(frame.buffers[key], original.buffers[key]):
                    raise ValueError(f"Lighting variant changed visible {key}; move emitter outside the camera")


def prepare_experiment(config_path: str | Path, output: str | Path, executable: str | Path,
                       renderer_root: str | Path | None = None) -> dict:
    from .reference_convergence import analyze_reference_convergence, capture_jobs, write_convergence_report

    config_path, output, executable = Path(config_path).resolve(), Path(output).resolve(), Path(executable).resolve()
    config = experiment_config(config_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(config_path.read_bytes())
    for name in ("capture_configs", "capture_logs", "data", "reference_convergence"):
        (output / name).mkdir()
    renderer_root = Path(renderer_root).resolve() if renderer_root else executable.parent.parent
    protocol_sha = file_sha256(config_path)
    def capture(description: dict, destination: Path, name: str) -> None:
        filename = output / "capture_configs" / (name + ".json")
        write_json(filename, description)
        with (output / "capture_logs" / (name + ".log")).open("w") as log:
            subprocess.run([str(executable), str(filename), str(destination)], cwd=renderer_root,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if config["reuse_reference_captures"]:
        reuse_root = (config_path.parent / config["reuse_reference_captures"]).resolve()
        original = json.loads((reuse_root.parent.parent / "protocol.json").read_text())
        for key in ("width", "height", "samples", "max_depth", "lighting", "reference_counts", "replicates"):
            if original[key] != config[key]:
                raise ValueError("Reusable reference protocol changed: " + key)
        if original["cameras"]["validation"] != config["cameras"]["validation"]:
            raise ValueError("Reusable calibration camera changed")
    reports, budgets, intervals = [], {}, []
    calibration_cursor = 100000000
    reused = []
    for condition in config["lighting"]:
        counts = config["reference_counts"] + config["extra_reference_counts"].get(condition, [])
        group = "validation_" + condition
        plan = {"sample_counts": counts, "replicates": config["replicates"],
                "max_noise_fraction": config["max_noise_fraction"], "sample_start": calibration_cursor,
                "split_manifest_sha256": protocol_sha, "partitions": {group: "validation"},
                "cases": [{"id": condition, "group": group,
                           "capture": capture_config(config, condition, config["cameras"]["validation"][:1])}]}
        for case, count, replica, description, relative in capture_jobs(plan):
            destination = output / "reference_convergence/captures" / relative
            name = f"reference_{condition}_{count}_{replica}"
            reusable = (config_path.parent / config["reuse_reference_captures"] / relative) if config["reuse_reference_captures"] else None
            if reusable and reusable.is_dir():
                frame = load_frame(reusable / "frame_000000")
                metadata = frame.metadata
                if (metadata["render"]["sample_start"] != description["sample_start"] or
                    metadata["render"]["samples"] != config["samples"] or
                    metadata["render"]["reference_samples"] != count):
                    raise ValueError("Reusable reference RNG ranges disagree with the extended plan")
                shutil.copytree(reusable, destination)
                write_json(output / "capture_configs" / (name + ".json"), description)
                reused.append(str(relative))
                print(f"Reuse calibration {condition}: {count} SPP, replica {replica}", flush=True)
            else:
                print(f"Calibrate {condition}: {count} SPP, replica {replica}", flush=True)
                capture(description, destination, name)
            end = description["sample_start"] + config["samples"] + count
            intervals.append((description["sample_start"], end))
            calibration_cursor = end
        report = analyze_reference_convergence(plan, output / "reference_convergence/captures")
        case = report["cases"][0]
        # Keep the original ladder's highest-level 4-SPP input fixed while
        # extending the reference budget, so adaptive qualification does not
        # improve just by selecting a different, noisier input realization.
        frames = [load_frame(output / "reference_convergence/captures" / condition / f"s{counts[-1]}" / f"r{i}" / "frame_000000") for i in range(config["replicates"])]
        anchor = np.mean([f.buffers["reference"].astype(np.float64) for f in frames], axis=0)
        original_input = load_frame(output / "reference_convergence/captures" / condition / f"s{config['reference_counts'][-1]}" / "r0/frame_000000").buffers["radiance"].astype(np.float64)
        luminance = anchor @ np.array([0.2126, 0.7152, 0.0722])
        highlight = (luminance >= np.quantile(luminance, 0.95)) & (luminance > 0)
        input_error = float(np.sqrt(np.square(original_input - anchor).mean()))
        highlight_error = float(np.sqrt(np.square(original_input[highlight] - anchor[highlight]).mean())) if highlight.any() else 0.0
        case["input_rmse_against_anchor"] = input_error
        case["fixed_input_reference_level"] = config["reference_counts"][-1]
        for row in case["levels"]:
            row["noise_fraction_of_input"] = row["replica_noise_rmse"] / input_error if input_error else None
            row["highlight_noise_fraction_of_input"] = row["highlight_noise_rmse"] / highlight_error if highlight_error else None
            row["qualifies"] = (row["noise_fraction_of_input"] is not None and row["highlight_noise_fraction_of_input"] is not None and
                                max(row["noise_fraction_of_input"], row["highlight_noise_fraction_of_input"]) <= config["max_noise_fraction"])
        selected = next((level["samples"] for i, level in enumerate(case["levels"])
                         if all(r["qualifies"] for r in case["levels"][i:])), None)
        case["selected_reference_samples"] = selected
        budgets[condition] = max(config["minimum_reference_samples"], selected) if selected else None
        reports.append(case)
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])) or calibration_cursor >= 200000000:
        raise ValueError("Calibration RNG ranges overlap")
    report = {"format": "wavefront-lighting-reference-convergence", "version": 1,
              "protocol_sha256": protocol_sha, "scope": "camera_sampling_at_fixed_integrator",
              "split_kind": "within_scene_lighting_and_camera_ablation", "cases": reports,
              "max_noise_fraction": config["max_noise_fraction"], "replicates": config["replicates"],
              "reference_samples_by_condition": budgets, "reused_captures": reused}
    write_convergence_report(report, output / "reference_convergence")
    if any(budget is None for budget in budgets.values()):
        raise ValueError("Reference budget did not qualify; inspect reference_convergence/report.json")
    groups = training_groups(config)
    names, cursor = {}, 200000000
    jobs = []
    for arm, ids in groups.items():
        for group in ids:
            if group not in names:
                condition = config["control_condition"] if arm == "control" else group.removeprefix("train_diverse_")
                names[group] = condition
                jobs.append((group, condition, "train"))
    for partition in ("validation", "probe"):
        for condition in config["lighting"]:
            group = partition + "_" + condition
            names[group] = condition
            jobs.append((group, condition, partition))
    inventory, paths = {}, {}
    for group, condition, partition in jobs:
        description = capture_config(config, condition, config["cameras"][partition])
        budget = budgets[condition]
        description.update(sequence_id=group, reference_samples=budget, sample_start=cursor)
        cursor += (config["samples"] + budget) * len(description["frames"])
        if cursor > 0xFFFFFFFF:
            raise ValueError("Experiment exceeds RNG sample range")
        print(f"Capture {group}: {len(description['frames'])} cameras, {budget} reference SPP", flush=True)
        capture(description, output / "data" / group, group)
        paths[group] = sorted((output / "data" / group).glob("*/manifest.json"))
        inventory[group] = [{"manifest": str(p.relative_to(output)), "manifest_sha256": file_sha256(p),
                             "image_sha256": read_manifest(p)["image_sha256"]} for p in paths[group]]
    # Train/validation image integrity and visible-guide invariance are checked
    # now. Probe pixels stay unopened until all checkpoints have been frozen.
    assert_fixed_guides([paths[name] for name in groups["control"] + groups["diverse"]])
    assert_fixed_guides([paths["validation_" + name] for name in config["lighting"]])
    index = {"format": "wavefront-lighting-data", "version": 1, "protocol_sha256": protocol_sha,
             "split_kind": "within_scene_lighting_and_camera_ablation", "reference_samples_by_condition": budgets,
             "training_groups": groups, "condition_by_group": names, "inventory": inventory,
             "probe_pixels_loaded": False, "guide_invariance_train_validation": True,
             "capture_executable_sha256": file_sha256(executable)}
    write_json(output / "dataset.json", index)
    return index


def paired_comparison(results: list[dict], seeds: list[int], conditions: list[str]) -> dict:
    comparisons = {}
    for condition in conditions:
        pairs = []
        for seed in seeds:
            values = {row["arm"]: row["probe"][condition]["denoised_mean"]
                      for row in results if row["seed"] == seed and row["arm"] in ("control", "diverse")}
            if set(values) != {"control", "diverse"}:
                raise ValueError("Missing matched training seed")
            control, diverse = values["control"], values["diverse"]
            pairs.append({"seed": seed, "control": control, "diverse": diverse,
                          "hdr_rmse_change_fraction": diverse["linear_rmse"] / control["linear_rmse"] - 1,
                          "display_psnr_gain_db": diverse["display_psnr_db"] - control["display_psnr_db"]})
        comparisons[condition] = {"pairs": pairs,
            "mean_hdr_rmse_change_fraction": float(np.mean([p["hdr_rmse_change_fraction"] for p in pairs])),
            "mean_display_psnr_gain_db": float(np.mean([p["display_psnr_gain_db"] for p in pairs]))}
    return comparisons
