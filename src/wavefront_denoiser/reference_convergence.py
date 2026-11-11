"""Independent Monte Carlo reference replicates at fixed scene/camera/settings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from .dataset_splits import read_split_manifest
from .frames import load_frame
from .metrics import image_metrics, mean_metrics


def convergence_plan(path: str | Path) -> dict:
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict) or plan.get("format") != "wavefront-reference-plan" or plan.get("version") != 1:
        raise ValueError("Unsupported reference convergence plan")
    counts = plan.get("sample_counts")
    if not isinstance(counts, list) or len(counts) < 2 or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("Need at least two positive integer sample counts")
    if counts != sorted(set(counts)):
        raise ValueError("Reference sample counts must be unique and increasing")
    replicates = plan.get("replicates", 2)
    if type(replicates) is not int or replicates < 2:
        raise ValueError("At least two independent reference replicates are required")
    fraction = plan.get("max_noise_fraction", 0.1)
    if type(fraction) not in (int, float) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("max_noise_fraction must be between zero and one")
    start = plan.get("sample_start", 1000000)
    if type(start) is not int or not 0 <= start <= 0xFFFFFFFF:
        raise ValueError("Invalid convergence sample_start")
    spec = read_split_manifest(path.parent / plan["split_manifest"])
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Need nonempty calibration cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
            raise ValueError("Invalid calibration case ID")
        if case["id"] in ids:
            raise ValueError("Duplicate calibration case ID")
        ids.add(case["id"])
        group = case.get("group")
        if group not in spec["groups"] or spec["membership"][group] == "test":
            raise ValueError("Calibration may use only registered train/validation groups")
        capture = case.get("capture")
        if not isinstance(capture, dict) or capture.get("scene") != spec["groups"][group]["scene"]:
            raise ValueError("Calibration scene disagrees with split declaration")
        if not isinstance(capture.get("frames"), list) or len(capture["frames"]) != 1:
            raise ValueError("Each calibration case needs exactly one fixed camera")
        samples = capture.get("samples", 4)
        if type(samples) is not int or samples < 1:
            raise ValueError("Calibration input sample count must be positive")
    plan = dict(plan, replicates=replicates, max_noise_fraction=fraction, sample_start=start)
    plan["split_manifest"] = spec["manifest_path"]
    plan["split_manifest_sha256"] = spec["manifest_sha256"]
    plan["partitions"] = spec["membership"]
    return plan


def capture_jobs(plan: dict):
    cursor = plan["sample_start"]
    for case in plan["cases"]:
        for samples in plan["sample_counts"]:
            for replica in range(plan["replicates"]):
                config = dict(case["capture"])
                config["samples"] = config.get("samples", 4)
                config.update(sequence_id=f"reference_{case['id']}_s{samples}_r{replica}",
                              sample_start=cursor, reference_samples=samples)
                cursor += config["samples"] + samples
                if cursor > 0xFFFFFFFF:
                    raise ValueError("Convergence experiment exceeds the RNG sample index range")
                yield case, samples, replica, config, Path(case["id"]) / f"s{samples}" / f"r{replica}"


def _ratio(noise: float, input_error: float) -> float | None:
    if input_error == 0:
        return 0.0 if noise == 0 else None
    return noise / input_error


def analyze_reference_convergence(plan: dict, captures: str | Path) -> dict:
    captures = Path(captures)
    jobs = list(capture_jobs(plan))
    loaded = {}
    for case, samples, replica, config, relative in jobs:
        frame = load_frame(captures / relative / "frame_000000")
        render = frame.metadata["render"]
        if (render["sample_start"], render["samples"], render["reference_sample_start"], render["reference_samples"]) != (
                config["sample_start"], config["samples"], config["sample_start"] + config["samples"], samples):
            raise ValueError("Capture sample ranges disagree with the independent-replicate plan")
        if not frame.metadata.get("scene_fingerprint"):
            raise ValueError("Reference captures require scene_fingerprint")
        loaded[case["id"], samples, replica] = frame
    reports = []
    for case in plan["cases"]:
        counts = plan["sample_counts"]
        highest = counts[-1]
        anchor_frames = [loaded[case["id"], highest, replica] for replica in range(plan["replicates"])]
        base = anchor_frames[0]
        if base.metadata["scene"] != case["capture"]["scene"]:
            raise ValueError("Actual calibration scene disagrees with the declared scene")
        anchor_stack = np.stack([frame.buffers["reference"].astype(np.float64) for frame in anchor_frames])
        anchor = anchor_stack.mean(axis=0)
        noisy = base.buffers["radiance"].astype(np.float64)
        input_rmse = float(np.sqrt(np.square(noisy - anchor).mean()))
        luminance = anchor @ np.array([0.2126, 0.7152, 0.0722])
        highlight = (luminance >= np.quantile(luminance, 0.95)) & (luminance > 0)
        highlight_input_rmse = float(np.sqrt(np.square(noisy[highlight] - anchor[highlight]).mean())) if highlight.any() else 0.0
        rows = []
        # Sampling identity changes across levels; estimator settings must not.
        ignored = {"sample_start", "samples", "reference_sample_start", "reference_samples", "sample_stride"}
        settings = {key: value for key, value in base.metadata["render"].items() if key not in ignored}
        for samples in counts:
            frames = [loaded[case["id"], samples, replica] for replica in range(plan["replicates"])]
            for frame in frames:
                if (frame.metadata["scene_fingerprint"], frame.metadata["camera"], frame.metadata["conventions"],
                    frame.metadata["width"], frame.metadata["height"]) != (
                        base.metadata["scene_fingerprint"], base.metadata["camera"], base.metadata["conventions"],
                        base.metadata["width"], base.metadata["height"]):
                    raise ValueError("Scene, camera, dimensions, or conventions changed during convergence capture")
                if {key: value for key, value in frame.metadata["render"].items() if key not in ignored} != settings:
                    raise ValueError("Integrator settings changed during convergence capture")
            estimates = np.stack([frame.buffers["reference"].astype(np.float64) for frame in frames])
            variance = np.var(estimates, axis=0, ddof=1)
            noise = float(np.sqrt(variance.mean()))
            highlight_noise = float(np.sqrt(variance[highlight].mean())) if highlight.any() else 0.0
            fraction = _ratio(noise, input_rmse)
            highlight_fraction = _ratio(highlight_noise, highlight_input_rmse)
            # Highest-level comparisons use other replicas, never an anchor
            # containing the very image whose error is being measured.
            comparisons = [image_metrics(estimate, anchor if samples != highest else
                np.delete(estimates, replica, axis=0).mean(axis=0)) for replica, estimate in enumerate(estimates)]
            rows.append({"samples": samples, "replica_noise_rmse": noise,
                         "noise_fraction_of_input": fraction, "highlight_noise_rmse": highlight_noise,
                         "highlight_noise_fraction_of_input": highlight_fraction,
                         "qualifies": fraction is not None and highlight_fraction is not None and
                                      max(fraction, highlight_fraction) <= plan["max_noise_fraction"],
                         "comparison_target": "independent_highest_spp_mean" if samples != highest else "other_highest_spp_replicas",
                         "comparison_metrics_mean": mean_metrics(comparisons),
                         "replicate_image_sha256": [frame.metadata["image_sha256"] for frame in frames]})
        reports.append({"id": case["id"], "group": case["group"], "partition": plan["partitions"][case["group"]],
                        "scene": base.metadata["scene"], "scene_fingerprint": base.metadata["scene_fingerprint"],
                        "render_settings": settings,
                        "camera": base.metadata["camera"], "input_samples": base.metadata["render"]["samples"],
                        "input_rmse_against_anchor": input_rmse,
                        "highest_spp_anchor_noise_rmse_estimate": rows[-1]["replica_noise_rmse"] / np.sqrt(plan["replicates"]),
                        "highlight_pixel_count": int(highlight.sum()), "levels": rows,
                        "photon_map_convergence_measured": False})
    selected = next((samples for index, samples in enumerate(plan["sample_counts"])
                     if all(all(row["qualifies"] for row in case["levels"][index:]) for case in reports)), None)
    return {"format": "wavefront-reference-convergence", "version": 1,
            "scope": "camera_sampling_at_fixed_integrator", "split_manifest_sha256": plan["split_manifest_sha256"],
            "sample_counts": plan["sample_counts"], "replicates": plan["replicates"],
            "max_noise_fraction": plan["max_noise_fraction"], "cases": reports,
            "selected_reference_samples": selected,
            "status": "qualified_for_calibration_views" if selected is not None else "insufficient_budget",
            "interpretation": "Empirical reference noise relative to one low-SPP input; not proof of an unbiased or fully converged solution."}


def write_convergence_report(report: dict, directory: str | Path) -> None:
    directory = Path(directory)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "samples", "noise_rmse", "noise_fraction", "highlight_noise_fraction", "qualifies"])
        writer.writeheader()
        for case in report["cases"]:
            for row in case["levels"]:
                writer.writerow({"case": case["id"], "samples": row["samples"], "noise_rmse": row["replica_noise_rmse"],
                                 "noise_fraction": row["noise_fraction_of_input"],
                                 "highlight_noise_fraction": row["highlight_noise_fraction_of_input"], "qualifies": row["qualifies"]})


def plot_reference_convergence(report: dict, filename: str | Path) -> None:
    """Optional publication/export artifact; core analysis does not need matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for case in report["cases"]:
        counts = [row["samples"] for row in case["levels"]]
        for axis, field in zip(axes, ("noise_fraction_of_input", "highlight_noise_fraction_of_input")):
            values = [row[field] for row in case["levels"]]
            axis.loglog(counts, values, marker="o", label=case["id"])
    for axis, title in zip(axes, ("Whole image", "Brightest 5% of pixels")):
        axis.axhline(report["max_noise_fraction"], color="black", linestyle="--", linewidth=1, label="Qualification threshold")
        axis.set_title(title)
        axis.set_xlabel("Reference samples per pixel")
        axis.set_xticks(report["sample_counts"], labels=[str(n) for n in report["sample_counts"]])
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Estimated reference noise / low-SPP input error")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"Independent reference replicas; selected budget: {report['selected_reference_samples']} SPP")
    figure.savefig(filename, dpi=160)
    plt.close(figure)


def run_reference_convergence(config: str | Path, output: str | Path, executable: str | Path,
                              renderer_root: str | Path | None = None) -> dict:
    plan = convergence_plan(config)
    jobs = list(capture_jobs(plan))
    output = Path(output).resolve()
    exe = Path(executable).resolve()
    cwd = Path(renderer_root).resolve() if renderer_root else exe.parent.parent
    output.mkdir(parents=True, exist_ok=False)
    (output / "capture_configs").mkdir()
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for case, samples, replica, capture, relative in jobs:
        print(f"Reference calibration {case['id']}: {samples} SPP, replicate {replica + 1}/{plan['replicates']}", flush=True)
        filename = output / "capture_configs" / f"{capture['sequence_id']}.json"
        filename.write_text(json.dumps(capture, indent=2) + "\n")
        with filename.with_suffix(".log").open("w") as log:
            subprocess.run([str(exe), str(filename), str(output / "captures" / relative)], cwd=cwd,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = analyze_reference_convergence(plan, output / "captures")
    write_convergence_report(report, output)
    return report
