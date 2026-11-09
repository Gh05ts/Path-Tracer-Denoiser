from copy import deepcopy
import json

import numpy as np
import pytest

from wavefront_denoiser.reference_convergence import (analyze_reference_convergence, capture_jobs,
                                                      convergence_plan, write_convergence_report)


@pytest.fixture
def synthetic_references(make_frame, tmp_path):
    split = {"format": "wavefront-dataset-split", "version": 1, "data_root": "dataset",
             "partitions": {"train": ["train"], "validation": ["val"], "test": ["test"]},
             "groups": {key: {"scene": key, "family": key, "capture_config": key + ".json"}
                        for key in ("train", "val", "test")}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    raw = {"format": "wavefront-reference-plan", "version": 1, "split_manifest": "split.json",
           "sample_counts": [2, 8], "replicates": 2, "sample_start": 1000, "max_noise_fraction": 0.1,
           "cases": [{"id": "train", "group": "train", "capture": {"scene": "train", "samples": 4, "frames": [{}]}}]}
    filename = tmp_path / "plan.json"
    filename.write_text(json.dumps(raw))
    plan = convergence_plan(filename)
    for case, samples, replica, config, relative in capture_jobs(plan):
        delta = (1 if samples == 2 else 0.1) * (-1 if replica == 0 else 1)

        def pixels(channels, n=samples, change=delta):
            for component in "RGB":
                channels[f"reference.{component}"].fill(8 + change)
            channels["reference_count.N"].fill(n)

        def metadata(value, settings=config):
            value.update(scene="train", scene_fingerprint="abc")
            value["render"].update(sample_start=settings["sample_start"], samples=settings["samples"],
                                   reference_sample_start=settings["sample_start"] + settings["samples"],
                                   reference_samples=settings["reference_samples"])

        make_frame(str("captures" / relative / "frame_000000"), sequence=config["sequence_id"],
                   mutate=pixels, metadata_mutate=metadata)
    return filename, raw, plan, tmp_path / "captures"


def test_reference_replicates_and_rng_intervals_are_independent(synthetic_references):
    _, _, plan, _ = synthetic_references
    intervals = [(config["sample_start"], config["sample_start"] + config["samples"] + samples)
                 for _, samples, _, config, _ in capture_jobs(plan)]
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_noise_estimate_and_budget_selection(synthetic_references, tmp_path):
    _, _, plan, captures = synthetic_references
    report = analyze_reference_convergence(plan, captures)
    levels = report["cases"][0]["levels"]
    assert np.isclose(levels[0]["replica_noise_rmse"], np.sqrt(2), rtol=1e-6)
    assert report["selected_reference_samples"] == 8
    # The highest-SPP image is compared to the OTHER replicate, not an average
    # containing itself (which would halve the measured difference).
    assert np.isclose(levels[-1]["comparison_metrics_mean"]["linear_rmse"], 0.2, rtol=1e-5)
    write_convergence_report(report, tmp_path)
    assert (tmp_path / "report.json").exists() and (tmp_path / "summary.csv").exists()


def test_sample_stride_is_sampling_allocation_not_an_integrator_change(synthetic_references):
    _, _, plan, captures = synthetic_references
    for path in captures.rglob("manifest.json"):
        metadata = json.loads(path.read_text())
        metadata["render"]["sample_stride"] = metadata["render"]["samples"] + metadata["render"]["reference_samples"]
        path.write_text(json.dumps(metadata))
    assert analyze_reference_convergence(plan, captures)["selected_reference_samples"] == 8


def test_insufficient_budget_is_reported(synthetic_references):
    _, _, plan, captures = synthetic_references
    plan = dict(plan, max_noise_fraction=0.0001)
    report = analyze_reference_convergence(plan, captures)
    assert report["selected_reference_samples"] is None
    assert report["status"] == "insufficient_budget"


def test_test_scene_cannot_be_used_for_calibration(synthetic_references):
    filename, raw, _, _ = synthetic_references
    raw["cases"][0]["group"] = "test"
    raw["cases"][0]["capture"]["scene"] = "test"
    filename.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="train/validation"):
        convergence_plan(filename)


@pytest.mark.parametrize("changes", [{"sample_counts": [8, 2]}, {"sample_counts": [2, 2]},
                                      {"replicates": 1}, {"max_noise_fraction": 0}])
def test_invalid_calibration_plan_rejected(synthetic_references, changes):
    filename, raw, _, _ = synthetic_references
    raw.update(changes)
    filename.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        convergence_plan(filename)


def test_camera_drift_during_convergence_is_rejected(synthetic_references):
    _, _, plan, captures = synthetic_references
    path = captures / "train/s2/r0/frame_000000/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["camera"]["position"] = [1, 0, 0]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="camera"):
        analyze_reference_convergence(plan, captures)


def test_reused_sample_interval_is_rejected(synthetic_references):
    _, _, plan, captures = synthetic_references
    path = captures / "train/s2/r1/frame_000000/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["render"].update(sample_start=1000, reference_sample_start=1004)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sample ranges"):
        analyze_reference_convergence(plan, captures)
