from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from wavefront_denoiser.dataset_generation import (
    MOTIONS, asset_files, build_configs, capture_lock, digest, expected_camera,
    missing_ranges, prepare, read_plan, run_captures, trajectory, verify_frame,
)

PLAN = Path(__file__).resolve().parents[1] / "dataset_generation/plan.json"


def test_production_plan_is_scene_independent_and_report_samples_align():
    plan = read_plan(PLAN)
    standard = build_configs(PLAN)
    reports = build_configs(PLAN, reference_samples=4096)
    partitions = {s["id"]: s["partition"] for s in plan["scenes"]}
    counts = {p: sum(partitions[c["scene"]] == p for c in standard.values()) for p in ("train", "validation", "test")}
    assert counts == {"train": 90, "validation": 24, "test": 12}
    assert {s["id"] for s in plan["scenes"] if s["partition"] == "train"} == {
        "bmw", "bistro", "living_room", "breakfast_room", "dragon"}
    ranges = []
    for group, config in standard.items():
        assert (config["width"], config["height"], config["samples"], config["reference_samples"]) == (1920, 1080, 4, 1024)
        assert len(config["frames"]) == 8
        assert config["frames"] == reports[group]["frames"]
        assert config["sample_start"] == reports[group]["sample_start"]
        assert config["sample_stride"] == reports[group]["sample_stride"]
        ranges.append((config["sample_start"], config["sample_start"]+8*config["sample_stride"]))
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


@pytest.mark.parametrize("motion", MOTIONS)
def test_trajectories_are_finite_smooth_and_move(motion):
    for scene in read_plan(PLAN)["scenes"]:
        frames = trajectory(scene, scene["anchors"][0], motion)
        positions = np.array([f["position"] for f in frames])
        targets = np.array([f["target"] for f in frames])
        assert np.isfinite(positions).all() and np.isfinite(targets).all()
        assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < scene["motion_scale"]*1.1
        directions = targets-positions
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        assert np.max(np.linalg.norm(np.diff(directions, axis=0), axis=1)) < .1
        assert np.ptp(positions, axis=0).max() > 0 or np.ptp(directions, axis=0).max() > 0


def test_missing_ranges_and_output_lock(tmp_path):
    assert missing_ranges([0, 1, 3, 6, 7]) == [(0, 2), (3, 4), (6, 8)]
    assert missing_ranges([]) == []
    with capture_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another capture"):
            with capture_lock(tmp_path):
                pass


@pytest.fixture
def small_config():
    config = deepcopy(next(iter(build_configs(PLAN, width=8, height=8, reference_samples=16).values())))
    config["sequence_id"] = "fixture"
    return config


def create_frame(make_frame, output, config, i):
    def metadata(m):
        m.update(sequence_id=config["sequence_id"], scene=config["scene"], scene_fingerprint="123",
                 camera={k: np.asarray(v).tolist() for k, v in expected_camera(config["frames"][i], config).items()},
                 previous_camera={k: np.asarray(v).tolist() for k, v in expected_camera(config["frames"][i-1], config).items()} if i else None)
        start = config["sample_start"]+i*config["sample_stride"]
        m["render"].update(sample_start=start, reference_sample_start=start+4,
            sample_stride=config["sample_stride"], max_depth=config["max_depth"],
            russian_roulette_start_depth=config["russian_roulette_start_depth"], caustics=False)
        if "indirect_radiance_clamp" in config:
            m["render"]["indirect_radiance_clamp"] = config["indirect_radiance_clamp"]
    return make_frame(str(output/config["sequence_id"]/f"frame_{i:06d}"), index=i,
                      sequence=config["sequence_id"], previous=i > 0, width=8, height=8, metadata_mutate=metadata)


def test_resume_skips_valid_frames_and_recovers_interrupted_export(make_frame, tmp_path, small_config, monkeypatch):
    output = tmp_path / "captures"
    config = small_config
    create_frame(make_frame, output, config, 0)
    create_frame(make_frame, output, config, 2)
    partial = output/config["sequence_id"]/"frame_000001.partial"
    partial.mkdir()
    (partial/"frame.exr").write_bytes(b"interrupted")
    commands = []
    def render(command, **kwargs):
        first, end = int(command[-3]), int(command[-1])
        commands.append((first, end))
        for i in range(first, end):
            create_frame(make_frame, output, config, i)
    monkeypatch.setattr(subprocess, "run", render)
    result = run_captures({config["sequence_id"]: config}, [config["sequence_id"]], output, Path("fake-renderer"), tmp_path)
    assert commands == [(1, 2), (3, 8)]
    assert result["skipped_frames"] == 2 and result["captured_frames"] == 6
    assert list((output/"recovery.partial"/config["sequence_id"]).glob("*.partial-*"))
    commands.clear()
    result = run_captures({config["sequence_id"]: config}, [config["sequence_id"]], output, Path("fake-renderer"), tmp_path)
    assert commands == [] and result["skipped_frames"] == 8
    frame = output/config["sequence_id"]/"frame_000000/frame.exr"
    frame.write_bytes(b"corrupt completed image")
    with pytest.raises(ValueError, match="checksum"):
        run_captures({config["sequence_id"]: config}, [config["sequence_id"]], output, Path("fake-renderer"), tmp_path)
    assert commands == []


def test_changed_camera_is_not_silently_skipped(make_frame, tmp_path, small_config):
    output = tmp_path/"captures"
    directory = create_frame(make_frame, output, small_config, 0)
    small_config["frames"][0]["position"][0] += 1
    with pytest.raises(ValueError, match="Camera disagrees"):
        verify_frame(directory, small_config, 0)


@pytest.mark.parametrize("recorded", [None, 0, 20])
def test_unmatched_clamp_is_not_silently_skipped(make_frame, tmp_path, small_config, recorded):
    small_config["indirect_radiance_clamp"] = 10.
    directory = create_frame(make_frame, tmp_path/"captures", small_config, 0)
    verify_frame(directory, small_config, 0)
    manifest = directory/"manifest.json"
    metadata = json.loads(manifest.read_text())
    if recorded is None:
        del metadata["render"]["indirect_radiance_clamp"]
    else:
        metadata["render"]["indirect_radiance_clamp"] = recorded
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Frame metadata disagrees"):
        verify_frame(directory, small_config, 0)


@pytest.mark.parametrize("clamp", [-1, float("nan"), float("inf"), True, "10"])
def test_invalid_plan_clamp_rejected(tmp_path, clamp):
    plan = read_plan(PLAN)
    plan["indirect_radiance_clamp"] = clamp
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="indirect_radiance_clamp"):
        read_plan(path)


def test_provenance_tracks_external_bytes_and_binds_renderer_after_plan(tmp_path):
    plan = read_plan(PLAN)
    scene = tmp_path/"scene.json"
    asset = tmp_path/"mesh.gltf"
    texture = tmp_path/"texture.png"
    texture.write_bytes(b"texture original")
    asset.write_text(json.dumps({"images": [{"uri": "texture.png"}]}))
    for s in plan["scenes"]:
        p = tmp_path/f"{s['id']}.json"
        p.write_text(json.dumps({"name": s["id"], "object": {"file": "mesh.gltf"}}))
        s["scene_file"] = p.name
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(plan))
    configs = build_configs(path)
    output = tmp_path/"data"
    assert texture in asset_files(tmp_path/"bmw.json")
    prepare(path, output, configs, None)
    executable = tmp_path/"renderer"
    executable.write_bytes(b"renderer version 1")
    prepare(path, output, configs, executable)
    assert json.loads((output/"capture_lock.json").read_text())["renderer_sha256"] == digest(executable)
    prepare(path, output, configs, None)
    prepare(path, output, configs, executable)
    texture.write_bytes(b"changed external texture")
    with pytest.raises(ValueError, match="assets or renderer changed"):
        prepare(path, output, configs, executable)


def test_v3_profiles_preserve_v2_cameras_and_sample_ranges():
    from collections import Counter, defaultdict
    from wavefront_denoiser.dataset_generation import DEFAULT_PLAN, DEFAULT_DATASET_ROOT
    assert DEFAULT_DATASET_ROOT == Path(__file__).resolve().parents[2]/"dataset"
    plan = read_plan(DEFAULT_PLAN)
    configs = build_configs(DEFAULT_PLAN)
    assert plan["dataset_id"] == "denoiser-v3"
    assert plan["indirect_radiance_clamp"] == 10.
    assert {g: {k: v for k, v in c.items() if k != "indirect_radiance_clamp"}
            for g, c in configs.items()} == build_configs(DEFAULT_PLAN.with_name("plan-v2.json"))
    assert all(c["indirect_radiance_clamp"] == 10. for c in configs.values())
    assert len(configs) == 126
    for scene in plan["scenes"]:
        groups = [g for g, c in configs.items() if c["scene"] == scene["id"]]
        counts = Counter(g.rsplit("_", 1)[1] for g in groups)
        assert set(counts) == {"slow", "medium", "fast"}
        if scene["partition"] == "train":
            assert counts == {"slow": 6, "medium": 6, "fast": 6}
            per_motion = defaultdict(set)
            for g in groups:
                suffix = g.split("_view_", 1)[1].split("_", 1)[1]
                motion, speed = suffix.rsplit("_", 1)
                per_motion[motion].add(speed)
            assert all(speeds == {"slow", "medium", "fast"} for speeds in per_motion.values())
    scene = plan["scenes"][0]
    distances = []
    for name in ("slow", "medium", "fast"):
        frames = trajectory(scene, scene["anchors"][0], "linear", motion_multiplier=plan["motion_profiles"][name])
        distances.append(np.linalg.norm(np.asarray(frames[-1]["position"])-frames[0]["position"]))
    assert distances[0] < distances[1] < distances[2]
    assert np.isclose(distances[2]/distances[0], 2.8/.6)
    reports = build_configs(DEFAULT_PLAN, reference_samples=4096)
    for g, c in configs.items():
        assert c["frames"] == reports[g]["frames"]
        assert c["sample_start"] == reports[g]["sample_start"]
        assert reports[g]["indirect_radiance_clamp"] == c["indirect_radiance_clamp"]


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf"), True])
def test_invalid_motion_multiplier_rejected(multiplier):
    scene = read_plan(PLAN)["scenes"][0]
    with pytest.raises(ValueError, match="multiplier"):
        trajectory(scene, scene["anchors"][0], "linear", motion_multiplier=multiplier)


def test_cli_defaults_to_v3_under_workspace_dataset(tmp_path, monkeypatch):
    from wavefront_denoiser import dataset_generation as generation
    called = {}
    def prepare_plan(path, output, configs, executable):
        called.update(path=path, output=output, configs=configs, executable=executable)
        return {"declared_reference_samples": 1024}
    monkeypatch.setattr(generation, "DEFAULT_DATASET_ROOT", tmp_path/"dataset")
    monkeypatch.setattr(generation, "prepare", prepare_plan)
    assert generation.main(["--stage", "plan"]) == 0
    assert called["path"] == generation.DEFAULT_PLAN
    assert called["output"] == tmp_path/"dataset/denoiser-v3"
    assert called["executable"] is None
    assert all(g.rsplit("_", 1)[1] in ("slow", "medium", "fast") for g in called["configs"])
