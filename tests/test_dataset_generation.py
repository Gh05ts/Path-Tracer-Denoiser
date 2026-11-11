from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

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


@pytest.mark.parametrize("skip_missing_rendered", [False, True])
def test_resume_skips_valid_frames_and_recovers_interrupted_export(make_frame, tmp_path, small_config, monkeypatch,
                                                                 skip_missing_rendered):
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
    options = {"skip_missing_rendered": skip_missing_rendered}
    result = run_captures({config["sequence_id"]: config}, [config["sequence_id"]], output, Path("fake-renderer"), tmp_path,
                          **options)
    assert commands == [(1, 2), (3, 8)]
    assert result["skipped_frames"] == 2 and result["captured_frames"] == 6
    assert list((output/"recovery.partial"/config["sequence_id"]).glob("*.partial-*"))
    commands.clear()
    result = run_captures({config["sequence_id"]: config}, [config["sequence_id"]], output, Path("fake-renderer"), tmp_path,
                          **options)
    assert commands == [] and result["skipped_frames"] == 8
    frame = output/config["sequence_id"]/"frame_000000/frame.exr"
    frame.write_bytes(b"corrupt completed image")
    with pytest.raises(ValueError, match="checksum"):
        run_captures({config["sequence_id"]: config}, [config["sequence_id"]], output, Path("fake-renderer"), tmp_path,
                     **options)
    assert commands == []


def test_archive_resume_trusts_manifests_and_legacy_logs_but_captures_new_frames(
        make_frame, tmp_path, small_config, monkeypatch):
    output = tmp_path / "captures"
    config = small_config
    group = config["sequence_id"]
    first = create_frame(make_frame, output, config, 0)
    (first / "frame.exr").unlink()
    second = create_frame(make_frame, output, config, 1)
    shutil.rmtree(second)
    create_frame(make_frame, output, config, 2)
    partial = output / group / "frame_000003.partial"
    partial.mkdir()
    (partial / "frame.exr").write_bytes(b"interrupted")
    logs = output / "logs"
    logs.mkdir()
    # Existing renderer logs are retained even when whole frame directories move.
    # A different absolute root is allowed; unrelated groups and truncated writes
    # must not cause never-rendered frames to be skipped.
    (logs / f"{group}.log").write_text(
        f'Capturing frame 1: 4 input + 16 reference SPP\n'
        f'Wrote "/old/dataset/{group}/frame_000001"\n'
        f'Wrote "/old/dataset/other_group/frame_000004"\n'
        f'Wrote "/old/dataset/{group}/frame_000005\n')
    commands = []
    def render(command, **kwargs):
        first, end = int(command[-3]), int(command[-1])
        commands.append((first, end))
        for i in range(first, end):
            create_frame(make_frame, output, config, i)
    monkeypatch.setattr(subprocess, "run", render)
    args = ({group: config}, [group], output, Path("fake-renderer"), tmp_path)
    with pytest.raises(FileNotFoundError):
        run_captures(*args)
    assert commands == []
    # The opt-in behavior intentionally does not read an archived manifest.
    (first / "manifest.json").write_text("not validated in archive mode")
    result = run_captures(*args, skip_missing_rendered=True)
    assert commands == [(3, 8)]
    assert result["archived_frames"] == 2
    assert result["skipped_frames"] == 3 and result["captured_frames"] == 5
    assert not (first / "frame.exr").exists() and not second.exists()
    assert list((output / "recovery.partial" / group).glob("*.partial-*"))
    commands.clear()
    result = run_captures(*args, skip_missing_rendered=True)
    assert commands == [] and result["skipped_frames"] == 8
    assert result["archived_frames"] == 2
    assert json.loads((output / "last-capture.json").read_text())["archived_frames"] == 2


def test_default_resume_rerenders_missing_directory_despite_completion_log(
        make_frame, tmp_path, small_config, monkeypatch):
    config = small_config
    group = config["sequence_id"]
    output = tmp_path / "captures"
    (output / "logs").mkdir(parents=True)
    (output / "logs" / f"{group}.log").write_text(
        f'Wrote "{output / group / "frame_000000"}"\n')
    commands = []
    def render(command, **kwargs):
        commands.append(command)
        create_frame(make_frame, output, config, 0)
    monkeypatch.setattr(subprocess, "run", render)
    result = run_captures({group: config}, [group], output, Path("fake-renderer"), tmp_path, end_frame=1)
    assert len(commands) == 1
    assert result["captured_frames"] == 1 and result["skipped_frames"] == 0
    assert "archived_frames" not in result


@pytest.mark.parametrize("skip_missing_rendered", [False, True])
def test_cli_archive_flag_controls_final_local_audit(tmp_path, small_config, monkeypatch, capsys,
                                                    skip_missing_rendered):
    from wavefront_denoiser import dataset_generation as generation
    config = small_config
    group = config["sequence_id"]
    executable = tmp_path / "renderer"
    executable.touch()
    output = tmp_path / "captures"
    calls = []
    monkeypatch.setattr(generation, "build_configs", lambda *args, **kwargs: {group: config})
    monkeypatch.setattr(generation, "prepare", lambda *args: {"declared_reference_samples": 16})
    def capture(*args, **kwargs):
        calls.append(kwargs["skip_missing_rendered"])
        return {"captured_frames": 0, "skipped_frames": 8, "archived_frames": 8 if skip_missing_rendered else 0}
    monkeypatch.setattr(generation, "run_captures", capture)
    def audit(path):
        calls.append("audit")
        return {"status": "ready"}
    monkeypatch.setattr(generation, "audit_dataset_split", audit)
    args = ["--renderer", str(executable), "--output", str(output)]
    if skip_missing_rendered:
        args.append("--skip-missing-rendered")
    assert generation.main(args) == 0
    assert calls == ([True] if skip_missing_rendered else [False, "audit"])
    assert (output / "split_audit.json").exists() == (not skip_missing_rendered)
    assert ("archived files were not validated" in capsys.readouterr().out) == skip_missing_rendered


@pytest.mark.parametrize("provider", ["b2", "r2"])
def test_captures_archive_at_scene_boundaries_with_all_planned_clips(tmp_path, monkeypatch, provider):
    from wavefront_denoiser import dataset_generation as generation
    configs = {group: {"scene": scene, "frames": [{}]} for group, scene in (
        ("bmw_first", "bmw"), ("bmw_last", "bmw"), ("bmw_unselected", "bmw"), ("bistro_first", "bistro"))}
    events = []
    def render(command, **kwargs):
        events.append(("render", Path(command[2]).name))
    monkeypatch.setattr(subprocess, "run", render)
    monkeypatch.setattr(generation, "verify_frame", lambda *args, **kwargs: {"image_sha256": "abc"})
    class Archiver:
        destination = "b2:bucket/dataset"
        def archive_scene(self, scene, scene_configs):
            events.append(("archive", scene))
            assert set(scene_configs) == {g for g, c in configs.items() if c["scene"] == scene}
            return scene != "bmw"  # The real archiver defers incomplete/unselected clips.
    archiver = Archiver()
    archiver.provider = provider
    archiver.destination = f"{provider}://bucket/dataset"
    result = generation.run_captures(configs, ["bmw_first", "bmw_last", "bistro_first"], tmp_path,
                                     Path("renderer"), tmp_path, scene_archiver=archiver)
    assert events == [("render", "bmw_first"), ("render", "bmw_last"), ("archive", "bmw"),
                      ("render", "bistro_first"), ("archive", "bistro")]
    assert result[f"{provider}_archived_scenes"] == ["bistro"]
    assert result[f"{provider}_destination"] == archiver.destination


@pytest.mark.parametrize("destination", ["b2:bucket/data", "b2://bucket/data"])
def test_cli_b2_archive_preflights_before_capture_and_enables_archive_resume(tmp_path, small_config, monkeypatch,
                                                                           destination):
    from wavefront_denoiser import dataset_generation as generation
    executable = tmp_path / "renderer"
    executable.touch()
    output = tmp_path / "captures"
    events = []
    class Archiver:
        def __init__(self, root, target, *, rclone):
            assert root == output and target == destination and rclone == "/custom/rclone"
        def preflight(self):
            events.append("preflight")
    monkeypatch.setattr(generation, "B2SceneArchiver", Archiver)
    monkeypatch.setattr(generation, "build_configs", lambda *args, **kwargs: {"fixture": small_config})
    monkeypatch.setattr(generation, "prepare", lambda *args: {"declared_reference_samples": 16})
    def capture(*args, **kwargs):
        assert kwargs["skip_missing_rendered"] and isinstance(kwargs["scene_archiver"], Archiver)
        events.append("capture")
        return {}
    monkeypatch.setattr(generation, "run_captures", capture)
    monkeypatch.setattr(generation, "audit_dataset_split", lambda *args: pytest.fail("Local audit after B2 archival"))
    assert generation.main(["--renderer", str(executable), "--output", str(output),
                            "--b2-archive", destination, "--rclone", "/custom/rclone"]) == 0
    assert events == ["preflight", "capture"]


@pytest.mark.parametrize("stage", ["plan", "pilot", "preview"])
@pytest.mark.parametrize("flag", ["--b2-archive", "--r2-archive"])
def test_cloud_archive_rejects_non_capture_stages(stage, flag):
    from wavefront_denoiser import dataset_generation as generation
    with pytest.raises(SystemExit) as error:
        generation.main(["--stage", stage, flag, f"{flag[2:4]}://bucket/data"])
    assert error.value.code == 2


def test_cli_r2_preflights_before_capture_and_skips_local_audit(tmp_path, small_config, monkeypatch):
    from wavefront_denoiser import dataset_generation as generation
    executable = tmp_path / "renderer"
    executable.touch()
    output = tmp_path / "captures"
    events = []
    class Archiver:
        def __init__(self, root, destination):
            assert root == output and destination == "r2://bucket/data"
        def preflight(self):
            events.append("preflight")
    monkeypatch.setattr(generation, "R2SceneArchiver", Archiver)
    monkeypatch.setattr(generation, "build_configs", lambda *args, **kwargs: {"fixture": small_config})
    monkeypatch.setattr(generation, "prepare", lambda *args: {"declared_reference_samples": 16})
    def capture(*args, **kwargs):
        assert kwargs["skip_missing_rendered"] and isinstance(kwargs["scene_archiver"], Archiver)
        events.append("capture")
        return {}
    monkeypatch.setattr(generation, "run_captures", capture)
    monkeypatch.setattr(generation, "audit_dataset_split", lambda *args: pytest.fail("Local audit after R2 archive"))
    assert generation.main(["--renderer", str(executable), "--output", str(output),
                            "--r2-archive", "r2://bucket/data"]) == 0
    assert events == ["preflight", "capture"]


def test_cli_cannot_delete_locally_after_archiving_to_two_destinations():
    from wavefront_denoiser import dataset_generation as generation
    with pytest.raises(SystemExit) as error:
        generation.main(["--b2-archive", "b2://bucket/data", "--r2-archive", "r2://bucket/data"])
    assert error.value.code == 2


def test_sigterm_stops_and_reaps_active_subprocess(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    child_code = f"import os, time; from pathlib import Path; Path({str(child_pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    parent_code = (
        "import subprocess, sys\n"
        "from wavefront_denoiser.dataset_generation import terminate_subprocesses_on_sigterm\n"
        "with terminate_subprocesses_on_sigterm():\n"
        f"    subprocess.run([sys.executable, '-c', {child_code!r}], check=True)\n")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    parent = subprocess.Popen([sys.executable, "-c", parent_code], env=environment)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text())
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=5) == 128 + signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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
    assert (output / "train.json").is_file()
    assert not (output / "train-spatial.json").exists()
    assert not (output / "train-recurrent.json").exists()
    training = json.loads((output / "train.json").read_text())
    assert training["window_length"] == 5 and training["base_channels"] == 48
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
    from wavefront_denoiser.dataset_generation import DEFAULT_DATASET_ROOT
    v3_plan = PLAN.with_name("plan-v3.json")
    assert DEFAULT_DATASET_ROOT == Path(__file__).resolve().parents[2]/"dataset"
    plan = read_plan(v3_plan)
    configs = build_configs(v3_plan)
    assert plan["dataset_id"] == "denoiser-v3"
    assert plan["indirect_radiance_clamp"] == 10.
    assert {g: {k: v for k, v in c.items() if k != "indirect_radiance_clamp"}
            for g, c in configs.items()} == build_configs(PLAN.with_name("plan-v2.json"))
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
    reports = build_configs(v3_plan, reference_samples=4096)
    for g, c in configs.items():
        assert c["frames"] == reports[g]["frames"]
        assert c["sample_start"] == reports[g]["sample_start"]
        assert reports[g]["indirect_radiance_clamp"] == c["indirect_radiance_clamp"]


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf"), True])
def test_invalid_motion_multiplier_rejected(multiplier):
    scene = read_plan(PLAN)["scenes"][0]
    with pytest.raises(ValueError, match="multiplier"):
        trajectory(scene, scene["anchors"][0], "linear", motion_multiplier=multiplier)


def test_v4_expands_viewpoints_and_motion_without_changing_paused_plan():
    from collections import Counter
    from wavefront_denoiser.dataset_generation import DEFAULT_PLAN
    plan = read_plan(DEFAULT_PLAN)
    old_plan = read_plan(PLAN.with_name("plan-v3.json"))
    configs = build_configs(DEFAULT_PLAN)
    reports = build_configs(DEFAULT_PLAN, reference_samples=4096)
    assert plan["dataset_id"] == "denoiser-v4"
    assert len(configs) == 234
    membership = {s["id"]: s["partition"] for s in plan["scenes"]}
    assert Counter(membership[c["scene"]] for c in configs.values()) == {
        "train": 180, "validation": 36, "test": 18}
    for scene, old in zip(plan["scenes"], old_plan["scenes"]):
        assert scene["partition"] == old["partition"]
        assert scene["anchors"][:len(old["anchors"])] == old["anchors"]
        assert len(scene["anchors"]) == (6 if scene["partition"] == "train" else 3)
        origins = []
        for anchor in scene["anchors"]:
            # Pan/tilt keeps the origin fixed and reveals each actual viewpoint.
            frames = trajectory(scene, anchor, "pan_tilt")
            origins.append(frames[0]["position"])
            if "camera" in anchor:
                np.testing.assert_allclose(frames[0]["position"], anchor["camera"]["position"])
        assert len({tuple(o) for o in origins}) == len(origins)
        for anchor in scene["anchors"][len(old["anchors"]):]:
            assert min(np.linalg.norm(np.array(anchor["camera"]["position"])-o)
                       for o in origins[:len(old["anchors"])]) > .3
    assert plan["motion_profiles"]["slow"] == old_plan["motion_profiles"]["slow"]
    for speed in ("medium", "fast"):
        assert 1.2 < plan["motion_profiles"][speed]/old_plan["motion_profiles"][speed] < 1.35
    old_end = max(c["sample_start"]+8*c["sample_stride"]
                  for c in build_configs(PLAN.with_name("plan-v3.json")).values())
    ranges = []
    for group, config in configs.items():
        assert config["sample_start"] > old_end
        assert config["frames"] == reports[group]["frames"]
        assert config["sample_start"] == reports[group]["sample_start"]
        assert config["indirect_radiance_clamp"] == 10.
        positions = np.array([f["position"] for f in config["frames"]])
        targets = np.array([f["target"] for f in config["frames"]])
        assert np.isfinite(positions).all() and np.isfinite(targets).all()
        assert np.all(np.linalg.norm(targets-positions, axis=1) > 0)
        ranges.append((config["sample_start"], config["sample_start"]+8*config["sample_stride"]))
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    # Independent reference-convergence replicas reserve samples from 1e9.
    assert ranges[-1][1] < 1_000_000_000


def test_explicit_anchor_camera_is_independent_of_scene_camera():
    scene = read_plan(PLAN)["scenes"][0]
    camera = {"position": [4, 2, 3], "target": [0, 1, 0], "up": [0, 1, 0],
              "vertical_fov_degrees": 55.}
    anchor = {"name": "remote", "camera": camera}
    for motion in MOTIONS:
        frames = trajectory(scene, anchor, motion)
        moved_scene = deepcopy(scene)
        moved_scene["camera"]["position"] = [100, 100, 100]
        assert frames == trajectory(moved_scene, anchor, motion)
        assert all(f["vertical_fov_degrees"] == 55. for f in frames)


def test_cli_defaults_to_v4_under_workspace_dataset(tmp_path, monkeypatch):
    from wavefront_denoiser import dataset_generation as generation
    called = {}
    def prepare_plan(path, output, configs, executable):
        called.update(path=path, output=output, configs=configs, executable=executable)
        return {"declared_reference_samples": 1024}
    monkeypatch.setattr(generation, "DEFAULT_DATASET_ROOT", tmp_path/"dataset")
    monkeypatch.setattr(generation, "prepare", prepare_plan)
    assert generation.main(["--stage", "plan"]) == 0
    assert called["path"] == generation.DEFAULT_PLAN
    assert called["output"] == tmp_path/"dataset/denoiser-v4"
    assert called["executable"] is None
    assert all(g.rsplit("_", 1)[1] in ("slow", "medium", "fast") for g in called["configs"])
