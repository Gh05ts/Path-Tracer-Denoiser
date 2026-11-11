"""v5 metadata, immutable source reuse, companion integrity and sequence tests."""

from copy import deepcopy
import json
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
import pytest

from wavefront_denoiser.adaptive_data import (
    NEW_LAYER_SPP, RANGE_FIELDS, V5SequenceDataset, layer_layout, load_v5_frame,
    read_layer_capture, verify_v5_record,
)
from wavefront_denoiser.adaptive_dataset import capture_v5, prepare_v5
from wavefront_denoiser.dataset_generation import build_configs, digest, expected_camera, prepare
from wavefront_denoiser.frames import FrameFormatError, read_manifest


def fixture_plan(tmp_path, *, width=9, height=7):
    plan = {"format": "wavefront-denoiser-capture-plan", "version": 1, "dataset_id": "source-fixture",
        "width": width, "height": height, "frames_per_clip": 8, "samples": 4,
        "reference_samples": 16, "report_reference_samples": 64, "sample_start": 10000,
        "sample_stride": 128, "clip_sample_stride": 2048, "max_depth": 4,
        "russian_roulette_start_depth": 3, "tile_width": 5, "tile_height": 4,
        "caustics": False, "indirect_radiance_clamp": 10., "motions": ["linear"], "scenes": []}
    for partition in ("train", "validation", "test"):
        name = f"{partition}_scene"
        (tmp_path / f"{name}.json").write_text(json.dumps({"name": name, "preset": "demo"}))
        plan["scenes"].append({"id": name, "partition": partition, "scene_file": f"{name}.json",
            "motion_scale": .1, "focus_distance": 4,
            "camera": {"position": [-4, 1.2, 5], "target": [0, .2, 0], "up": [0, 1, 0], "vertical_fov_degrees": 45},
            "anchors": [{"name": "view_00", "offset": [0, 0, 0]}]})
    source_plan = tmp_path / "source-plan.json"
    source_plan.write_text(json.dumps(plan))
    v5 = tmp_path / "plan-v5.json"
    v5.write_text(json.dumps({"format": "wavefront-ntasad-companion-plan", "version": 1,
        "dataset_id": "v5-fixture", "source_plan": source_plan.name, "reuse_spp": 4,
        "new_layer_spp": list(NEW_LAYER_SPP)}))
    return source_plan, v5


def fake_layers(directory, base, layout):
    directory.mkdir(parents=True)
    h, w = base["height"], base["width"]
    channels = {f"spp_{item['samples']}.{c}": np.full((h, w), item["samples"] + n/10, np.float32)
                for item in layout for n, c in enumerate("RGB")}
    header = OpenEXR.Header(w, h)
    header["channels"] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in channels}
    image = OpenEXR.OutputFile(str(directory / "layers.exr"), header)
    image.writePixels({name: values.tobytes() for name, values in channels.items()})
    image.close()
    metadata = deepcopy(base)
    metadata.update(schema="cuda-wavefront-sample-layers", image="layers.exr",
        image_sha256=digest(directory / "layers.exr"),
        render={k: v for k, v in base["render"].items() if k not in RANGE_FIELDS},
        layers=[{"samples": item["samples"], "sample_start": base["render"]["sample_start"] + item["offset"],
                 "channels": [f"spp_{item['samples']}.{c}" for c in "RGB"]} for item in layout])
    (directory / "manifest.json").write_text(json.dumps(metadata))


@pytest.fixture
def source_dataset(tmp_path, make_frame):
    source_plan, plan = fixture_plan(tmp_path)
    renderer = tmp_path / "renderer"
    renderer.write_bytes(b"renderer fixture")
    source, output = tmp_path / "v4", tmp_path / "v5"
    configs = build_configs(source_plan)
    prepare(source_plan, source, configs, renderer)
    for group, config in configs.items():
        for index in range(8):
            def metadata(m, config=config, index=index):
                camera = lambda i: {k: np.asarray(v).tolist() for k, v in expected_camera(config["frames"][i], config).items()}
                m.update(scene=config["scene"], scene_fingerprint="fixture_scene", camera=camera(index),
                         previous_camera=camera(index-1) if index else None)
                start = config["sample_start"] + index*config["sample_stride"]
                m["render"].update(sample_start=start, reference_sample_start=start+4,
                    sample_stride=config["sample_stride"], max_depth=config["max_depth"],
                    russian_roulette_start_depth=config["russian_roulette_start_depth"], caustics=False,
                    indirect_radiance_clamp=10., integrator_version=11)
            make_frame(str(source / group / f"frame_{index:06d}"), index=index, sequence=group,
                previous=index > 0, width=config["width"], height=config["height"], metadata_mutate=metadata)
    prepare_v5(plan, source, output, renderer=renderer)
    return plan, source, output, renderer, configs


def mock_capture(monkeypatch, output):
    calls = []
    def run(command, **kwargs):
        config = json.loads(Path(command[1]).read_text())
        first, end = int(command[-3]), int(command[-1])
        calls.append((config["sequence_id"], first, end))
        for index in range(first, end):
            relative = Path(config["sequence_id"]) / f"frame_{index:06d}"
            base = read_manifest(output / "base" / relative)
            fake_layers(output / "layers" / relative, base, config["sample_layers"])
    monkeypatch.setattr("wavefront_denoiser.adaptive_dataset.subprocess.run", run)
    return calls


def test_production_v5_inherits_all_v4_cameras_without_source_pixels(tmp_path):
    root = Path(__file__).resolve().parents[1]
    before = digest(root / "dataset_generation/plan-v4.json")
    configs = build_configs(root / "dataset_generation/plan-v4.json")
    output = tmp_path / "v5"
    value = prepare_v5(root / "dataset_generation/plan-v5.json", tmp_path / "not-finished-v4", output)
    assert len(value["groups"]) == 234
    assert sum(g["frame_count"] for g in value["groups"].values()) == 1872
    for group, info in value["groups"].items():
        assert info["source_config"] == configs[group]
        assert info["sample_layers"] == [dict(samples=n, offset=o) for n, o in zip(NEW_LAYER_SPP, [4100,4101,4103,4111,4127])]
    assert before == digest(root / "dataset_generation/plan-v4.json")
    assert not (output / "base").exists() and not (output / "capture_binding.json").exists()
    assert json.loads((output / "train-joint-plan.json").read_text())["status"] == "planned_requires_adaptive_trainer"


def test_layer_ranges_exclude_input_and_both_reference_budgets():
    config = {"samples": 4, "reference_samples": 1024, "sample_start": 0,
              "sample_stride": 8192, "frames": [{}]*8}
    layout = layer_layout(config, 4096)
    spans = [(0,4), (4,4100)] + [(p["offset"],p["offset"]+p["samples"]) for p in layout]
    assert all(a[1] <= b[0] for a,b in zip(spans,spans[1:]))
    assert sum(p["samples"] for p in layout) == 59
    with pytest.raises(ValueError, match="no room"):
        layer_layout(dict(config, sample_stride=4100), 4096)
    with pytest.raises(ValueError, match="4-SPP"):
        layer_layout(dict(config, samples=8), 4096)


def test_copy_capture_resume_and_sequence_reader(source_dataset, monkeypatch):
    plan, source, output, renderer, configs = source_dataset
    groups = list(configs)[:2]
    source_hashes = {p: digest(p) for p in source.rglob("*") if p.is_file()}
    calls = mock_capture(monkeypatch, output)
    result = capture_v5(output, source, renderer, source.parent, groups)
    assert result["captured_frames"] == 16 and result["pending_source_frames"] == 0
    assert calls == [(groups[0],0,8),(groups[1],0,8)]
    first = load_v5_frame(output, groups[0], 0)
    assert first.layers.shape == (6,7,9,3)
    np.testing.assert_array_equal(first.layers[2], first.base.buffers["radiance"])
    assert first.layers[5,0,0,0] == 32
    for group in groups:
        for i in range(8):
            for name in ("frame.exr", "manifest.json"):
                relative = Path(group) / f"frame_{i:06d}" / name
                assert digest(source / relative) == digest(output / "base" / relative)
    calls.clear()
    resumed = capture_v5(output, source, renderer, source.parent, groups)
    assert resumed["skipped_frames"] == 16 and not calls
    dataset = V5SequenceDataset(output, partition="train", tensors=False)
    assert len(dataset) == 4
    window = dataset[0]
    assert window["layers"].shape == (5,6,7,9,3)
    assert (window["buffers"]["history_valid"][0] == 0).all()
    tensors = V5SequenceDataset(output, partition="train")[0]
    assert tuple(tensors["layers"].shape) == (5,6,3,7,9)
    assert all(digest(p) == sha for p,sha in source_hashes.items())
    assert not (output / "base" / list(configs)[2]).exists()


def test_partial_frame_recovery_and_sparse_range_resume(source_dataset, monkeypatch):
    _, source, output, renderer, configs = source_dataset
    group = next(iter(configs))
    partial = output / "base" / group / "frame_000003.partial"
    partial.mkdir(parents=True)
    (partial / "incomplete").write_text("keep this evidence")
    partial_layer = output / "layers" / group / "frame_000003.partial"
    partial_layer.mkdir(parents=True)
    calls = mock_capture(monkeypatch, output)
    capture_v5(output, source, renderer, source.parent, [group], first_frame=3, end_frame=5)
    assert calls == [(group,3,5)]
    assert list((output / "recovery.partial/base" / group).glob("*.partial-*"))
    assert list((output / "recovery.partial/layers" / group).glob("*.partial-*"))
    calls.clear()
    capture_v5(output, source, renderer, source.parent, [group])
    assert calls == [(group,0,3),(group,5,8)]


@pytest.mark.parametrize("target", ["base", "layers"])
def test_corrupt_completed_images_are_rejected(source_dataset, monkeypatch, target):
    _, source, output, renderer, configs = source_dataset
    group = next(iter(configs))
    calls = mock_capture(monkeypatch, output)
    capture_v5(output, source, renderer, source.parent, [group], end_frame=1)
    filename = "frame.exr" if target == "base" else "layers.exr"
    (output / target / group / "frame_000000" / filename).write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="checksum"):
        capture_v5(output, source, renderer, source.parent, [group], end_frame=1)
    assert not calls


@pytest.mark.parametrize("change", ["camera", "integrator", "ranges", "dtype"])
def test_layer_mismatch_fails_before_completion(source_dataset, monkeypatch, change):
    _, source, output, renderer, configs = source_dataset
    group = next(iter(configs))
    mock_capture(monkeypatch, output)
    capture_v5(output, source, renderer, source.parent, [group], end_frame=1)
    relative = Path(group) / "frame_000000"
    base = read_manifest(output / "base" / relative)
    path = output / "layers" / relative / "manifest.json"
    metadata = json.loads(path.read_text())
    if change == "camera": metadata["camera"]["position"][0] += 1
    if change == "integrator": metadata["render"]["integrator_version"] += 1
    if change == "ranges": metadata["layers"][0]["sample_start"] -= 1
    if change == "dtype":
        image = output / "layers" / relative / "layers.exr"
        header = OpenEXR.Header(base["width"], base["height"])
        names = [c for layer in metadata["layers"] for c in layer["channels"]]
        header["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)) for c in names}
        exr = OpenEXR.OutputFile(str(image), header)
        exr.writePixels({c: np.zeros((base["height"],base["width"]),np.float16).tobytes() for c in names})
        exr.close()
        metadata["image_sha256"] = digest(image)
    path.write_text(json.dumps(metadata))
    with pytest.raises(FrameFormatError):
        read_layer_capture(path.parent, base, layer_layout(configs[group],64))


def test_source_or_renderer_changes_are_not_adopted(source_dataset):
    plan, source, output, renderer, configs = source_dataset
    renderer.write_bytes(b"different renderer")
    with pytest.raises(ValueError, match="changed"):
        prepare_v5(plan, source, output, renderer=renderer)
    with pytest.raises(ValueError, match="separate"):
        prepare_v5(plan, source, source)
    with pytest.raises(ValueError, match="separate"):
        prepare_v5(plan, source, source / "v5")


def test_missing_source_frame_is_pending_not_synthesized(source_dataset, monkeypatch):
    _, source, output, renderer, configs = source_dataset
    group = next(iter(configs))
    (source / group / "frame_000002").rename(source / group / "frame_000002.partial")
    calls = mock_capture(monkeypatch, output)
    result = capture_v5(output, source, renderer, source.parent, [group])
    assert result["pending_source_frames"] == 1 and result["captured_frames"] == 7
    assert calls == [(group,0,2),(group,3,8)]
    assert not (output / "frames" / group / "frame_000002").exists()


def test_changed_source_camera_and_assets_fail(source_dataset, monkeypatch):
    plan, source, output, renderer, configs = source_dataset
    group = next(iter(configs))
    manifest = source / group / "frame_000000/manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata["camera"]["position"][0] += 1
    manifest.write_text(json.dumps(metadata))
    calls = mock_capture(monkeypatch, output)
    with pytest.raises(ValueError, match="Camera"):
        capture_v5(output, source, renderer, source.parent, [group], end_frame=1)
    assert not calls
    Path(configs[group]["scene_file"]).write_text('{"name":"changed"}')
    with pytest.raises(ValueError):
        prepare_v5(plan, source, output, renderer=renderer)
