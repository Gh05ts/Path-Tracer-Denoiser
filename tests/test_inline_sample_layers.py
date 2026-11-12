from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from wavefront_denoiser.b2_archive import B2SceneArchiver
from wavefront_denoiser.dataset_generation import digest, run_captures, write_json
from wavefront_denoiser.inline_sample_layers import InlineSampleLayers
from wavefront_denoiser.adaptive_data import layer_layout
from test_adaptive_dataset import fake_layers
from test_b2_archive import fake_rclone
from test_dataset_generation import create_frame, small_config


SPPS = [1, 2, 8, 16, 32]


@pytest.fixture
def inline_data(tmp_path, make_frame, small_config):
    output = tmp_path/"dataset"
    configs = {}
    for n, group in enumerate(("old", "boundary", "new")):
        config = deepcopy(small_config)
        config.update(sequence_id=group, sample_start=10000+n*100000, frames=config["frames"][:2])
        configs[group] = config
        for i in range(2):
            create_frame(make_frame, output, config, i)
        write_json(output/"capture_configs"/(group+".json"), config)
    write_json(output/"capture_lock.json", {"configs": configs})
    renderer = tmp_path/"renderer"
    renderer.write_bytes(b"layer renderer")
    return output, configs, renderer


def manager(data):
    output, configs, renderer = data
    return InlineSampleLayers(output, configs, renderer, output, spps=SPPS,
                              after_group="boundary", reserved_reference_samples=64)


def capture_layers(data, monkeypatch):
    output, _, _ = data
    calls = []
    def run(command, **kwargs):
        config = json.loads(Path(command[1]).read_text())
        first, end = int(command[-3]), int(command[-1])
        calls.append((first, end))
        for i in range(first, end):
            relative = Path(config["sequence_id"])/f"frame_{i:06d}"
            base = json.loads((output/relative/"manifest.json").read_text())
            fake_layers(output/"sample_layers"/relative, base, config["sample_layers"])
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_boundary_backfill_and_layer_resume_preserve_base_bytes(inline_data, monkeypatch):
    output, configs, _ = inline_data
    before = {p: digest(p) for p in output.glob("*/frame_*/*")}
    capture = manager(inline_data)
    calls = capture_layers(inline_data, monkeypatch)
    for group in configs:
        capture.capture_group(group)
    assert calls == [(0, 2)]
    assert {p: digest(p) for p in before} == before
    assert not (output/"sample_layers"/"boundary").exists()
    readme = (output/"SAMPLE_LAYERS_BACKFILL.md").read_text()
    assert "4 frames in 2 clips" in readme and "| `boundary`" in readme and "| `new`" not in readme
    capture = manager(inline_data)
    capture.capture_group("new")
    assert calls == [(0, 2)] and capture.skipped == 2
    assert json.loads((output/"sample_layers/new/frame_000000/manifest.json").read_text())["layers"][-1]["samples"] == 32
    (output/"sample_layers/new/frame_000000/layers.exr").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        capture.capture_group("new")


def test_interrupted_layer_export_recovers_only_unfinished_frames(inline_data, monkeypatch):
    output, _, _ = inline_data
    capture = manager(inline_data)
    calls = capture_layers(inline_data, monkeypatch)
    capture.capture_group("new", end_frame=1)
    partial = output/"sample_layers/new/frame_000001.partial"
    partial.mkdir()
    (partial/"layers.exr").write_bytes(b"incomplete")
    capture.capture_group("new")
    assert calls == [(0, 1), (1, 2)]
    assert list((output/"recovery.partial/sample_layers/new").glob("frame_000001.partial-*/layers.exr"))
    changed = list(SPPS[:-1])
    with pytest.raises(ValueError, match="settings"):
        InlineSampleLayers(output, inline_data[1], inline_data[2], output, spps=changed,
                           after_group="boundary", reserved_reference_samples=64)


def test_layer_archive_blocks_incomplete_scene_and_verifies_layers_before_deleting(inline_data, monkeypatch, fake_rclone):
    output, configs, _ = inline_data
    capture = manager(inline_data)
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    archiver.preflight()
    assert not archiver.archive_scene("bmw", configs)
    assert list(output.glob("*/frame_*/frame.exr"))
    with monkeypatch.context() as local:
        capture_layers(inline_data, local)
        capture.capture_group("new")
    assert archiver.archive_scene("bmw", configs)
    assert not list(output.glob("*/frame_*/frame.exr"))
    assert not list(output.glob("sample_layers/*/frame_*/layers.exr"))
    receipt = json.loads((output/"b2-archives/bmw.json").read_text())
    assert "sample_layers/new/frame_000000/layers.exr" in receipt["files"]
    assert "sample_layers_lock.json" in receipt["files"]
    assert "SAMPLE_LAYERS_BACKFILL.md" in receipt["files"]
    capture = manager(inline_data)
    capture.capture_group("new")
    assert capture.archived == 2
    assert archiver.archive_scene("bmw", configs)


def test_group_layers_are_completed_before_scene_archive(inline_data, monkeypatch):
    output, configs, renderer = inline_data
    events = []
    class Layers:
        def capture_group(self, group, **kwargs):
            events.append(("layers", group))
    class Archive:
        provider = "r2"
        destination = "r2://bucket/run"
        def archive_scene(self, scene, configs):
            events.append(("archive", scene))
            return True
    run_captures(configs, list(configs), output, renderer, output, scene_archiver=Archive(),
                 sample_layer_capture=Layers())
    assert events == [("layers", "old"), ("layers", "boundary"), ("layers", "new"), ("archive", "bmw")]


def test_separate_backfill_is_required_and_included_in_scene_archive(inline_data, monkeypatch, fake_rclone):
    output, configs, _ = inline_data
    capture = manager(inline_data)
    lock_before = (output/"sample_layers_lock.json").read_bytes()
    with monkeypatch.context() as local:
        capture_layers(inline_data, local)
        capture.capture_group("new")
    config = dict(configs["old"], sample_layers=layer_layout(configs["old"], 64), reserved_reference_samples=64)
    write_json(output/"sample_layer_configs/old.json", config)
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    archiver.preflight()
    assert not archiver.archive_scene("bmw", configs)
    assert list(output.glob("*/frame_*/frame.exr"))
    for index in range(2):
        relative = Path("old")/f"frame_{index:06d}"
        base = json.loads((output/relative/"manifest.json").read_text())
        fake_layers(output/"sample_layers"/relative, base, config["sample_layers"])
    assert archiver.archive_scene("bmw", configs)
    receipt = json.loads((output/"b2-archives/bmw.json").read_text())
    assert "sample_layer_configs/old.json" in receipt["files"]
    assert "sample_layers/old/frame_000000/layers.exr" in receipt["files"]
    assert not list(output.glob("sample_layers/*/frame_*/layers.exr"))
    assert (output/"sample_layers_lock.json").read_bytes() == lock_before
