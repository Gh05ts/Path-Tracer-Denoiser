"""Opt-in integration tests against the actual renderer, never a mock renderer."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from wavefront_denoiser import load_frame, discover_sequences


@pytest.fixture(scope="module")
def capture_tool():
    executable = os.environ.get("WAVEFRONT_CAPTURE_EXECUTABLE")
    if not executable:
        pytest.skip("Set WAVEFRONT_CAPTURE_EXECUTABLE to enable CUDA capture integration")
    executable = Path(executable).resolve()
    # Out-of-tree builds (e.g. /tmp) do not sit beside the preset manifests.
    renderer_root = Path(os.environ.get("WAVEFRONT_RENDERER_ROOT",
                        Path(__file__).resolve().parents[2] / "Wavefront-Path-Tracer")).resolve()
    return executable, renderer_root


def run_capture(capture_tool, tmp_path, name, config):
    executable, renderer_root = capture_tool
    filename = tmp_path / f"{name}.json"
    filename.write_text(json.dumps(config))
    output = tmp_path / name
    subprocess.run([str(executable), str(filename), str(output)], cwd=renderer_root,
                   check=True, timeout=120, capture_output=True, text=True)
    return [load_frame(path) for path in discover_sequences(output)[config["sequence_id"]]]


def test_tiling_and_execution_modes_agree(capture_tool, tmp_path):
    config = {"sequence_id": "mode_equivalence", "scene": "demo", "width": 65, "height": 49,
              "tile_width": 32, "tile_height": 24, "samples": 2, "reference_samples": 8,
              "max_depth": 8, "frames": [{}, {"position": [-9.8, 0.35, 0.8]}]}
    tiled = run_capture(capture_tool, tmp_path, "tiled", config)
    full = run_capture(capture_tool, tmp_path, "full", dict(config, tile_width=65, tile_height=49))
    persistent = run_capture(capture_tool, tmp_path, "persistent", dict(config, persistent_wavefront=True))
    for reference, full_frame, persistent_frame in zip(tiled, full, persistent):
        for name, values in reference.buffers.items():
            np.testing.assert_allclose(full_frame.buffers[name], values, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(persistent_frame.buffers[name], values, rtol=1e-5, atol=1e-6)
    assert tiled[0].buffers["guide_valid"].any()
    assert not tiled[0].buffers["history_valid"].any()
    assert tiled[1].buffers["history_valid"].any()
    assert np.abs(tiled[1].buffers["motion"]).max() > 0.1
    assert tiled[0].metadata["render"]["reference_sample_start"] == 2
    assert tiled[1].metadata["render"]["sample_start"] == 10


def test_fixed_photon_map_capture_preserves_hdr(capture_tool, tmp_path):
    config = {"sequence_id": "caustic_smoke", "scene": "prism", "width": 33, "height": 25,
              "tile_width": 16, "tile_height": 12, "samples": 2, "reference_samples": 8,
              "max_depth": 8, "caustics": True, "caustic_photons": 4096, "frames": [{}]}
    frames = run_capture(capture_tool, tmp_path, "caustics", config)
    assert frames[0].metadata["render"]["caustics"]
    assert frames[0].buffers["radiance"].max() > 1.0


@pytest.mark.parametrize("changes", [{"width": -1}, {"samples": 0}, {"reference_samples": 1.5},
                                       {"sample_start": 4294967295},
                                       {"frames": [{"position": [0, 0, 0], "target": [0, 0, 0]}]}])
def test_invalid_capture_config_exits_before_creating_output(capture_tool, tmp_path, changes):
    executable, renderer_root = capture_tool
    config = dict({"sequence_id": "invalid", "scene": "demo", "width": 8, "height": 8,
                   "samples": 1, "reference_samples": 1, "frames": [{}]}, **changes)
    filename = tmp_path / "bad.json"
    filename.write_text(json.dumps(config))
    output = tmp_path / "must-not-exist"
    result = subprocess.run([str(executable), str(filename), str(output)], cwd=renderer_root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert not output.exists()


def test_capture_lighting_preserves_sky_and_handles_nee_backends(capture_tool, tmp_path):
    config = {'sequence_id': 'lighting_check', 'scene': 'demo', 'width': 33, 'height': 25,
              'tile_width': 16, 'tile_height': 12, 'samples': 2, 'reference_samples': 16,
              'max_depth': 8, 'frames': [{'position': [-4, 1.2, 5], 'target': [0, 0.2, 0]}]}
    plain = run_capture(capture_tool, tmp_path, 'plain', config)[0]
    sky = run_capture(capture_tool, tmp_path, 'sky', dict(config, capture_lighting={'sky': True}))[0]
    for name, pixels in plain.buffers.items():
        np.testing.assert_array_equal(sky.buffers[name], pixels)
    assert sky.metadata['geometry_fingerprint'] == plain.metadata['scene_fingerprint']
    assert sky.metadata['scene_fingerprint'] != plain.metadata['scene_fingerprint']
    area = {'sky': False, 'area': True, 'center': [-1, 6, 0], 'size': 4, 'emission': [24, 24, 24]}
    tiled = run_capture(capture_tool, tmp_path, 'area_tiled', dict(config, capture_lighting=area))[0]
    full = run_capture(capture_tool, tmp_path, 'area_full', dict(config, capture_lighting=area, tile_width=33, tile_height=25))[0]
    persistent = run_capture(capture_tool, tmp_path, 'area_persistent', dict(config, capture_lighting=area, persistent_wavefront=True))[0]
    for name, pixels in tiled.buffers.items():
        np.testing.assert_allclose(full.buffers[name], pixels, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(persistent.buffers[name], pixels, rtol=1e-5, atol=1e-6)
    for name in ['albedo', 'normal', 'geometric_normal', 'depth', 'guide_valid']:
        np.testing.assert_array_equal(tiled.buffers[name], sky.buffers[name])
    assert not np.allclose(tiled.buffers['reference'], sky.buffers['reference'])
    assert tiled.metadata['capture_lighting']['emission'] == [24, 24, 24]


def test_frame_range_resume_and_report_keep_identical_inputs(capture_tool, tmp_path):
    executable, renderer_root = capture_tool
    config = {"sequence_id": "stride_resume", "scene": "demo", "width": 33, "height": 25,
              "tile_width": 16, "tile_height": 12, "samples": 4, "reference_samples": 16,
              "sample_stride": 128, "sample_start": 1000, "max_depth": 8,
              "frames": [{}, {"position": [-9.9, .35, .8]}, {"position": [-9.8, .35, .8]}]}
    baseline = run_capture(capture_tool, tmp_path, "baseline", config)
    filename = tmp_path/"resume.json"
    filename.write_text(json.dumps(config))
    output = tmp_path/"resumed"
    for first, end in ((0, 1), (2, 3), (1, 2)):
        subprocess.run([str(executable), str(filename), str(output), "--start-frame", str(first), "--end-frame", str(end)],
                       cwd=renderer_root, check=True, timeout=120, capture_output=True, text=True)
    resumed = [load_frame(path) for path in discover_sequences(output)[config["sequence_id"]]]
    reports = run_capture(capture_tool, tmp_path, "reports", dict(config, reference_samples=64))
    for i, (expected, actual, report) in enumerate(zip(baseline, resumed, reports)):
        assert actual.metadata == expected.metadata
        assert actual.metadata["render"]["sample_start"] == 1000+i*128
        for name, pixels in expected.buffers.items():
            np.testing.assert_array_equal(actual.buffers[name], pixels)
            if name not in ("reference", "reference_count"):
                np.testing.assert_array_equal(report.buffers[name], pixels)
    duplicate = subprocess.run([str(executable), str(filename), str(output), "--start-frame", "0", "--end-frame", "1"],
                               cwd=renderer_root, timeout=30, capture_output=True, text=True)
    assert duplicate.returncode == 1 and "already exists" in duplicate.stderr


@pytest.mark.parametrize("changes", [{"sample_stride": 3}, {"sample_stride": -1}])
def test_overlapping_sample_stride_rejected(capture_tool, tmp_path, changes):
    executable, renderer_root = capture_tool
    config = dict({"sequence_id": "invalid_stride", "scene": "demo", "width": 8, "height": 8,
                   "samples": 4, "reference_samples": 16, "frames": [{}]}, **changes)
    filename = tmp_path/"bad-stride.json"
    filename.write_text(json.dumps(config))
    output = tmp_path/"must-not-exist"
    result = subprocess.run([str(executable), str(filename), str(output)], cwd=renderer_root,
                            timeout=30, capture_output=True, text=True)
    assert result.returncode == 1 and not output.exists()
