"""Opt-in CUDA validation of layer capture, v5 reuse, replay and resume."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_adaptive_dataset import fixture_plan
from wavefront_denoiser.adaptive_data import V5SequenceDataset, load_v5_frame, read_layer_capture
from wavefront_denoiser.adaptive_dataset import capture_v5, prepare_v5
from wavefront_denoiser.dataset_generation import build_configs, digest, prepare, run_captures
from wavefront_denoiser.frames import load_frame


@pytest.fixture(scope="module")
def renderer_tool():
    value = os.environ.get("WAVEFRONT_CAPTURE_EXECUTABLE")
    root = os.environ.get("WAVEFRONT_RENDERER_ROOT")
    if not value or not root:
        pytest.skip("Set WAVEFRONT_CAPTURE_EXECUTABLE and WAVEFRONT_RENDERER_ROOT for actual GPU tests")
    return Path(value).resolve(), Path(root).resolve()


def run(renderer_tool, config, path, *, first=0, end=1):
    executable, root = renderer_tool
    path.parent.mkdir(parents=True, exist_ok=True)
    filename = path.with_suffix(".json")
    filename.write_text(json.dumps(config))
    subprocess.run([str(executable), str(filename), str(path), "--start-frame", str(first), "--end-frame", str(end)],
                   cwd=root, check=True, timeout=120, capture_output=True, text=True)
    return path / f"frame_{first:06d}"


def test_gpu_v5_full_clip_pipeline_reuses_source_and_resumes(renderer_tool, tmp_path):
    renderer, root = renderer_tool
    source_plan, plan = fixture_plan(tmp_path, width=17, height=13)
    configs = build_configs(source_plan)
    groups = list(configs)[:2]  # Explicit train and validation clips; no test pixels.
    source, output = tmp_path / "v4", tmp_path / "v5"
    prepare(source_plan, source, configs, renderer)
    run_captures(configs, groups, source, renderer, root)
    before = {p: digest(p) for p in source.rglob("*") if p.is_file()}
    prepare_v5(plan, source, output, renderer=renderer, groups=groups)
    partial = capture_v5(output, source, renderer, root, groups, end_frame=3)
    assert partial["captured_frames"] == 6
    # Preserve abandoned exports and render only their missing frame ranges.
    interrupted = output / "layers" / groups[0] / "frame_000003.partial"
    interrupted.mkdir(parents=True)
    (interrupted / "interrupted").write_bytes(b"keep")
    completed = capture_v5(output, source, renderer, root, groups)
    assert completed["captured_frames"] == 10 and completed["skipped_frames"] == 6
    assert list((output / "recovery.partial/layers" / groups[0]).glob("*.partial-*"))
    resumed = capture_v5(output, source, renderer, root, groups)
    assert resumed["captured_frames"] == 0 and resumed["skipped_frames"] == 16
    assert all(digest(path) == sha for path, sha in before.items())
    window = V5SequenceDataset(output, partition="train")[0]
    assert tuple(window["layers"].shape) == (5,6,3,13,17)
    import torch
    from wavefront_denoiser.sample_layers import replay_sample_layers
    layers = window["layers"].to("cuda")
    counts = torch.full((5,1,13,17), 4, dtype=torch.int64, device="cuda")
    torch.testing.assert_close(replay_sample_layers(layers, counts), window["buffers"]["radiance"].to("cuda"), rtol=0, atol=0)
    counts[..., 0, 0] = 0
    counts[..., -1, -1] = 63
    replay = replay_sample_layers(layers, counts)
    assert torch.isfinite(replay).all() and (replay[..., 0, 0] == 0).all()
    assert not (output / "base" / list(configs)[2]).exists()


def test_gpu_layer_values_equal_independent_ranges_and_backends(renderer_tool, tmp_path):
    _, plan = fixture_plan(tmp_path)
    source_plan = tmp_path / "source-plan.json"
    config = next(iter(build_configs(source_plan).values()))
    base_dir = run(renderer_tool, config, tmp_path / "base", first=3, end=4)
    base = load_frame(base_dir)
    from wavefront_denoiser.adaptive_data import layer_layout
    layout = layer_layout(config, 64)
    layer_config = dict(config, sample_layers=layout, reserved_reference_samples=64)
    layer_dir = run(renderer_tool, layer_config, tmp_path / "tiled", first=3, end=4)
    metadata, images = read_layer_capture(layer_dir, base.metadata, layout)
    assert "reference_samples" not in metadata["render"]
    for item in layout:
        direct = dict(config, samples=item["samples"], reference_samples=1,
                      sample_start=config["sample_start"]+item["offset"])
        frame = load_frame(run(renderer_tool, direct, tmp_path / f"direct-{item['samples']}", first=3, end=4))
        np.testing.assert_allclose(images[item["samples"]], frame.buffers["radiance"], rtol=1e-5, atol=1e-6)
    for name, changes in (("full", {"tile_width":config["width"],"tile_height":config["height"]}),
                          ("persistent", {"persistent_wavefront":True})):
        variant_dir = run(renderer_tool, dict(layer_config, **changes), tmp_path / name, first=3, end=4)
        expected = deepcopy(base.metadata)
        variant_meta = json.loads((variant_dir / "manifest.json").read_text())
        expected["render_fingerprint"] = variant_meta["render_fingerprint"]
        expected["render"].update(changes)
        _, variant = read_layer_capture(variant_dir, expected, layout)
        for spp in images:
            np.testing.assert_allclose(variant[spp], images[spp], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("layers", [[], [{"samples":3,"offset":68}],
    [{"samples":1,"offset":0}], [{"samples":1,"offset":68},{"samples":2,"offset":68}],
    [{"samples":32,"offset":120}]])
def test_gpu_capture_rejects_invalid_layers_before_output(renderer_tool, tmp_path, layers):
    fixture_plan(tmp_path)
    config = next(iter(build_configs(tmp_path / "source-plan.json").values()))
    config.update(sample_layers=layers, reserved_reference_samples=64)
    with pytest.raises(subprocess.CalledProcessError):
        run(renderer_tool, config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
