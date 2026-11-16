"""The audited BMW exception preserves actual provenance and stays narrowly bound."""

from copy import deepcopy
import json

import pytest

from test_adaptive_dataset import fake_layers
from test_decoded_cache import cache_data
from test_dataset_generation import small_config
from test_inline_sample_layers import inline_data
from wavefront_denoiser.adaptive_data import layer_layout, read_layer_capture
from wavefront_denoiser.dataset_generation import write_json
from wavefront_denoiser.decoded_cache import decode_frame, verify_decoded_frame
from wavefront_denoiser.distributed.backfill import capture_layers
from wavefront_denoiser.frames import FrameFormatError, read_manifest
from wavefront_denoiser.layer_compatibility import bmw_source_binding, integrator12_fingerprint


def test_fingerprint_changes_only_final_integrator_field():
    def fingerprint(version, settings):
        result = 1469598103934665603
        for byte in settings + version.to_bytes(4, "little"):
            result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
        return format(result, "x")
    assert integrator12_fingerprint(fingerprint(11, b"locked settings")) == fingerprint(12, b"locked settings")
    assert integrator12_fingerprint(fingerprint(11, b"locked settings")) != fingerprint(12, b"changed settings")


def bridge_frame(output, group, index):
    base_path = output / group / f"frame_{index:06d}/manifest.json"
    base = json.loads(base_path.read_text())
    base["render"]["integrator_version"] = 11
    base["render_fingerprint"] = "12abc"
    write_json(base_path, base)
    layer_path = output / "sample_layers" / group / f"frame_{index:06d}/manifest.json"
    layer = json.loads(layer_path.read_text())
    layer["render"]["integrator_version"] = 12
    layer["render_fingerprint"] = integrator12_fingerprint(base["render_fingerprint"])
    layer["source_compatibility"] = bmw_source_binding(base)
    write_json(layer_path, layer)
    return base, layer, layer_path


def test_decoded_bridge_keeps_original_base_and_actual_layer_versions(cache_data):
    output, _, _, _ = cache_data
    base, layer, _ = bridge_frame(output, "new", 0)
    source_bytes = (output / "new/frame_000000/manifest.json").read_bytes()
    assert decode_frame(output, "new", 0)
    verify_decoded_frame(output, "new", 0)
    assert (output / "new/frame_000000/manifest.json").read_bytes() == source_bytes
    assert base["render"]["integrator_version"] == 11
    assert layer["render"]["integrator_version"] == 12


def test_backfill_binds_raw_integrator12_output_without_rewriting_provenance(cache_data, tmp_path):
    output, configs, renderer, _ = cache_data
    base, layer, path = bridge_frame(output, "new", 0)
    del layer["source_compatibility"]
    write_json(path, layer)
    config = dict(configs["new"], sample_layers=layer_layout(configs["new"], 64), reserved_reference_samples=64)
    from pathlib import Path
    source_workspace = str(Path(config["scene_file"]).parent)
    plan = {"source_workspace": source_workspace,
            "tasks": {"new": {"frame_count": 1, "layer_config": config, "bmw_integrator_bridge": True}}}
    capture_layers(plan, "new", output, tmp_path, renderer)
    actual = json.loads(path.read_text())
    assert actual["source_compatibility"] == bmw_source_binding(base)
    assert actual["render"] == layer["render"]
    assert actual["render_fingerprint"] == layer["render_fingerprint"]


@pytest.mark.parametrize("corruption", ["binding", "fingerprint", "render", "scene", "missing-binding"])
def test_bridge_rejects_unrelated_changes(cache_data, corruption):
    output, _, _, _ = cache_data
    base, layer, path = bridge_frame(output, "new", 0)
    if corruption == "binding":
        layer["source_compatibility"]["source_image_sha256"] = "0" * 64
    elif corruption == "fingerprint":
        layer["render_fingerprint"] = "abcd"
    elif corruption == "render":
        layer["render"]["max_depth"] += 1
    elif corruption == "scene":
        base["scene"] = layer["scene"] = "bistro"
    else:
        del layer["source_compatibility"]
    write_json(path, layer)
    layout = layer_layout(json.loads((output / "capture_configs/new.json").read_text()), 64)
    with pytest.raises(FrameFormatError):
        read_layer_capture(path.parent, base, layout, pixels=False)
