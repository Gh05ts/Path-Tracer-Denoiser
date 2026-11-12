import json
from pathlib import Path
import subprocess
import sys

import pytest

from wavefront_denoiser.dataset_generation import build_configs, digest, prepare, read_plan, write_json
from test_dataset_generation import create_frame


@pytest.mark.parametrize("corrupt", [False, True])
def test_scope_migration_preserves_frames_and_rejects_corrupt_data_before_mutation(tmp_path, make_frame, corrupt):
    root = Path(__file__).resolve().parents[1]
    plan = read_plan(root/"dataset_generation/plan-v4.json")
    plan.update(width=8, height=8, reference_samples=16)
    for scene in plan["scenes"]:
        scene_path = tmp_path/(scene["id"]+".json")
        write_json(scene_path, {"name": scene["id"]})
        scene["scene_file"] = str(scene_path)
    original = tmp_path/"original.json"
    write_json(original, plan)
    configs = build_configs(original)
    renderer = tmp_path/"renderer"
    renderer.write_bytes(b"unchanged renderer")
    output = tmp_path/"dataset"
    prepare(original, output, configs, renderer)
    captures = [configs["bmw_view_00_linear_slow"], configs["bistro_view_00_linear_slow"]]
    manifests = []
    for config in captures:
        directory = create_frame(make_frame, output, config, 0)
        manifests.append(directory/"manifest.json")
    before = {p: p.read_bytes() for p in manifests}
    original_lock = (output/"capture_lock.json").read_bytes()
    for scene in plan["scenes"]:
        if scene["id"] != "bmw":
            for a, anchor in enumerate(scene["anchors"]):
                anchor["motion_speeds"] = [[motion, ("slow", "medium", "fast")[(a+m) % 3]]
                                           for m, motion in enumerate(plan["motions"][:3])]
    plan["sample_starts"] = {g: c["sample_start"] for g, c in configs.items()}
    reduced = tmp_path/"reduced.json"
    write_json(reduced, plan)
    report = tmp_path/"selection.json"
    write_json(report, {"plan_sha256": digest(reduced), "renderer_sha256": digest(renderer)})
    if corrupt:
        (manifests[0].parent/"frame.exr").write_bytes(b"broken")
    result = subprocess.run([sys.executable, str(root/"tools/apply_dataset_scope.py"),
        "--plan", str(reduced), "--output", str(output), "--renderer", str(renderer),
        "--selection-report", str(report)], capture_output=True, text=True)
    assert {p: p.read_bytes() for p in manifests} == before
    if corrupt:
        assert result.returncode != 0 and "checksum" in result.stderr
        assert (output/"capture_lock.json").read_bytes() == original_lock
        assert not (output/"capture-history").exists()
    else:
        assert result.returncode == 0, result.stderr
        lock = json.loads((output/"capture_lock.json").read_text())
        assert (output/lock["revision"]["parent_lock"]).read_bytes() == original_lock
        assert lock["revision"]["preserved_groups"] == sorted(c["sequence_id"] for c in captures)
        prepare(reduced, output, build_configs(reduced), renderer)
