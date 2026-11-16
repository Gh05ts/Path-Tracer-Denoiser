"""Backfill restoration, durable ownership and scene receipt upgrades on fake R2."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dataset_generation import small_config
from test_distributed_clips import (distributed_data, fake_render, patch_workers,
                                    second_machine)
from wavefront_denoiser.b2_archive import file_hashes
from wavefront_denoiser.dataset_generation import digest, write_json
from wavefront_denoiser.distributed import backfill, worker
from wavefront_denoiser.distributed import backfill_cli
from wavefront_denoiser.distributed.coordinator import Coordinator
from wavefront_denoiser.distributed.plan import install_bundle, prepare
from wavefront_denoiser.r2_native import R2RequestError


def _completed_data(distributed_data, make_frame, monkeypatch, tmp_path, *, decode_only=False, defer_all=False):
    d = distributed_data
    if decode_only:
        # A clip captured after layer enablement but before cache enablement.
        path = d.output / "sample_layers_lock.json"
        layers = json.loads(path.read_text())
        layers["groups"].insert(0, d.first)
        config = dict(d.configs[d.first], sample_layers=backfill.layer_layout(d.configs[d.first], 64),
                      reserved_reference_samples=64)
        layers["configs"][d.first] = config
        write_json(path, layers)
        write_json(d.output / "sample_layer_configs" / (d.first + ".json"), config)
        decoded = json.loads((d.output / "decoded_cache_lock.json").read_text())
        decoded["sample_layers_lock_sha256"] = digest(path)
        write_json(d.output / "decoded_cache_lock.json", decoded)
    if defer_all:
        path = d.output / "decoded_cache_lock.json"
        decoded = json.loads(path.read_text())
        decoded["groups"] = []
        write_json(path, decoded)
    # Fixtures use tiny EXRs; real v4 manifests declare this provenance field.
    for path in (d.output / d.first).glob("frame_*/manifest.json"):
        meta = json.loads(path.read_text())
        meta["render"]["integrator_version"] = 12
        write_json(path, meta)
    d.prepared = tmp_path / "backfill-source-prepared"
    d.plan = prepare(d.output, d.root, d.prepared, primary_worker="3090-a")
    d.client.objects.clear()
    d.coordinator = Coordinator(d.store, d.plan)
    d.coordinator.initialize(d.prepared)
    render = fake_render(make_frame, [])

    def render_with_version(command, **kwargs):
        result = render(command, **kwargs)
        config = json.loads(Path(command[1]).read_text())
        if "sample_layers" not in config:
            for path in Path(command[2]).glob("frame_*/manifest.json"):
                meta = json.loads(path.read_text())
                meta["render"]["integrator_version"] = 12
                write_json(path, meta)
        return result

    patch_workers(d, monkeypatch, render_with_version)
    worker.run_worker(d.plan, d.output, d.root, d.renderer, "r2://bucket/v4", "3090-a")
    assert d.coordinator.status()["counts"]["complete"] == 2
    d.backfill_plan = backfill.prepare_plan(d.store, d.plan, d.output, run_id="v4-backfill",
                                          destination="r2://bucket/v4", endpoint=d.client.endpoint)
    d.backfill_coordinator = backfill.BackfillCoordinator(d.store, d.backfill_plan)
    d.backfill_coordinator.initialize()
    monkeypatch.setattr(backfill, "connection", lambda output, destination: (
        SimpleNamespace(destination=destination, native=d.client), d.store))
    monkeypatch.setattr(backfill, "terminate_subprocesses_on_sigterm", nullcontext)

    def download(key, target, expected):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(d.client.objects[key])
        assert file_hashes(target) == expected

    monkeypatch.setattr(d.store, "download", download)
    d.calls = []
    monkeypatch.setattr(backfill.subprocess, "run", fake_render(make_frame, d.calls))
    return d


@pytest.fixture(params=[False, True], ids=["missing-layers", "decode-only"])
def completed_data(distributed_data, make_frame, monkeypatch, request, tmp_path):
    return _completed_data(distributed_data, make_frame, monkeypatch, tmp_path, decode_only=request.param)


def test_backfill_restores_sources_and_upgrades_receipt_without_recapture(completed_data, tmp_path):
    d = completed_data
    source_objects = dict(d.client.objects)
    root, output, renderer = second_machine(d, tmp_path)
    result = backfill.run_worker(d.backfill_plan, output, root, {12: renderer}, "r2://bucket/v4", "3090-b")
    assert result["counts"] == {"pending": 0, "claimed": 0, "complete": 1}
    assert all(layer for _, layer, *_ in d.calls)
    assert bool(d.calls) == d.backfill_plan["tasks"][d.first]["new_layers"]
    # Existing source EXRs/manifests and already complete clips stay byte-identical.
    for key, value in source_objects.items():
        if not key.startswith(("_", "r2-archives/")):
            assert d.client.objects[key] == value
    receipt = d.store.get("r2-archives/scene_a.json")[0]
    assert receipt["backfill_run_id"] == "v4-backfill"
    assert receipt["capture_lock_sha256"] == digest(output / "capture_lock.json")
    assert f"sample_layer_configs/{d.first}.json" in receipt["files"]
    for index in range(8):
        assert f"decoded/{d.first}/frame_{index:06d}/float.npy" in receipt["files"]
    assert not list((output / d.first).rglob("*.exr"))
    assert not list((output / "decoded").rglob("*.npy"))
    # Receipts repair/idempotence and completed task skipping.
    assert backfill.run_worker(d.backfill_plan, output, root, {12: renderer}, "r2://bucket/v4", "3090-b")["processed_clips"] == 0
    assert d.store.get(receipt["source_receipt"])[0] == d.backfill_plan["source_receipts"]["scene_a"]


def test_backfill_upload_failure_retains_files_and_resumes_without_rendering(completed_data):
    d = completed_data
    d.client.fail_key = f"decoded/{d.first}/frame_000003/uint.npy"
    with pytest.raises(R2RequestError, match="ServiceUnavailable"):
        backfill.run_worker(d.backfill_plan, d.output, d.root, {12: d.renderer}, "r2://bucket/v4", "3090-a")
    assert (d.output / "decoded" / d.first / "frame_000003/uint.npy").exists()
    assert d.backfill_coordinator.status()["counts"]["claimed"] == 1
    assert d.store.get("r2-archives/scene_a.json")[0] == d.backfill_plan["source_receipts"]["scene_a"]
    calls = len(d.calls)
    d.client.fail_key = None
    assert backfill.run_worker(d.backfill_plan, d.output, d.root, {12: d.renderer}, "r2://bucket/v4", "3090-a")["processed_clips"] == 1
    assert len(d.calls) == calls


def test_backfill_claim_race_and_wrong_renderer_rejected_before_claim(completed_data, tmp_path):
    d = completed_data
    c = d.backfill_coordinator
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda n: c.claim(f"worker-{n}", f"machine-{n}"), (1, 2)))
    winner = next(claim for claim in claims if claim)
    assert sum(claim is not None for claim in claims) == 1
    c.release(d.first, winner["token"])
    if d.backfill_plan["tasks"][d.first]["new_layers"]:
        renderer = tmp_path / "wrong-renderer"
        renderer.write_bytes(b"another integrator")
        with pytest.raises(ValueError, match="Renderer binary differs"):
            backfill.run_worker(d.backfill_plan, d.output, d.root, {12: renderer}, "r2://bucket/v4", "3090-b")
        assert c.status()["counts"]["pending"] == 1
        with pytest.raises(ValueError, match="Integrator 12 renderer is required"):
            backfill.validate_backfill_machine(d.backfill_plan, d.root, {})
    else:
        # Decoding already captured layers needs no renderer or scene assets.
        backfill.validate_backfill_machine(d.backfill_plan, d.root, {})


def test_conflicting_remote_payload_never_overwritten(completed_data):
    d = completed_data
    key = f"decoded/{d.first}/frame_000000/float.npy"
    d.client.objects[key] = b"conflicting object"
    with pytest.raises(ValueError, match="conflicts"):
        backfill.run_worker(d.backfill_plan, d.output, d.root, {12: d.renderer}, "r2://bucket/v4", "3090-a")
    assert d.client.objects[key] == b"conflicting object"
    assert (d.output / "decoded" / d.first / "frame_000000/float.npy").exists()


def test_refuses_initialization_before_generation_finishes(distributed_data):
    d = distributed_data
    with pytest.raises(ValueError, match="Finish all generation"):
        backfill.prepare_plan(d.store, d.plan, d.output, run_id="backfill", destination="r2://bucket/v4", endpoint=d.client.endpoint)
    assert d.store.get("_backfill/active.json") is None


def test_two_machines_complete_distinct_backfill_clips(distributed_data, make_frame, monkeypatch, tmp_path):
    d = _completed_data(distributed_data, make_frame, monkeypatch, tmp_path, defer_all=True)
    root_b, output_b, renderer_b = second_machine(d, tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(backfill.run_worker, d.backfill_plan, d.output, d.root, {12: d.renderer},
                            "r2://bucket/v4", "3090-a", max_clips=1),
                pool.submit(backfill.run_worker, d.backfill_plan, output_b, root_b, {12: renderer_b},
                            "r2://bucket/v4", "3090-b", max_clips=1)]
        assert [job.result()["processed_clips"] for job in jobs] == [1, 1]
    assert d.backfill_coordinator.status()["counts"] == {"pending": 0, "claimed": 0, "complete": 2}
    assert {d.store.get(d.backfill_coordinator.key(g))[0]["worker"] for g in d.configs} == {"3090-a", "3090-b"}
    assert all(layer and group == d.first for group, layer, *_ in d.calls)


def test_receipt_failure_is_repaired_without_recapture(completed_data):
    d = completed_data
    key = "r2-archives/scene_a.json"
    d.client.fail_key = key
    with pytest.raises(R2RequestError, match="ServiceUnavailable"):
        backfill.run_worker(d.backfill_plan, d.output, d.root, {12: d.renderer}, "r2://bucket/v4", "3090-a")
    assert d.backfill_coordinator.status()["counts"]["complete"] == 1
    assert (d.output / "decoded" / d.first / "frame_000000/float.npy").exists()
    calls = len(d.calls)
    d.client.fail_key = None
    backfill.run_worker(d.backfill_plan, d.output, d.root, {12: d.renderer}, "r2://bucket/v4", "3090-a")
    assert d.store.get(key)[0]["backfill_run_id"] == "v4-backfill"
    assert len(d.calls) == calls
    assert not (d.output / "decoded" / d.first / "frame_000000/float.npy").exists()


def test_cli_initialization_check_worker_and_status(completed_data, monkeypatch, capsys):
    d = completed_data
    monkeypatch.setattr(backfill_cli, "connection", backfill.connection)
    common = ["--destination", "r2://bucket/v4", "--run-id", "v4-backfill", "--output", str(d.output)]
    assert backfill_cli.main(["init", *common]) == 0
    assert backfill_cli.main(["init", *common]) == 0
    assert backfill_cli.main(["check", *common, "--workspace", str(d.root), "--renderer", f"12={d.renderer}"]) == 0
    assert d.backfill_coordinator.status()["counts"]["pending"] == 1
    assert backfill_cli.main(["worker", *common, "--workspace", str(d.root), "--renderer", f"12={d.renderer}",
                              "--worker-id", "3090-a", "--max-clips", "1"]) == 0
    assert backfill_cli.main(["status", *common]) == 0
    assert '"complete": 1' in capsys.readouterr().out


def test_history_requires_checksum_configs_and_reuse_evidence(completed_data):
    d = completed_data
    path = d.output / "capture_lock.json"
    original = json.loads(path.read_text())
    sha = digest(path)
    write_json(d.output / f"capture-history/{sha}.json", original)
    evidence = d.output / "capture-history/reuse.json"
    write_json(evidence, {"verified": True})
    current = deepcopy(original)
    current["renderer_sha256"] = "f" * 64
    current["revision"] = {"parent_sha256": sha, "parent_lock": f"capture-history/{sha}.json",
                           "reused_scenes": ["scene_a"], "reset_scenes": [],
                           "validation": {"scene_a": {"path": "capture-history/reuse.json", "sha256": digest(evidence)}}}
    write_json(path, current)
    assert backfill.source_lock(d.output, "scene_a", sha)["renderer_sha256"] == original["renderer_sha256"]
    with pytest.raises(ValueError, match="authorized capture ancestor"):
        backfill.source_lock(d.output, "scene_b", sha)
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="reuse validation"):
        backfill.source_lock(d.output, "scene_a", sha)
