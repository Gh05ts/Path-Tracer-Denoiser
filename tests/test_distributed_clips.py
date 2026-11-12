"""Two isolated workers sharing R2 coordination; no renderer/GPU required."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

from test_adaptive_dataset import fake_layers
from test_dataset_generation import small_config, create_frame
from wavefront_denoiser.b2_archive import file_hashes
from wavefront_denoiser.dataset_generation import digest, write_json
from wavefront_denoiser.decoded_cache import BackgroundDecodedCache
from wavefront_denoiser.inline_sample_layers import InlineSampleLayers
from wavefront_denoiser.r2_native import R2NativeClient, R2RequestError
from wavefront_denoiser.distributed.coordinator import Coordinator
from wavefront_denoiser.distributed.plan import prepare, install_bundle
from wavefront_denoiser.distributed.store import R2Store
from wavefront_denoiser.distributed import worker


class MemoryClient(R2NativeClient):
    """Exercise native conditional headers/ETags against atomic shared storage."""
    def __init__(self):
        super().__init__("bucket", "v4")
        self.endpoint = "https://test.r2.cloudflarestorage.com"
        self.objects = {}
        self.lock = threading.RLock()
        self.fail_key = None
        self.ambiguous = False

    def _request(self, method, relative=None, *, path=None, headers=None, **kwargs):
        with self.lock:
            old = self.objects.get(relative)
            etag = hashlib.md5(old, usedforsecurity=False).hexdigest() if old is not None else None
            if method in ("GET", "HEAD"):
                if old is None:
                    raise R2RequestError(404, "NoSuchKey")
                return (old if method == "GET" else b""), {"etag": f'"{etag}"', "content-length": str(len(old))}
            assert method == "PUT"
            if relative == self.fail_key:
                raise R2RequestError(503, "ServiceUnavailable")
            if headers.get("if-none-match") == "*" and old is not None:
                raise R2RequestError(412, "PreconditionFailed")
            if "if-match" in headers and headers["if-match"] != f'"{etag}"':
                raise R2RequestError(412, "PreconditionFailed")
            body = Path(path).read_bytes()
            self.objects[relative] = body
            if self.ambiguous:
                self.ambiguous = False
                raise R2RequestError(412, "PreconditionFailed")
            return b"", {"etag": '"' + hashlib.md5(body, usedforsecurity=False).hexdigest() + '"'}

    def list_files(self):
        with self.lock:
            return {n: {"etag": hashlib.md5(b, usedforsecurity=False).hexdigest(), "size": len(b)}
                    for n, b in self.objects.items()}


@pytest.fixture
def distributed_data(tmp_path, make_frame, small_config, request):
    workspace = tmp_path / "machine-a"
    (workspace / "Wavefront-Path-Tracer").mkdir(parents=True)
    renderer = workspace / "renderer"
    renderer.write_bytes(b"identical renderer")
    output = workspace / "dataset"
    output.mkdir()
    configs, assets = {}, {}
    scenes = ("scene_a", "scene_a" if getattr(request, "param", False) else "scene_b")
    for ordinal, scene in enumerate(scenes):
        path = workspace / "assets" / (scene + ".json")
        write_json(path, {"name": scene, "preset": "demo"})
        assets[str(path)] = digest(path)
        group = scene + f"_view_{ordinal:02d}_linear"
        config = deepcopy(small_config)
        config.update(sequence_id=group, scene=scene, scene_file=str(path), sample_start=10000+ordinal*100000)
        configs[group] = config
        write_json(output / "capture_configs" / (group + ".json"), config)
    capture = {"format": "wavefront-capture-lock", "version": 1, "configs": configs,
        "assets": assets, "renderer_sha256": digest(renderer),
        "plan": {"dataset_id": "denoiser-v4", "scenes": [
            {"id": "scene_a", "partition": "train"}, {"id": "scene_b", "partition": "validation"}]}}
    write_json(output / "capture_lock.json", capture)
    first = next(iter(configs))
    for index in (0, 1):
        create_frame(make_frame, output, configs[first], index)
    InlineSampleLayers(output, configs, renderer, workspace, spps=[1, 2, 8, 16, 32],
                       after_group=first, reserved_reference_samples=64)
    BackgroundDecodedCache(output, configs, after_group=first).close()
    prepared = tmp_path / "prepared"
    plan = prepare(output, workspace, prepared, primary_worker="3090-a")
    client = MemoryClient()
    store = R2Store(client)
    coordinator = Coordinator(store, plan)
    coordinator.initialize(prepared)
    return SimpleNamespace(root=workspace, output=output, renderer=renderer, configs=configs,
        prepared=prepared, plan=plan, client=client, store=store, coordinator=coordinator,
        first=first, second=list(configs)[1])


def second_machine(data, tmp_path):
    root = tmp_path / "machine-b"
    (root / "Wavefront-Path-Tracer").mkdir(parents=True)
    shutil.copytree(data.root / "assets", root / "assets")
    shutil.copy2(data.renderer, root / "renderer")
    output = root / "dataset"
    install_bundle(data.prepared / "metadata.tar", output, data.plan)
    return root, output, root / "renderer"


def test_prepare_is_offline_repeatable_and_preserves_partial_progress(distributed_data):
    d = distributed_data
    before = {p: p.read_bytes() for p in d.output.glob("*/frame_*/*")}
    assert prepare(d.output, d.root, d.prepared, primary_worker="3090-a") == d.plan
    assert d.plan["tasks"][d.first]["affinity"] == "3090-a"
    assert len(d.plan["tasks"][d.first]["local_manifests"]) == 2
    assert d.plan["tasks"][d.second]["affinity"] is None
    assert {p: p.read_bytes() for p in before} == before


def test_atomic_claim_race_and_resume_ownership(distributed_data):
    d = distributed_data
    barrier = threading.Barrier(2)
    def claim(number):
        barrier.wait()
        return d.coordinator.claim(f"worker-{number}", f"machine-{number}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (1, 2)))
    winners = [r for r in results if r]
    assert len(winners) == 1 and winners[0]["group"] == d.second
    winner = winners[0]
    assert d.coordinator.claim(winner["worker"], winner["machine"])["token"] == winner["token"]
    with pytest.raises(ValueError, match="different local workspace"):
        d.coordinator.claim(winner["worker"], "another-disk")
    with pytest.raises(ValueError, match="current claim token"):
        d.coordinator.release(d.second, "wrong")
    d.coordinator.release(d.second, winner["token"])
    new = d.coordinator.claim("replacement", "replacement-disk")
    with pytest.raises(ValueError, match="ownership was lost"):
        d.coordinator.owned(winner)
    assert new["token"] != winner["token"]


def test_conditional_write_reconciles_lost_response_and_never_overwrites(distributed_data, tmp_path):
    d = distributed_data
    d.client.ambiguous = True
    assert d.store.put("test.json", {"write_id": "unique"})
    assert not d.store.put("test.json", {"write_id": "other"})
    path = tmp_path / "payload.npy"
    path.write_bytes(b"original")
    d.store.upload(path, "immutable.npy", file_hashes(path))
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="conflicts"):
        d.store.upload(path, "immutable.npy", file_hashes(path))
    assert d.client.objects["immutable.npy"] == b"original"


def fake_render(make_frame, calls, barrier=None):
    def render(command, **kwargs):
        config = json.loads(Path(command[1]).read_text())
        target = Path(command[2])
        first, end = int(command[-3]), int(command[-1])
        layer = "sample_layers" in config
        calls.append((config["sequence_id"], layer, first, end, config["scene_file"]))
        if barrier and not layer:
            barrier.wait(timeout=10)
        for index in range(first, end):
            if layer:
                output = target.parent.parent
                base = json.loads((output / config["sequence_id"] / f"frame_{index:06d}" / "manifest.json").read_text())
                fake_layers(target / f"frame_{index:06d}", base, config["sample_layers"])
            else:
                create_frame(make_frame, target.parent, config, index)
        return SimpleNamespace(returncode=0)
    return render


def patch_workers(d, monkeypatch, render):
    def connect(output, destination):
        return SimpleNamespace(destination=destination, native=d.client), d.store
    monkeypatch.setattr(worker, "connection", connect)
    monkeypatch.setattr(worker, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"]))
    monkeypatch.setattr(worker, "terminate_subprocesses_on_sigterm", nullcontext)
    monkeypatch.setattr(worker.subprocess, "run", render)


def test_two_machines_render_distinct_whole_clips_and_publish_verified_caches(distributed_data, tmp_path, make_frame, monkeypatch):
    d = distributed_data
    root_b, output_b, renderer_b = second_machine(d, tmp_path)
    calls = []
    patch_workers(d, monkeypatch, fake_render(make_frame, calls, threading.Barrier(2)))
    before = file_hashes(d.output / d.first / "frame_000000/frame.exr")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker.run_worker, d.plan, d.output, d.root, d.renderer,
                              "r2://bucket/v4", "3090-a", max_clips=1),
                   pool.submit(worker.run_worker, d.plan, output_b, root_b, renderer_b,
                              "r2://bucket/v4", "3090-b", max_clips=1)]
        for future in futures:
            assert future.result()["processed_clips"] == 1
    assert {(g, layer, first, end) for g, layer, first, end, _ in calls} == {
        (d.first, False, 2, 8), (d.second, False, 0, 8), (d.second, True, 0, 8)}
    assert all(str(root_b) in path for group, _, _, _, path in calls if group == d.second)
    assert d.coordinator.status()["counts"] == {"pending": 0, "claimed": 0, "complete": 2}
    assert hashlib.sha256(d.client.objects[f"{d.first}/frame_000000/frame.exr"]).hexdigest() == before["sha256"]
    assert f"decoded/{d.second}/frame_000007/float.npy" in d.client.objects
    for scene in ("scene_a", "scene_b"):
        receipt = json.loads(d.client.objects[f"r2-archives/{scene}.json"])
        assert receipt["status"] == "verified"
        d.client.verify_files(receipt["files"])
    assert not list(d.output.rglob("*.exr")) and not list(output_b.rglob("*.npy"))


def test_failed_upload_resumes_same_claim_without_rerender(distributed_data, tmp_path, make_frame, monkeypatch):
    d = distributed_data
    root, output, renderer = second_machine(d, tmp_path)
    calls = []
    patch_workers(d, monkeypatch, fake_render(make_frame, calls))
    d.client.fail_key = f"decoded/{d.second}/frame_000000/float.npy"
    with pytest.raises(R2RequestError):
        worker.run_worker(d.plan, output, root, renderer, "r2://bucket/v4", "3090-b", max_clips=1)
    assert len(list(output.rglob("*.exr"))) == 16
    assert len(list(output.rglob("*.npy"))) == 16
    assert d.coordinator.status()["counts"]["claimed"] == 1
    d.client.fail_key = None
    worker.run_worker(d.plan, output, root, renderer, "r2://bucket/v4", "3090-b", max_clips=1)
    assert len(calls) == 2  # Exactly one base render and one layer render, across both runs.
    assert not list(output.rglob("*.exr")) and not list(output.rglob("*.npy"))


def test_worker_rejects_changed_assets_or_renderer(distributed_data, tmp_path):
    d = distributed_data
    root, _, renderer = second_machine(d, tmp_path)
    worker.validate_machine(d.plan, root, renderer)
    renderer.write_bytes(b"different executable")
    with pytest.raises(ValueError, match="Renderer binary differs"):
        worker.validate_machine(d.plan, root, renderer)
    shutil.copy2(d.renderer, renderer)
    (root / "assets/scene_b.json").write_text('{"name":"changed"}')
    with pytest.raises(ValueError, match="differs"):
        worker.validate_machine(d.plan, root, renderer)


def test_existing_archive_is_seeded_complete_without_rerender(distributed_data, tmp_path, make_frame):
    d = distributed_data
    for index in range(2, 8):
        create_frame(make_frame, d.output, d.configs[d.first], index)
    files = {p.relative_to(d.output).as_posix(): file_hashes(p)
             for p in (d.output / d.first).glob("frame_*/*")}
    receipt = {"format": "wavefront-r2-scene-archive", "version": 1, "status": "verified",
               "scene": "scene_a", "capture_lock_sha256": digest(d.output / "capture_lock.json"), "files": files}
    path = d.output / "r2-archives/scene_a.json"
    write_json(path, receipt)
    d.client.objects["r2-archives/scene_a.json"] = path.read_bytes()
    for name in files:
        d.client.objects[name] = (d.output / name).read_bytes()
    prepared = tmp_path / "seeded-preparation"
    plan = prepare(d.output, d.root, prepared, primary_worker="3090-a", run_id="seeded-v4")
    client = MemoryClient()
    client.objects = {n: b for n, b in d.client.objects.items() if not n.startswith("_distributed/")}
    coordinator = Coordinator(R2Store(client), plan)
    coordinator.initialize(prepared)
    assert coordinator.status()["counts"]["complete"] == 1
    assert coordinator.claim("3090-a", "disk-a")["group"] == d.second
    assert not coordinator.finalize_scene("scene_a", destination="r2://bucket/v4", endpoint=d.client.endpoint)
    assert d.client.objects["r2-archives/scene_a.json"] == path.read_bytes()


def test_failed_reprepare_preserves_original_bundle(distributed_data):
    d = distributed_data
    before = (d.prepared / "metadata.tar").read_bytes()
    plan_bytes = (d.prepared / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="Prepared distributed plan changed"):
        prepare(d.output, d.root, d.prepared, primary_worker="different-primary")
    assert (d.prepared / "metadata.tar").read_bytes() == before
    assert (d.prepared / "plan.json").read_bytes() == plan_bytes


def test_distributed_workspace_rejects_sequential_capture(distributed_data, tmp_path, capsys):
    d = distributed_data
    _, output, renderer = second_machine(d, tmp_path)
    from wavefront_denoiser.dataset_generation import main
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output), "--renderer", str(renderer)])
    assert error.value.code == 2
    assert "managed by distributed clip workers" in capsys.readouterr().err


def test_settings_reader_never_executes_the_launcher(tmp_path, monkeypatch):
    from wavefront_denoiser.distributed.cli import settings
    import os
    monkeypatch.setattr(os, "environ", {"R2_SECRET_ACCESS_KEY": "environment-secret"})
    path = tmp_path / "settings.py"
    path.write_text("ACCOUNT_ID = 'account'\nACCESS_KEY_ID = 'key'\nSECRET_ACCESS_KEY = 'file-secret'\nraise RuntimeError('must not execute')\n")
    settings(path)
    assert os.environ["R2_ACCOUNT_ID"] == "account"
    assert os.environ["R2_SECRET_ACCESS_KEY"] == "environment-secret"


def test_coordinator_cannot_complete_a_partial_clip(distributed_data):
    d = distributed_data
    claim = d.coordinator.claim("3090-b", "disk-b")
    with pytest.raises(ValueError, match="whole clip"):
        d.coordinator.complete(claim, {})
    assert d.coordinator.owned(claim)[0]["status"] == "claimed"


def test_a_second_run_cannot_create_an_independent_queue_for_the_same_prefix(distributed_data):
    d = distributed_data
    plan = dict(d.plan, run_id="another-run")
    with pytest.raises(ValueError, match="Another distributed queue"):
        Coordinator(d.store, plan).initialize(d.prepared)
    assert d.coordinator.status()["counts"] == {"pending": 2, "claimed": 0, "complete": 0}


@pytest.mark.parametrize("distributed_data", [True], indirect=True)
def test_scene_receipt_waits_for_clips_from_both_machines(distributed_data, tmp_path, make_frame, monkeypatch):
    d = distributed_data
    root_b, output_b, renderer_b = second_machine(d, tmp_path)
    calls = []
    patch_workers(d, monkeypatch, fake_render(make_frame, calls))
    worker.run_worker(d.plan, output_b, root_b, renderer_b, "r2://bucket/v4", "3090-b", max_clips=1)
    assert "r2-archives/scene_a.json" not in d.client.objects
    worker.run_worker(d.plan, d.output, d.root, d.renderer, "r2://bucket/v4", "3090-a", max_clips=1)
    receipt = json.loads(d.client.objects["r2-archives/scene_a.json"])
    assert f"{d.first}/frame_000007/frame.exr" in receipt["files"]
    assert f"{d.second}/frame_000007/frame.exr" in receipt["files"]
    assert f"decoded/{d.second}/frame_000007/float.npy" in receipt["files"]
    d.client.verify_files(receipt["files"])
