"""One GPU capture stream plus a bounded CPU decode/upload process per machine."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
import uuid

from ..adaptive_data import read_layer_capture
from ..b2_archive import file_hashes
from ..dataset_generation import (asset_files, capture_lock, digest, missing_ranges,
                                  terminate_subprocesses_on_sigterm, verify_frame, write_json)
from ..decoded_cache import _decode_clip, verify_decoded_frame
from ..frames import read_manifest
from ..r2_archive import R2SceneArchiver
from .coordinator import Coordinator
from .plan import belongs
from .store import R2Store


def connection(output, destination):
    archiver = R2SceneArchiver(Path(output), destination)
    archiver.preflight()
    return archiver, R2Store(archiver.native)


def validate_machine(plan, workspace, renderer, *, scenes=None):
    workspace, renderer = Path(workspace).resolve(), Path(renderer).resolve()
    if digest(renderer) != plan["renderer_sha256"]:
        raise ValueError("Renderer binary differs from locked v4; copy the exact persistent renderer")
    checked = set()
    for task in plan["tasks"].values():
        if scenes is not None and task["scene"] not in scenes:
            continue
        original = Path(task["config"]["scene_file"])
        scene = workspace / original.relative_to(plan["source_workspace"])
        if scene in checked:
            continue
        for path in asset_files(scene):
            relative = path.relative_to(workspace).as_posix()
            if plan["assets"].get(relative) != digest(path):
                raise ValueError(f"Scene/texture/geometry differs from the locked v4 assets: {relative}")
        checked.add(scene)


def _recover(output, path, category, group):
    if path.exists():
        parent = Path(output) / "recovery.partial" / category / group
        parent.mkdir(parents=True, exist_ok=True)
        path.rename(parent / (path.name + f"-{time.time_ns()}"))


def capture_clip(plan, claim, output, workspace, renderer):
    """Use locked camera/sample settings; rebase only the runtime scene path."""
    output, workspace, renderer = Path(output), Path(workspace), Path(renderer)
    group = claim["group"]
    task = plan["tasks"][group]
    validate_machine(plan, workspace, renderer, scenes={task["scene"]})
    base_config = task["config"]
    runtime = deepcopy(base_config)
    runtime["scene_file"] = str(workspace / Path(runtime["scene_file"]).relative_to(plan["source_workspace"]))
    configurations = [(False, base_config, runtime, output / group, group + ".log")]
    if task["layers"]:
        locked_layers = json.loads((output / "sample_layer_configs" / (group + ".json")).read_text())
        runtime_layers = dict(locked_layers, scene_file=runtime["scene_file"])
        configurations.append((True, locked_layers, runtime_layers,
                               output / "sample_layers" / group, "layers-" + group + ".log"))
    for layers, locked, actual, directory, logfile in configurations:
        missing = []
        for index in range(8):
            frame = directory / f"frame_{index:06d}"
            if frame.exists():
                if layers:
                    read_layer_capture(frame, read_manifest(output / group / frame.name), locked["sample_layers"], pixels=False)
                else:
                    verify_frame(frame, locked, index)
            else:
                _recover(output, frame.with_name(frame.name + ".partial"), "sample_layers" if layers else "base", group)
                missing.append(index)
        runtime_path = output / "distributed/runtime_configs" / ("layers-" if layers else "") / (group + ".json")
        write_json(runtime_path, actual)
        log = output / "logs" / logfile
        log.parent.mkdir(parents=True, exist_ok=True)
        for first, end in missing_ranges(missing):
            command = [str(renderer), str(runtime_path.resolve()), str(directory.resolve()),
                       "--start-frame", str(first), "--end-frame", str(end)]
            with log.open("a") as stream:
                stream.write(json.dumps({"command": command, "started_unix": time.time(), "distributed_clip": group}) + "\n")
                stream.flush()
                subprocess.run(command, cwd=workspace / "Wavefront-Path-Tracer", stdout=stream, stderr=subprocess.STDOUT, check=True)
        for index in range(8):
            frame = directory / f"frame_{index:06d}"
            if layers:
                read_layer_capture(frame, read_manifest(output / group / frame.name), locked["sample_layers"], pixels=True)
            else:
                verify_frame(frame, locked, index, pixels=True)


def clip_inventory(plan, group, output):
    """Validate complete frames and build an allowlist of this task's files."""
    output = Path(output)
    task = plan["tasks"][group]
    files = []
    for index in range(8):
        relative = Path(group) / f"frame_{index:06d}"
        base = verify_frame(output / relative, task["config"], index)
        files += [relative / "frame.exr", relative / "manifest.json"]
        if task["layers"]:
            layer_config = json.loads((output / "sample_layer_configs" / (group + ".json")).read_text())
            read_layer_capture(output / "sample_layers" / relative, base, layer_config["sample_layers"], pixels=False)
            files += [Path("sample_layers") / relative / name for name in ("layers.exr", "manifest.json")]
        if task["decoded"]:
            verify_decoded_frame(output, group, index)
            files += [Path("decoded") / relative / name for name in ("float.npy", "uint.npy", "manifest.json")]
    for name in (group + ".log", "layers-" + group + ".log"):
        if (output / "logs" / name).exists():
            files.append(Path("logs") / name)
    inventory, stamps = {}, {}
    for relative in files:
        path = output / relative
        before = path.stat()
        inventory[relative.as_posix()] = file_hashes(path)
        after = path.stat()
        stamp = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns)
        if stamp(before) != stamp(after):
            raise ValueError("Clip file changed during upload preparation")
        stamps[relative.as_posix()] = stamp(after)
    return inventory, stamps


def publish_clip(plan, claim, output, coordinator, archiver):
    output = Path(output)
    group = claim["group"]
    inventory, stamps = clip_inventory(plan, group, output)
    coordinator.phase(claim, "uploading")
    for name, hashes in inventory.items():
        if not belongs(name, group) and name not in (f"logs/{group}.log", f"logs/layers-{group}.log"):
            raise ValueError("Clip upload attempted to write outside its task")
        coordinator.owned(claim)
        coordinator.store.upload(output / name, name, hashes)
    coordinator.complete(claim, inventory)
    # Publish the deterministic scene receipt before cleaning the final clip.
    coordinator.finalize_scene(plan["tasks"][group]["scene"], destination=archiver.destination,
                               endpoint=archiver.native.endpoint)
    write_json(output / "distributed/receipts" / (group + ".json"), {
        "status": "verified", "group": group, "claim_token": claim["token"], "files": inventory})
    payloads = [name for name in inventory if Path(name).suffix in (".exr", ".npy")]
    for name in payloads:
        stat = (output / name).stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != stamps[name]:
            raise ValueError("Clip payload changed during upload; keeping local payloads")
    for name in payloads:
        (output / name).unlink()
    print(f"Distributed clip verified/uploaded: {group} ({len(payloads)} local payloads released)", flush=True)


def finish_clip(plan, claim, output, destination):
    archiver, store = connection(output, destination)
    coordinator = Coordinator(store, plan)
    coordinator.owned(claim)
    if plan["tasks"][claim["group"]]["decoded"]:
        coordinator.phase(claim, "decoding")
        _decode_clip(output, claim["group"], 2)
    publish_clip(plan, claim, output, coordinator, archiver)


def run_worker(plan, output, workspace, renderer, destination, worker, *, max_clips=None):
    output, workspace, renderer = map(lambda p: Path(p).resolve(), (output, workspace, renderer))
    archiver, store = connection(output, destination)
    coordinator = Coordinator(store, plan)
    coordinator.require_ready()
    if digest(output / "capture_lock.json") != plan["capture_lock_sha256"]:
        raise ValueError("Local dataset is not the shared locked v4 dataset")
    with capture_lock(output), terminate_subprocesses_on_sigterm():
        machine_file = output / "distributed/machine.json"
        if not machine_file.exists():
            write_json(machine_file, {"machine_id": uuid.uuid4().hex})
        machine = json.loads(machine_file.read_text())["machine_id"]
        pending = []
        captured = 0
        # Decode and upload the previous clip while the main process renders.
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
            while max_clips is None or captured < max_clips:
                for group, future in list(pending):
                    if future.done():
                        future.result()
                        pending.remove((group, future))
                if len(pending) >= 2:
                    pending.pop(0)[1].result()
                claim = coordinator.claim(worker, machine, exclude={g for g, _ in pending})
                if claim is None:
                    break
                group = claim["group"]
                task = plan["tasks"][group]
                for name, expected in task.get("local_manifests", {}).items():
                    if digest(output / name) != expected:
                        raise ValueError("Primary worker must use its original local dataset/progress")
                print(f"Distributed GPU task: {group} (8 frames), worker={worker}", flush=True)
                capture_clip(plan, claim, output, workspace, renderer)
                future = executor.submit(finish_clip, plan, claim, str(output), destination)
                pending.append((group, future))
                captured += 1
            for _, future in pending:
                future.result()
        # This also repairs a scene-receipt interruption after all clips committed.
        for scene in {t["scene"] for t in plan["tasks"].values()}:
            coordinator.finalize_scene(scene, destination=archiver.destination, endpoint=archiver.native.endpoint)
        return {"processed_clips": captured, **coordinator.status()}
