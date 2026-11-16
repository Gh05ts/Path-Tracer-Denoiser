"""Deferred v4 layers and decoded caches, without recapturing source images."""

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import uuid

from ..adaptive_data import layer_layout, read_layer_capture
from ..b2_archive import file_hashes
from ..dataset_generation import (capture_lock, digest, missing_ranges,
                                  terminate_subprocesses_on_sigterm, verify_frame, write_json)
from ..decoded_cache import _decode_clip, read_cache_lock
from ..frames import read_manifest
from ..layer_compatibility import (BMW11_RENDERER, BMW12_RENDERER, BMW_VALIDATION,
                                   bmw_source_binding)
from .coordinator import Coordinator
from .plan import identifier, install_bundle, prefix
from .worker import _recover, clip_inventory, connection, validate_machine


def restore(store, output, name, expected):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe backfill source path")
    target = Path(output) / name
    if target.exists():
        if file_hashes(target) != expected:
            raise ValueError(f"Local backfill source differs from verified archive: {name}")
    else:
        store.download(name, target, expected)
    return target


def source_lock(output, scene, expected):
    """Follow checksum-bound reuse edges without resolving old absolute asset paths."""
    output = Path(output)
    path = output / "capture_lock.json"
    seen = set()
    while digest(path) != expected:
        lock = json.loads(path.read_text())
        revision = lock.get("revision", {})
        sha = revision.get("parent_sha256", "")
        name = revision.get("parent_lock")
        if (scene not in revision.get("reused_scenes", []) or scene in revision.get("reset_scenes", [])
                or len(sha) != 64 or name != f"capture-history/{sha}.json" or sha in seen or len(seen) >= 16):
            raise ValueError(f"Archive is not an authorized capture ancestor: {scene}")
        seen.add(sha)
        parent_path = output / name
        if digest(parent_path) != sha:
            raise ValueError("Capture ancestor checksum mismatch")
        parent = json.loads(parent_path.read_text())
        current_configs = {g: c for g, c in lock["configs"].items() if c["scene"] == scene}
        old_configs = {g: c for g, c in parent["configs"].items() if c["scene"] == scene}
        if not current_configs or current_configs != old_configs:
            raise ValueError("Capture ancestor changed source configs")
        evidence = revision.get("validation", {}).get(scene, {})
        evidence_path = PurePosixPath(evidence.get("path", ""))
        if (evidence_path.is_absolute() or ".." in evidence_path.parts
                or not evidence_path.parts or evidence_path.parts[0] != "capture-history"
                or digest(output / evidence_path) != evidence.get("sha256")):
            raise ValueError("Capture reuse validation is missing or changed")
        path = parent_path
    return json.loads(path.read_text())


def generation_ready(store, generation):
    coordinator = Coordinator(store, generation)
    coordinator.require_ready()
    if coordinator.status()["counts"] != {"pending": 0, "claimed": 0, "complete": len(generation["tasks"])}:
        raise ValueError("Finish all generation tasks before initializing backfills")


def prepare_plan(store, generation, output, *, run_id, destination, endpoint):
    """Freeze deferred membership and the verified source scene receipts."""
    identifier(run_id)
    generation_ready(store, generation)
    output = Path(output)
    capture = json.loads((output / "capture_lock.json").read_text())
    layers = json.loads((output / "sample_layers_lock.json").read_text())
    decoded = read_cache_lock(output)
    if (digest(output / "capture_lock.json") != generation["capture_lock_sha256"]
            or layers["base_capture_lock_sha256"] != generation["capture_lock_sha256"]
            or layers["spps"] != [1, 2, 8, 16, 32]
            or decoded["configs"] != capture["configs"]):
        raise ValueError("Backfill source locks disagree with the completed generation")
    deferred = [g for g in capture["configs"] if g not in layers["groups"] or g not in decoded["groups"]]
    if not deferred:
        raise ValueError("No deferred backfill clips in this dataset")
    receipts, tasks = {}, {}
    for group in deferred:
        identifier(group)
        config = capture["configs"][group]
        scene = identifier(config["scene"])
        if scene not in receipts:
            record = store.get(f"r2-archives/{scene}.json")
            receipt = record[0] if record else {}
            if (receipt.get("format") != "wavefront-r2-scene-archive" or receipt.get("version") != 1
                    or receipt.get("status") != "verified" or receipt.get("scene") != scene
                    or receipt.get("destination") != destination or receipt.get("r2_endpoint") != endpoint):
                raise ValueError(f"Missing or incompatible verified source scene receipt: {scene}")
            source_lock(output, scene, receipt.get("capture_lock_sha256"))
            receipts[scene] = receipt
        receipt = receipts[scene]
        original = source_lock(output, scene, receipt["capture_lock_sha256"])
        sources, integrators = {}, set()
        new_layers = group not in layers["groups"]
        for index in range(len(config["frames"])):
            frame = f"{group}/frame_{index:06d}"
            for name in (f"{frame}/manifest.json", f"{frame}/frame.exr"):
                if name not in receipt["files"]:
                    raise ValueError(f"Source archive is missing a required frame: {name}")
                sources[name] = receipt["files"][name]
            base_path = restore(store, output, f"{frame}/manifest.json", sources[f"{frame}/manifest.json"])
            base = read_manifest(base_path)
            if (base["sequence_id"] != group or base["frame_index"] != index
                    or base["image_sha256"] != sources[f"{frame}/frame.exr"]["sha256"]):
                raise ValueError("Source manifest disagrees with archive identity/checksum")
            version = base["render"].get("integrator_version")
            if type(version) is not int or version < 1:
                raise ValueError("Backfill source must declare its integrator_version")
            integrators.add(version)
            if not new_layers:
                for filename in ("manifest.json", "layers.exr"):
                    name = f"sample_layers/{frame}/{filename}"
                    if name not in receipt["files"]:
                        raise ValueError(f"Source archive is missing existing companion layers: {name}")
                    sources[name] = receipt["files"][name]
        if len(integrators) != 1:
            raise ValueError("A backfill clip mixes source integrators")
        source_version = integrators.pop()
        version, renderer_sha = source_version, original["renderer_sha256"]
        bridge = new_layers and source_version == 11
        if bridge:
            if (scene != "bmw" or renderer_sha != BMW11_RENDERER
                    or generation["renderer_sha256"] != BMW12_RENDERER
                    or digest(output / "capture-history/bmw-comparison.json") != BMW_VALIDATION):
                raise ValueError("Integrator-12 BMW backfill requires the original audited binary pair and comparison evidence")
            version, renderer_sha = 12, generation["renderer_sha256"]
        tasks[group] = {"scene": scene, "config": config, "frame_count": len(config["frames"]),
                        "layers": True, "decoded": True, "affinity": None,
                        "new_layers": new_layers, "sources": sources,
                        "integrator_version": version, "renderer_sha256": renderer_sha,
                        "source_integrator_version": source_version, "bmw_integrator_bridge": bridge,
                        "assets": {Path(n).relative_to(generation["source_workspace"]).as_posix(): h
                                   for n, h in original["assets"].items()},
                        "layer_config": dict(config, sample_layers=layer_layout(config, layers["reserved_reference_samples"]),
                                             reserved_reference_samples=layers["reserved_reference_samples"])}
    inventory = {n: h for receipt in receipts.values() for n, h in receipt["files"].items()}
    inventory.update(generation["metadata_files"])
    store.client.verify_files(inventory)
    return {"format": "wavefront-distributed-clips", "version": 1, "kind": "backfill",
            "run_id": run_id, "generation_run_id": generation["run_id"],
            "capture_lock_sha256": generation["capture_lock_sha256"],
            "source_workspace": generation["source_workspace"], "assets": generation["assets"],
            "metadata_files": generation["metadata_files"], "bundle": generation["bundle"],
            "task_order": list(tasks), "tasks": tasks, "source_receipts": receipts,
            "destination": destination, "endpoint": endpoint}


class BackfillCoordinator(Coordinator):
    def __init__(self, store, plan):
        super().__init__(store, plan)
        if plan.get("kind") != "backfill":
            raise ValueError("Expected a backfill plan")
        self.prefix = "_backfill/" + identifier(plan["run_id"])

    @property
    def binding(self):
        return {"run_id": self.plan["generation_run_id"],
                "capture_lock_sha256": self.plan["capture_lock_sha256"]}

    def initialize(self):
        active = self.store.get("_distributed/active.json")
        if active is None or active[0] != self.binding:
            raise ValueError("Generation queue binding changed")
        marker = {**self.binding, "backfill_run_id": self.plan["run_id"]}
        if not self.store.put("_backfill/active.json", marker):
            raise ValueError("Another backfill queue already manages this dataset")
        if not self.store.put(f"{self.prefix}/plan.json", self.plan):
            raise ValueError("Backfill plan changed; refusing to replace frozen sources")
        for group in self.plan["task_order"]:
            if self.store.get(self.key(group)) is None:
                state = {"group": group, "capture_lock_sha256": self.plan["capture_lock_sha256"], "status": "pending"}
                if not self.store.put(self.key(group), state):
                    raise ValueError("Backfill task initialization conflict")
        for scene, receipt in self.plan["source_receipts"].items():
            if not self.store.put(f"{self.prefix}/source-receipts/{scene}.json", receipt):
                raise ValueError("Frozen source receipt changed")
        if not self.store.put(f"{self.prefix}/ready.json", marker):
            raise ValueError("Backfill initialization conflict")

    def require_ready(self):
        active = self.store.get("_distributed/active.json")
        marker = {**self.binding, "backfill_run_id": self.plan["run_id"]}
        ready = self.store.get(f"{self.prefix}/ready.json")
        current = self.store.get("_backfill/active.json")
        if active is None or active[0] != self.binding or ready is None or ready[0] != marker or current is None or current[0] != marker:
            raise ValueError("Backfill queue is not initialized or its binding changed")

    def complete(self, claim, inventory):
        group = claim["group"]
        config_name = f"sample_layer_configs/{group}.json"
        required = {config_name}
        for index in range(self.plan["tasks"][group]["frame_count"]):
            frame = f"{group}/frame_{index:06d}"
            required.update(f"{frame}/{n}" for n in ("frame.exr", "manifest.json"))
            required.update(f"sample_layers/{frame}/{n}" for n in ("layers.exr", "manifest.json"))
            required.update(f"decoded/{frame}/{n}" for n in ("float.npy", "uint.npy", "manifest.json"))
        if set(inventory) != required:
            raise ValueError("Backfill completion must include the complete clip and its layer config")
        state, etag = self.owned(claim)
        self.store.verify(inventory)
        complete = dict(state, status="complete", phase="verified", files=inventory)
        if not self.store.put(self.key(group), complete, etag=etag) or self.store.get(self.key(group))[0] != complete:
            raise ValueError("Backfill completion readback failed; retaining local payloads")

    def finalize_scene(self, scene, *, destination, endpoint):
        if destination != self.plan["destination"] or endpoint != self.plan["endpoint"]:
            raise ValueError("Backfill archive destination changed")
        inventory = dict(self.plan["source_receipts"][scene]["files"])
        # Current locks supersede historical receipt metadata; source image and
        # manifest hashes remain identical. Frozen old receipts stay in history.
        inventory.update(self.plan["metadata_files"])
        for group, task in self.plan["tasks"].items():
            if task["scene"] != scene:
                continue
            record = self.store.get(self.key(group))
            if record is None or record[0]["status"] != "complete":
                return False
            for name, hashes in record[0]["files"].items():
                if name in inventory and inventory[name] != hashes:
                    raise ValueError("Backfill would replace an existing source object")
                inventory[name] = hashes
        self.store.client.verify_files(inventory)
        receipt = {"format": "wavefront-r2-scene-archive", "version": 1, "status": "verified",
                   "scene": scene, "destination": destination, "r2_endpoint": endpoint,
                   "capture_lock_sha256": self.plan["capture_lock_sha256"], "files": inventory,
                   "backfill_run_id": self.plan["run_id"],
                   "source_receipt": f"{self.prefix}/source-receipts/{scene}.json"}
        key = f"r2-archives/{scene}.json"
        old = self.store.get(key)
        if old is not None and old[0] == receipt:
            return True
        if old is None or old[0] != self.plan["source_receipts"][scene]:
            raise ValueError("Scene receipt changed outside this backfill; retained results")
        if not self.store.put(key, receipt, etag=old[1]) or self.store.get(key)[0] != receipt:
            raise ValueError("Scene receipt publication/readback failed; retained results")
        return True


def validate_backfill_machine(plan, workspace, renderers, *, scenes=None):
    selected = {g: t for g, t in plan["tasks"].items() if scenes is None or t["scene"] in scenes}
    if not selected:
        raise ValueError("No backfill clips match the requested scenes")
    for group, task in selected.items():
        if not task["new_layers"]:
            continue
        version = task["integrator_version"]
        renderer = renderers.get(version)
        if renderer is None:
            raise ValueError(f"Integrator {version} renderer is required for {group}; pass --renderer {version}=PATH")
        check = dict(plan, tasks={group: task}, renderer_sha256=task["renderer_sha256"], assets=task["assets"])
        validate_machine(check, workspace, renderer)
    return selected


def capture_layers(plan, group, output, workspace, renderer):
    task = plan["tasks"][group]
    config = task["layer_config"]
    directory = Path(output) / "sample_layers" / group
    runtime = deepcopy(config)
    runtime["scene_file"] = str(Path(workspace) / Path(config["scene_file"]).relative_to(plan["source_workspace"]))
    path = Path(output) / "backfill/runtime_configs" / (group + ".json")
    write_json(path, runtime)
    missing = []
    def verify_layer(index, *, pixels):
        frame = directory / f"frame_{index:06d}"
        base = read_manifest(Path(output) / group / frame.name)
        if task["bmw_integrator_bridge"]:
            metadata = json.loads((frame / "manifest.json").read_text())
            binding = bmw_source_binding(base)
            if "source_compatibility" not in metadata:
                # Keep the renderer's actual version and fingerprint. Add only
                # the audited source binding, after validating the candidate.
                candidate = dict(metadata, source_compatibility=binding)
                from ..layer_compatibility import compatible_bmw_layer
                if not compatible_bmw_layer(candidate, base):
                    raise ValueError("Integrator-12 layer fingerprint differs beyond the audited BMW version change")
                write_json(frame / "manifest.json", candidate)
        read_layer_capture(frame, base, config["sample_layers"], pixels=pixels)
    for index in range(task["frame_count"]):
        frame = directory / f"frame_{index:06d}"
        if frame.exists():
            verify_layer(index, pixels=False)
        else:
            _recover(output, frame.with_name(frame.name + ".partial"), "sample_layers", group)
            missing.append(index)
    log = Path(output) / "backfill/logs" / (group + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    for first, end in missing_ranges(missing):
        command = [str(Path(renderer).resolve()), str(path.resolve()), str(directory.resolve()),
                   "--start-frame", str(first), "--end-frame", str(end)]
        with log.open("a") as stream:
            stream.write(json.dumps({"command": command, "started_unix": time.time()}) + "\n")
            stream.flush()
            subprocess.run(command, cwd=Path(workspace) / "Wavefront-Path-Tracer", stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    for index in range(task["frame_count"]):
        verify_layer(index, pixels=True)


def process_clip(plan, claim, output, workspace, renderers, coordinator):
    """Keep disk use bounded to one clip, reuse validated work after interruption."""
    group = claim["group"]
    task = plan["tasks"][group]
    coordinator.phase(claim, "restoring")
    for name, hashes in task["sources"].items():
        coordinator.owned(claim)
        restore(coordinator.store, output, name, hashes)
    for index in range(task["frame_count"]):
        verify_frame(Path(output) / group / f"frame_{index:06d}", task["config"], index)
    config_path = Path(output) / "sample_layer_configs" / (group + ".json")
    if config_path.exists() and json.loads(config_path.read_text()) != task["layer_config"]:
        raise ValueError("Backfill layer config changed")
    # Keep identical config bytes on retries and when restoring an existing config.
    if not config_path.exists():
        write_json(config_path, task["layer_config"])
    if task["new_layers"]:
        coordinator.phase(claim, "rendering")
        capture_layers(plan, group, output, workspace, renderers[task["integrator_version"]])
    coordinator.phase(claim, "decoding")
    _decode_clip(output, group, 2)
    inventory, stamps = clip_inventory(plan, group, output)
    # Generation logs belong to the original archive, not this new task.
    inventory = {n: h for n, h in inventory.items() if not n.startswith("logs/")}
    config_name = config_path.relative_to(output).as_posix()
    inventory[config_name] = file_hashes(config_path)
    coordinator.phase(claim, "uploading")
    for name, hashes in inventory.items():
        coordinator.owned(claim)
        coordinator.store.upload(Path(output) / name, name, hashes)
    coordinator.complete(claim, inventory)
    coordinator.finalize_scene(task["scene"], destination=plan["destination"], endpoint=plan["endpoint"])
    write_json(Path(output) / "backfill/receipts" / (group + ".json"), {"status": "verified", "files": inventory})
    cleanup_payloads(output, inventory, stamps=stamps)
    print(f"Backfill verified/uploaded: {group}; local payloads released", flush=True)


def cleanup_payloads(output, inventory, *, stamps=None):
    output = Path(output)
    payloads = [n for n in inventory if Path(n).suffix in (".exr", ".npy") and (output / n).exists()]
    observed = {}
    for name in payloads:
        stat = (output / name).stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamps is not None and stamp != stamps[name]:
            raise ValueError("Backfill payload changed during upload; retained local files")
        if stamps is None and file_hashes(output / name) != inventory[name]:
            raise ValueError("Retained payload differs from verified backfill; keeping local files")
        observed[name] = stamp
    for name in payloads:
        stat = (output / name).stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != observed[name]:
            raise ValueError("Backfill payload changed before cleanup; retained local files")
    for name in payloads:
        (output / name).unlink()


def run_worker(plan, output, workspace, renderers, destination, worker, *, scenes=None, max_clips=None):
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    selected = validate_backfill_machine(plan, workspace, renderers, scenes=scenes)
    archiver, store = connection(output, destination)
    coordinator = BackfillCoordinator(store, plan)
    coordinator.require_ready()
    if digest(output / "capture_lock.json") != plan["capture_lock_sha256"]:
        raise ValueError("Local backfill capture lock changed")
    with capture_lock(output), terminate_subprocesses_on_sigterm():
        machine_path = output / "backfill/machine.json"
        if not machine_path.exists():
            write_json(machine_path, {"machine_id": uuid.uuid4().hex})
        machine = json.loads(machine_path.read_text())["machine_id"]
        processed = 0
        while max_clips is None or processed < max_clips:
            claim = coordinator.claim(worker, machine, exclude=set(plan["tasks"]) - set(selected))
            if claim is None:
                break
            print(f"Backfill task: {claim['group']}, worker={worker}", flush=True)
            process_clip(plan, claim, output, workspace, renderers, coordinator)
            processed += 1
        for scene in plan["source_receipts"]:
            if coordinator.finalize_scene(scene, destination=archiver.destination, endpoint=archiver.native.endpoint):
                write_json(output / "r2-archives" / (scene + ".json"), store.get(f"r2-archives/{scene}.json")[0])
        # Repair cleanup after a crash between the durable completion record,
        # scene receipt publication and local payload deletion.
        for group in selected:
            record = store.get(coordinator.key(group))[0]
            if record["status"] == "complete" and record.get("machine") == machine:
                store.verify(record["files"])
                cleanup_payloads(output, record["files"])
        return {"processed_clips": processed, **coordinator.status()}


def install_metadata(store, generation, output):
    output = Path(output)
    archive = output / "backfill/metadata.tar"
    if not archive.exists() or file_hashes(archive) != generation["bundle"]:
        store.download(f"{prefix(generation['run_id'])}/metadata.tar", archive, generation["bundle"])
    install_bundle(archive, output, generation)
