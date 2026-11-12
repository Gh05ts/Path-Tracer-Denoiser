"""Freeze the existing v4 capture contract and preserve machine-local progress."""

from pathlib import Path, PurePosixPath
import json
import re
import shutil
import tarfile
import tempfile

from ..b2_archive import file_hashes
from ..capture_history import compatible_receipt_lock
from ..dataset_generation import capture_lock, digest, write_json
from ..decoded_cache import read_cache_lock
from ..frames import read_manifest


ROOT_FILES = ("capture_lock.json", "sample_layers_lock.json", "decoded_cache_lock.json", "split.json",
              "train.json", "train-joint-decoded.json", "reference-check-plan.json",
              "reference_policy.json", "reference_convergence.json", "SAMPLE_LAYERS_BACKFILL.md",
              "DECODED_BACKFILL.md")


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
        raise ValueError("Use a nonempty identifier containing letters, digits, underscores or hyphens")
    return value


def prefix(run_id):
    return "_distributed/" + identifier(run_id)


def belongs(relative, group):
    path = PurePosixPath(relative)
    parts = path.parts
    return ((len(parts) >= 3 and parts[0] == group and parts[1].startswith("frame_"))
            or (len(parts) >= 4 and parts[0] in ("sample_layers", "decoded") and parts[1] == group))


def _prepare(output, workspace, prepared, *, primary_worker, run_id="denoiser-v4"):
    output, workspace, prepared = map(lambda p: Path(p).resolve(), (output, workspace, prepared))
    identifier(primary_worker)
    identifier(run_id)
    with capture_lock(output):
        capture = json.loads((output / "capture_lock.json").read_text())
        layers = json.loads((output / "sample_layers_lock.json").read_text())
        decoded = read_cache_lock(output)
        if capture.get("format") != "wavefront-capture-lock" or capture.get("version") != 1:
            raise ValueError("Distributed capture requires the existing locked v4 dataset")
        if layers["base_capture_lock_sha256"] != digest(output / "capture_lock.json"):
            raise ValueError("Sample layer provenance disagrees with v4")
        if capture["renderer_sha256"] != layers["renderer_sha256"]:
            raise ValueError("Distributed workers require one matching base/layer renderer binary")
        if layers["spps"] != [1, 2, 8, 16, 32]:
            raise ValueError("Distributed v4 requires all five companion layers")
        assets = {}
        for path, sha in capture["assets"].items():
            assets[Path(path).relative_to(workspace).as_posix()] = sha
        archived = {}
        for path in sorted((output / "r2-archives").glob("*.json")):
            receipt = json.loads(path.read_text())
            scene = path.stem
            if (receipt.get("format") != "wavefront-r2-scene-archive" or receipt.get("version") != 1
                    or receipt.get("status") != "verified" or receipt.get("scene") != scene
                    or not compatible_receipt_lock(output, scene, receipt.get("capture_lock_sha256"),
                                                   digest(output / "capture_lock.json"))):
                raise ValueError(f"Untrusted existing scene receipt: {scene}")
            archived[scene] = receipt
        tasks = {}
        for group, config in capture["configs"].items():
            identifier(group)
            if len(config["frames"]) != 8:
                raise ValueError("A distributed task must contain the entire eight-frame clip")
            scene = config["scene"]
            local = any((output / group).glob("frame_*")) if (output / group).exists() else False
            seed = None
            if scene in archived:
                files = {n: h for n, h in archived[scene]["files"].items() if belongs(n, group)}
                for i in range(8):
                    for filename in ("frame.exr", "manifest.json"):
                        if f"{group}/frame_{i:06d}/{filename}" not in files:
                            raise ValueError(f"Archived clip is missing a source frame: {group}")
                seed = {"status": "complete", "source": "existing_scene_archive", "files": files}
            else:
                for path in (output / group).glob("frame_*/manifest.json"):
                    meta = read_manifest(path)
                    if meta["sequence_id"] != group:
                        raise ValueError("Existing clip identity differs from its locked config")
            tasks[group] = {"scene": scene, "config": config, "frame_count": 8,
                            "layers": group in layers["groups"], "decoded": group in decoded["groups"],
                            "affinity": primary_worker if local and seed is None else None,
                            "local_manifests": {p.relative_to(output).as_posix(): digest(p)
                                                for p in sorted((output / group).glob("frame_*/manifest.json"))}
                                               if local and seed is None else {},
                            "seed": seed}
        names = [name for name in ROOT_FILES if (output / name).is_file()]
        for directory in ("capture_configs", "sample_layer_configs", "capture-history"):
            names += [p.relative_to(output).as_posix() for p in sorted((output / directory).rglob("*"))
                      if p.is_file() and p.suffix in (".json", ".md", ".log")]
        prepared.mkdir(parents=True, exist_ok=True)
        metadata = prepared / "metadata"
        metadata.mkdir(exist_ok=True)
        for name in sorted(set(names)):
            source = output / name
            if source.is_symlink():
                raise ValueError("Metadata bundle must contain regular files")
            target = metadata / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files = {name: file_hashes(metadata / name) for name in sorted(set(names))}
        archive = prepared / "metadata.tar"
        def portable_header(info):
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            return info
        with tarfile.open(archive, "w") as bundle:
            for name in files:
                bundle.add(metadata / name, arcname=name, recursive=False, filter=portable_header)
        plan = {"format": "wavefront-distributed-clips", "version": 1, "run_id": run_id,
                "dataset_id": capture["plan"]["dataset_id"], "source_workspace": str(workspace),
                "primary_worker": primary_worker, "capture_lock_sha256": digest(output / "capture_lock.json"),
                "task_order": list(tasks),
                "renderer_sha256": capture["renderer_sha256"], "assets": assets, "tasks": tasks,
                "metadata_files": files, "bundle": file_hashes(archive), "archived_scenes": archived}
        existing = prepared / "plan.json"
        if existing.exists() and json.loads(existing.read_text()) != plan:
            raise ValueError("Prepared distributed plan changed; use a new preparation directory")
        write_json(existing, plan)
        return plan


def prepare(output, workspace, prepared, *, primary_worker, run_id="denoiser-v4"):
    prepared = Path(prepared).resolve()
    prepared.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="distributed-prepare-", dir=prepared.parent) as name:
        staging = Path(name)
        plan = _prepare(output, workspace, staging, primary_worker=primary_worker, run_id=run_id)
        if (prepared / "plan.json").exists():
            if json.loads((prepared / "plan.json").read_text()) != plan:
                raise ValueError("Prepared distributed plan changed; use a new preparation directory")
            return plan
        if prepared.exists() and any(prepared.iterdir()):
            raise ValueError("Preparation directory must be empty")
        shutil.copytree(staging, prepared, dirs_exist_ok=True)
        return plan


def install_bundle(archive, output, plan):
    """Install only the frozen metadata allowlist, without adopting other captures."""
    output = Path(output)
    active = output / "distributed/active.json"
    binding = {"run_id": plan["run_id"], "capture_lock_sha256": plan["capture_lock_sha256"]}
    if active.exists() and json.loads(active.read_text()) != binding:
        raise ValueError("Local dataset is already managed by a different distributed queue")
    allowed = plan["metadata_files"]
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        if len(members) != len(allowed) or {m.name for m in members} != set(allowed):
            raise ValueError("Distributed metadata bundle inventory mismatch")
        contents = {}
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or member.size != allowed[member.name]["size"]):
                raise ValueError("Unsafe distributed metadata bundle entry")
            target = output / member.name
            data = bundle.extractfile(member).read()
            import hashlib
            if hashlib.sha256(data).hexdigest() != allowed[member.name]["sha256"]:
                raise ValueError("Distributed metadata checksum mismatch")
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Local metadata differs from the shared v4 dataset: {member.name}")
            contents[member.name] = data
        for name, data in contents.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_name(target.name + ".partial")
                temporary.write_bytes(data)
                temporary.replace(target)
    write_json(active, binding)
