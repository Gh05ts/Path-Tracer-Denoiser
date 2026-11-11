"""Explicit capture revisions preserve archived scenes under their source lock."""
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision_parent(output: Path, lock: dict) -> str | None:
    revision = lock.get("revision")
    if revision is None:
        return None
    name = revision.get("parent_lock", "")
    digest = revision.get("parent_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or name != f"capture-history/{digest}.json":
        raise ValueError("Invalid capture revision parent")
    path = output / name
    if not path.is_file() or sha256(path) != digest:
        raise ValueError("Capture revision parent checksum mismatch")
    parent = json.loads(path.read_text())
    reused = revision.get("reused_scenes", [])
    reset = revision.get("reset_scenes", [])
    if not reused or set(reused) & set(reset):
        raise ValueError("Capture revision must distinguish reused and reset scenes")
    from .dataset_generation import asset_files
    for scene in reused:
        current = {g: c for g, c in lock["configs"].items() if c["scene"] == scene}
        original = {g: c for g, c in parent["configs"].items() if c["scene"] == scene}
        if not current or current != original:
            raise ValueError(f"Reused scene configs changed: {scene}")
        for scene_file in {c["scene_file"] for c in current.values()}:
            for asset in asset_files(Path(scene_file)):
                name = str(asset)
                if (name not in lock["assets"] or lock["assets"][name] != parent["assets"].get(name)
                        or sha256(asset) != lock["assets"][name]):
                    raise ValueError(f"Reused scene assets changed: {scene}")
        evidence = revision.get("validation", {}).get(scene, {})
        relative = evidence.get("path", "")
        if (not relative.startswith("capture-history/") or ".." in Path(relative).parts
                or not (output / relative).is_file()
                or sha256(output / relative) != evidence.get("sha256")):
            raise ValueError(f"Missing or changed reuse validation: {scene}")
    return revision["parent_lock"]


def compatible_receipt_lock(output: Path, scene: str, receipt_lock: str, current_lock: str) -> bool:
    if receipt_lock == current_lock:
        return True
    lock = json.loads((output / "capture_lock.json").read_text())
    if revision_parent(output, lock) is None:
        return False
    revision = lock["revision"]
    return scene in revision["reused_scenes"] and receipt_lock == revision["parent_sha256"]
