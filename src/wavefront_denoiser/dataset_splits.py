"""Explicit whole-scene partitions and metadata-only leakage audits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from .frames import read_manifest


PARTITIONS = ("train", "validation", "test")


def json_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict) or spec.get("format") != "wavefront-dataset-split" or spec.get("version") != 1:
        raise ValueError("Unsupported dataset split manifest")
    if not isinstance(spec.get("data_root"), str) or not spec["data_root"]:
        raise ValueError("Split manifest needs data_root")
    if "declared_reference_samples" in spec and (type(spec["declared_reference_samples"]) is not int
                                                  or spec["declared_reference_samples"] < 1):
        raise ValueError("Declared reference samples must be a positive integer")
    groups = spec.get("groups")
    partitions = spec.get("partitions")
    if not isinstance(groups, dict) or not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
        raise ValueError("Split manifest needs groups and train/validation/test partitions")
    membership, families = {}, {}
    for partition in PARTITIONS:
        ids = partitions[partition]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{partition} must contain explicit capture IDs")
        for group in ids:
            if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", group) or group not in groups or group in membership:
                raise ValueError("Capture groups must exist and belong to exactly one partition")
            entry = groups[group]
            if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) and entry[key]
                                                    for key in ("scene", "family", "capture_config")):
                raise ValueError(f"Invalid group declaration: {group}")
            if "frame_count" in entry and (type(entry["frame_count"]) is not int or entry["frame_count"] < 1):
                raise ValueError("Declared frame_count must be a positive integer")
            if entry["family"] in families and families[entry["family"]] != partition:
                raise ValueError(f"Scene family crosses partitions: {entry['family']}")
            families[entry["family"]] = partition
            membership[group] = partition
    if membership.keys() != groups.keys():
        raise ValueError("Every declared group must belong to a partition")
    # Labels alone cannot permit a known scene to appear on both sides.
    scenes = {}
    for group, entry in groups.items():
        partition = membership[group]
        if entry["scene"] in scenes and scenes[entry["scene"]] != partition:
            raise ValueError(f"Scene crosses partitions: {entry['scene']}")
        scenes[entry["scene"]] = partition
    spec = dict(spec)
    spec["data_root"] = str((path.parent / spec["data_root"]).resolve())
    spec["manifest_path"] = str(path)
    spec["manifest_sha256"] = json_digest(path)
    spec["membership"] = membership
    return spec


def audit_dataset_split(path: str | Path, *, allow_missing_test: bool = False) -> dict:
    """Inspect JSON metadata only; test EXR pixel data is never opened here."""
    spec = read_split_manifest(path)
    root = Path(spec["data_root"])
    found = defaultdict(list)
    for manifest in sorted(root.rglob("manifest.json")):
        if any(part.endswith(".partial") for part in manifest.parts):
            continue
        metadata = read_manifest(manifest)
        group = metadata["sequence_id"]
        if group not in spec["groups"]:
            raise ValueError(f"Unregistered capture group: {group}")
        found[group].append((manifest, metadata))
    fingerprints, image_digests, records = {}, {}, {}
    minimum_reference = None
    policy_path = root / "reference_policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if policy.get("split_manifest_sha256") != spec["manifest_sha256"]:
            raise ValueError("Reference policy belongs to a different dataset split")
        minimum_reference = policy.get("reference_samples")
        if type(minimum_reference) is not int or minimum_reference < 1:
            raise ValueError("Invalid reference policy sample count")
        if json_digest(root / "reference_convergence.json") != policy.get("convergence_report_sha256"):
            raise ValueError("Reference convergence report checksum mismatch")
        evidence = json.loads((root / "reference_convergence.json").read_text())
        if evidence.get("selected_reference_samples") != minimum_reference or evidence.get("split_manifest_sha256") != spec["manifest_sha256"]:
            raise ValueError("Reference policy disagrees with convergence evidence")
    elif spec.get("require_reference_policy", False):
        raise ValueError("This dataset split requires qualified reference convergence evidence")
    for group, entry in spec["groups"].items():
        partition = spec["membership"][group]
        frames = found[group]
        if not frames:
            if not (allow_missing_test and partition == "test"):
                raise ValueError(f"Missing dataset group: {group}")
            records[group] = {"partition": partition, "scene": entry["scene"], "family": entry["family"],
                              "frames": 0, "status": "reserved"}
            continue
        identities, frame_indices, inventory = set(), set(), []
        for manifest, metadata in frames:
            if not (manifest.parent / metadata["image"]).is_file():
                raise ValueError(f"Missing image file for {manifest}")
            if metadata["scene"] != entry["scene"]:
                raise ValueError(f"Capture scene disagrees with split declaration: {group}")
            identity = metadata.get("scene_fingerprint")
            if not isinstance(identity, str) or not identity or any(c not in "0123456789abcdef" for c in identity):
                raise ValueError("Strict splits require scene_fingerprint; recapture with the updated renderer")
            identities.add(identity)
            if metadata["frame_index"] in frame_indices:
                raise ValueError(f"Duplicate frame index in {group}")
            frame_indices.add(metadata["frame_index"])
            for value, seen, label in ((identity, fingerprints, "Scene identity"),
                                       (metadata["image_sha256"], image_digests, "Image content")):
                if value in seen and seen[value] != partition:
                    raise ValueError(f"{label} crosses dataset partitions")
                seen[value] = partition
            if minimum_reference and metadata["render"]["reference_samples"] < minimum_reference:
                raise ValueError(f"Reference budget below the qualified policy: {manifest}")
            if metadata["render"]["reference_samples"] < spec.get("declared_reference_samples", 1):
                raise ValueError(f"Reference budget below the declared dataset budget: {manifest}")
            measured = next((row.get("render_settings") for row in evidence["cases"] if row["group"] == group), None) if minimum_reference else None
            if measured:
                ignored = {"tile_width", "tile_height", "persistent_wavefront", "sample_stride"}
                if any(metadata["render"].get(key) != value for key, value in measured.items() if key not in ignored):
                    raise ValueError(f"Integrator settings differ from reference calibration: {group}")
            inventory.append({"manifest": str(manifest.relative_to(root)), "manifest_sha256": json_digest(manifest),
                              "image_sha256": metadata["image_sha256"]})
        if len(identities) != 1:
            raise ValueError(f"Scene settings changed within group {group}")
        if "frame_count" in entry and frame_indices != set(range(entry["frame_count"])):
            raise ValueError(f"Incomplete frame indices in group {group}")
        records[group] = {"partition": partition, "scene": entry["scene"], "family": entry["family"],
                          "scene_fingerprint": next(iter(identities)), "frames": len(frames),
                          "status": "captured", "inventory": inventory}
    return {"format": "wavefront-split-audit", "version": 1,
            "split_manifest_sha256": spec["manifest_sha256"], "data_root": str(root),
            "partitions": spec["partitions"], "groups": records,
            "reference_samples": minimum_reference,
            **({"declared_reference_samples": spec["declared_reference_samples"]}
               if "declared_reference_samples" in spec else {}),
            "test_pixels_loaded": False,
            "status": "test_reserved" if any(row["status"] == "reserved" for row in records.values()) else "ready"}


def capture_independent_dataset(split_path: str | Path, convergence_path: str | Path,
                                executable: str | Path, renderer_root: str | Path | None = None) -> dict:
    spec = read_split_manifest(split_path)
    evidence = json.loads(Path(convergence_path).read_text())
    if evidence.get("format") != "wavefront-reference-convergence" or evidence.get("version") != 1:
        raise ValueError("Unsupported convergence evidence")
    budget = evidence.get("selected_reference_samples")
    if evidence.get("split_manifest_sha256") != spec["manifest_sha256"] or type(budget) is not int or budget < 1:
        raise ValueError("Need a qualifying convergence report for this split")
    # The policy must have been measured without opening the test scene.
    cases = evidence.get("cases", [])
    required = set(spec["partitions"]["train"] + spec["partitions"]["validation"])
    if not cases or {row.get("group") for row in cases} != required or any(
            row.get("partition") != spec["membership"].get(row.get("group")) or row.get("partition") == "test"
            for row in cases):
        raise ValueError("Convergence calibration must exclude the test partition")
    if evidence.get("status") != "qualified_for_calibration_views" or evidence.get("scope") != "camera_sampling_at_fixed_integrator":
        raise ValueError("Reference convergence evidence is not qualified for this workflow")
    root = Path(spec["data_root"])
    exe = Path(executable).resolve()
    cwd = Path(renderer_root).resolve() if renderer_root else exe.parent.parent
    configs = {}
    for group, entry in spec["groups"].items():
        config_path = Path(spec["manifest_path"]).parent / entry["capture_config"]
        config = json.loads(config_path.read_text())
        if config.get("sequence_id") != group or config.get("scene") != entry["scene"]:
            raise ValueError(f"Capture configuration disagrees with split: {group}")
        measured = next((row.get("render_settings", {}) for row in cases if row["group"] == group), {})
        for config_key, render_key in (("max_depth", "max_depth"), ("russian_roulette_start_depth", "russian_roulette_start_depth"),
                                       ("caustics", "caustics"), ("caustic_photons", "caustic_photons"), ("caustic_radius", "caustic_radius")):
            if config_key in config and render_key in measured:
                actual, expected = config[config_key], measured[render_key]
                close_float = isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and (
                    isinstance(actual, float) or isinstance(expected, float)) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
                if actual != expected and not close_float:
                    raise ValueError(f"Capture integrator differs from calibration: {group}")
        config["reference_samples"] = budget
        configs[group] = config
    root.mkdir(parents=True, exist_ok=False)
    (root / "capture_configs").mkdir()
    (root / "reference_convergence.json").write_bytes(Path(convergence_path).read_bytes())
    policy = {"reference_samples": budget, "split_manifest_sha256": spec["manifest_sha256"],
              "convergence_report_sha256": json_digest(convergence_path),
              "scope": "camera_sampling_at_fixed_integrator"}
    (root / "reference_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    for partition in PARTITIONS:
        for group in spec["partitions"][partition]:
            config_path = root / "capture_configs" / f"{group}.json"
            config_path.write_text(json.dumps(configs[group], indent=2) + "\n")
            print(f"Capturing {partition}: {group}, {budget} reference SPP", flush=True)
            with (root / f"{group}.log").open("w") as log:
                subprocess.run([str(exe), str(config_path), str(root / group)], cwd=cwd,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = audit_dataset_split(split_path)
    (root / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
