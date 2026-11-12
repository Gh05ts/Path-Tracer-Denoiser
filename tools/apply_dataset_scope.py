#!/usr/bin/env python3
"""Apply a camera/scope revision, preserving captured frames and lock ancestry.

Requires an idle dataset. Validates every retained local EXR before replacing
sidecars, preserves original lock/configs/split, and never deletes images/logs.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from wavefront_denoiser.capture_history import compatible_receipt_lock, revision_ancestors
from wavefront_denoiser.dataset_generation import (
    asset_files, build_configs, capture_lock, digest, prepare, read_plan, verify_frame, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    args = parser.parse_args()
    output, plan_path = args.output.resolve(), args.plan.resolve()
    configs, plan = build_configs(plan_path), read_plan(plan_path)
    report = json.loads(args.selection_report.read_text())
    if report["plan_sha256"] != digest(plan_path) or report["renderer_sha256"] != digest(args.renderer):
        raise ValueError("Candidate selection report does not match plan/renderer")
    with capture_lock(output):
        old_path = output/"capture_lock.json"
        old = json.loads(old_path.read_text())
        revision_ancestors(output, old)
        if old["renderer_sha256"] != digest(args.renderer):
            raise ValueError("Scope changes cannot replace the renderer")
        assets = set().union(*(asset_files(Path(c["scene_file"])) for c in configs.values()))
        if {str(p): digest(p) for p in assets} != old["assets"]:
            raise ValueError("Scope changes cannot replace assets")
        preserved, checked = set(), []
        for manifest in sorted(output.glob("*/frame_*/manifest.json")):
            group = manifest.parent.parent.name
            if group not in configs or configs[group] != old["configs"].get(group):
                raise ValueError(f"Scope change would drop or change captured group: {group}")
            preserved.add(group)
            image = manifest.parent/"frame.exr"
            if image.exists():
                metadata = verify_frame(manifest.parent, configs[group], int(manifest.parent.name[6:]))
                checked.append({"group": group, "frame": metadata["frame_index"],
                                "image_sha256": metadata["image_sha256"]})
        reused = []
        for scene in plan["scenes"]:
            name = scene["id"]
            current = {g: c for g, c in configs.items() if c["scene"] == name}
            previous = {g: c for g, c in old["configs"].items() if c["scene"] == name}
            if current == previous:
                reused.append(name)
        for receipt_path in sorted((output/"r2-archives").glob("*.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt["scene"] not in reused or not compatible_receipt_lock(
                    output, receipt["scene"], receipt["capture_lock_sha256"], digest(old_path)):
                raise ValueError(f"Existing archive would lose provenance: {receipt_path}")
            # Verify retained manifest hashes against the original archive receipt.
            for relative, hashes in receipt["files"].items():
                if relative.endswith("/manifest.json") and digest(output/relative) != hashes["sha256"]:
                    raise ValueError(f"Archived manifest changed: {relative}")
        history = output/"capture-history"
        parent_hash = digest(old_path)
        backup = history/f"scope-before-{parent_hash}"
        backup.mkdir(parents=True, exist_ok=False)
        for name in ("capture_lock.json", "split.json", "plan-summary.json", "reference-check-plan.json"):
            shutil.copy2(output/name, backup/name)
        shutil.copytree(output/"capture_configs", backup/"capture_configs")
        shutil.copy2(old_path, history/f"{parent_hash}.json")
        evidence = history/f"scope-validation-{parent_hash}.json"
        write_json(evidence, {"local_frames_verified": checked, "preserved_groups": sorted(preserved),
                              "archive_receipts_verified": [p.name for p in (output/"r2-archives").glob("*.json")],
                              "selection": report})
        relative = evidence.relative_to(output).as_posix()
        new = deepcopy(old)
        new.update(plan=plan, configs=configs, revision={
            "kind": "scope_reduction", "parent_lock": f"capture-history/{parent_hash}.json",
            "parent_sha256": parent_hash, "reused_scenes": reused, "reset_scenes": [],
            "preserved_groups": sorted(preserved),
            "validation": {s: {"path": relative, "sha256": digest(evidence)} for s in reused}})
        revision_ancestors(output, new)
        # Sidecars are recoverable from the immutable backup if interrupted here.
        write_json(old_path, new)
        for group, config in configs.items():
            write_json(output/"capture_configs"/f"{group}.json", config)
        from wavefront_denoiser.dataset_generation import split_spec
        write_json(output/"split.json", split_spec(plan, configs, output))
        prepare(plan_path, output, configs, args.renderer.resolve())
        print(json.dumps({"clips": len(configs), "local_frames_verified": len(checked),
                          "preserved_groups": len(preserved), "backup": str(backup)}, indent=2))


if __name__ == "__main__":
    main()
