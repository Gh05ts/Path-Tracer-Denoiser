"""Capture independent SPP layers after each base clip, before scene archiving."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from .adaptive_data import layer_layout, read_layer_capture
from .capture_history import compatible_receipt_lock
from .dataset_generation import digest, missing_ranges, write_json
from .frames import read_manifest


LOCK_NAME = "sample_layers_lock.json"


def required_layer_pairs(output, configs):
    """The archive must include every required companion image and manifest."""
    path = output/LOCK_NAME
    if not path.exists():
        return []
    lock = json.loads(path.read_text())
    if lock["base_capture_lock_sha256"] != digest(output/"capture_lock.json"):
        raise ValueError("Sample-layer binding disagrees with the base capture lock")
    return [(output/"sample_layers"/g/f"frame_{i:06d}"/"layers.exr",
             output/"sample_layers"/g/f"frame_{i:06d}"/"manifest.json")
            for g, c in configs.items() if g in lock["groups"] for i in range(len(c["frames"]))]


class InlineSampleLayers:
    def __init__(self, output, configs, renderer, renderer_root, *, spps, after_group,
                 reserved_reference_samples):
        self.output = Path(output)
        self.renderer, self.renderer_root = Path(renderer).resolve(), Path(renderer_root).resolve()
        if after_group not in configs:
            raise ValueError("Unknown clip boundary for sample layers")
        ordered = list(configs)
        boundary = ordered.index(after_group)+1
        self.groups = ordered[boundary:]
        if not self.groups:
            raise ValueError("No clips remain after the sample-layer boundary")
        self.configs = {}
        for group in self.groups:
            config = deepcopy(configs[group])
            config.update(sample_layers=layer_layout(config, reserved_reference_samples, spps=spps),
                          reserved_reference_samples=reserved_reference_samples)
            self.configs[group] = config
        lock = {"format": "wavefront-inline-sample-layers", "version": 1,
                "base_capture_lock_sha256": digest(self.output/"capture_lock.json"),
                "renderer": str(self.renderer), "renderer_sha256": digest(self.renderer),
                "spps": list(spps), "after_group": after_group,
                "reserved_reference_samples": reserved_reference_samples,
                "groups": self.groups, "configs": self.configs}
        path = self.output/LOCK_NAME
        if path.exists():
            if json.loads(path.read_text()) != lock:
                raise ValueError("Sample-layer settings, renderer or base lock changed")
        elif (self.output/"sample_layers").exists():
            raise ValueError("Refusing to adopt sample layers without a provenance lock")
        else:
            write_json(path, lock)
        for group, config in self.configs.items():
            path = self.output/"sample_layer_configs"/(group+".json")
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError(f"Stored sample-layer config changed: {group}")
            write_json(path, config)
        # This is a fixed backfill list, including the unfinished frames of the
        # boundary clip. It deliberately never expands on a later restart.
        rows = []
        for group in ordered[:boundary]:
            count = len(configs[group]["frames"])
            completed = sum((self.output/group/f"frame_{i:06d}"/"manifest.json").exists() for i in range(count))
            rows.append(f"| `{group}` | 0–{count-1} | {completed}/{count} |")
        missing_count = sum(len(configs[g]["frames"]) for g in ordered[:boundary])
        text = ("# Frames awaiting additional SPP layers\n\n"
                f"The {', '.join(map(str, spps))}-SPP layers begin **after** `{after_group}`. "
                "Its remaining base frames finish at the original 4+1024-SPP settings. "
                "The existing 4-SPP input, references, guides and sample IDs are retained.\n\n"
                f"The following **{missing_count} frames in {boundary} clips** intentionally have no "
                "additional layers in this run. Backfill them separately later, using their stored "
                "capture configs and original renderer/integrator provenance. BMW's integrator-11 "
                "source captures require a matching integrator-11 layer renderer; Bistro uses integrator 12.\n\n"
                "New layers are independent linear HDR RGB means in "
                "`sample_layers/GROUP/frame_NNNNNN/layers.exr`, with channels "
                "`spp_N.R/G/B`. Their manifest records exact sample ranges. They reuse the base frame's "
                "camera/guides/reference rather than duplicating those buffers.\n\n"
                "| Clip | Frame indices needing backfill | Base manifests at enablement |\n"
                "| --- | --- | ---: |\n" + "\n".join(rows) + "\n")
        missing_path = self.output/"SAMPLE_LAYERS_BACKFILL.md"
        if not missing_path.exists():
            missing_path.write_text(text)
        self.captured = self.skipped = self.archived = 0

    def _archived(self, scene, image, manifest):
        for provider in ("r2", "b2"):
            path = self.output/(provider+"-archives")/(scene+".json")
            if not path.is_file():
                continue
            receipt = json.loads(path.read_text())
            if (receipt.get("status") != "verified"
                    or not compatible_receipt_lock(self.output, scene, receipt.get("capture_lock_sha256"),
                                                   digest(self.output/"capture_lock.json"))):
                continue
            files = receipt.get("files", {})
            relative_image = image.relative_to(self.output).as_posix()
            relative_manifest = manifest.relative_to(self.output).as_posix()
            if (relative_image in files and relative_manifest in files
                    and digest(manifest) == files[relative_manifest]["sha256"]
                    and json.loads(manifest.read_text())["image_sha256"] == files[relative_image]["sha256"]):
                return True
        return False

    def capture_group(self, group, *, first_frame=0, end_frame=None):
        if group not in self.configs:
            return
        config = self.configs[group]
        end_frame = end_frame or len(config["frames"])
        missing = []
        for i in range(first_frame, end_frame):
            base = read_manifest(self.output/group/f"frame_{i:06d}")
            directory = self.output/"sample_layers"/group/f"frame_{i:06d}"
            image, manifest = directory/"layers.exr", directory/"manifest.json"
            if manifest.exists():
                archived = not image.exists() and self._archived(config["scene"], image, manifest)
                read_layer_capture(directory, base, config["sample_layers"], pixels=False, verify_digest=not archived)
                self.skipped += 1
                self.archived += int(archived)
            elif directory.exists():
                raise ValueError(f"Incomplete or corrupt published sample layers: {directory}")
            else:
                missing.append(i)
                partial = directory.with_name(directory.name+".partial")
                if partial.exists():
                    recovery = self.output/"recovery.partial"/"sample_layers"/group
                    recovery.mkdir(parents=True, exist_ok=True)
                    partial.rename(recovery/(partial.name+f"-{time.time_ns()}"))
        print(f"  Sample layers {group}: {len(missing)} frames to capture "
              f"({','.join(str(p['samples']) for p in config['sample_layers'])} SPP)", flush=True)
        for first, end in missing_ranges(missing):
            log_path = self.output/"logs"/("layers-"+group+".log")
            log_path.parent.mkdir(exist_ok=True)
            command = [str(self.renderer), str(self.output/"sample_layer_configs"/(group+".json")),
                       str(self.output/"sample_layers"/group), "--start-frame", str(first), "--end-frame", str(end)]
            with log_path.open("a") as log:
                log.write(json.dumps({"command": command, "started_unix": time.time()})+"\n")
                log.flush()
                subprocess.run(command, cwd=self.renderer_root, stdout=log, stderr=subprocess.STDOUT, check=True)
            for i in range(first, end):
                base = read_manifest(self.output/group/f"frame_{i:06d}")
                read_layer_capture(self.output/"sample_layers"/group/f"frame_{i:06d}", base,
                                   config["sample_layers"], pixels=True)
                self.captured += 1
        write_json(self.output/"last-sample-layers.json", {"last_group": group,
            "captured_frames": self.captured, "skipped_frames": self.skipped, "archived_frames": self.archived})
