"""Metadata-only snapshots and synchronous, one-frame-at-a-time R2 replay."""
import json
from pathlib import Path
import re
import shutil
import tempfile

from .adaptive_data import layer_layout, load_v5_frame, read_json, read_v5_dataset
from .dataset_generation import digest, split_spec, write_json
from .joint_data import joint_inventory
from .r2_archive import R2SceneArchiver


LAYOUTS = ("v5", "inline")
SOURCE_RECORD = "r2-source.json"


def image_keys(group, index, layout):
    relative = f"{group}/frame_{index:06d}"
    if layout not in LAYOUTS:
        raise ValueError("R2 layout must be v5 or inline")
    return ((f"base/{relative}/frame.exr", f"layers/{relative}/layers.exr") if layout == "v5"
            else (f"{relative}/frame.exr", f"sample_layers/{relative}/layers.exr"))


class R2FrameSource:
    """Keep metadata locally; download and release two EXRs per frame read.

    This deliberately has no persistent image cache, workers or prefetch queue.
    Crop/augmentation and all training math stay in the existing dataset/trainer.
    """
    def __init__(self, root, source, *, layout="v5", spool_directory=None):
        self.root = Path(root)
        self.layout = layout
        if layout not in LAYOUTS:
            raise ValueError("R2 layout must be v5 or inline")
        remote = R2SceneArchiver(self.root, source)
        self.source, self.client = remote.destination, remote.native
        self.spool_directory = spool_directory
        if spool_directory is not None:
            Path(spool_directory).mkdir(parents=True, exist_ok=True)
        record = read_json(self.root/SOURCE_RECORD)
        if record.get("source") != self.source or record.get("layout") != layout:
            raise ValueError("R2 source/layout disagrees with prepared metadata snapshot")
        self.selected = set(record["selected"]["train"]+record["selected"]["validation"])
        self.stats = {"downloaded_bytes": 0, "downloaded_objects": 0, "decoded_frames": 0,
                      "persistent_image_cache": False, "prefetch": False}

    def preflight(self):
        self.client.preflight()
        if read_json(self.root/SOURCE_RECORD).get("r2_endpoint") != self.client.endpoint:
            raise ValueError("R2 account endpoint disagrees with prepared metadata snapshot")

    def load_frame(self, group, index):
        # Group/index validation precedes path construction or remote requests.
        descriptor = read_v5_dataset(self.root)
        if group not in self.selected or type(index) is not int or not 0 <= index < descriptor["groups"][group]["frame_count"]:
            raise ValueError("Unknown R2 frame")
        relative = Path(group)/f"frame_{index:06d}"
        with tempfile.TemporaryDirectory(prefix="ntasad-r2-frame-", dir=self.spool_directory) as name:
            staging = Path(name)
            shutil.copyfile(self.root/"dataset.json", staging/"dataset.json")
            for category in ("base", "layers", "frames"):
                directory = staging/category/relative
                directory.mkdir(parents=True)
                shutil.copyfile(self.root/category/relative/"manifest.json", directory/"manifest.json")
            for category, filename, key in zip(("base", "layers"), ("frame.exr", "layers.exr"),
                                                image_keys(group, index, self.layout)):
                metadata = read_json(staging/category/relative/"manifest.json")
                result = self.client.download_file(key, staging/category/relative/filename,
                                                   sha256=metadata["image_sha256"])
                self.stats["downloaded_bytes"] += result["size"]
                self.stats["downloaded_objects"] += 1
            frame = load_v5_frame(staging, group, index)
            self.stats["decoded_frames"] += 1
            return frame


def prepare_r2_metadata(source, output, *, layout="v5", train_groups=None, validation_groups=None):
    """Freeze selected completed clips from R2, without GETs of any EXR.

    Inline archives are normalized to v5 metadata locally; source object keys
    and source pixels are never rewritten or uploaded. Only current-lock scene
    receipts are accepted in this first implementation. Earlier clips may use
    separately published, receipt-bound sample_layer_configs/GROUP.json files.
    """
    if layout not in LAYOUTS:
        raise ValueError("R2 layout must be v5 or inline")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Metadata snapshot already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    remote = R2SceneArchiver(output, source)
    client = remote.native
    client.preflight()
    files = client.list_files()
    with tempfile.TemporaryDirectory(prefix=".r2-metadata-", dir=output.parent) as name:
        root = Path(name)/"snapshot"
        root.mkdir()

        def fetch(key, destination=None, expected=None):
            if key not in files:
                raise ValueError(f"Missing R2 metadata: {key}")
            target = root/(destination or key)
            client.download_file(key, target, size=files[key]["size"],
                                 sha256=expected, max_bytes=64*1024*1024)
            return read_json(target)

        if layout == "v5":
            descriptor = fetch("dataset.json")
            read_v5_dataset(root)
            layer_lock = None
        else:
            lock = fetch("capture_lock.json")
            split = fetch("split.json")
            layer_lock = fetch("sample_layers_lock.json")
            if (lock.get("format") != "wavefront-capture-lock" or lock.get("version") != 1
                    or split != split_spec(lock["plan"], lock["configs"], root)
                    or layer_lock.get("format") != "wavefront-inline-sample-layers"
                    or layer_lock.get("version") != 1
                    or layer_lock.get("base_capture_lock_sha256") != digest(root/"capture_lock.json")
                    or layer_lock.get("spps") != [1, 2, 8, 16, 32]):
                raise ValueError("Inline R2 capture/split/sample-layer locks disagree")
            reserved = layer_lock["reserved_reference_samples"]
            descriptor = {"format": "wavefront-ntasad-v5-dataset", "version": 1,
                "dataset_id": lock["plan"]["dataset_id"]+"-streaming",
                "reference_status": split["reference_status"], "partitions": split["partitions"],
                "groups": {g: {"scene": c["scene"], "frame_count": len(c["frames"]),
                    "source_config": c, "reserved_reference_samples": reserved,
                    "sample_layers": layer_layout(c, reserved)} for g,c in lock["configs"].items()}}
            write_json(root/"dataset.json", descriptor)
            read_v5_dataset(root)
        selected = {p: names if names is not None else descriptor["partitions"][p]
                    for p,names in (("train", train_groups), ("validation", validation_groups))}
        for partition, groups in selected.items():
            if (not groups or len(set(groups)) != len(groups)
                    or set(groups)-set(descriptor["partitions"][partition])):
                raise ValueError("R2 selections must belong to the declared train/validation partitions")
        receipts = {}
        for group in selected["train"]+selected["validation"]:
            info = descriptor["groups"][group]
            scene = info["scene"]
            if not isinstance(scene, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", scene):
                raise ValueError("Invalid R2 scene identifier")
            receipt = None
            if layout == "inline":
                config = info["source_config"]
                expected = dict(config, sample_layers=info["sample_layers"],
                                reserved_reference_samples=info["reserved_reference_samples"])
                inline_group = group in layer_lock["groups"]
                if inline_group and layer_lock["configs"].get(group) != expected:
                    raise ValueError(f"Missing or incompatible inline sample layers: {group}")
                if scene not in receipts:
                    receipt = fetch(f"r2-archives/{scene}.json")
                    if (receipt.get("format") != "wavefront-r2-scene-archive" or receipt.get("version") != 1
                            or receipt.get("status") != "verified" or receipt.get("scene") != scene
                            or receipt.get("destination") != remote.destination
                            or receipt.get("r2_endpoint") != client.endpoint
                            or receipt.get("capture_lock_sha256") != digest(root/"capture_lock.json")):
                        raise ValueError("Inline streaming requires a verified receipt for the current capture lock")
                    receipts[scene] = receipt
                receipt = receipts[scene]
                if not inline_group:
                    # Backfills retain the capture-time boundary/renderer lock.
                    # Their own per-clip config is bound by the scene receipt.
                    key = f"sample_layer_configs/{group}.json"
                    record = receipt["files"].get(key)
                    if record is None or key not in files:
                        raise ValueError(f"Missing or incompatible inline sample layers: {group}; backfill config required")
                    if fetch(key, expected=record["sha256"]) != expected:
                        raise ValueError(f"Backfill sample-layer config disagrees with source: {group}")
            for index in range(info["frame_count"]):
                relative = Path(group)/f"frame_{index:06d}"
                for category, key in zip(("base", "layers"), image_keys(group, index, layout)):
                    if key not in files:
                        raise ValueError(f"Missing R2 image (clip is not complete): {key}")
                    manifest_key = key.rsplit("/", 1)[0]+"/manifest.json"
                    if receipt is not None:
                        records = receipt["files"]
                        if manifest_key not in records or key not in records:
                            raise ValueError("Inline receipt does not include required sample layers/images")
                        image_record = records[key]
                        if files[key]["size"] != image_record["size"]:
                            raise ValueError("R2 image size disagrees with archive receipt")
                    metadata = fetch(manifest_key, str(Path(category)/relative/"manifest.json"),
                                     receipt["files"][manifest_key]["sha256"] if receipt else None)
                    if receipt and metadata["image_sha256"] != receipt["files"][key]["sha256"]:
                        raise ValueError("Image manifest disagrees with archive receipt")
                if layout == "v5":
                    fetch(str(Path("frames")/relative/"manifest.json"))
                else:
                    write_json(root/"frames"/relative/"manifest.json", {
                        "format": "wavefront-ntasad-v5-frame", "version": 1,
                        "sequence_id": group, "frame_index": index,
                        "source_manifest_sha256": digest(root/"base"/relative/"manifest.json"),
                        "layer_manifest_sha256": digest(root/"layers"/relative/"manifest.json")})
        # Audit known test identity from JSON only, including unselected scenes.
        for group in descriptor["partitions"]["test"]:
            for index in range(descriptor["groups"][group]["frame_count"]):
                prefix = "base/" if layout == "v5" else ""
                relative = f"{group}/frame_{index:06d}/manifest.json"
                if prefix+relative in files:
                    fetch(prefix+relative, "base/"+relative)
        inventory = joint_inventory(root, selected, verify_images=False)
        write_json(root/SOURCE_RECORD, {"source": remote.destination, "layout": layout,
            "r2_endpoint": client.endpoint, "selected": selected,
            "inventory": inventory, "image_verification": "sha256_on_every_download"})
        from .training_defaults import joint_training_config
        pilot = joint_training_config(".")
        pilot.update(epochs=1, batch_size=1, denoiser_channels=8, sampler_channels=8,
            patch_size=32, patches_per_window=1, validation_patch_size=32, validation_crops_per_window=1,
            train_groups=selected["train"], validation_groups=selected["validation"],
            r2_streaming={"source": remote.destination, "layout": layout})
        write_json(root/"train-streaming-pilot.json", pilot)
        root.rename(output)
    return {"metadata_root": str(output), "source": remote.destination, "layout": layout,
            "selected": selected, "images_downloaded": 0}
