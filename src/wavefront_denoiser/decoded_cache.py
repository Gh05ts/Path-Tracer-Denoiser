"""Lossless, crop-addressable capture caches and bounded background decoding."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .adaptive_data import LAYER_SPP, NEW_LAYER_SPP, layer_layout, read_layer_capture
from .dataset_generation import digest, write_json
from .frames import CHANNELS, COUNT_BUFFERS, Frame, FrameFormatError, load_frame, read_manifest


LOCK_NAME = "decoded_cache_lock.json"
FLOAT_NAMES = [c for name, channels in CHANNELS.items() if name not in COUNT_BUFFERS for c in channels]
FLOAT_NAMES += [f"spp_{spp}.{c}" for spp in NEW_LAYER_SPP for c in "RGB"]
UINT_NAMES = [c for name, channels in CHANNELS.items() if name in COUNT_BUFFERS for c in channels]
ARRAYS = {"float.npy": (np.dtype("float32"), FLOAT_NAMES),
          "uint.npy": (np.dtype("uint32"), UINT_NAMES)}


@lru_cache(maxsize=16)
def _cached_json(path, stamp):
    text = Path(path).read_text()
    return json.loads(text), digest(Path(path))


def _json_and_hash(path):
    path = Path(path)
    stat = path.stat()
    return _cached_json(str(path), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def read_cache_lock(output):
    output = Path(output)
    lock, _ = _json_and_hash(output / LOCK_NAME)
    if (lock.get("format") != "wavefront-decoded-cache-lock" or lock.get("version") != 1
            or lock.get("capture_lock_sha256") != _json_and_hash(output / "capture_lock.json")[1]
            or lock.get("sample_layers_lock_sha256") != _json_and_hash(output / "sample_layers_lock.json")[1]):
        raise ValueError("Decoded cache provenance lock changed")
    return lock


def required_decoded_pairs(output, configs):
    output = Path(output)
    if not (output / LOCK_NAME).exists():
        return []
    lock = read_cache_lock(output)
    return [(output / "decoded" / group / f"frame_{index:06d}" / filename,
             output / "decoded" / group / f"frame_{index:06d}" / "manifest.json")
            for group, config in configs.items() if group in lock["groups"]
            for index in range(len(config["frames"])) for filename in ARRAYS]


def decoded_dataset_descriptor(output):
    """Expose the original partitions to joint training without downloading EXRs."""
    output = Path(output)
    lock = read_cache_lock(output)
    capture, _ = _json_and_hash(output / "capture_lock.json")
    layers, _ = _json_and_hash(output / "sample_layers_lock.json")
    plan = capture["plan"]
    membership = {scene["id"]: scene["partition"] for scene in plan["scenes"]}
    return {"format": "wavefront-ntasad-v5-dataset", "version": 1, "storage": "decoded",
            "dataset_id": plan["dataset_id"], "reference_status": "fixed_budget_not_convergence_qualified",
            "partitions": {p: [g for g, c in lock["configs"].items() if membership[c["scene"]] == p]
                           for p in ("train", "validation", "test")},
            "groups": {g: {"scene": c["scene"], "frame_count": len(c["frames"]), "source_config": c,
                           "reserved_reference_samples": layers["reserved_reference_samples"],
                           "sample_layers": layer_layout(c, layers["reserved_reference_samples"])}
                       for g, c in lock["configs"].items()}}


def verify_decoded_frame(output, group, index, *, verify_hashes=True, allow_missing=False):
    """Verify cached bytes once for inventory/archive, never once per crop."""
    output = Path(output)
    lock = read_cache_lock(output)
    relative = Path(group) / f"frame_{index:06d}"
    directory = output / "decoded" / relative
    record = json.loads((directory / "manifest.json").read_text())
    base_path = output / relative / "manifest.json"
    layer_path = output / "sample_layers" / relative / "manifest.json"
    if (record.get("format") != "wavefront-decoded-frame" or record.get("version") != 1
            or record.get("cache_lock_sha256") != _json_and_hash(output / LOCK_NAME)[1]
            or record.get("sequence_id") != group or record.get("frame_index") != index
            or record.get("source_manifest_sha256") != digest(base_path)
            or record.get("layer_manifest_sha256") != digest(layer_path)):
        raise FrameFormatError("Decoded cache metadata/source binding mismatch")
    base = read_manifest(base_path)
    if base["sequence_id"] != group or base["frame_index"] != index:
        raise FrameFormatError("Decoded cache source identity mismatch")
    layers, _ = _json_and_hash(output / "sample_layers_lock.json")
    layout = layer_layout(lock["configs"][group], layers["reserved_reference_samples"])
    read_layer_capture(layer_path.parent, base, layout,
                       pixels=False, verify_digest=False)
    if set(record.get("arrays", {})) != set(ARRAYS):
        raise FrameFormatError("Decoded cache array inventory mismatch")
    for filename, (dtype, channels) in ARRAYS.items():
        info = record["arrays"][filename]
        shape = (base["height"], base["width"], len(channels))
        if (info.get("channels") != channels or info.get("shape") != list(shape)
                or info.get("dtype") != dtype.str or not isinstance(info.get("sha256"), str)
                or len(info["sha256"]) != 64):
            raise FrameFormatError("Decoded cache array layout mismatch")
        path = directory / filename
        if not path.exists() and allow_missing:
            continue
        if path.stat().st_size != info.get("size"):
            raise FrameFormatError("Decoded cache array size mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
            raise FrameFormatError("Decoded cache array header mismatch")
        del array
        if verify_hashes and digest(path) != info["sha256"]:
            raise FrameFormatError("Decoded cache array checksum mismatch")
    return base, record


def load_decoded_crop(output, group, index, *, top=0, left=0, size=None):
    """Read only the requested pixels from read-only memory maps.

    Call verify_decoded_frame with hashes at dataset inventory time. This path
    checks metadata and array headers, but does not hash or decode whole images.
    """
    output = Path(output)
    base, _ = verify_decoded_frame(output, group, index, verify_hashes=False)
    h, w = base["height"], base["width"]
    if size is None:
        height, width = h, w
    else:
        height = width = size
    if (any(type(v) is not int for v in (top, left, height, width))
            or min(top, left) < 0 or min(height, width) < 1
            or top + height > h or left + width > w):
        raise ValueError("Invalid decoded cache crop")
    directory = output / "decoded" / group / f"frame_{index:06d}"
    floats = np.load(directory / "float.npy", mmap_mode="r", allow_pickle=False)
    uints = np.load(directory / "uint.npy", mmap_mode="r", allow_pickle=False)
    cropped_float = floats[top:top+height, left:left+width]
    cropped_uint = uints[top:top+height, left:left+width]
    buffers = {}
    for name, channels in CHANNELS.items():
        names, values = (UINT_NAMES, cropped_uint) if name in COUNT_BUFFERS else (FLOAT_NAMES, cropped_float)
        first = names.index(channels[0])
        buffers[name] = np.array(values[..., first:first+len(channels)], copy=True, order="C")
    images = {4: buffers["radiance"]}
    for spp in NEW_LAYER_SPP:
        first = FLOAT_NAMES.index(f"spp_{spp}.R")
        images[spp] = cropped_float[..., first:first+3]
    layers = np.stack([images[spp] for spp in LAYER_SPP])
    return Frame(output / group / f"frame_{index:06d}" / "manifest.json", base, buffers), layers


def decode_frame(output, group, index):
    output = Path(output)
    read_cache_lock(output)
    relative = Path(group) / f"frame_{index:06d}"
    directory = output / "decoded" / relative
    if directory.exists():
        verify_decoded_frame(output, group, index)
        return False
    base = load_frame(output / relative)
    lock = read_cache_lock(output)
    layers, _ = _json_and_hash(output / "sample_layers_lock.json")
    layout = layer_layout(lock["configs"][group], layers["reserved_reference_samples"])
    _, images = read_layer_capture(output / "sample_layers" / relative, base.metadata,
                                  layout)
    if set(images) != set(NEW_LAYER_SPP):
        raise ValueError("Decoded cache requires all five additional SPP layers")
    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        recovery = output / "recovery.partial" / "decoded" / group
        recovery.mkdir(parents=True, exist_ok=True)
        staging.rename(recovery / f"{staging.name}-{time.time_ns()}")
    staging.mkdir(parents=True)
    record = {"format": "wavefront-decoded-frame", "version": 1,
              "sequence_id": group, "frame_index": index,
              "cache_lock_sha256": digest(output / LOCK_NAME),
              "source_manifest_sha256": digest(output / relative / "manifest.json"),
              "layer_manifest_sha256": digest(output / "sample_layers" / relative / "manifest.json"),
              "arrays": {}}
    # Write directly into disk-backed arrays; retain only one decoded frame.
    for filename, (dtype, channels) in ARRAYS.items():
        shape = (base.metadata["height"], base.metadata["width"], len(channels))
        target = np.lib.format.open_memmap(staging / filename, mode="w+", dtype=dtype, shape=shape)
        offset = 0
        for name, names in CHANNELS.items():
            if (name in COUNT_BUFFERS) != (filename == "uint.npy"):
                continue
            values = base.buffers[name]
            target[..., offset:offset+len(names)] = values
            offset += len(names)
        if filename == "float.npy":
            for spp in NEW_LAYER_SPP:
                target[..., offset:offset+3] = images[spp]
                offset += 3
        target.flush()
        # Read back to verify equality before publishing the directory.
        offset = 0
        for name, names in CHANNELS.items():
            if (name in COUNT_BUFFERS) == (filename == "uint.npy"):
                if not np.array_equal(target[..., offset:offset+len(names)], base.buffers[name]):
                    raise ValueError("Decoded cache differs from source pixels")
                offset += len(names)
        if filename == "float.npy":
            for spp in NEW_LAYER_SPP:
                if not np.array_equal(target[..., offset:offset+3], images[spp]):
                    raise ValueError("Decoded cache differs from companion pixels")
                offset += 3
        del target
        path = staging / filename
        record["arrays"][filename] = {"shape": list(shape), "dtype": dtype.str,
                                      "channels": channels, "size": path.stat().st_size,
                                      "sha256": digest(path)}
    # Source manifests are immutable; catch unrelated writers before publication.
    if (record["source_manifest_sha256"] != digest(output / relative / "manifest.json")
            or record["layer_manifest_sha256"] != digest(output / "sample_layers" / relative / "manifest.json")):
        raise ValueError("Source manifests changed while decoding")
    write_json(staging / "manifest.json", record)
    staging.rename(directory)
    verify_decoded_frame(output, group, index)
    return True


def _decode_clip(output, group, threads):
    import OpenEXR
    if hasattr(OpenEXR, "set_global_thread_count"):
        OpenEXR.set_global_thread_count(threads)
    start = time.monotonic()
    lock = read_cache_lock(output)
    count = sum(decode_frame(output, group, index) for index in range(len(lock["configs"][group]["frames"])))
    print(f"  Decoded cache {group}: {count} frames written/verified in {time.monotonic()-start:.1f}s", flush=True)
    return count


class BackgroundDecodedCache:
    """One spawned CPU worker, at most two queued/running clips, scene barrier."""

    def __init__(self, output, configs, *, after_group):
        self.output = Path(output)
        ordered = list(configs)
        if after_group not in configs:
            raise ValueError("Unknown decoded-cache clip boundary")
        groups = ordered[ordered.index(after_group)+1:]
        layers = json.loads((self.output / "sample_layers_lock.json").read_text())
        if not groups or set(groups) - set(layers["groups"]) or layers["spps"] != list(NEW_LAYER_SPP):
            raise ValueError("Decoded caches require complete five-layer capture for every enabled clip")
        lock = {"format": "wavefront-decoded-cache-lock", "version": 1,
                "capture_lock_sha256": digest(self.output / "capture_lock.json"),
                "sample_layers_lock_sha256": digest(self.output / "sample_layers_lock.json"),
                "after_group": after_group, "groups": groups, "configs": configs,
                "float_channels": FLOAT_NAMES, "uint_channels": UINT_NAMES,
                "worker_processes": 1, "decoder_threads": 2, "max_pending_clips": 2}
        path = self.output / LOCK_NAME
        if path.exists():
            if json.loads(path.read_text()) != lock:
                raise ValueError("Decoded cache configuration changed")
        else:
            if (self.output / "decoded").exists():
                raise ValueError("Refusing to adopt decoded files without a provenance lock")
            write_json(path, lock)
        self.lock = read_cache_lock(self.output)
        self.pending = []
        self.executor = None
        training = self.output / "train-joint-decoded.json"
        if not training.exists():
            from .training_defaults import joint_training_config
            write_json(training, joint_training_config())
        backfill = self.output / "DECODED_BACKFILL.md"
        if not backfill.exists():
            deferred = ordered[:ordered.index(after_group)+1]
            rows = []
            for group in deferred:
                count = len(configs[group]["frames"])
                done = sum((self.output / group / f"frame_{i:06d}" / "manifest.json").exists() for i in range(count))
                rows.append(f"| `{group}` | 0–{count-1} | {done}/{count} |")
            total = sum(len(configs[g]["frames"]) for g in deferred)
            backfill.write_text("# Decoded cache backfill before joint training\n\n"
                f"Automatic decoding starts after `{after_group}`. All **{total} frames in {len(deferred)} clips** "
                "below, including the complete boundary clip, require separate decoded backfill. "
                "Completed EXRs are reused, not rerendered. Download archived EXRs when needed.\n\n"
                "First generate any missing independent layers listed in SAMPLE_LAYERS_BACKFILL.md, "
                "with the original renderer/integrator provenance (BMW integrator 11; Bistro integrator 12). "
                "Then generate and verify lossless decoded arrays and upload them with their metadata before training. "
                "Do not replace existing EXRs or change their sample ranges.\n\n"
                "Cache layout: `decoded/GROUP/frame_NNNNNN/{float.npy,uint.npy,manifest.json}`. "
                "The arrays retain all 37 source channels as float32/uint32; memory mapping reads crops "
                "without whole-image decompression. Both payload types must be remotely verified before local cleanup.\n\n"
                "| Clip | Frame indices needing decoded backfill | Base manifests at enablement |\n"
                "| --- | --- | ---: |\n" + "\n".join(rows) + "\n")

    def check(self):
        remaining = []
        for group, future in self.pending:
            if future.done():
                future.result()
            else:
                remaining.append((group, future))
        self.pending = remaining

    def submit_group(self, group):
        self.check()
        if group not in self.lock["groups"]:
            return
        # A verified archive may have removed the entire local payload already.
        config = self.lock["configs"][group]
        pairs = required_decoded_pairs(self.output, {group: config})
        if pairs and all(not image.exists() and manifest.exists() for image, manifest in pairs):
            from .capture_history import compatible_receipt_lock
            for provider in ("r2", "b2"):
                receipt_path = self.output / (provider + "-archives") / (config["scene"] + ".json")
                if not receipt_path.exists():
                    continue
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("status") == "verified"
                        and compatible_receipt_lock(self.output, config["scene"], receipt.get("capture_lock_sha256"),
                                                    digest(self.output / "capture_lock.json"))
                        and all(str(image.relative_to(self.output)) in receipt["files"]
                                and digest(manifest) == receipt["files"].get(str(manifest.relative_to(self.output)), {}).get("sha256")
                                for image, manifest in pairs)):
                    for index in range(len(config["frames"])):
                        verify_decoded_frame(self.output, group, index, verify_hashes=False, allow_missing=True)
                    return
        if len(self.pending) >= self.lock["max_pending_clips"]:
            self.pending.pop(0)[1].result()
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        future = self.executor.submit(_decode_clip, str(self.output), group, self.lock["decoder_threads"])
        self.pending.append((group, future))
        print(f"  Queued CPU decoding: {group}", flush=True)

    def finish_scene(self, scene):
        for group, future in self.pending:
            if self.lock["configs"][group]["scene"] == scene:
                future.result()
        self.check()

    def close(self):
        try:
            for _, future in self.pending:
                future.result()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
