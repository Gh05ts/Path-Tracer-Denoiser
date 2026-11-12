"""Archive completed scenes to B2 after hash verification, then remove local EXRs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlsplit

from .b2_native import B2NativeClient
from .capture_history import compatible_receipt_lock


def file_hashes(path: Path) -> dict:
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha1.update(block)
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return {"sha1": sha1.hexdigest(), "sha256": sha256.hexdigest(), "md5": md5.hexdigest(), "size": size}


class B2SceneArchiver:
    provider = "b2"
    receipt_format = "wavefront-b2-scene-archive"

    def __init__(self, output: Path, destination: str, *, rclone: str = "rclone"):
        self.native = None
        if destination.startswith("b2://"):
            parsed = urlsplit(destination)
            prefix = parsed.path.strip("/")
            if (not re.fullmatch(r"[A-Za-z0-9-]+", parsed.netloc) or parsed.query or parsed.fragment
                    or ".." in PurePosixPath(prefix).parts or "\n" in destination or "\r" in destination):
                raise ValueError("--b2-archive must be b2://BUCKET/PREFIX")
            self.native = B2NativeClient(parsed.netloc, prefix)
            self.output = output.resolve()
            self.destination = f"b2://{parsed.netloc}" + (f"/{prefix}" if prefix else "")
            self.rclone = str(rclone)
            self.directory = self.output / "b2-archives"
            return
        remote, separator, location = destination.partition(":")
        if (not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", remote)
                or not location or location.startswith("/") or ".." in PurePosixPath(location).parts
                or "\n" in location or "\r" in location):
            raise ValueError("--b2-archive must be a configured rclone REMOTE:BUCKET/PREFIX")
        self.output = output.resolve()
        self.destination = destination.rstrip("/")
        self.rclone = str(rclone)
        self.directory = self.output / "b2-archives"

    def _run(self, *args, **kwargs):
        try:
            return subprocess.run([self.rclone, *map(str, args)], check=True, **kwargs)
        except FileNotFoundError as error:
            raise RuntimeError("B2 archiving requires rclone; install it or supply --rclone PATH") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"B2 archive {args[0]} failed; local EXRs were retained. "
                               "Fix the connection/configuration and resume the same command.") from error

    def preflight(self):
        """Check credentials and hash support before spending GPU time."""
        if self.native is not None:
            self.native.preflight()
            return
        result = self._run("backend", "features", self.destination, capture_output=True, text=True)
        features = json.loads(result.stdout)
        if "sha1" not in {name.lower().replace("-", "") for name in features.get("Hashes", [])}:
            raise ValueError("B2 archiving requires a remote with SHA1 checksums; configure rclone's native B2 backend")
        bucket = self.destination.split("/", 1)[0]
        self._run("lsf", bucket, "--max-depth", "1", stdout=subprocess.DEVNULL)

    def _verify_remote(self, paths: dict[str, dict], stem: str):
        if self.native is not None:
            self.native.verify_files(paths)
            return
        listing = self.directory / f"{stem}.files"
        sums = self.directory / f"{stem}.remote.sha1"
        listing.write_text("".join(f"{path}\n" for path in sorted(paths)))
        # Do not let a stale report from an interrupted attempt count as success.
        sums.unlink(missing_ok=True)
        self._run("hashsum", "sha1", self.destination, "--files-from-raw", listing,
                  "--output-file", sums)
        remote = {}
        for line in sums.read_text().splitlines():
            match = re.fullmatch(r"([0-9a-fA-F]{40})  (.+)", line)
            if not match or match[2] in remote:
                raise ValueError("Invalid or missing remote SHA1; local EXRs were retained")
            remote[match[2]] = match[1].lower()
        expected = {path: record["sha1"] for path, record in paths.items()}
        if remote != expected:
            raise ValueError("B2 archive checksum mismatch or missing remote files; local EXRs were retained")

    def receipt_context(self) -> dict:
        return {}

    def archive_scene(self, scene: str, configs: dict[str, dict]) -> bool:
        """Archive only a complete planned scene, retaining manifests and logs.

        A verified receipt is written before removing any EXR. If interrupted
        during deletion, the receipt supplies hashes for already-removed files
        while the remaining local files are uploaded/checked again on resume.
        """
        from .dataset_generation import digest, write_json
        from .frames import read_manifest

        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene):
            raise ValueError("Invalid archive scene ID")
        lock_hash = digest(self.output / "capture_lock.json")
        context = self.receipt_context()
        receipt_path = self.directory / f"{scene}.json"
        previous = {}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("format") != self.receipt_format or receipt.get("version") != 1
                    or receipt.get("status") != "verified" or receipt.get("scene") != scene
                    or receipt.get("destination") != self.destination
                    or not compatible_receipt_lock(self.output, scene, receipt.get("capture_lock_sha256"), lock_hash)
                    or any(receipt.get(key) != value for key, value in context.items())):
                raise ValueError(f"Archive receipt does not match this dataset/destination: {receipt_path}")
            previous = receipt["files"]

        images = []
        manifests = []
        for group, config in configs.items():
            for i in range(len(config["frames"])):
                frame = self.output / group / f"frame_{i:06d}"
                image, manifest = frame / "frame.exr", frame / "manifest.json"
                if not manifest.is_file() or (not image.is_file() and str(image.relative_to(self.output)) not in previous):
                    print(f"Scene {scene} is not fully available for {self.provider.upper()} archiving; keeping local files.", flush=True)
                    return False
                images.append(image)
                manifests.append(manifest)

        from .inline_sample_layers import required_layer_pairs
        for image, manifest in required_layer_pairs(self.output, configs):
            if not manifest.is_file() or (not image.is_file() and str(image.relative_to(self.output)) not in previous):
                print(f"Scene {scene} sample layers are incomplete; keeping local files.", flush=True)
                return False
            images.append(image)
            manifests.append(manifest)

        if previous and all(not image.exists() for image in images):
            print(f"Scene {scene} already archived to {self.destination}; skipping.", flush=True)
            return True

        print(f"Archiving completed scene {scene} to {self.destination}", flush=True)
        self.directory.mkdir(exist_ok=True)
        sidecars = [self.output / name for name in (
            "capture_lock.json", "split.json", "train.json", "reference-check-plan.json",
            "reference_policy.json", "reference_convergence.json", "sample_layers_lock.json",
            "SAMPLE_LAYERS_BACKFILL.md") if (self.output / name).is_file()]
        sidecars.extend(sorted((self.output / "capture_configs").glob("*.json")))
        sidecars.extend(sorted((self.output / "sample_layer_configs").glob("*.json")))
        sidecars.extend(sorted(p for p in (self.output / "capture-history").rglob("*") if p.is_file()))
        sidecars.extend(self.output / "logs" / f"{group}.log" for group in configs
                        if (self.output / "logs" / f"{group}.log").is_file())
        sidecars.extend(self.output / "logs" / f"layers-{group}.log" for group in configs
                        if (self.output / "logs" / f"layers-{group}.log").is_file())
        inventory, available, stamps = {}, [], {}
        for path in [*images, *manifests, *sidecars]:
            relative = path.relative_to(self.output).as_posix()
            if path.is_file():
                before = path.stat()
                inventory[relative] = file_hashes(path)
                after = path.stat()
                stamp = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ino)
                if stamp(before) != stamp(after):
                    raise ValueError(f"File changed while preparing {self.provider.upper()} archive: {path}")
                stamps[path] = stamp(after)
                available.append(relative)
            else:
                inventory[relative] = previous[relative]
        for image, manifest in zip(images, manifests):
            if image.name == "layers.exr":
                from .adaptive_data import read_layer_capture
                group = manifest.parent.parent.name
                base = read_manifest(self.output/group/manifest.parent.name)
                layer_config = json.loads((self.output/"sample_layer_configs"/(group+".json")).read_text())
                metadata, _ = read_layer_capture(manifest.parent, base, layer_config["sample_layers"],
                                                 pixels=False, verify_digest=False)
            else:
                metadata = read_manifest(manifest)
            if inventory[image.relative_to(self.output).as_posix()]["sha256"] != metadata["image_sha256"]:
                raise ValueError(f"Local EXR checksum mismatch before {self.provider.upper()} upload: {image}")

        if self.native is not None:
            self.native.upload_files(self.output, available, inventory)
        else:
            files = self.directory / f"{scene}.upload.files"
            files.write_text("".join(f"{path}\n" for path in sorted(available)))
            self._run("copy", self.output, self.destination, "--files-from-raw", files, "--checksum",
                      "--transfers", "4", "--stats", "30s")
        self._verify_remote(inventory, scene)
        receipt = {"format": self.receipt_format, "version": 1, "status": "verified",
                   "scene": scene, "destination": self.destination, "capture_lock_sha256": lock_hash,
                   "files": inventory, **context}
        write_json(receipt_path, receipt)
        relative_receipt = receipt_path.relative_to(self.output).as_posix()
        receipt_hashes = file_hashes(receipt_path)
        if self.native is not None:
            self.native.upload_file(receipt_path, relative_receipt, receipt_hashes)
        else:
            self._run("copyto", receipt_path, f"{self.destination}/{relative_receipt}", "--checksum")
        self._verify_remote({relative_receipt: receipt_hashes}, f"{scene}.receipt")

        # Capture holds the dataset lock, but also refuse to remove local files
        # modified by an unrelated writer while the upload was running.
        local_images = [image for image in images if image in stamps]
        for image in local_images:
            stat = image.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != stamps[image]:
                raise ValueError(f"Local EXR changed during upload; keeping local files: {image}")
        for image in local_images:
            image.unlink()
        removed = sum(inventory[image.relative_to(self.output).as_posix()]["size"] for image in local_images)
        print(f"Archived scene {scene}: removed {len(local_images)} verified local EXRs "
              f"({removed / 1e9:.2f} GB); kept manifests, logs and receipt.", flush=True)
        return True
