import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from wavefront_denoiser.b2_archive import B2SceneArchiver


@pytest.fixture
def scene_data(tmp_path, make_frame):
    output = tmp_path / "dataset"
    group = "bmw_clip"
    for i in range(2):
        make_frame(str(output / group / f"frame_{i:06d}"), index=i, sequence=group, previous=i > 0)
    (output / "capture_lock.json").write_text('{"renderer_sha256": "locked"}')
    (output / "split.json").write_text('{"data_root": "."}')
    (output / "train.json").write_text('{"epochs": 1000}')
    (output / "capture_configs").mkdir()
    (output / "capture_configs" / f"{group}.json").write_text('{"frames": [{}, {}]}')
    (output / "logs").mkdir()
    (output / "logs" / f"{group}.log").write_text("completed capture\n")
    return output, {group: {"scene": "bmw", "frames": [{}, {}]}}


@pytest.fixture
def fake_rclone(tmp_path, monkeypatch):
    """Exercise transfers/hashes against a separate directory, without B2 credentials."""
    class Remote:
        destination = "b2:bucket/denoiser-v4"
        root = tmp_path / "cloud"
        fail = None
        corrupt = False
        missing_hash = False
        fail_receipt_hash = False
        after_receipt = None
        commands = []

        def location(self, remote):
            return self.root / remote.split(":", 1)[1]

        def run(self, command, **kwargs):
            self.commands.append(command)
            operation = command[1]
            if operation == self.fail:
                raise subprocess.CalledProcessError(1, command)
            if operation == "backend":
                return SimpleNamespace(stdout='{"Hashes": ["sha1"]}')
            if operation == "lsf":
                return SimpleNamespace(returncode=0)
            if operation == "copy":
                source, destination = Path(command[2]), self.location(command[3])
                files = Path(command[command.index("--files-from-raw") + 1]).read_text().splitlines()
                for name in files:
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
                    if self.corrupt and name.endswith(".exr"):
                        target.write_bytes(b"corrupt uploaded content")
            elif operation == "copyto":
                target = self.location(command[3])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(command[2], target)
            elif operation == "hashsum":
                root = self.location(command[3])
                files = Path(command[command.index("--files-from-raw") + 1]).read_text().splitlines()
                sums = Path(command[command.index("--output-file") + 1])
                if self.fail_receipt_hash and ".receipt." in sums.name:
                    raise subprocess.CalledProcessError(1, command)
                sums.write_text("".join(
                    f'{"" if self.missing_hash else hashlib.sha1((root / name).read_bytes()).hexdigest()}  {name}\n'
                    for name in files if (root / name).is_file()))
                if self.after_receipt and ".receipt." in sums.name:
                    self.after_receipt()
            else:
                raise AssertionError(command)
            return SimpleNamespace(returncode=0)
    remote = Remote()
    monkeypatch.setattr(subprocess, "run", remote.run)
    return remote


def test_scene_upload_verifies_every_file_before_removing_exrs(scene_data, fake_rclone):
    output, configs = scene_data
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    archiver.preflight()
    assert archiver.archive_scene("bmw", configs)
    assert not list(output.glob("*/frame_*/frame.exr"))
    assert len(list(output.glob("*/frame_*/manifest.json"))) == 2
    receipt = json.loads((output / "b2-archives/bmw.json").read_text())
    assert receipt["status"] == "verified"
    remote = fake_rclone.location(fake_rclone.destination)
    for name, expected in receipt["files"].items():
        assert hashlib.sha256((remote / name).read_bytes()).hexdigest() == expected["sha256"]
    assert json.loads((remote / "b2-archives/bmw.json").read_text()) == receipt
    assert (output / "logs/bmw_clip.log").is_file()
    before = len(fake_rclone.commands)
    assert archiver.archive_scene("bmw", configs)
    assert len(fake_rclone.commands) == before  # Missing archived EXRs need no remote revalidation.


@pytest.mark.parametrize("failure", ["copy", "hashsum", "copyto", "corrupt", "missing_hash", "fail_receipt_hash"])
def test_upload_or_verification_failure_never_removes_local_exrs(scene_data, fake_rclone, failure):
    output, configs = scene_data
    if failure in ("corrupt", "missing_hash", "fail_receipt_hash"):
        setattr(fake_rclone, failure, True)
    else:
        fake_rclone.fail = failure
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    with pytest.raises((RuntimeError, ValueError)):
        archiver.archive_scene("bmw", configs)
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 2
    fake_rclone.fail, fake_rclone.corrupt, fake_rclone.missing_hash = None, False, False
    fake_rclone.fail_receipt_hash = False
    assert archiver.archive_scene("bmw", configs)
    assert not list(output.glob("*/frame_*/frame.exr"))


def test_incomplete_scene_is_not_uploaded_or_deleted(scene_data, fake_rclone):
    output, configs = scene_data
    # An unrendered clip still belongs to this scene, even with --groups selection.
    configs["bmw_other_clip"] = {"scene": "bmw", "frames": [{}, {}]}
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    assert not archiver.archive_scene("bmw", configs)
    assert not fake_rclone.commands
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 2


def test_corrupt_local_exr_is_not_uploaded_or_deleted(scene_data, fake_rclone):
    output, configs = scene_data
    next(output.glob("*/frame_*/frame.exr")).write_bytes(b"local corruption")
    with pytest.raises(ValueError, match="Local EXR checksum mismatch"):
        B2SceneArchiver(output, fake_rclone.destination).archive_scene("bmw", configs)
    assert not fake_rclone.commands
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 2


def test_resume_after_interrupted_deletion_uses_receipt_for_missing_exrs(scene_data, fake_rclone, monkeypatch):
    output, configs = scene_data
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    unlink = Path.unlink
    removed = []
    def interrupted_unlink(path, *args, **kwargs):
        if path.name == "frame.exr":
            if removed:
                raise KeyboardInterrupt
            removed.append(path)
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted_unlink)
        with pytest.raises(KeyboardInterrupt):
            archiver.archive_scene("bmw", configs)
    assert len(removed) == 1 and not removed[0].exists()
    assert (output / "b2-archives/bmw.json").is_file()
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 1
    assert archiver.archive_scene("bmw", configs)
    assert not list(output.glob("*/frame_*/frame.exr"))


def test_exr_modified_during_upload_is_not_removed(scene_data, fake_rclone):
    output, configs = scene_data
    image = next(output.glob("*/frame_*/frame.exr"))
    fake_rclone.after_receipt = lambda: image.write_bytes(b"changed during upload")
    with pytest.raises(ValueError, match="Local EXR changed during upload"):
        B2SceneArchiver(output, fake_rclone.destination).archive_scene("bmw", configs)
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 2


def test_preflight_missing_rclone_is_actionable(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="requires rclone"):
        B2SceneArchiver(tmp_path, "b2:bucket/data").preflight()


def test_preflight_rejects_remote_without_sha1(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout='{"Hashes": ["md5"]}'))
    with pytest.raises(ValueError, match="native B2 backend"):
        B2SceneArchiver(tmp_path, "b2:bucket/data").preflight()


@pytest.mark.parametrize("change", ["destination", "capture_lock"])
def test_archived_scene_cannot_silently_switch_destination_or_dataset(scene_data, fake_rclone, change):
    output, configs = scene_data
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    archiver.archive_scene("bmw", configs)
    if change == "destination":
        archiver = B2SceneArchiver(output, "b2:other-bucket/dataset")
    else:
        (output / "capture_lock.json").write_text('{"renderer_sha256": "changed"}')
    with pytest.raises(ValueError, match="receipt does not match"):
        archiver.archive_scene("bmw", configs)


@pytest.mark.parametrize("destination", ["/tmp/local", "b2:", "b2:/bucket", "b2:bucket/../elsewhere", "b2:bucket\nother"])
def test_invalid_destinations_are_rejected(tmp_path, destination):
    with pytest.raises(ValueError, match="REMOTE:BUCKET/PREFIX"):
        B2SceneArchiver(tmp_path, destination)
