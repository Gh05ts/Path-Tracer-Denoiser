"""Lossless caches, real CPU workers, archive barriers and cache-only training."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_adaptive_dataset import fake_layers, source_dataset
from test_b2_archive import fake_rclone
from test_dataset_generation import small_config
from test_inline_sample_layers import inline_data, manager
from test_joint_training import joint_dataset, config_for
from test_r2_native import r2_api
from wavefront_denoiser.b2_archive import B2SceneArchiver
from wavefront_denoiser.dataset_generation import digest, run_captures, write_json
from wavefront_denoiser.decoded_cache import (
    BackgroundDecodedCache, decode_frame, load_decoded_crop, verify_decoded_frame,
)
from wavefront_denoiser.frames import FrameFormatError, load_frame, read_manifest


@pytest.fixture
def cache_data(inline_data):
    output, configs, renderer = inline_data
    capture_lock = json.loads((output / "capture_lock.json").read_text())
    capture_lock["plan"] = {"dataset_id": "fixture", "scenes": [{"id": "bmw", "partition": "train"}]}
    write_json(output / "capture_lock.json", capture_lock)
    layers = manager(inline_data)
    for index in range(2):
        relative = Path("new") / f"frame_{index:06d}"
        fake_layers(output / "sample_layers" / relative, read_manifest(output / relative),
                    layers.configs["new"]["sample_layers"])
    cache = BackgroundDecodedCache(output, configs, after_group="boundary")
    yield output, configs, renderer, cache
    cache.close()


def test_lossless_crop_reads_without_exrs_or_full_array_hashes(cache_data, monkeypatch):
    output, _, _, _ = cache_data
    source = load_frame(output / "new/frame_000001")
    assert decode_frame(output, "new", 1)
    assert not decode_frame(output, "new", 1)
    verify_decoded_frame(output, "new", 1)
    (output / "new/frame_000001/frame.exr").unlink()
    (output / "sample_layers/new/frame_000001/layers.exr").unlink()
    import OpenEXR
    monkeypatch.setattr(OpenEXR, "InputFile", lambda *a, **k: pytest.fail("EXR decoded during crop read"))
    from wavefront_denoiser import decoded_cache
    original_digest = decoded_cache.digest
    def metadata_only(path):
        assert Path(path).suffix != ".npy", "Full array hashed during crop read"
        return original_digest(path)
    monkeypatch.setattr(decoded_cache, "digest", metadata_only)
    frame, layers = load_decoded_crop(output, "new", 1, top=2, left=1, size=4)
    for name, value in source.buffers.items():
        np.testing.assert_array_equal(frame.buffers[name], value[2:6, 1:5])
        assert frame.buffers[name].dtype == value.dtype
    np.testing.assert_array_equal(layers[2], frame.buffers["radiance"])
    for index, spp in enumerate((1, 2, 4, 8, 16, 32)):
        if spp != 4:
            expected = np.empty((4, 4, 3), np.float32)
            expected[:] = [spp, spp + .1, spp + .2]
            np.testing.assert_array_equal(layers[index], expected)
    frame.buffers["radiance"].fill(0)
    again, _ = load_decoded_crop(output, "new", 1, top=2, left=1, size=4)
    np.testing.assert_array_equal(again.buffers["radiance"], source.buffers["radiance"][2:6, 1:5])


def test_spawned_decoder_and_resume_preserve_source_bytes(cache_data):
    output, _, _, cache = cache_data
    before = {p: digest(p) for p in output.rglob("*.exr")}
    cache.submit_group("boundary")
    assert cache.executor is None
    cache.submit_group("new")
    assert cache.executor is not None
    cache.finish_scene("bmw")
    assert not cache.pending
    for index in range(2):
        verify_decoded_frame(output, "new", index)
    cache.submit_group("new")
    cache.close()
    assert {p: digest(p) for p in before} == before
    text = (output / "DECODED_BACKFILL.md").read_text()
    assert "4 frames in 2 clips" in text and "| `boundary`" in text and "| `new`" not in text


def test_interrupted_cache_recovers_and_published_corruption_is_rejected(cache_data):
    output, _, _, _ = cache_data
    staging = output / "decoded/new/frame_000000.partial"
    staging.mkdir(parents=True)
    (staging / "float.npy").write_bytes(b"unfinished")
    decode_frame(output, "new", 0)
    assert list((output / "recovery.partial/decoded/new").glob("frame_000000.partial-*/float.npy"))
    path = output / "decoded/new/frame_000000/float.npy"
    array = np.load(path, mmap_mode="r+")
    array[0, 0, 0] += 1
    array.flush()
    del array
    with pytest.raises(FrameFormatError, match="checksum"):
        decode_frame(output, "new", 0)


def test_decoder_failure_propagates_and_preserves_sources(cache_data):
    output, _, _, cache = cache_data
    (output / "sample_layers/new/frame_000000/layers.exr").write_bytes(b"corrupt")
    cache.submit_group("new")
    with pytest.raises(FrameFormatError, match="checksum"):
        cache.finish_scene("bmw")
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 6
    # The expected error has been observed; avoid repeating it in fixture cleanup.
    cache.pending.clear()


def test_archive_requires_and_verifies_decoded_payloads_before_cleanup(cache_data, fake_rclone):
    output, configs, _, cache = cache_data
    archiver = B2SceneArchiver(output, fake_rclone.destination)
    assert not archiver.archive_scene("bmw", configs)
    assert not fake_rclone.commands
    cache.submit_group("new")
    cache.finish_scene("bmw")
    assert archiver.archive_scene("bmw", configs)
    assert not list(output.rglob("*.exr"))
    assert not list((output / "decoded").rglob("*.npy"))
    receipt = json.loads((output / "b2-archives/bmw.json").read_text())
    remote = fake_rclone.location(fake_rclone.destination)
    for name in ("decoded/new/frame_000000/float.npy", "decoded/new/frame_000000/uint.npy",
                 "decoded/new/frame_000000/manifest.json", "decoded_cache_lock.json", "DECODED_BACKFILL.md"):
        assert name in receipt["files"]
        assert digest(remote / name) == receipt["files"][name]["sha256"]
    cache.submit_group("new")
    assert not cache.pending
    assert archiver.archive_scene("bmw", configs)


@pytest.mark.parametrize("failure", ["local_corruption", "remote_corruption", "receipt_verification"])
def test_cache_upload_failure_never_deletes_sources_or_arrays(cache_data, fake_rclone, monkeypatch, failure):
    output, configs, _, cache = cache_data
    cache.submit_group("new")
    cache.finish_scene("bmw")
    if failure == "local_corruption":
        path = output / "decoded/new/frame_000000/float.npy"
        array = np.load(path, mmap_mode="r+")
        array[0, 0, 0] += 1
        array.flush()
        del array
    elif failure == "remote_corruption":
        original = fake_rclone.run
        def corrupt_upload(command, **kwargs):
            result = original(command, **kwargs)
            if command[1] == "copy":
                (fake_rclone.location(fake_rclone.destination) / "decoded/new/frame_000000/float.npy").write_bytes(b"bad")
            return result
        monkeypatch.setattr("subprocess.run", corrupt_upload)
    else:
        fake_rclone.fail_receipt_hash = True
    with pytest.raises((ValueError, RuntimeError)):
        B2SceneArchiver(output, fake_rclone.destination).archive_scene("bmw", configs)
    assert len(list(output.glob("*/frame_*/frame.exr"))) == 6
    assert len(list((output / "sample_layers").rglob("*.exr"))) == 2
    assert len(list((output / "decoded").rglob("*.npy"))) == 4


def test_native_r2_upload_includes_caches_and_verified_receipt(cache_data, r2_api):
    from wavefront_denoiser.r2_archive import R2SceneArchiver
    output, configs, _, cache = cache_data
    cache.submit_group("new")
    cache.finish_scene("bmw")
    archiver = R2SceneArchiver(output, "r2://bucket/cache-test")
    archiver.preflight()
    assert archiver.archive_scene("bmw", configs)
    receipt = json.loads((output / "r2-archives/bmw.json").read_text())
    remote = r2_api.objects
    for name, expected in receipt["files"].items():
        import hashlib
        assert hashlib.sha256(remote["cache-test/" + name]).hexdigest() == expected["sha256"]
    assert "cache-test/decoded/new/frame_000001/float.npy" in remote
    assert json.loads(remote["cache-test/r2-archives/bmw.json"]) == receipt
    assert not list((output / "decoded").rglob("*.npy"))


def test_clip_decode_is_submitted_before_next_clip_and_awaited_before_archive(inline_data):
    output, configs, renderer = inline_data
    events = []
    class Cache:
        def check(self):
            pass
        def submit_group(self, group):
            events.append(("decode", group))
        def finish_scene(self, scene):
            events.append(("wait", scene))
        def close(self):
            events.append(("close", None))
    class Archive:
        provider = "r2"
        destination = "r2://bucket/run"
        def archive_scene(self, scene, configs):
            events.append(("upload", scene))
            return True
    run_captures(configs, list(configs), output, renderer, output,
                 decoded_cache=Cache(), scene_archiver=Archive())
    assert events == [("decode", g) for g in configs] + [("wait", "bmw"), ("upload", "bmw"), ("close", None)]


def test_joint_training_uses_identical_crops_with_only_cache_and_metadata(joint_dataset, tmp_path, monkeypatch):
    import OpenEXR
    import torch
    from wavefront_denoiser.inline_sample_layers import InlineSampleLayers
    from wavefront_denoiser.joint_data import JointPatchDataset, joint_inventory
    from wavefront_denoiser.joint_training import train_joint

    original = joint_dataset
    source = Path(json.loads((original / "v5_lock.json").read_text())["source_root"])
    capture = json.loads((source / "capture_lock.json").read_text())
    configs = capture["configs"]
    output = tmp_path / "decoded-only"
    output.mkdir()
    shutil.copy2(source / "capture_lock.json", output / "capture_lock.json")
    for group in configs:
        if (original / "base" / group).exists():
            shutil.copytree(original / "base" / group, output / group)
    renderer = tmp_path / "cache-fixture-renderer"
    renderer.write_bytes(b"fixture")
    boundary = next(iter(configs))
    InlineSampleLayers(output, configs, renderer, output, spps=[1, 2, 8, 16, 32],
                       after_group=boundary, reserved_reference_samples=64)
    for group in configs:
        if (original / "layers" / group).exists():
            shutil.copytree(original / "layers" / group, output / "sample_layers" / group)
    cache = BackgroundDecodedCache(output, configs, after_group=boundary)
    cache.close()
    for group in configs:
        if (output / group).exists():
            for index in range(8):
                decode_frame(output, group, index)
    expected = {}
    for partition in ("train", "validation"):
        reader = JointPatchDataset(original, partition=partition, patch_size=5,
                                   patches_per_window=2, augment=False)
        expected[partition] = [reader[i] for i in range(len(reader))]
    for path in output.rglob("*.exr"):
        path.unlink()
    monkeypatch.setattr(OpenEXR, "InputFile", lambda *a, **k: pytest.fail("Joint training decoded an EXR"))
    inventory = joint_inventory(output)
    assert not inventory["test_pixels_loaded"]
    for partition in ("train", "validation"):
        reader = JointPatchDataset(output, partition=partition, patch_size=5,
                                   patches_per_window=2, augment=False)
        for index, baseline in enumerate(expected[partition]):
            sample = reader[index]
            torch.testing.assert_close(sample["layers"], baseline["layers"], rtol=0, atol=0)
            for name, value in baseline["buffers"].items():
                torch.testing.assert_close(sample["buffers"][name], value, rtol=0, atol=0)
    config, _ = config_for(output)
    summary = train_joint(config, tmp_path / "cache-training", "cpu")
    assert summary["completed_epochs"] == 2 and summary["global_step"] == 8
