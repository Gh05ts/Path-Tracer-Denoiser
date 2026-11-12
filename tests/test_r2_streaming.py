"""Real EXR/trainer parity through a fake signed R2 transport, without cloud writes."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from test_r2_native import r2_api
from wavefront_denoiser.adaptive_data import read_json
from wavefront_denoiser.dataset_generation import digest
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset
from wavefront_denoiser.joint_training import train_joint
from wavefront_denoiser.r2_native import R2NativeClient
from wavefront_denoiser.r2_streaming import R2FrameSource, prepare_r2_metadata


@pytest.fixture
def cloud(joint_dataset, r2_api):
    for path in joint_dataset.rglob("*"):
        if path.is_file():
            r2_api.objects["dataset/"+path.relative_to(joint_dataset).as_posix()] = path.read_bytes()
    descriptor = read_json(joint_dataset/"dataset.json")
    for group in descriptor["partitions"]["test"]:
        for path in (joint_dataset.parent/"v4"/group).rglob("manifest.json"):
            key = "dataset/base/"+path.relative_to(joint_dataset.parent/"v4").as_posix()
            r2_api.objects[key] = path.read_bytes()
            r2_api.objects[key.removesuffix("manifest.json")+"frame.exr"] = b"reserved-test-must-not-be-read"
    return joint_dataset, r2_api


def test_metadata_only_and_streamed_crops_match_local(cloud, tmp_path):
    root, api = cloud
    snapshot = tmp_path/"metadata"
    report = prepare_r2_metadata("r2://bucket/dataset", snapshot)
    assert report["images_downloaded"] == 0
    assert not any(urlsplit(u).path.endswith(".exr") for _,u in api.calls)
    assert not list(snapshot.rglob("*.exr"))
    assert api.uploads == 0
    source = R2FrameSource(snapshot, "r2://bucket/dataset", spool_directory=tmp_path/"spool")
    source.preflight()
    for augment in (False, True):
        local = JointPatchDataset(root, partition="train", patch_size=5, augment=augment)
        streamed = JointPatchDataset(snapshot, partition="train", patch_size=5, augment=augment, frame_source=source)
        torch.manual_seed(19)
        a = local[1]
        torch.manual_seed(19)
        b = streamed[1]
        torch.testing.assert_close(a["layers"], b["layers"], rtol=0, atol=0)
        for key in a["buffers"]:
            torch.testing.assert_close(a["buffers"][key], b["buffers"][key], rtol=0, atol=0)
    assert not list((tmp_path/"spool").iterdir())
    assert source.stats["decoded_frames"] == 10
    assert source.stats["downloaded_objects"] == 20
    assert not any("test_scene" in unquote(u) and u.endswith(".exr") for _,u in api.calls)


def test_streaming_training_matches_local_and_exact_resume(cloud, tmp_path):
    root, api = cloud
    snapshot = tmp_path/"metadata"
    prepare_r2_metadata("r2://bucket/dataset", snapshot)
    config_path, config = config_for(root)
    train_joint(config_path, tmp_path/"local", "cpu")
    config["data_root"] = str(snapshot)
    config["r2_streaming"] = {"source": "r2://bucket/dataset", "spool_directory": str(tmp_path/"spool")}
    config_path.write_text(json.dumps(config))
    full = train_joint(config_path, tmp_path/"streamed", "cpu")
    config["epochs"] = 1
    config_path.write_text(json.dumps(config))
    train_joint(config_path, tmp_path/"first", "cpu")
    config["epochs"] = 2
    config_path.write_text(json.dumps(config))
    resumed = train_joint(config_path, tmp_path/"resumed", "cpu", resume=tmp_path/"first/last.pt")
    local, _ = load_joint_checkpoint(tmp_path/"local/last.pt")
    streamed, payload = load_joint_checkpoint(tmp_path/"streamed/last.pt")
    continued, resumed_payload = load_joint_checkpoint(tmp_path/"resumed/last.pt")
    for name, value in local.state_dict().items():
        torch.testing.assert_close(value, streamed.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(value, continued.state_dict()[name], rtol=0, atol=0)
    for key in ("torch", "generator"):
        assert torch.equal(payload["training"]["rng"][key], resumed_payload["training"]["rng"][key])
    assert full["global_step"] == resumed["global_step"] == 8
    assert full["r2_streaming"]["downloaded_bytes"] > 0
    assert not full["test_pixels_loaded"]
    history = read_json(tmp_path/"streamed/summary.json")
    assert history["r2_streaming"]["persistent_image_cache"] is False
    first_epoch = json.loads((tmp_path/"streamed/history.jsonl").read_text().splitlines()[0])
    assert first_epoch["train"]["sampler_gradient_norm"] > 0
    assert first_epoch["train"]["denoiser_gradient_norm"] > 0
    assert not list((tmp_path/"spool").iterdir())
    assert api.uploads == 0


def test_corruption_and_missing_assets_fail_without_residue(cloud, tmp_path):
    root, api = cloud
    snapshot = tmp_path/"metadata"
    prepare_r2_metadata("r2://bucket/dataset", snapshot)
    group = read_json(root/"dataset.json")["partitions"]["train"][0]
    source = R2FrameSource(snapshot, "r2://bucket/dataset", spool_directory=tmp_path/"spool")
    source.preflight()
    key = f"dataset/layers/{group}/frame_000000/layers.exr"
    api.objects[key] = b"corrupt"
    with pytest.raises(ValueError, match="SHA-256"):
        source.load_frame(group, 0)
    assert not list((tmp_path/"spool").iterdir())
    del api.objects[key]
    with pytest.raises(ValueError, match="not complete"):
        prepare_r2_metadata("r2://bucket/dataset", tmp_path/"incomplete")
    assert not (tmp_path/"incomplete").exists()
    assert not list(tmp_path.glob(".r2-metadata-*"))


def test_streaming_rejects_changed_snapshot_or_endpoint(cloud, tmp_path, monkeypatch):
    _, api = cloud
    snapshot = tmp_path/"metadata"
    prepare_r2_metadata("r2://bucket/dataset", snapshot)
    with pytest.raises(ValueError, match="source/layout"):
        R2FrameSource(snapshot, "r2://bucket/elsewhere")
    source = R2FrameSource(snapshot, "r2://bucket/dataset")
    monkeypatch.setenv("R2_ACCOUNT_ID", "b"*32)
    with pytest.raises(ValueError, match="endpoint"):
        source.preflight()
    monkeypatch.setenv("R2_ACCOUNT_ID", "a"*32)
    config_path, config = config_for(snapshot)
    config["r2_streaming"] = {"source": "r2://bucket/dataset"}
    config_path.write_text(json.dumps(config))
    (snapshot/"dataset.json").write_text((snapshot/"dataset.json").read_text()+"\n")
    with pytest.raises(ValueError, match="inventory/selections"):
        train_joint(config_path, tmp_path/"changed", "cpu")
    assert not (tmp_path/"changed").exists()


def test_preparation_rejects_reserved_scene_overlap_without_test_pixels(cloud, tmp_path):
    root, api = cloud
    descriptor = read_json(root/"dataset.json")
    train_group = descriptor["partitions"]["train"][0]
    test_group = descriptor["partitions"]["test"][0]
    identity = read_json(root/"base"/train_group/"frame_000000/manifest.json")["scene_fingerprint"]
    key = f"dataset/base/{test_group}/frame_000000/manifest.json"
    metadata = json.loads(api.objects[key])
    metadata["scene_fingerprint"] = identity
    api.objects[key] = json.dumps(metadata).encode()
    with pytest.raises(ValueError, match="Reserved test identity"):
        prepare_r2_metadata("r2://bucket/dataset", tmp_path/"overlap")
    assert not (tmp_path/"overlap").exists()
    assert not any(u.endswith(".exr") for _,u in api.calls)


@pytest.fixture
def inline_cloud(cloud):
    root, api = cloud
    descriptor = read_json(root/"dataset.json")
    v4 = root.parent/"v4"
    lock = read_json(v4/"capture_lock.json")
    api.objects.clear()
    for name in ("capture_lock.json", "split.json"):
        api.objects["dataset/"+name] = (v4/name).read_bytes()
    layer_lock = {"format": "wavefront-inline-sample-layers", "version": 1,
        "base_capture_lock_sha256": digest(v4/"capture_lock.json"), "spps": [1,2,8,16,32],
        "reserved_reference_samples": 64, "groups": list(descriptor["groups"]),
        "configs": {g: dict(info["source_config"], sample_layers=info["sample_layers"],
                           reserved_reference_samples=64) for g,info in descriptor["groups"].items()}}
    api.objects["dataset/sample_layers_lock.json"] = json.dumps(layer_lock).encode()
    for group, info in descriptor["groups"].items():
        records = {}
        for category, prefix in (("base", ""), ("layers", "sample_layers/")):
            for path in (root/category/group).rglob("*"):
                if path.is_file():
                    key = prefix+path.relative_to(root/category).as_posix()
                    data = path.read_bytes()
                    api.objects["dataset/"+key] = data
                    records[key] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        receipt = {"format": "wavefront-r2-scene-archive", "version": 1, "status": "verified",
            "scene": info["scene"], "destination": "r2://bucket/dataset",
            "r2_endpoint": "https://"+"a"*32+".r2.cloudflarestorage.com",
            "capture_lock_sha256": digest(v4/"capture_lock.json"), "files": records}
        api.objects[f"dataset/r2-archives/{info['scene']}.json"] = json.dumps(receipt).encode()
    return root, api


def test_inline_archive_normalizes_without_copying_images(inline_cloud, tmp_path):
    root, api = inline_cloud
    snapshot = tmp_path/"inline"
    prepare_r2_metadata("r2://bucket/dataset", snapshot, layout="inline")
    assert not any(u.endswith(".exr") for _,u in api.calls)
    source = R2FrameSource(snapshot, "r2://bucket/dataset", layout="inline")
    source.preflight()
    local = JointPatchDataset(root, partition="train", patch_size=5, augment=False)[0]
    streamed = JointPatchDataset(snapshot, partition="train", patch_size=5, augment=False, frame_source=source)[0]
    torch.testing.assert_close(local["layers"], streamed["layers"], rtol=0, atol=0)
    assert any("sample_layers/" in u and u.endswith("layers.exr") for _,u in api.calls)
    assert api.uploads == 0


def test_inline_requires_real_layers_and_current_receipt(inline_cloud, tmp_path):
    _, api = inline_cloud
    key = "dataset/sample_layers_lock.json"
    original = api.objects[key]
    lock = json.loads(original)
    lock["groups"] = []
    api.objects[key] = json.dumps(lock).encode()
    with pytest.raises(ValueError, match="Missing or incompatible"):
        prepare_r2_metadata("r2://bucket/dataset", tmp_path/"no-layers", layout="inline")
    api.objects[key] = original
    receipt_key = next(k for k in api.objects if k.startswith("dataset/r2-archives/"))
    receipt = json.loads(api.objects[receipt_key])
    receipt["capture_lock_sha256"] = "f"*64
    api.objects[receipt_key] = json.dumps(receipt).encode()
    with pytest.raises(ValueError, match="current capture lock"):
        prepare_r2_metadata("r2://bucket/dataset", tmp_path/"old-lock", layout="inline")


@pytest.fixture
def backfill_cloud(inline_cloud):
    root, api = inline_cloud
    descriptor = read_json(root/"dataset.json")
    group = descriptor["partitions"]["train"][0]
    key = "dataset/sample_layers_lock.json"
    lock = json.loads(api.objects[key])
    config = lock["configs"].pop(group)
    lock["groups"].remove(group)
    api.objects[key] = json.dumps(lock).encode()
    config_key = f"sample_layer_configs/{group}.json"
    data = json.dumps(config).encode()
    api.objects["dataset/"+config_key] = data
    receipt_key = f"dataset/r2-archives/{descriptor['groups'][group]['scene']}.json"
    receipt = json.loads(api.objects[receipt_key])
    receipt["files"][config_key] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    api.objects[receipt_key] = json.dumps(receipt).encode()
    return root, api, group, receipt_key


def test_v4_backfill_outside_original_boundary_streams_without_v5_dataset(backfill_cloud, tmp_path):
    root, api, group, _ = backfill_cloud
    original_lock = api.objects["dataset/sample_layers_lock.json"]
    snapshot = tmp_path/"backfilled-v4"
    assert "dataset/dataset.json" not in api.objects
    prepare_r2_metadata("r2://bucket/dataset", snapshot, layout="inline")
    assert not any(u.endswith(".exr") for _,u in api.calls)
    assert api.objects["dataset/sample_layers_lock.json"] == original_lock
    source = R2FrameSource(snapshot, "r2://bucket/dataset", layout="inline")
    source.preflight()
    local = JointPatchDataset(root, partition="train", patch_size=5, augment=False)[0]
    streamed = JointPatchDataset(snapshot, partition="train", patch_size=5, augment=False, frame_source=source)[0]
    torch.testing.assert_close(local["layers"], streamed["layers"], rtol=0, atol=0)
    torch.testing.assert_close(streamed["layers"][:,2], streamed["buffers"]["radiance"], rtol=0, atol=0)
    assert api.uploads == 0


@pytest.mark.parametrize("problem", ["unreceipted", "config", "integrator"])
def test_v4_backfill_requires_bound_config_and_matching_integrator(backfill_cloud, tmp_path, problem):
    _, api, group, receipt_key = backfill_cloud
    receipt = json.loads(api.objects[receipt_key])
    config_key = f"sample_layer_configs/{group}.json"
    if problem == "unreceipted":
        del receipt["files"][config_key]
        expected_error = "backfill config required"
    else:
        key = config_key if problem == "config" else f"sample_layers/{group}/frame_000000/manifest.json"
        value = json.loads(api.objects["dataset/"+key])
        if problem == "config":
            value["sample_layers"][0]["offset"] += 1
            expected_error = "Backfill sample-layer config"
        else:
            value["render"]["integrator_version"] += 1
            expected_error = "integrator/render settings"
        data = json.dumps(value).encode()
        api.objects["dataset/"+key] = data
        receipt["files"][key] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    api.objects[receipt_key] = json.dumps(receipt).encode()
    with pytest.raises(ValueError, match=expected_error):
        prepare_r2_metadata("r2://bucket/dataset", tmp_path/problem, layout="inline")
    assert not (tmp_path/problem).exists()
    assert not any(u.endswith(".exr") for _,u in api.calls)


def test_download_retries_from_zero_and_limits_memory(r2_api, tmp_path, monkeypatch):
    from wavefront_denoiser import r2_native
    client = R2NativeClient("bucket", "dataset")
    client.preflight()
    data = b"x"*(2*1024*1024+3)
    r2_api.objects["dataset/frame.exr"] = data
    original = r2_native.urlopen
    reads, attempts = [], []
    class Interrupted(BytesIO):
        headers = {"Content-Length": str(len(data))}
        def read(self, size=-1):
            reads.append(size)
            assert size == 1024*1024
            if self.tell() > 0:
                raise OSError("connection interrupted")
            return super().read(size)
    def request(req, timeout):
        attempts.append(req.full_url)
        return Interrupted(data) if len(attempts) == 1 else original(req, timeout)
    monkeypatch.setattr(r2_native, "urlopen", request)
    target = tmp_path/"frame.exr"
    result = client.download_file("frame.exr", target, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    assert target.read_bytes() == data
    assert result["size"] == len(data) and len(attempts) == 2
    assert not list(tmp_path.glob("*.partial"))
    target.write_bytes(b"keep-existing")
    with pytest.raises(ValueError, match="size"):
        client.download_file("frame.exr", target, max_bytes=10)
    assert target.read_bytes() == b"keep-existing"
    assert not list(tmp_path.glob("*.partial"))
