from copy import deepcopy
import hashlib
import json
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
import pytest

torch = pytest.importorskip("torch")

from test_adaptive_dataset import source_dataset, fake_layers
from wavefront_denoiser.adaptive_data import read_json, read_v5_dataset
from wavefront_denoiser.adaptive_dataset import capture_v5
from wavefront_denoiser.dataset_generation import digest
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset, joint_inventory
from wavefront_denoiser.joint_training import train_joint
from wavefront_denoiser.training_defaults import joint_training_config
from wavefront_denoiser.cli import main


def rewrite_exr(directory,filename,delta):
    path = directory/filename
    source = OpenEXR.InputFile(str(path))
    header = source.header()
    values = {}
    for name,info in header["channels"].items():
        dtype = np.uint32 if info.type.v == Imath.PixelType.UINT else np.float32
        values[name] = np.frombuffer(source.channel(name,info.type),dtype=dtype).copy()
        if name.endswith(".R") or name.endswith(".G") or name.endswith(".B"):
            values[name] += delta + np.arange(values[name].size,dtype=np.float32)*.001
    source.close()
    output = OpenEXR.OutputFile(str(path),header)
    output.writePixels({key:value.tobytes() for key,value in values.items()})
    output.close()


@pytest.fixture
def joint_dataset(source_dataset,monkeypatch):
    _,source,output,renderer,configs = source_dataset
    groups = list(configs)
    for ordinal,group in enumerate(groups):
        for index in range(8):
            directory = source/group/f"frame_{index:06d}"
            rewrite_exr(directory,"frame.exr",ordinal+index*.02)
            manifest = read_json(directory/"manifest.json")
            manifest["scene_fingerprint"] = format(ordinal+1,"x")
            manifest["image_sha256"] = digest(directory/"frame.exr")
            (directory/"manifest.json").write_text(json.dumps(manifest))
    def render(command,**kwargs):
        config = read_json(Path(command[1]))
        group = config["sequence_id"]
        for index in range(int(command[-3]),int(command[-1])):
            relative = Path(group)/f"frame_{index:06d}"
            metadata = read_json(output/"base"/relative/"manifest.json")
            directory = output/"layers"/relative
            fake_layers(directory,metadata,config["sample_layers"])
            rewrite_exr(directory,"layers.exr",groups.index(group)*.7+index*.03)
            manifest = read_json(directory/"manifest.json")
            manifest["image_sha256"] = digest(directory/"layers.exr")
            (directory/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("wavefront_denoiser.adaptive_dataset.subprocess.run",render)
    capture_v5(output,source,renderer,source.parent,groups[:2])
    return output


def config_for(root,**options):
    config = joint_training_config(str(root))
    config.update(epochs=2,batch_size=1,patch_size=5,patches_per_window=1,
        validation_patch_size=5,validation_crops_per_window=1,cpu_threads=1,
        denoiser_channels=4,sampler_channels=4)
    config.update(options)
    path = root.parent/"joint-train.json"
    path.write_text(json.dumps(config))
    return path,config


def test_joint_crops_keep_layer4_and_guides_aligned(joint_dataset):
    torch.manual_seed(7)
    dataset = JointPatchDataset(joint_dataset,partition="train",patch_size=5)
    sample = dataset[0]
    torch.testing.assert_close(sample["layers"][:,2],sample["buffers"]["radiance"],rtol=0,atol=0)
    assert sample["layers"].shape == (5,6,3,5,5)
    assert (sample["buffers"]["history_valid"][0] == 0).all()
    state = torch.get_rng_state().clone()
    validation = JointPatchDataset(joint_dataset,partition="validation",patch_size=5,augment=False)
    first = validation[0]
    torch.testing.assert_close(first["layers"],validation[0]["layers"],rtol=0,atol=0)
    assert torch.equal(state,torch.get_rng_state())


def test_two_epochs_equal_one_epoch_plus_resume_cpu(joint_dataset,tmp_path):
    path,config = config_for(joint_dataset)
    full = train_joint(path,tmp_path/"full","cpu")
    config["epochs"] = 1
    path.write_text(json.dumps(config))
    assert main(["train-adaptive",str(path),str(tmp_path/"first"),"--device","cpu"]) == 0
    config["epochs"] = 2
    path.write_text(json.dumps(config))
    resumed = train_joint(path,tmp_path/"resumed","cpu",resume=tmp_path/"first/last.pt")
    assert full["global_step"] == resumed["global_step"] == 8
    a,payload_a = load_joint_checkpoint(tmp_path/"full/last.pt")
    b,payload_b = load_joint_checkpoint(tmp_path/"resumed/last.pt")
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
    for key in ("torch","generator"):
        assert torch.equal(payload_a["training"]["rng"][key],payload_b["training"]["rng"][key])
    history = [json.loads(line) for line in (tmp_path/"full/history.jsonl").read_text().splitlines()]
    assert history[0]["train"]["sampler_gradient_norm"] > 0
    assert history[0]["train"]["denoiser_gradient_norm"] > 0
    assert full["test_pixels_loaded"] is False


def test_joint_inventory_never_opens_reserved_test_bytes(joint_dataset,monkeypatch):
    root = joint_dataset
    test_group = read_v5_dataset(root)["partitions"]["test"][0]
    original = Path.open
    def guarded(self,*args,**kwargs):
        if test_group in self.parts and self.suffix == ".exr":
            raise AssertionError("Reserved test pixels were opened")
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    audit = joint_inventory(root)
    assert audit["test_pixels_loaded"] is False
    with pytest.raises(ValueError,match="partition"):
        JointPatchDataset(root,partition="train",sequence_ids=[test_group],patch_size=5)


def test_joint_reader_rejects_capture_writer_and_restores_flags(joint_dataset,tmp_path):
    from wavefront_denoiser.dataset_generation import capture_lock
    path,_ = config_for(joint_dataset,epochs=1)
    before = torch.are_deterministic_algorithms_enabled()
    with capture_lock(joint_dataset):
        with pytest.raises(RuntimeError,match="capture writer"):
            train_joint(path,tmp_path/"locked","cpu")
    assert torch.are_deterministic_algorithms_enabled() == before
    assert not (tmp_path/"locked").exists()


def test_resume_rejects_changed_policy_before_output(joint_dataset,tmp_path):
    path,config = config_for(joint_dataset,epochs=1)
    train_joint(path,tmp_path/"first","cpu")
    config.update(epochs=2,zero_policy="zero")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="changed"):
        train_joint(path,tmp_path/"wrong","cpu",resume=tmp_path/"first/last.pt")
    assert not (tmp_path/"wrong").exists()


def test_legacy_config_resume_requires_explicit_reconstruction(joint_dataset,tmp_path):
    from wavefront_denoiser.joint_checkpoint import LEGACY_IDENTITY
    path,config = config_for(joint_dataset,epochs=1,reconstruction_version="sample_mass_v1")
    train_joint(path,tmp_path/"old","cpu")
    checkpoint = tmp_path/"old/last.pt"
    payload = torch.load(checkpoint,weights_only=True)
    payload.update(LEGACY_IDENTITY)
    payload["config"].pop("reconstruction_version")
    payload["model_spec"].pop("reconstruction_version")
    torch.save(payload,checkpoint)
    config.update(epochs=2,reconstruction_version="pixel_mean_v2")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="changed"):
        train_joint(path,tmp_path/"wrong","cpu",resume=checkpoint)
    assert not (tmp_path/"wrong").exists()
    config["reconstruction_version"] = "sample_mass_v1"
    path.write_text(json.dumps(config))
    result = train_joint(path,tmp_path/"legacy-resumed","cpu",resume=checkpoint)
    assert result["global_step"] == 8


def test_resume_rejects_changed_inventory_before_output(joint_dataset,tmp_path):
    path,config = config_for(joint_dataset,epochs=1)
    train_joint(path,tmp_path/"first","cpu")
    config["epochs"] = 2
    path.write_text(json.dumps(config))
    descriptor = joint_dataset/"dataset.json"
    descriptor.write_text(descriptor.read_text()+"\n")
    with pytest.raises(ValueError,match="inventory changed"):
        train_joint(path,tmp_path/"wrong","cpu",resume=tmp_path/"first/last.pt")
    assert not (tmp_path/"wrong").exists()


def test_cross_partition_scene_identity_is_rejected(joint_dataset):
    root = joint_dataset
    descriptor = read_v5_dataset(root)
    group = descriptor["partitions"]["validation"][0]
    for index in range(8):
        relative = Path(group)/f"frame_{index:06d}"
        for kind in ("base","layers"):
            path = root/kind/relative/"manifest.json"
            value = read_json(path)
            value["scene_fingerprint"] = "1"
            path.write_text(json.dumps(value))
        record = root/"frames"/relative/"manifest.json"
        value = read_json(record)
        value["source_manifest_sha256"] = digest(root/"base"/relative/"manifest.json")
        value["layer_manifest_sha256"] = digest(root/"layers"/relative/"manifest.json")
        record.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="Scene identity"):
        joint_inventory(root)


def test_joint_evaluation_prediction_is_reference_free(joint_dataset):
    torch.set_num_threads(1)
    sample = JointPatchDataset(joint_dataset,partition="train",patch_size=5,augment=False)[0]
    buffers = {k:v.unsqueeze(0) for k,v in sample["buffers"].items()}
    layers = sample["layers"].unsqueeze(0)
    model = JointNTASAD(4,4).eval()
    with torch.no_grad():
        first = model.forward_sequence(buffers,layers)
        second = model.forward_sequence(dict(buffers,reference=torch.full_like(buffers["reference"],float("nan"))),layers)
    torch.testing.assert_close(first.radiance,second.radiance,rtol=0,atol=0)
    assert first.counts.max() <= 63
