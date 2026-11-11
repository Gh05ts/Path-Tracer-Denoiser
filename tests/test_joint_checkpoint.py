import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import (
    joint_optimizer, load_joint_checkpoint, restore_joint_training, save_joint_checkpoint,
)
from wavefront_denoiser.checkpoint import load_checkpoint


def test_joint_checkpoint_roundtrip_optimizer_and_rng(tmp_path):
    torch.manual_seed(11); random.seed(11); np.random.seed(11)
    model = JointNTASAD(4,4)
    optimizer = joint_optimizer(model)
    generator = torch.Generator().manual_seed(12)
    # Populate Adam moments for every parameter of both networks.
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    path = tmp_path/"joint.pt"
    save_joint_checkpoint(path,model,optimizer=optimizer,epoch=1,global_step=3,
                          config=model.specification(),dataset_split={"fixture":True},generator=generator)
    random_expected = (random.random(),np.random.rand(),torch.rand(2),torch.rand(2,generator=generator))
    restored,payload = load_joint_checkpoint(path)
    restored_optimizer = joint_optimizer(restored)
    restored_generator = torch.Generator()
    restore_joint_training(restored,restored_optimizer,payload,generator=restored_generator)
    assert random.random() == random_expected[0] and np.random.rand() == random_expected[1]
    assert torch.equal(torch.rand(2),random_expected[2])
    assert torch.equal(torch.rand(2,generator=restored_generator),random_expected[3])
    for original,loaded in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(original,loaded,rtol=0,atol=0)
    # Same next update, including both networks' Adam moments and groups.
    for m,o in ((model,optimizer),(restored,restored_optimizer)):
        o.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in m.parameters()).backward()
        o.step()
    for original,loaded in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(original,loaded,rtol=0,atol=0)
    with pytest.raises(ValueError,match="format"):
        load_checkpoint(path,torch.device("cpu"))


@pytest.mark.parametrize("field,value", [("version",3),("gradient_version","other"),
    ("feature_version",2),("architecture","uniform"),("warp_version","grid_sample"),("layer_spp",[1,2,4])])
def test_joint_checkpoint_rejects_incompatible_semantics(tmp_path,field,value):
    path = tmp_path/"joint.pt"
    save_joint_checkpoint(path,JointNTASAD(4,4))
    payload = torch.load(path,weights_only=True)
    payload[field] = value
    torch.save(payload,path)
    with pytest.raises(ValueError,match="Incompatible"):
        load_joint_checkpoint(path)


def test_weights_only_and_wrong_optimizer_cannot_resume(tmp_path):
    model = JointNTASAD(4,4)
    path = tmp_path/"joint.pt"
    save_joint_checkpoint(path,model)
    loaded,payload = load_joint_checkpoint(path)
    with pytest.raises(ValueError,match="resumable"):
        restore_joint_training(loaded,joint_optimizer(loaded),payload)
    save_joint_checkpoint(path,model,optimizer=joint_optimizer(model))
    loaded,payload = load_joint_checkpoint(path)
    with pytest.raises(ValueError,match="ordering"):
        restore_joint_training(loaded,torch.optim.Adam(loaded.parameters()),payload)


def test_joint_partial_write_preserves_old_checkpoint(tmp_path,monkeypatch):
    path = tmp_path/"joint.pt"
    model = JointNTASAD(4,4)
    save_joint_checkpoint(path,model)
    before = path.read_bytes()
    def fail(payload,stream):
        stream.write(b"partial")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(torch,"save",fail)
    with pytest.raises(RuntimeError,match="interrupted"):
        save_joint_checkpoint(path,model)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.partial"))


def test_nonfinite_joint_weights_and_optimizer_are_rejected(tmp_path):
    model = JointNTASAD(4,4)
    optimizer = joint_optimizer(model)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    first = next(iter(optimizer.state.values()))
    first["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError,match="optimizer state"):
        save_joint_checkpoint(tmp_path/"bad.pt",model,optimizer=optimizer)
    first["exp_avg"].zero_()
    with torch.no_grad():
        next(model.parameters()).fill_(float("inf"))
    with pytest.raises(ValueError,match="weights"):
        save_joint_checkpoint(tmp_path/"bad.pt",model)
    assert not (tmp_path/"bad.pt").exists()


def test_original_v1_checkpoint_retains_legacy_reconstruction_and_resume(tmp_path):
    from wavefront_denoiser.joint_checkpoint import LEGACY_IDENTITY
    model = JointNTASAD(4,4,reconstruction_version="sample_mass_v1")
    optimizer = joint_optimizer(model)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    path = tmp_path/"legacy.pt"
    save_joint_checkpoint(path,model,optimizer=optimizer,epoch=2,global_step=3)
    # Exact schema produced before this audit, with no reconstruction policy.
    payload = torch.load(path,weights_only=True)
    payload.update(LEGACY_IDENTITY)
    payload["model_spec"].pop("reconstruction_version")
    torch.save(payload,path)
    loaded,payload = load_joint_checkpoint(path)
    assert loaded.denoiser.reconstruction_version == "sample_mass_v1"
    assert loaded.specification() == model.specification()
    restored_optimizer = joint_optimizer(loaded)
    assert restore_joint_training(loaded,restored_optimizer,payload) == {"epoch":2,"global_step":3}
    for m,o in ((model,optimizer),(loaded,restored_optimizer)):
        o.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in m.parameters()).backward()
        o.step()
    for original,restored in zip(model.parameters(),loaded.parameters()):
        torch.testing.assert_close(original,restored,rtol=0,atol=0)
    # Saving a legacy model in the new envelope still preserves its semantics.
    save_joint_checkpoint(tmp_path/"resaved.pt",loaded)
    resaved,new_payload = load_joint_checkpoint(tmp_path/"resaved.pt")
    assert new_payload["version"] == 2
    assert resaved.denoiser.reconstruction_version == "sample_mass_v1"
    with pytest.raises(ValueError,match="differ"):
        restore_joint_training(JointNTASAD(4,4),joint_optimizer(JointNTASAD(4,4)),payload)


def test_checkpoint_reconstruction_identity_must_match_specification(tmp_path):
    path = tmp_path/"joint.pt"
    save_joint_checkpoint(path,JointNTASAD(4,4))
    payload = torch.load(path,weights_only=True)
    payload["model_spec"]["reconstruction_version"] = "sample_mass_v1"
    torch.save(payload,path)
    with pytest.raises(ValueError,match="Incompatible"):
        load_joint_checkpoint(path)
