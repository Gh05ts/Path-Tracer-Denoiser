import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from test_joint_preview import preview_data, preview_module
from test_r2_streaming import decoded_cloud, inline_cloud, cloud, r2_api
from wavefront_denoiser.adaptive_data import read_v5_dataset, V5SequenceDataset
from wavefront_denoiser.dataset_generation import digest
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
from wavefront_denoiser.joint_data import joint_inventory
from wavefront_denoiser.joint_training import train_joint, training_config
from wavefront_denoiser.model import INPUT_CHANNELS


def test_budget_stop_saves_final_validation_and_resumable_trained_models(joint_dataset,tmp_path):
    path,config = config_for(joint_dataset,epochs=100,max_training_seconds=.001,
        validation_every_epochs=100,validation_warmup_epochs=0,checkpoint_every_epochs=100)
    run = tmp_path/"budget"
    summary = train_joint(path,run,"cpu")
    assert summary["completed_epochs"] == 1
    assert summary["stop_reason"] == "time_budget_epoch_boundary"
    assert summary["best_trained_epoch"] == 1
    history = [json.loads(s) for s in (run/"history.jsonl").read_text().splitlines()]
    assert history[-1]["validation"] is not None
    assert history[-1]["timing"]["resume_checkpoint_saved"]
    for name,epoch in (("initial.pt",0),("best-trained.pt",1),("last.pt",1)):
        _,payload = load_joint_checkpoint(run/name)
        assert payload["epoch"] == epoch
        assert payload["training"]
    # A stopped run retains the configured ceiling for exact epoch-boundary
    # resume, but a new invocation gets a new time budget.
    again = train_joint(path,tmp_path/"continued","cpu",resume=run/"last.pt")
    assert again["completed_epochs"] == 2
    assert again["global_step"] == 2*summary["global_step"]


@pytest.mark.parametrize("value",[0,-1,True,float("nan"),float("inf"),"7200"])
def test_budget_rejects_invalid_values(tmp_path,value):
    path,_ = config_for(tmp_path,max_training_seconds=value)
    with pytest.raises(ValueError,match="max_training_seconds|Non-finite JSON"):
        training_config(path)


def test_split_overlay_pins_source_preserves_test_and_cannot_split_scene(joint_dataset):
    original = read_v5_dataset(joint_dataset)
    partitions = dict(original["partitions"],train=original["partitions"]["validation"],
                      validation=original["partitions"]["train"])
    split = {"format":"wavefront-experiment-split","version":1,
        "source_manifest_sha256":digest(joint_dataset/"dataset.json"),
        "source_partitions":original["partitions"],"partitions":partitions}
    path = joint_dataset/"experiment-split.json"
    path.write_text(json.dumps(split))
    inventory = joint_inventory(joint_dataset)
    assert inventory["experiment_split_sha256"] == digest(path)
    assert inventory["selected"]["train"] == original["partitions"]["validation"]
    assert inventory["partitions"]["test"] == original["partitions"]["test"]
    assert digest(joint_dataset/"dataset.json") == split["source_manifest_sha256"]
    split["partitions"]["test"] = []
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError,match="reserved test"):
        read_v5_dataset(joint_dataset)
    split["partitions"] = original["partitions"]
    split["source_manifest_sha256"] = "0"*64
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError,match="provenance"):
        read_v5_dataset(joint_dataset)


def test_native_preview_matches_whole_sequence_and_uniform_control_is_reference_free(preview_data,tmp_path,preview_module,monkeypatch):
    torch.set_num_threads(1)
    _,config = config_for(preview_data)
    run = tmp_path/"run"; run.mkdir()
    model = JointNTASAD(4,4).eval()
    inventory = joint_inventory(preview_data)
    group = inventory["selected"]["validation"][0]
    save_joint_checkpoint(run/"last.pt",model,config=config,dataset_split=inventory,epoch=1,global_step=4)
    sample = V5SequenceDataset(preview_data,partition="validation",sequence_ids=[group])[0]
    with torch.inference_mode():
        expected = model.forward_sequence({k:sample["buffers"][k][None] for k in INPUT_CHANNELS},sample["layers"][None])
    # Both the learned sampler and the fixed-denoiser uniform control must be
    # reference-free, including the whole-image low-level recurrence path.
    original = type(model.denoiser)._step
    def guarded(self,buffers,*args,**kwargs):
        assert "reference" not in buffers
        return original(self,buffers,*args,**kwargs)
    monkeypatch.setattr(type(model.denoiser),"_step",guarded)
    images,count,valid,animation,metrics = preview_module.full_frame_images(model,preview_data,group,0,5,"cpu",1.)
    actual = next(rgb for name,_,rgb in images if name == "denoised")
    np.testing.assert_array_equal(actual,expected.radiance[0,-1].numpy().transpose(1,2,0))
    torch.testing.assert_close(count,expected.counts[:,-1],rtol=0,atol=0)
    assert len(animation) == 5
    assert metrics["frames"][-1]["uniform_paths"] == 4*7*9
    result = preview_module.export_preview(run,full_frame=True,device_name="cpu")
    assert result["mode"] == "validation_full_frame" and result["crop"] is None
    assert Image.open(run/"preview/denoised.png").size == (9,7)
    assert Image.open(run/"preview/comparison.png").size == (45,31)
    assert (run/"preview/comparison.gif").is_file()
    assert np.load(run/"preview/sample-count.npy").shape == (7,9)
    assert result["joint_metrics"]["spatial_l1"] >= 0


def test_decoded_experiment_view_shares_arrays_and_keeps_source_inventory_intact(decoded_cloud,tmp_path):
    root,_ = decoded_cloud
    script = Path(__file__).parents[1]/"tools/run_quality_pilot.py"
    spec = importlib.util.spec_from_file_location("quality_runner",script)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    descriptor = read_v5_dataset(root)
    before = joint_inventory(root)
    train,validation = descriptor["partitions"]["train"],descriptor["partitions"]["validation"]
    plan = {"train_scenes":[descriptor["groups"][validation[0]]["scene"]],
            "validation_scenes":[descriptor["groups"][train[0]]["scene"]]}
    view = tmp_path/"view"
    runner.make_view(root,view,{"train_groups":validation,"validation_groups":train},plan)
    assert (view/"decoded").is_symlink()
    assert joint_inventory(root) == before
    after = joint_inventory(view)
    assert after["train"][0]["source_image_sha256"] == before["validation"][0]["source_image_sha256"]
    assert after["partitions"]["test"] == before["partitions"]["test"]
    assert "experiment_split_sha256" in after


def test_quality_preparation_reuses_verified_arrays_without_test_payloads(decoded_cloud,tmp_path):
    root,api = decoded_cloud
    script = Path(__file__).parents[1]/"tools/run_quality_pilot.py"
    spec = importlib.util.spec_from_file_location("quality_prepare",script)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    descriptor = read_v5_dataset(root)
    train,validation = descriptor["partitions"]["train"],descriptor["partitions"]["validation"]
    config = {"train_groups":validation,"validation_groups":train}
    plan = {"source":"r2://bucket/dataset","reuse_arrays_from":str(root),
            "train_scenes":[descriptor["groups"][validation[0]]["scene"]],
            "validation_scenes":[descriptor["groups"][train[0]]["scene"]]}
    base = tmp_path/"quality"; base.mkdir()
    runner.prepare(base,config,plan)
    report = json.loads((base/"materialization.json").read_text())
    assert report["downloaded_objects"] == 0 and report["reused_objects"] > 0
    assert (base/"source-data/decoded"/train[0]/"frame_000000/float.npy").is_symlink()
    assert joint_inventory(base/"data")["selected"]["train"] == validation
    assert not (root/"experiment-split.json").exists()
    assert not any(url.endswith((".npy",".exr")) for _,url in api.calls)
    # Completed metadata/download preparation can be rerun without rewriting
    # source provenance or creating additional array copies.
    runner.prepare(base,config,plan)
    assert json.loads((base/"materialization.json").read_text())["downloaded_objects"] == 0
