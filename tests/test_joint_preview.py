import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from test_r2_streaming import decoded_cloud, inline_cloud, cloud, r2_api
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import save_joint_checkpoint
from wavefront_denoiser.joint_data import joint_inventory


@pytest.fixture
def preview_module():
    path = Path(__file__).resolve().parents[1]/"tools/preview_joint_pilot.py"
    spec = importlib.util.spec_from_file_location("preview_joint_pilot",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["v5","decoded"])
def preview_data(request):
    return (request.getfixturevalue("joint_dataset") if request.param == "v5"
            else request.getfixturevalue("decoded_cloud")[0])


def test_joint_preview_exports_aligned_recurrent_validation_crop(preview_data,tmp_path,preview_module,monkeypatch):
    torch.set_num_threads(1)
    _, config = config_for(preview_data)
    run = tmp_path/"run"
    run.mkdir()
    model = JointNTASAD(4,4,allocation_policy=config["allocation_policy"])
    inventory = joint_inventory(preview_data)
    save_joint_checkpoint(run/"last.pt",model,config=config,dataset_split=inventory,epoch=2,global_step=8)
    original = JointNTASAD.forward_sequence
    def guarded(self,buffers,layers,**kwargs):
        assert "reference" not in buffers and kwargs.get("training_reference") is None
        return original(self,buffers,layers,**kwargs)
    monkeypatch.setattr(JointNTASAD,"forward_sequence",guarded)
    result = preview_module.export_preview(run,device_name="cpu")
    assert result["frame_indices"] == [0,1,2,3,4]
    assert result["crop"] == {"top":1,"left":2,"size":5}
    assert result["checkpoint_global_step"] == 8
    assert not result["reference_supplied_to_prediction"] and not result["test_pixels_loaded"]
    assert Image.open(run/"preview/comparison.png").size == (20,29)
    assert (run/"preview/denoised.exr").is_file()
    assert np.asarray(Image.open(run/"preview/denoised.png")).shape == (5,5,3)
    assert json.loads((run/"preview/preview.json").read_text())["mode"] == "validation_center_crop"
    with pytest.raises(FileExistsError):
        preview_module.export_preview(run,device_name="cpu")
    test_group = inventory["partitions"]["test"][0]
    with pytest.raises(ValueError,match="validation clip"):
        preview_module.export_preview(run,output=tmp_path/"test-preview",sequence_id=test_group,device_name="cpu")
    assert not (tmp_path/"test-preview").exists()
    category = "decoded" if (preview_data/"decoded_cache_lock.json").exists() else "frames"
    record = preview_data/category/result["sequence_id"]/"frame_000000/manifest.json"
    record.write_text(record.read_text()+"\n")
    with pytest.raises(ValueError,match="provenance"):
        preview_module.export_preview(run,output=tmp_path/"changed-preview",device_name="cpu")
    assert not (tmp_path/"changed-preview").exists()
