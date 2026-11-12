"""Crop-aware two-network NTASAD training with epoch-boundary resume."""

import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .adaptive_data import read_json, read_v5_dataset
from .adaptive_model import RECONSTRUCTION_VERSIONS
from .device import select_device
from .joint import JointNTASAD
from .joint_checkpoint import joint_optimizer, load_joint_checkpoint, restore_joint_training, save_joint_checkpoint
from .joint_data import JointPatchDataset, joint_inventory, joint_dataset_read_lock
from .loss import sequence_loss
from .model import INPUT_CHANNELS
from .training_defaults import joint_training_config


def training_config(path):
    path = Path(path).resolve()
    root = read_json(path)
    defaults = joint_training_config()
    if (root.get("format") != defaults["format"] or type(root.get("version")) is not int or root["version"] != 1
            or root.keys()-(defaults.keys() | {"r2_streaming"})):
        raise ValueError("Invalid or unknown joint training options; use train-joint.json, not the historical plan")
    config = dict(defaults,**root)
    if not isinstance(config["data_root"],str) or not config["data_root"]:
        raise ValueError("Joint training needs data_root")
    config["data_root"] = str((path.parent/config["data_root"]).resolve())
    if "r2_streaming" in config:
        streaming = config["r2_streaming"]
        if (not isinstance(streaming, dict) or set(streaming)-{"source", "layout", "spool_directory"}
                or not isinstance(streaming.get("source"), str)
                or streaming.get("layout", "v5") not in ("v5", "inline")):
            raise ValueError("r2_streaming needs source r2://BUCKET/PREFIX and layout v5 or inline")
        from .r2_archive import R2SceneArchiver
        streaming["source"] = R2SceneArchiver(Path(config["data_root"]), streaming["source"]).destination
        streaming.setdefault("layout", "v5")
        if "spool_directory" in streaming:
            if not isinstance(streaming["spool_directory"], str) or not streaming["spool_directory"]:
                raise ValueError("R2 spool_directory must be a nonempty local path")
            streaming["spool_directory"] = str((path.parent/streaming["spool_directory"]).resolve())
    for key in ("epochs","batch_size","patch_size","patches_per_window","window_length",
                "validation_patch_size","validation_crops_per_window","cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["window_length"] < 3 or config["validation_crops_per_window"] > 5:
        raise ValueError("Need at least three frames and one to five validation crops")
    if type(config["seed"]) is not int or config["seed"] < 0 or type(config["num_workers"]) is not int or config["num_workers"] != 0:
        raise ValueError("Need a nonnegative seed; joint exact-resume currently requires num_workers=0")
    if type(config["learning_rate"]) not in (int,float) or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for key in ("train_groups","validation_groups"):
        names = config[key]
        if names is not None and (not isinstance(names,list) or not names or any(not isinstance(g,str) or not g for g in names)
                                  or len(set(names)) != len(names)):
            raise ValueError(f"{key} must be null or unique explicit group IDs")
    # Validate the model specification without initializing random weights or
    # changing RNG before dataset/provenance checks.
    from .allocation import POLICIES
    from .renderer_gradient import ZERO_POLICIES
    if (any(type(config[k]) is not int or not 4 <= config[k] <= 128 for k in ("denoiser_channels","sampler_channels"))
            or type(config["average_spp"]) is not int or type(config["max_spp"]) is not int
            or not 1 <= config["average_spp"] <= config["max_spp"] <= 63
            or config["allocation_policy"] not in POLICIES or config["zero_policy"] not in ZERO_POLICIES
            or config["reconstruction_version"] not in RECONSTRUCTION_VERSIONS):
        raise ValueError("Invalid joint widths, sample capacity or method policies")
    return config


def _batch(batch,device):
    return ({key:batch["buffers"][key].to(device) for key in INPUT_CHANNELS},
            batch["layers"].to(device),batch["buffers"]["reference"].to(device))


def _metrics(prediction,reference,max_spp):
    metrics = sequence_loss(prediction.radiance,reference,transformed=True)
    actual = prediction.counts[:,1:].to(torch.float32)
    metrics.update(actual_spp=actual.mean(),expected_spp=prediction.expected[:,1:].mean(),
                   zero_fraction=(actual==0).to(torch.float32).mean(),
                   saturated_fraction=(actual==max_spp).to(torch.float32).mean(),
                   unavailable_fraction=(~prediction.valid[:,1:]).to(torch.float32).mean())
    return metrics


@torch.inference_mode()
def evaluate_joint(model,dataset,device):
    model.eval()
    totals = {}
    paths = pixels = 0
    loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,generator=torch.Generator().manual_seed(0))
    for batch in loader:
        buffers,layers,reference = _batch(batch,device)
        # References stay outside prediction during evaluation.
        prediction = model.forward_sequence(buffers,layers,transformed=True)
        for key,value in _metrics(prediction,reference,model.max_spp).items():
            totals[key] = totals.get(key,0.)+float(value)
        paths += int(prediction.counts.sum())
        pixels += prediction.counts.numel()
    return {**{k:v/len(dataset) for k,v in totals.items()},"replayed_paths":paths,"replayed_pixels":pixels}


def _train_joint(config,output,device_name="auto",*,resume=None,frame_source=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Joint training output already exists: {output}")
    descriptor = read_v5_dataset(config["data_root"])
    selected = {p:config[p+"_groups"] or descriptor["partitions"][p] for p in ("train","validation")}
    provenance = joint_inventory(config["data_root"],selected,verify_images=frame_source is None)
    if frame_source is not None:
        from .r2_streaming import SOURCE_RECORD
        from .dataset_generation import digest
        source_record = Path(config["data_root"])/SOURCE_RECORD
        if provenance != read_json(source_record)["inventory"]:
            raise ValueError("R2 metadata inventory/selections changed; prepare a new snapshot")
        provenance["r2_snapshot_sha256"] = digest(source_record)
        provenance["image_verification"] = "sha256_on_every_download"
    train = JointPatchDataset(config["data_root"],partition="train",sequence_ids=selected["train"],
        length=config["window_length"],patch_size=config["patch_size"],patches_per_window=config["patches_per_window"],
        frame_source=frame_source)
    validation = JointPatchDataset(config["data_root"],partition="validation",sequence_ids=selected["validation"],
        length=config["window_length"],patch_size=config["validation_patch_size"],
        patches_per_window=config["validation_crops_per_window"],augment=False,frame_source=frame_source)
    device = select_device(device_name)
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config["seed"])
    keys = ("denoiser_channels","sampler_channels","average_spp","max_spp","allocation_policy","zero_policy",
            "reconstruction_version")
    epoch0 = global_step = 0
    if resume is None:
        model = JointNTASAD(**{key:config[key] for key in keys}).to(device)
        payload = None
    else:
        model,payload = load_joint_checkpoint(resume,device)
        # Missing policy in historical configs means sample-mass v1, never the
        # new default. Continuing old training requires selecting it explicitly.
        comparable = lambda c:{k:v for k,v in {"reconstruction_version":"sample_mass_v1",**c}.items() if k != "epochs"}
        if (comparable(config) != comparable(payload["config"]) or provenance != payload["dataset_split"]
                or model.specification() != {k:config[k] for k in keys}):
            raise ValueError("Joint resume configuration, policies or dataset inventory changed")
        if payload["epoch"] >= config["epochs"]:
            raise ValueError("Joint resume needs a larger total target epoch count")
    optimizer = joint_optimizer(model,config["learning_rate"])
    if payload is not None:
        progress = restore_joint_training(model,optimizer,payload,generator=generator)
        epoch0,global_step = progress["epoch"],progress["global_step"]
    loader = torch.utils.data.DataLoader(train,batch_size=config["batch_size"],shuffle=True,
                                        num_workers=0,generator=generator)
    initial = evaluate_joint(model,validation,device)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (("config.json",config),("inventory.json",provenance)):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    best_loss,best_epoch,best = initial["loss"],epoch0,initial
    def save(name,epoch):
        save_joint_checkpoint(output/name,model,optimizer=optimizer,epoch=epoch,global_step=global_step,
                              config=config,dataset_split=provenance,generator=generator)
    save("model.pt",epoch0)
    for epoch in range(epoch0+1,config["epochs"]+1):
        model.train()
        totals = {}
        paths = pixels = windows = 0
        for batch in loader:
            buffers,layers,reference = _batch(batch,device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_sequence(buffers,layers,training_reference=reference,transformed=True)
            metrics = _metrics(prediction,reference,model.max_spp)
            metrics["loss"].backward()
            for name,network in (("sampler",model.sampler),("denoiser",model.denoiser)):
                gradients = [p.grad for p in network.parameters() if p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError(f"Non-finite {name} training gradient")
                metrics[name+"_gradient_norm"] = torch.stack([g.detach().norm() for g in gradients]).norm() if gradients else reference.new_tensor(0.)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Non-finite joint weights after optimizer update")
            global_step += 1
            count = reference.shape[0]
            windows += count
            for key,value in metrics.items():
                totals[key] = totals.get(key,0.)+float(value.detach())*count
            paths += int(prediction.counts.sum())
            pixels += prediction.counts.numel()
        measured = evaluate_joint(model,validation,device)
        record = {"epoch":epoch,"global_step":global_step,
            "train":{**{k:v/windows for k,v in totals.items()},"replayed_paths":paths,"replayed_pixels":pixels},
            "validation":measured}
        with (output/"history.jsonl").open("a") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
        print(json.dumps(record,allow_nan=False),flush=True)
        if measured["loss"] < best_loss:
            best_loss,best_epoch,best = measured["loss"],epoch,measured
            save("model.pt",epoch)
        save("last.pt",epoch)
    summary = {"device":str(device),"parameters":sum(p.numel() for p in model.parameters()),
        "completed_epochs":config["epochs"],"global_step":global_step,"best_epoch":best_epoch,
        "best_validation":best,"initial_validation":initial,"checkpoint":"model.pt","resume_checkpoint":"last.pt",
        "resumed_from":str(resume) if resume else None,"selection_scope":"this_run_including_initial_model",
        "config":config,"test_pixels_loaded":False}
    if frame_source is not None:
        summary["r2_streaming"] = dict(frame_source.stats)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def train_joint(config_path,output,device_name="auto",*,resume=None):
    """Train with a shared dataset lock and deterministic operator selection.

    Restore process-wide execution flags on exit. Checkpoints store the flags
    used during training, alongside optimizer/RNG state and dataset inventory.
    """
    config = training_config(config_path)
    if Path(output).exists():
        raise FileExistsError(f"Joint training output already exists: {output}")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark
    try:
        with joint_dataset_read_lock(config["data_root"]):
            frame_source = None
            if "r2_streaming" in config:
                from .r2_streaming import R2FrameSource
                frame_source = R2FrameSource(config["data_root"], **config["r2_streaming"])
                frame_source.preflight()
            torch.use_deterministic_algorithms(True)
            return _train_joint(config,output,device_name,resume=resume,frame_source=frame_source)
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn)
        torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark = cudnn
