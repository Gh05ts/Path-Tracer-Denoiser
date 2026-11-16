#!/usr/bin/env python3
"""Export a recurrent validation crop from the joint sampler/denoiser checkpoint."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from wavefront_denoiser.adaptive_data import read_v5_dataset, verify_v5_record
from wavefront_denoiser.device import select_device
from wavefront_denoiser.image_io import write_rgb_exr
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset, joint_dataset_read_lock
from wavefront_denoiser.model import INPUT_CHANNELS
from wavefront_denoiser.preview import display_rgb
from wavefront_denoiser.sample_layers import replay_sample_layers
from wavefront_denoiser.dataset_generation import digest


@torch.inference_mode()
def export_preview(run, output=None, *, checkpoint="last.pt", sequence_id=None, window=0,
                   patch_size=None, device_name="auto", exposure=1.):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run/"preview"
    if output.exists():
        raise FileExistsError(f"Preview output already exists: {output}")
    device = select_device(device_name)
    model, payload = load_joint_checkpoint(run/checkpoint, device)
    config = payload["config"]
    root = Path(config["data_root"])
    selected = payload["dataset_split"].get("selected", {}).get("validation", [])
    group = sequence_id or (selected[0] if selected else None)
    if group not in selected:
        raise ValueError("Preview needs a validation clip from this checkpoint's selected inventory")
    with joint_dataset_read_lock(root):
        descriptor = read_v5_dataset(root)
        manifest = "decoded_cache_lock.json" if descriptor.get("storage") == "decoded" else "dataset.json"
        if digest(root/manifest) != payload["dataset_split"]["dataset_manifest_sha256"]:
            raise ValueError("Preview dataset metadata differs from the checkpoint inventory")
        size = config["validation_patch_size"] if patch_size is None else patch_size
        dataset = JointPatchDataset(root, partition="validation", sequence_ids=[group],
            length=config["window_length"], patch_size=size, patches_per_window=1, augment=False)
        if type(window) is not int or not 0 <= window < len(dataset):
            raise ValueError("Preview window is outside the selected clip")
        group, start, length = dataset.sequences.windows[window]
        frozen = {(r["sequence_id"],r["frame_index"]):r for r in payload["dataset_split"]["validation"]}
        record_root = root/("decoded" if descriptor.get("storage") == "decoded" else "frames")
        for index in range(start,start+length):
            record = frozen.get((group,index))
            if record is None or digest(record_root/group/f"frame_{index:06d}/manifest.json") != record["completion_sha256"]:
                raise ValueError("Preview frame provenance differs from the checkpoint inventory")
            verify_v5_record(root,group,index)
        sample = dataset[window]
        dtype = next(model.parameters()).dtype
        buffers = {k:sample["buffers"][k].unsqueeze(0).to(device=device,
                   dtype=dtype if sample["buffers"][k].is_floating_point() else sample["buffers"][k].dtype)
                   for k in INPUT_CHANNELS}
        layers = sample["layers"].unsqueeze(0).to(device=device,dtype=dtype)
        # Targets remain outside the forward prediction and renderer replay.
        prediction = model.forward_sequence(buffers,layers)
        count = prediction.counts[:,-1]
        adaptive = replay_sample_layers(layers[:,-1],count)
        pixels = lambda tensor:tensor.detach().cpu().numpy().transpose(1,2,0)
        images = [("uniform-4spp", "Uniform 4 SPP", pixels(buffers["radiance"][0,-1])),
                  ("adaptive-input", "Adaptive noisy input", pixels(adaptive[0])),
                  ("denoised", "Joint denoised", pixels(prediction.radiance[0,-1])),
                  ("reference", "Reference", pixels(sample["buffers"]["reference"][-1]))]
        displays = [(name,label,display_rgb(image,exposure)) for name,label,image in images]
        metadata = descriptor["groups"][group]["source_config"]
        report = {"mode":"validation_center_crop", "sequence_id":group,
            "frame_indices":sample["frame_indices"],"displayed_frame":sample["frame_indices"][-1],
            "crop":{"top":(metadata["height"]-size)//2,"left":(metadata["width"]-size)//2,"size":size},
            "checkpoint":checkpoint,"checkpoint_sha256":digest(run/checkpoint),
            "checkpoint_epoch":payload["epoch"],"checkpoint_global_step":payload["global_step"],
            "exposure":exposure,"reference_supplied_to_prediction":False,"test_pixels_loaded":False,
            "expected_spp":float(prediction.expected[:,-1].mean()),"actual_spp":float(count.float().mean()),
            "unavailable_fraction":float((~prediction.valid[:,-1]).float().mean()),
            "sample_count_display_range":[0,model.max_spp],"diagnostic_only":True}
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".joint-preview-",dir=output.parent) as name:
            staging = Path(name)/"preview"
            staging.mkdir()
            canvas = Image.new("RGB",(size*len(images),size+24),(30,30,30))
            draw = ImageDraw.Draw(canvas)
            for index,(filename,label,display) in enumerate(displays):
                picture = Image.fromarray(display)
                picture.save(staging/(filename+".png"))
                canvas.paste(picture,(index*size,24))
                draw.text((index*size+3,5),label,fill="white")
            canvas.save(staging/"comparison.png")
            Image.fromarray(np.rint(count[0,0].cpu().numpy()/model.max_spp*255).astype(np.uint8)).save(staging/"sample-count.png")
            Image.fromarray(prediction.valid[0,-1,0].cpu().numpy().astype(np.uint8)*255).save(staging/"availability.png")
            write_rgb_exr(staging/"denoised.exr",images[2][2])
            (staging/"preview.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            staging.rename(output)
    return {"output":str(output),**report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--checkpoint",choices=("last.pt","model.pt"),default="last.pt")
    parser.add_argument("--sequence-id")
    parser.add_argument("--window",type=int,default=0)
    parser.add_argument("--patch-size",type=int)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--exposure",type=float,default=1.)
    args = parser.parse_args(argv)
    try:
        result = export_preview(args.run,args.output,checkpoint=args.checkpoint,sequence_id=args.sequence_id,
            window=args.window,patch_size=args.patch_size,device_name=args.device,exposure=args.exposure)
    except (OSError,RuntimeError,ValueError) as error:
        parser.exit(1,f"Joint preview failed: {error}\n")
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
