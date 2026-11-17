#!/usr/bin/env python3
"""Export a recurrent validation crop from the joint sampler/denoiser checkpoint."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from wavefront_denoiser.adaptive_data import read_v5_dataset, verify_v5_record, load_v5_frame
from wavefront_denoiser.allocation import allocate_samples
from wavefront_denoiser.loss import sequence_loss
from wavefront_denoiser.ntasad_radiance import decode_radiance
from wavefront_denoiser.device import select_device
from wavefront_denoiser.image_io import write_rgb_exr
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset, joint_dataset_read_lock
from wavefront_denoiser.model import INPUT_CHANNELS
from wavefront_denoiser.preview import display_rgb
from wavefront_denoiser.sample_layers import replay_sample_layers
from wavefront_denoiser.dataset_generation import digest


@torch.inference_mode()
def full_frame_images(model, root, group, start, length, device, exposure):
    """Whole-image allocation and recurrence, one native frame at a time.

    No tile-local softmax, resized guides, cropped history, or reference input.
    Uniform replay uses the same frozen denoiser and an independent history.
    """
    descriptor = read_v5_dataset(root)
    state = uniform_state = None
    predictions, uniforms, targets, animation, statistics = [], [], [], [], []
    pixels = lambda x: x.detach().cpu().numpy().transpose(1,2,0)
    for index in range(start, start+length):
        if descriptor.get("storage") == "decoded":
            from wavefront_denoiser.decoded_cache import load_decoded_crop
            base, layers = load_decoded_crop(root,group,index,top=0,left=0)
        else:
            loaded = load_v5_frame(root,group,index)
            base,layers = loaded.base,loaded.layers
        reference = torch.from_numpy(base.buffers["reference"].copy()).permute(2,0,1)
        frame = {k:torch.from_numpy(np.array(base.buffers[k],copy=True)).permute(2,0,1)
                 .unsqueeze(0).to(device=device, dtype=torch.int64 if k == "sample_count" else torch.float32)
                 for k in INPUT_CHANNELS}
        layer = torch.from_numpy(layers).permute(0,3,1,2).unsqueeze(0).to(device)
        h,w = reference.shape[-2:]
        uniform_count = torch.full((1,1,h,w),model.average_spp,dtype=torch.int64,device=device)
        if state is None:
            count = uniform_count
            expected = count.float()
        else:
            logits = model.sampler.predict_logits(frame,decode_radiance(state.radiance),
                history_available=state.valid,history_backend="gather")
            allocation = allocate_samples(logits,model.average_spp,policy=model.allocation_policy,max_spp=model.max_spp)
            expected,count = allocation.expected,allocation.counts
        adaptive = replay_sample_layers(layer,count)
        noisy = replay_sample_layers(layer,uniform_count)
        state = model.denoiser._step(dict(frame,radiance=adaptive,sample_count=count),state)
        uniform_state = model.denoiser._step(dict(frame,radiance=noisy,sample_count=uniform_count),uniform_state)
        denoised,uniform = decode_radiance(state.radiance)[0].cpu(),decode_radiance(uniform_state.radiance)[0].cpu()
        predictions.append(denoised); uniforms.append(uniform); targets.append(reference)
        images = [("uniform-4spp","Uniform 4 SPP",pixels(noisy[0])),
                  ("adaptive-input","Adaptive noisy input",pixels(adaptive[0])),
                  ("uniform-denoised","Uniform + same denoiser",pixels(uniform)),
                  ("denoised","Joint denoised",pixels(denoised)),
                  ("reference","Reference",pixels(reference))]
        # Small animation for presentation; PNG/EXR exports below stay native.
        thumbs = [Image.fromarray(display_rgb(rgb,exposure)) for _,_,rgb in images]
        thumb_w = min(w,480); thumb_h = max(1,round(h*thumb_w/w))
        strip = Image.new("RGB",(thumb_w*len(images),thumb_h+24),(30,30,30))
        draw = ImageDraw.Draw(strip)
        for j,(picture,(_,label,_)) in enumerate(zip(thumbs,images)):
            strip.paste(picture.resize((thumb_w,thumb_h)),(j*thumb_w,24))
            draw.text((j*thumb_w+3,5),label,fill="white")
        animation.append(strip)
        statistics.append({"frame":index,"expected_spp":float(expected.mean()),
            "actual_spp":float(count.float().mean()),"adaptive_paths":int(count.sum()),
            "uniform_paths":int(uniform_count.sum()),"zero_fraction":float((count==0).float().mean()),
            "saturated_fraction":float((count==model.max_spp).float().mean()),
            "unavailable_fraction":float((~state.valid).float().mean())})
    def metrics(values):
        scores = sequence_loss(torch.stack(values)[None],torch.stack(targets)[None])
        return {k:float(v) for k,v in scores.items()}
    report = {"frames":statistics,"joint_metrics":metrics(predictions),"uniform_denoiser_metrics":metrics(uniforms),
        "comparison":"same frozen denoiser; nominal 4 SPP; actual integer path totals reported, not exactly matched",
        "metric_domain":"gamma/log; final-frame spatial and final-pair temporal L1; no masked pixels"}
    return images,count,state.valid,animation,report


@torch.inference_mode()
def export_preview(run, output=None, *, checkpoint="last.pt", sequence_id=None, window=0,
                   patch_size=None, device_name="auto", exposure=1., full_frame=False):
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
        if descriptor.get("experiment_split_sha256") != payload["dataset_split"].get("experiment_split_sha256"):
            raise ValueError("Preview experiment split differs from the checkpoint inventory")
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
        extra = {}; animation = []
        if full_frame:
            images,count,valid,animation,extra = full_frame_images(model,root,group,start,length,device,exposure)
            height,width = images[0][2].shape[:2]
            frame_indices = list(range(start,start+length))
            expected_spp = extra["frames"][-1]["expected_spp"]
        else:
            sample = dataset[window]
            dtype = next(model.parameters()).dtype
            buffers = {k:sample["buffers"][k].unsqueeze(0).to(device=device,
                       dtype=dtype if sample["buffers"][k].is_floating_point() else sample["buffers"][k].dtype)
                       for k in INPUT_CHANNELS}
            layers = sample["layers"].unsqueeze(0).to(device=device,dtype=dtype)
            prediction = model.forward_sequence(buffers,layers)
            count,valid = prediction.counts[:,-1],prediction.valid[:,-1]
            adaptive = replay_sample_layers(layers[:,-1],count)
            pixels = lambda tensor:tensor.detach().cpu().numpy().transpose(1,2,0)
            images = [("uniform-4spp", "Uniform 4 SPP", pixels(buffers["radiance"][0,-1])),
                      ("adaptive-input", "Adaptive noisy input", pixels(adaptive[0])),
                      ("denoised", "Joint denoised", pixels(prediction.radiance[0,-1])),
                      ("reference", "Reference", pixels(sample["buffers"]["reference"][-1]))]
            height = width = size
            frame_indices = sample["frame_indices"]
            expected_spp = float(prediction.expected[:,-1].mean())
        displays = [(name,label,display_rgb(image,exposure)) for name,label,image in images]
        metadata = descriptor["groups"][group]["source_config"]
        report = {"mode":"validation_full_frame" if full_frame else "validation_center_crop", "sequence_id":group,
            "frame_indices":frame_indices,"displayed_frame":frame_indices[-1],
            "crop":None if full_frame else {"top":(metadata["height"]-size)//2,"left":(metadata["width"]-size)//2,"size":size},
            "native_width":width,"native_height":height,
            "checkpoint":checkpoint,"checkpoint_sha256":digest(run/checkpoint),
            "checkpoint_epoch":payload["epoch"],"checkpoint_global_step":payload["global_step"],
            "exposure":exposure,"reference_supplied_to_prediction":False,"test_pixels_loaded":False,
            "expected_spp":expected_spp,"actual_spp":float(count.float().mean()),
            "unavailable_fraction":float((~valid).float().mean()),
            "sample_count_display_range":[0,model.max_spp],"diagnostic_only":True,**extra}
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".joint-preview-",dir=output.parent) as name:
            staging = Path(name)/"preview"
            staging.mkdir()
            canvas = Image.new("RGB",(width*len(images),height+24),(30,30,30))
            draw = ImageDraw.Draw(canvas)
            for index,(filename,label,display) in enumerate(displays):
                picture = Image.fromarray(display)
                picture.save(staging/(filename+".png"))
                canvas.paste(picture,(index*width,24))
                draw.text((index*width+3,5),label,fill="white")
            canvas.save(staging/"comparison.png")
            Image.fromarray(np.rint(count[0,0].cpu().numpy()/model.max_spp*255).astype(np.uint8)).save(staging/"sample-count.png")
            Image.fromarray(valid[0,0].cpu().numpy().astype(np.uint8)*255).save(staging/"availability.png")
            np.save(staging/"sample-count.npy",count[0,0].cpu().numpy().astype(np.uint8),allow_pickle=False)
            write_rgb_exr(staging/"denoised.exr",next(rgb for name,_,rgb in images if name == "denoised"))
            if animation:
                animation[0].save(staging/"comparison.gif",save_all=True,append_images=animation[1:],duration=180,loop=0)
            (staging/"preview.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            staging.rename(output)
    return {"output":str(output),**report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--checkpoint",choices=("last.pt","model.pt","initial.pt","best-trained.pt"),default="last.pt")
    parser.add_argument("--full-frame",action="store_true",help="Native rectangular frames; global sampling and full history")
    parser.add_argument("--sequence-id")
    parser.add_argument("--window",type=int,default=0)
    parser.add_argument("--patch-size",type=int)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--exposure",type=float,default=1.)
    args = parser.parse_args(argv)
    try:
        result = export_preview(args.run,args.output,checkpoint=args.checkpoint,sequence_id=args.sequence_id,
            window=args.window,patch_size=args.patch_size,device_name=args.device,exposure=args.exposure,full_frame=args.full_frame)
    except (OSError,RuntimeError,ValueError) as error:
        parser.exit(1,f"Joint preview failed: {error}\n")
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
