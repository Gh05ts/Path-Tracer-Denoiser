#!/usr/bin/env python3
"""Restore R2 weights and infer native images, aligned crops and temporal GIFs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from wavefront_denoiser.adaptive_data import read_v5_dataset
from wavefront_denoiser.allocation import allocate_samples
from wavefront_denoiser.image_io import write_rgb_exr
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.model import INPUT_CHANNELS
from wavefront_denoiser.model_artifacts import download_bundle, verify_bundle
from wavefront_denoiser.ntasad_radiance import encode_radiance, decode_radiance
from wavefront_denoiser.preview import display_rgb
from wavefront_denoiser.r2_streaming import R2FrameSource, prepare_r2_test_metadata
from wavefront_denoiser.sample_layers import replay_sample_layers

LABELS = {'uniform-4spp': 'Uniform 4 SPP', 'adaptive-input': 'Adaptive input (mean 4 SPP)',
          'uniform-denoised': 'Uniform + same denoiser', 'adaptive-denoised': 'Adaptive + denoised',
          'reference': 'Reference (captured high SPP)'}
MAIN = ('uniform-4spp', 'adaptive-denoised', 'reference')
RESAMPLE = getattr(Image, 'Resampling', Image)
DITHER = getattr(Image, 'Dither', Image)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def crop_boxes(reference, size):
    """Center plus up to two detail regions ranked from reference alone."""
    h, w = reference.shape[:2]; size = min(size, h, w)
    center = [(w-size)//2, (h-size)//2, size, size]
    gray = display_rgb(reference).mean(2).astype(float)
    detail = np.zeros_like(gray)
    detail[1:] += np.abs(np.diff(gray, axis=0))
    detail[:, 1:] += np.abs(np.diff(gray, axis=1))
    xs = sorted(set(list(range(0, w-size+1, max(1, size//2)))+[w-size]))
    ys = sorted(set(list(range(0, h-size+1, max(1, size//2)))+[h-size]))
    choices = sorted([(-float(detail[y:y+size, x:x+size].mean()), y, x) for y in ys for x in xs])
    boxes = [center]
    for _, y, x in choices:
        if all(max(0, min(x+size, bx+bw)-max(x, bx))*max(0, min(y+size, by+bh)-max(y, by))
               < .5*size*size for bx, by, bw, bh in boxes):
            boxes.append([x, y, size, size])
            if len(boxes) == 3: break
    return boxes


def checked_boxes(boxes, width, height):
    if not boxes: raise ValueError('Need at least one crop')
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)):
            raise ValueError('Crops must be [x, y, width, height] integer lists')
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) < 1 or x+w > width or y+h > height:
            raise ValueError('Crop is outside the native image')
    return boxes


def strip(images, names=MAIN, width=480, boxes=None, note=None):
    first = images[names[0]]; w = min(width, first.width); h = max(1, round(first.height*w/first.width))
    font_size = max(10, min(28, w//32))
    try: font = ImageFont.truetype('DejaVuSans.ttf', font_size)
    except OSError: font = ImageFont.load_default()
    header = 2*font_size+16
    canvas = Image.new('RGB', (w*len(names), h+header), (25, 25, 25)); draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(names):
        view = images[name].copy()
        if boxes:
            marking = ImageDraw.Draw(view)
            for j, (x, y, cw, ch) in enumerate(boxes):
                marking.rectangle((x, y, x+cw-1, y+ch-1), outline=(255, 190, 30), width=max(1, first.width//600))
                marking.text((x+3, y+3), str(j+1), fill=(255, 190, 30))
        canvas.paste(view.resize((w, h), RESAMPLE.LANCZOS), (i*w, header))
        draw.text((i*w+5, 4), LABELS[name], fill='white', font=font)
    if note: draw.text((5, font_size+10), note, fill=(220, 220, 220), font=font)
    return canvas


def gif(path, frames, fps):
    # One common palette across time avoids independent-palette flicker.
    atlas = Image.new('RGB', (frames[0].width, sum(f.height for f in frames)))
    offset = 0
    for frame in frames: atlas.paste(frame, (0, offset)); offset += frame.height
    palette = atlas.quantize(colors=256)
    frames = [f.quantize(palette=palette, dither=DITHER.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=round(1000/fps), optimize=False, disposal=2)


def score(encoded, reference, previous):
    target = encode_radiance(reference)
    return {'spatial_encoded_l1': float((encoded-target).abs().mean()),
            'temporal_encoded_l1': None if previous is None else
            float(((encoded-previous[0])-(target-previous[1])).abs().mean())}, target


@torch.inference_mode()
def infer_clip(model, source, descriptor, group, output, *, device='cpu', exposure=1.,
               boxes=None, crop_size=160, fps=6., hdr=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    states = {'adaptive': None, 'uniform': None}; previous = {'adaptive': None, 'uniform': None}
    animations = []; records = []; last_images = None; last_linear = None
    count_frames = descriptor['groups'][group]['frame_count']
    for index in range(count_frames):
        loaded = source.load_frame(group, index)
        base, layers = loaded.base, loaded.layers
        reference = torch.from_numpy(np.array(base.buffers['reference'], copy=True)).permute(2, 0, 1)
        dtype = next(model.parameters()).dtype
        frame = {k: torch.from_numpy(np.array(base.buffers[k], copy=True)).permute(2, 0, 1)[None]
                 .to(device=device, dtype=torch.int64 if k == 'sample_count' else dtype) for k in INPUT_CHANNELS}
        bank = torch.from_numpy(np.array(layers, copy=True)).permute(0, 3, 1, 2)[None].to(device=device, dtype=dtype)
        uniform_counts = torch.full_like(frame['sample_count'], 4, dtype=torch.int64)
        if states['adaptive'] is None: expected, counts = uniform_counts.to(dtype), uniform_counts
        else:
            state = states['adaptive']
            logits = model.sampler.predict_logits(frame, decode_radiance(state.radiance),
                         history_available=state.valid, history_backend='gather')
            allocation = allocate_samples(logits, model.average_spp, policy=model.allocation_policy, max_spp=model.max_spp)
            expected, counts = allocation.expected, allocation.counts
        if int(counts.sum()) != 4*counts.numel(): raise ValueError('Adaptive frame does not have an exact 4-SPP budget')
        adaptive_input = replay_sample_layers(bank, counts)
        noisy = replay_sample_layers(bank, uniform_counts)
        row = {'frame': index, 'initialization_only': index == 0, 'adaptive_paths': int(counts.sum()),
               'uniform_paths': int(uniform_counts.sum()), 'actual_spp': float(counts.float().mean()),
               'expected_spp': float(expected.mean()), 'zero_fraction': float((counts == 0).float().mean()),
               'saturated_fraction': float((counts == model.max_spp).float().mean())}
        linear = {'uniform-4spp': noisy[0].cpu(), 'adaptive-input': adaptive_input[0].cpu(), 'reference': reference}
        for variant, image, allocation_counts in [('adaptive', adaptive_input, counts), ('uniform', noisy, uniform_counts)]:
            states[variant] = model.denoiser._step(dict(frame, radiance=image, sample_count=allocation_counts), states[variant])
            encoded = states[variant].radiance[0].cpu()
            metrics, target = score(encoded, reference, previous[variant])
            previous[variant] = (encoded, target)
            metrics['unavailable_fraction'] = float((~states[variant].valid).float().mean())
            row[variant] = metrics
            linear[variant+'-denoised'] = decode_radiance(encoded)
        images = {k: Image.fromarray(display_rgb(v.numpy().transpose(1, 2, 0), exposure)) for k, v in linear.items()}
        folder = output/f'frame_{index:06d}'; folder.mkdir()
        for name, image in images.items(): image.save(folder/(name+'.png'))
        np.save(folder/'sample-count.npy', counts[0, 0].cpu().numpy().astype(np.uint8))
        animations.append(strip(images, note=f'Frame {index}'+(' - uniform initialization' if index == 0 else '')))
        records.append(row); last_images, last_linear = images, linear
        del loaded, base, layers, bank, frame, adaptive_input, noisy, counts, uniform_counts
    reference = last_linear['reference'].numpy().transpose(1, 2, 0)
    h, w = reference.shape[:2]
    explicit_crops = boxes is not None
    boxes = checked_boxes(boxes if explicit_crops else crop_boxes(reference, crop_size), w, h)
    strip(last_images, width=w, boxes=boxes, note='Matched crops; one shared exposure').save(output/'comparison-native.png')
    strip(last_images, names=tuple(LABELS), width=480).save(output/'controls.png')
    for j, (x, y, cw, ch) in enumerate(boxes):
        folder = output/f'crop_{j+1:02d}'; folder.mkdir()
        crops = {name: image.crop((x, y, x+cw, y+ch)) for name, image in last_images.items()}
        for name, image in crops.items():
            image.save(folder/(name+'.png'))
            image.resize((cw*3, ch*3), RESAMPLE.NEAREST).save(folder/(name+'-3x.png'))
        strip(crops, width=cw).save(folder/'comparison.png')
        strip(crops, names=tuple(LABELS), width=cw).save(folder/'controls.png')
    if hdr:
        for name, value in last_linear.items():
            write_rgb_exr(output/(name+'.exr'), value.numpy().transpose(1, 2, 0))
    gif(output/'comparison.gif', animations, fps)
    partition = next(p for p, names in descriptor['partitions'].items() if group in names)
    report = {'group': group, 'scene': descriptor['groups'][group]['scene'], 'partition': partition, 'frames': records,
              'crops_xywh': boxes, 'exposure': exposure, 'display_transform': '(exposure*x/(1+exposure*x))**(1/2.2)',
              'crop_rule': 'explicit' if explicit_crops else 'center plus reference-gradient ranking',
              'reference_source_config': descriptor['groups'][group]['source_config'],
              'reference_status': descriptor.get('reference_status'), 'gif_playback_fps': fps,
              'reference_supplied_to_prediction': False, 'test_pixels_loaded': partition == 'test',
              'scope': 'Offline independent-layer replay; native global allocation, independent histories. Not live renderer speed.'}
    for variant in states:
        spatial = [r[variant]['spatial_encoded_l1'] for r in records[1:]]
        temporal = [r[variant]['temporal_encoded_l1'] for r in records[2:]]
        report[variant] = {'spatial_encoded_l1': sum(spatial)/len(spatial) if spatial else None,
                           'temporal_encoded_l1': sum(temporal)/len(temporal) if temporal else None}
    write_json(output/'metrics.json', report)
    return report


def generate(bundle, output, *, groups=None, device='cpu', exposure=1., crops=None,
             crop_size=160, fps=6., hdr=False, all_validation=False, include_test=False, all_test=False):
    bundle, output = Path(bundle), Path(output)
    if output.exists(): raise FileExistsError('Presentation output already exists')
    if not math.isfinite(exposure) or exposure <= 0 or not 0 < fps <= 60 or crop_size < 1:
        raise ValueError('Invalid exposure, FPS or crop size')
    if all_test and not include_test:
        raise ValueError('--all-test requires --include-test after checkpoint selection')
    verify_bundle(bundle)
    model, payload = load_joint_checkpoint(bundle/'weights.pt', torch.device(device))
    if model.average_spp != 4 or model.allocation_policy != 'minimum_one_exact_v1':
        raise ValueError('Presentation requires the final exact-budget 4-SPP model policy')
    metadata = bundle/'metadata'
    record = json.loads((metadata/'r2-source.json').read_text())
    # Validate the bundle metadata against the trained weights before pixel GETs.
    from wavefront_denoiser.joint_data import joint_inventory
    expected = dict(payload['dataset_split'])
    for key in ('r2_snapshot_sha256', 'image_verification'): expected.pop(key, None)
    if joint_inventory(metadata, record['selected'], verify_images=False) != expected:
        raise ValueError('Presentation metadata differs from trained inventory')
    descriptor = read_v5_dataset(metadata)
    selected = {'validation': payload['dataset_split']['selected']['validation']}
    if include_test:
        if record['layout'] != 'decoded':
            raise ValueError('Held-out presentation requires a decoded training snapshot')
        selected['test'] = descriptor['partitions']['test']
    allowed = {g for names in selected.values() for g in names}
    selection_rule = 'explicit groups' if groups is not None else 'first lexicographic clip per scene; all clips for requested partitions'
    if groups is None:
        groups = []
        for partition, names in selected.items():
            scenes = set()
            for group in sorted(names):
                scene = descriptor['groups'][group]['scene']
                if (all_validation if partition == 'validation' else all_test) or scene not in scenes:
                    groups.append(group); scenes.add(scene)
    if not groups or len(set(groups)) != len(groups) or set(groups)-allowed:
        raise ValueError('Choose unique validation clips from the frozen trained inventory; '
                         'reserved test clips also require --include-test')
    crop_spec = json.loads(Path(crops).read_text()) if crops else {}
    if not isinstance(crop_spec, dict) or set(crop_spec)-set(groups):
        raise ValueError('Crop specification must map selected clips to crop boxes')
    for group, boxes in crop_spec.items():
        config = descriptor['groups'][group]['source_config']
        checked_boxes(boxes, config['width'], config['height'])
    groups_by_partition = {p: [g for g in groups if g in names] for p, names in selected.items()}
    checkpoint_sha256 = hashlib.sha256((bundle/'weights.pt').read_bytes()).hexdigest()
    output.mkdir(parents=True)
    # Publish the chosen checkpoint and protocol before fetching any test metadata/pixels.
    protocol = {'checkpoint_sha256': checkpoint_sha256, 'epoch': payload['epoch'],
                'global_step': payload['global_step'], 'groups_by_partition': groups_by_partition,
                'selection': selection_rule, 'dataset_source': record['source'],
                'dataset_manifest_sha256': expected['dataset_manifest_sha256'],
                'exposure': exposure, 'crops': crop_spec, 'crop_size': crop_size, 'fps': fps, 'hdr': hdr,
                'weights_updated': False, 'test_used_for_checkpoint_selection': False}
    write_json(output/'evaluation-protocol.json', protocol)
    test_groups = groups_by_partition.get('test', [])
    test_source = None
    if test_groups:
        prepare_r2_test_metadata(record['source'], output/'test-metadata', groups=test_groups,
                                 training_metadata=metadata, trained_inventory=expected)
        test_source = R2FrameSource(output/'test-metadata', record['source'], layout='decoded',
                                   spool_directory=output/'spool', allow_test=True)
        test_source.preflight()
    source = R2FrameSource(metadata, record['source'], layout=record['layout'], spool_directory=output/'spool')
    if groups_by_partition['validation']: source.preflight()
    results = []
    torch.set_num_threads(4)
    for group in groups:
        frame_source = test_source if group in test_groups else source
        results.append(infer_clip(model, frame_source, descriptor, group, output/group, device=device, exposure=exposure,
                       boxes=crop_spec.get(group), crop_size=crop_size, fps=fps, hdr=hdr))
        print(f"Presentation clip ready: {group} ({results[-1]['partition']})", flush=True)
    report = {'groups': groups, 'selection': selection_rule,
              'groups_by_partition': groups_by_partition, 'checkpoint_sha256': checkpoint_sha256,
              'epoch': payload['epoch'], 'model_spec': model.specification(), 'r2': source.stats,
              'test_r2': test_source.stats if test_source else None,
              'clips': results, 'complete': True, 'test_pixels_loaded': bool(test_groups), 'weights_updated': False}
    write_json(output/'presentation.json', report)
    links = ['# Presentation outputs', '', 'Native frames, matched crops, controls and GIFs use one shared exposure.', '']
    for group in groups:
        partition = 'test' if group in test_groups else 'validation'
        links += [f'## {group} ({partition})', '', f'![comparison]({group}/comparison-native.png)', '',
                  f'![animation]({group}/comparison.gif)', '', f'[All controls]({group}/controls.png)', '']
    (output/'index.md').write_text('\n'.join(links))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    b = p.add_mutually_exclusive_group(required=True)
    b.add_argument('--bundle', type=Path); b.add_argument('--model-r2')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', nargs='+'); p.add_argument('--all-validation', action='store_true')
    p.add_argument('--include-test', action='store_true', help='Include held-out scenes after freezing checkpoint selection')
    p.add_argument('--all-test', action='store_true', help='With --include-test, render every reserved test clip')
    p.add_argument('--device', default='cuda'); p.add_argument('--exposure', type=float, default=1.)
    p.add_argument('--crops', type=Path); p.add_argument('--crop-size', type=int, default=160)
    p.add_argument('--fps', type=float, default=6.); p.add_argument('--hdr', action='store_true')
    a = p.parse_args()
    if a.model_r2:
        a.bundle = a.output.parent/(a.output.name+'-model')
        download_bundle(a.model_r2, a.bundle)
    generate(a.bundle, a.output, groups=a.groups, device=a.device, exposure=a.exposure, crops=a.crops,
             crop_size=a.crop_size, fps=a.fps, hdr=a.hdr, all_validation=a.all_validation,
             include_test=a.include_test, all_test=a.all_test)


if __name__ == '__main__': main()
