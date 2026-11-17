"""Global native allocation, scoring all learned frames and temporal pairs."""
import time
from pathlib import Path

import numpy as np
import torch

import diagnose_quality_collapse as h
from final_quality_model import ScreenModel
from wavefront_denoiser.model import INPUT_CHANNELS
from wavefront_denoiser.ntasad_radiance import encode_radiance, decode_radiance


def frame_metrics(image, reference, previous=None):
    target = encode_radiance(reference)
    spatial = float((image-target).abs().mean())
    temporal = None if previous is None else float(((image-previous[0])-(target-previous[1])).abs().mean())
    linear = decode_radiance(image)
    luma = lambda x: (x*x.new_tensor([.2126, .7152, .0722])[:, None, None]).sum(0)
    black = float(((luma(linear) <= 1e-6) & (luma(reference) > 1e-3)).float().mean())
    return {'spatial_l1': spatial, 'temporal_l1': temporal, 'unexpected_black_fraction': black}, target


def aggregate(frames, variant):
    # Frame zero is initialization only. The 0->1 pair is reported separately
    # from the primary temporal mean, which compares two learned frames.
    learned = [f[variant] for f in frames if f['frame'] >= 1]
    pairs = [f[variant] for f in frames if f['frame'] >= 2]
    if not learned or not pairs:
        return None
    spatial = sum(f['spatial_l1'] for f in learned)/len(learned)
    temporal = sum(f['temporal_l1'] for f in pairs)/len(pairs)
    return {'loss': spatial+temporal, 'spatial_l1': spatial, 'temporal_l1': temporal,
            'unexpected_black_fraction': sum(f['unexpected_black_fraction'] for f in learned)/len(learned)}


@torch.inference_mode()
def native(base, label, checkpoint, config, deadline, device='cuda'):
    from PIL import Image, ImageDraw
    from wavefront_denoiser.preview import display_rgb
    from wavefront_denoiser.adaptive_data import read_v5_dataset, load_v5_frame
    from wavefront_denoiser.decoded_cache import load_decoded_crop
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ScreenModel(payload['model_spec'], payload.get('zero_fraction', 0.),
                        payload.get('sparse_policy', 'forced'), payload.get('reserve_floor', 1.))
    model.load_state_dict(payload['state_dict']); model.to(device).eval()
    descriptor = read_v5_dataset(config['data_root'])
    output = base/'native'/label; output.mkdir(parents=True)
    records = {}
    preview_scenes = set()
    try:
        for group in config['validation_groups']:
            if time.monotonic() >= deadline:
                break
            scene = descriptor['groups'][group]['scene']
            save_visuals = scene not in preview_scenes
            preview_scenes.add(scene)
            target_frames = descriptor['groups'][group]['frame_count']
            states = {'learned': None, 'uniform': None}
            previous = {'learned': None, 'uniform': None}
            frames, animation = [], []
            for index in range(target_frames):
                if time.monotonic() >= deadline:
                    break
                if descriptor.get('storage') == 'decoded':
                    data, layers = load_decoded_crop(config['data_root'], group, index, top=0, left=0)
                else:
                    loaded = load_v5_frame(config['data_root'], group, index)
                    data, layers = loaded.base, loaded.layers
                reference = torch.from_numpy(np.array(data.buffers['reference'], copy=True)).permute(2, 0, 1)
                frame = {k: torch.from_numpy(np.array(data.buffers[k], copy=True)).permute(2, 0, 1)[None]
                         .to(device=device, dtype=torch.int64 if k == 'sample_count' else torch.float32) for k in INPUT_CHANNELS}
                layer = torch.from_numpy(np.array(layers, copy=True)).permute(0, 3, 1, 2)[None].to(device)
                row = {'frame': index}
                pictures = []
                for variant in states:
                    states[variant], _, counts = model.step(frame, layer, states[variant], variant=variant)
                    image = states[variant].radiance[0].cpu()
                    metrics, target = frame_metrics(image, reference, previous[variant])
                    previous[variant] = (image, target)
                    metrics.update(paths=int(counts.sum()), pixels=counts.numel(),
                                   zero_fraction=float((counts == 0).float().mean()),
                                   saturated_fraction=float((counts == 63).float().mean()),
                                   unavailable_fraction=float((~states[variant].valid).float().mean()))
                    row[variant] = metrics
                    if save_visuals:
                        pictures.append((variant, decode_radiance(image)))
                    if save_visuals and index == target_frames-1 and variant == 'learned':
                        folder = output/group; folder.mkdir(exist_ok=True)
                        np.save(folder/'sample-count.npy', counts[0, 0].cpu().numpy().astype(np.uint8))
                if row['learned']['paths'] != row['uniform']['paths']:
                    raise ValueError('Native integer budget mismatch')
                frames.append(row)
                if save_visuals:
                    pictures.append(('reference', reference))
                    height, width = reference.shape[-2:]
                    tw = min(width, 480); th = round(height*tw/width)
                    strip = Image.new('RGB', (tw*3, th+24), (24, 24, 24)); draw = ImageDraw.Draw(strip)
                    for j, (name, value) in enumerate(pictures):
                        picture = Image.fromarray(display_rgb(value.numpy().transpose(1, 2, 0), 1.))
                        strip.paste(picture.resize((tw, th)), (tw*j, 24)); draw.text((tw*j+3, 4), name, fill='white')
                        if index == target_frames-1:
                            folder = output/group; folder.mkdir(exist_ok=True); picture.save(folder/f'{name}.png')
                    animation.append(strip)
                del frame, layer, data, layers
            record = {'scene': scene, 'frames': frames, 'expected_frames': target_frames,
                      'complete': len(frames) == target_frames and target_frames >= 3,
                      'learned': aggregate(frames, 'learned'), 'uniform': aggregate(frames, 'uniform')}
            records[group] = record
            folder = output/group; folder.mkdir(exist_ok=True)
            h.write(folder/'metrics.json', record)
            if animation:
                animation[0].save(folder/'comparison.gif', save_all=True, append_images=animation[1:], duration=180, loop=0)
                animation[-1].save(folder/'comparison.png')
            h.write(base/'progress.json', {'stage': 'native', 'candidate': label, 'clip': group})
        result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': h.digest(Path(checkpoint)),
                  'zero_fraction_policy': model.zero_fraction, 'clips': records,
                  'sparse_policy': model.sparse_policy, 'reserve_floor': model.reserve_floor,
                  'complete': len(records) == len(config['validation_groups']) and all(r['complete'] for r in records.values()),
                  'scope': 'Every learned frame; temporal mean over pairs 1->2 onward. Seed frame and 0->1 pair recorded separately. Encoded L1, not paper PSNR.'}
        h.write(output/'summary.json', result)
        return result
    finally:
        del model
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
