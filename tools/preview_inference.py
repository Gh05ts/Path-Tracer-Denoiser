#!/usr/bin/env python3
"""Download frozen weights and generate selected-scene previews on a spare GPU."""
import argparse
import json
from pathlib import Path
import time

import torch

from generate_presentation import generate
from wavefront_denoiser.inference_preview import SCENE_PARTITIONS, separate_directories
from wavefront_denoiser.model_artifacts import download_bundle


def preview(*, output, download_dir, scenes, model_r2=None, bundle=None, device='cuda',
            exposure=1., crops=None, crop_size=160, fps=6., hdr=False):
    if bool(model_r2) == bool(bundle): raise ValueError('Choose exactly one of model_r2 or bundle')
    output, download_dir = Path(output), Path(download_dir)
    separate_directories(output, download_dir, *([bundle] if bundle else []))
    for path in (output, download_dir):
        if path.exists(): raise FileExistsError(f'Choose a new directory: {path}')
    if not scenes or len(set(scenes)) != len(scenes) or set(scenes)-set(SCENE_PARTITIONS):
        raise ValueError('Choose unique scenes: bathroom, classroom, sponza')
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; install CUDA-enabled PyTorch and check the NVIDIA driver')
    started = time.perf_counter()
    download_dir.mkdir(parents=True, exist_ok=False)
    if model_r2:
        bundle = download_dir/'model'
        download_bundle(model_r2, bundle)
    report = generate(bundle, output, scenes=scenes, device=device, exposure=exposure,
                      crops=crops, crop_size=crop_size, fps=fps, hdr=hdr,
                      download_directory=download_dir/'work', preview=True)
    summary = {'complete': True, 'purpose': 'visual_preview', 'scenes': scenes,
               'epoch': report['epoch'], 'checkpoint_sha256': report['checkpoint_sha256'],
               'bundle': str(Path(bundle).resolve()), 'downloads': str(download_dir.resolve()),
               'previews': str(output.resolve()), 'elapsed_seconds': time.perf_counter()-started,
               'device': str(device), 'gpu': torch.cuda.get_device_name(device)
                         if torch.device(device).type == 'cuda' else None,
               'peak_gpu_allocated_bytes': report['peak_gpu_allocated_bytes'],
               'coverage': report['coverage'],
               'download_cleanup': 'After completion, downloads may be removed independently of previews. '
                                   'Retain model/ if reusing these exact weights.'}
    (output/'preview.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('--model-r2', help='r2://denoiser/model-runs/UNIQUE_SNAPSHOT')
    model.add_argument('--bundle', type=Path, help='Reuse a previously downloaded verified model directory')
    parser.add_argument('--download-dir', type=Path, required=True, help='New disposable directory for downloads')
    parser.add_argument('--output', type=Path, required=True, help='New retained preview directory, separate from downloads')
    parser.add_argument('--scenes', nargs='+', choices=tuple(SCENE_PARTITIONS), required=True,
                        help='One clip per scene; Sponza uses the fixed circle-slow test clip')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--exposure', type=float, default=1.)
    parser.add_argument('--crops', type=Path, help='Optional JSON of clip IDs to [x,y,width,height] boxes')
    parser.add_argument('--crop-size', type=int, default=160)
    parser.add_argument('--fps', type=float, default=6.)
    parser.add_argument('--hdr', action='store_true', help='Also retain final-frame EXR images')
    args = parser.parse_args(argv)
    print(json.dumps(preview(**vars(args)), indent=2))


if __name__ == '__main__': main()
