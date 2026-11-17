#!/usr/bin/env python3
"""Prepare a portable full-data run from R2; never starts training."""
import argparse
import json
from pathlib import Path
import shutil

from wavefront_denoiser.adaptive_data import read_v5_dataset
from wavefront_denoiser.r2_streaming import prepare_r2_metadata, materialize_decoded_snapshot
from wavefront_denoiser.training_defaults import joint_training_config
from wavefront_denoiser.joint_training import training_config


def prepare(output, *, source='r2://denoiser/denoiser-v4', hours, epochs=1000,
            selection=None, materialize=False, batch_size=8):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Choose a new preparation directory')
    if not 0 < hours < 1e4 or epochs < 1 or batch_size < 1: raise ValueError('Invalid training budget/batch')
    selected = json.loads(Path(selection).read_text()) if selection else {'train': None, 'validation': None}
    if set(selected) != {'train', 'validation'}: raise ValueError('Selection needs train and validation lists')
    output.mkdir(parents=True)
    metadata = output/'data'
    prepare_r2_metadata(source, metadata, layout='decoded', train_groups=selected['train'],
                        validation_groups=selected['validation'])
    record = json.loads((metadata/'r2-source.json').read_text())
    descriptor = read_v5_dataset(metadata)
    arrays = [array for names in record['selected'].values() for group in names
              for path in (metadata/'decoded'/group).glob('frame_*/manifest.json')
              for array in json.loads(path.read_text())['arrays'].values()]
    total = sum(a['size'] for a in arrays)
    config = joint_training_config('data')
    config.update(epochs=epochs, batch_size=batch_size, learning_rate=1e-4, sampler_learning_rate=1e-5,
                  radiance_gradient_floor=1e-3, validation_batch_size=1,
                  train_groups=record['selected']['train'], validation_groups=record['selected']['validation'],
                  validation_every_epochs=5, validation_warmup_epochs=5, checkpoint_every_epochs=1,
                  finite_check_policy='grouped_v1', max_training_seconds=hours*3600,
                  loader_policy='prefetch_fused_v1', decoded_cache_frames=16, prefetch_batches=2)
    config_path = output/'train.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    training_config(config_path)
    plan = {'source': source, 'selected': record['selected'], 'payload_bytes': total,
            'array_objects': len(arrays), 'disk_free_bytes': shutil.disk_usage(output).free,
            'training_hours': hours, 'epochs_ceiling': epochs, 'test_pixels_loaded': False,
            'initialization': 'fresh; no checkpoint from the retiring machine required',
            'scenes': {p: sorted({descriptor['groups'][g]['scene'] for g in names})
                       for p,names in record['selected'].items()},
            'config': str(config_path), 'materialized': False}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan, indent=2), flush=True)
    if materialize:
        if shutil.disk_usage(output).free < total+8_000_000_000:
            raise ValueError('Insufficient disk: need selected payload plus 8 GB result headroom')
        report = materialize_decoded_snapshot(metadata)
        (output/'materialization.json').write_text(json.dumps(report, indent=2)+'\n')
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path); p.add_argument('--source', default='r2://denoiser/denoiser-v4')
    p.add_argument('--hours', type=float, required=True); p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--selection', type=Path, help='JSON with train and validation lists; default requests all original partitions')
    p.add_argument('--materialize', action='store_true', help='Download arrays after printing the audited plan')
    a = p.parse_args(); prepare(**vars(a))


if __name__ == '__main__': main()
