"""Fresh-machine R2 model round trip and reference-free native presentation."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import subprocess
from types import SimpleNamespace
REAL_RUN = subprocess.run
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image
import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from test_r2_streaming import decoded_cloud, inline_cloud, cloud, r2_api
from wavefront_denoiser.adaptive_data import read_v5_dataset
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import save_joint_checkpoint, load_joint_checkpoint
from wavefront_denoiser.model_artifacts import build_bundle, upload_bundle, download_bundle, safe_relative, verify_bundle
from wavefront_denoiser.r2_streaming import prepare_r2_metadata


def tool(name):
    path = Path(__file__).resolve().parents[1]/'tools'/(name+'.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_bundle(decoded_cloud, tmp_path, monkeypatch):
    monkeypatch.setattr("wavefront_denoiser.model_artifacts.subprocess", SimpleNamespace(run=REAL_RUN))
    torch.set_num_threads(1)
    _, api = decoded_cloud
    metadata = tmp_path/'metadata'
    prepare_r2_metadata('r2://bucket/dataset', metadata, layout='decoded')
    source = json.loads((metadata/'r2-source.json').read_text())
    _, config = config_for(metadata)
    model = JointNTASAD(4, 4, allocation_policy='minimum_one_exact_v1')
    checkpoint = tmp_path/'trained.pt'
    save_joint_checkpoint(checkpoint, model, config=config, dataset_split=source['inventory'], epoch=3, global_step=12)
    bundle = tmp_path/'bundle'
    build_bundle(checkpoint, metadata, bundle, include_resume=True)
    return bundle, api, metadata, checkpoint


def test_r2_roundtrip_infers_without_old_machine(model_bundle, tmp_path, monkeypatch):
    bundle, api, metadata, checkpoint = model_bundle
    original, _ = load_joint_checkpoint(checkpoint)
    receipt = upload_bundle(bundle, 'r2://bucket/model-runs/final')
    assert receipt['verified']
    puts = api.uploads
    upload_bundle(bundle, 'r2://bucket/model-runs/final')
    assert api.uploads == puts
    shutil.rmtree(metadata); checkpoint.unlink(); shutil.rmtree(bundle)
    restored = tmp_path/'restored'
    download_bundle('r2://bucket/model-runs/final', restored)
    loaded, payload = load_joint_checkpoint(restored/'weights.pt')
    assert payload['training'] is None and payload['epoch'] == 3
    for key, value in original.state_dict().items(): torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    assert (restored/'resume.pt').is_file()
    # References must stay outside all network prediction calls.
    original_logits = type(loaded.sampler).predict_logits
    original_step = type(loaded.denoiser)._step
    def logits(self, guides, *args, **kwargs):
        assert 'reference' not in guides
        return original_logits(self, guides, *args, **kwargs)
    def step(self, guides, *args, **kwargs):
        assert 'reference' not in guides
        return original_step(self, guides, *args, **kwargs)
    monkeypatch.setattr(type(loaded.sampler), 'predict_logits', logits)
    monkeypatch.setattr(type(loaded.denoiser), '_step', step)
    api.calls.clear()
    output = tmp_path/'presentation'
    report = tool('generate_presentation').generate(restored, output, device='cpu', crop_size=3, hdr=True)
    assert report['complete'] and not report['test_pixels_loaded']
    assert report['r2']['downloaded_objects'] == sum(len(c['frames'])*2 for c in report['clips'])
    descriptor = read_v5_dataset(restored/'metadata')
    requested = [unquote(urlsplit(url).path) for method, url in api.calls if method == 'GET']
    assert not any(path.endswith('.exr') for path in requested)
    assert not any('/'+g+'/' in path for g in descriptor['partitions']['test']+descriptor['partitions']['train'] for path in requested)
    assert not list((output/'spool').iterdir())
    for clip in report['clips']:
        folder = output/clip['group']
        for row in clip['frames']: assert row['adaptive_paths'] == row['uniform_paths']
        assert Image.open(folder/'comparison.gif').n_frames == len(clip['frames'])
        native = Image.open(folder/f"frame_{len(clip['frames'])-1:06d}/adaptive-denoised.png")
        for i, (x, y, w, h) in enumerate(clip['crops_xywh']):
            crop = Image.open(folder/f'crop_{i+1:02d}/adaptive-denoised.png')
            np.testing.assert_array_equal(crop, native.crop((x, y, x+w, y+h)))
        assert (folder/'adaptive-denoised.exr').is_file()


def test_bundle_rejects_changed_inventory_and_remote_collision(model_bundle, tmp_path):
    bundle, api, metadata, checkpoint = model_bundle
    with pytest.raises(ValueError, match='separate model-runs'):
        upload_bundle(bundle, 'r2://bucket/dataset/models')
    with pytest.raises(ValueError, match='dataset bucket'):
        upload_bundle(bundle, 'r2://other/models')
    api.objects['model-runs/final/weights.pt'] = b'other model'
    puts = api.uploads
    with pytest.raises(ValueError, match='different artifacts'):
        upload_bundle(bundle, 'r2://bucket/model-runs/final')
    assert api.uploads == puts
    (bundle/'weights.pt').write_bytes(b'corrupt local weights')
    with pytest.raises(ValueError, match='checksum changed'): verify_bundle(bundle)
    (metadata/'decoded_cache_lock.json').write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        build_bundle(checkpoint, metadata, tmp_path/'bad-bundle')


@pytest.mark.parametrize('path', ['../weights.pt', '/weights.pt', 'metadata/../../x', 'a\\b', '.', 'manifest.json', 'a//b'])
def test_bundle_rejects_path_traversal(path):
    with pytest.raises(ValueError): safe_relative(path)


def test_memory_guard_fails_below_user_limit():
    guard = tool('run_with_memory_guard')
    guard.check_memory({'host_used': 78, 'host_total': 100, 'gpu_used': 78, 'gpu_total': 100})
    with pytest.raises(RuntimeError, match='79%'): guard.check_memory({'host_used': 79, 'host_total': 100})


def test_prepare_full_inventory_never_downloads_arrays(decoded_cloud, tmp_path):
    _, api = decoded_cloud
    output = tmp_path/'final-prep'
    result = tool('prepare_final_run').prepare(output, source='r2://bucket/dataset', hours=24)
    assert result['payload_bytes'] > 0 and not result['test_pixels_loaded']
    assert not list(output.rglob('*.npy'))
    assert all(not urlsplit(url).path.endswith(('.npy', '.exr')) for _, url in api.calls)
    config = json.loads((output/'train.json').read_text())
    assert config['format'] == 'wavefront-ntasad-joint-training'
    assert config['denoiser_channels'] == config['sampler_channels'] == 48
    assert config['learning_rate'] == 1e-4 and config['sampler_learning_rate'] == 1e-5
    assert config['radiance_gradient_floor'] == 1e-3
    assert config['allocation_policy'] == 'minimum_one_exact_v1'
    assert config['checkpoint_every_epochs'] == 1 and config['max_training_seconds'] == 86400


def test_final_profile_cli_updates_both_networks(decoded_cloud, tmp_path):
    from wavefront_denoiser.cli import main
    from wavefront_denoiser.r2_streaming import materialize_decoded_snapshot

    output = tmp_path/'final-prep'
    tool('prepare_final_run').prepare(output, source='r2://bucket/dataset', hours=35)
    materialize_decoded_snapshot(output/'data')
    config = json.loads((output/'train.json').read_text())
    # A separate tiny technical smoke keeps the final model, learning rates,
    # allocation/backward policies and fused loader, reducing only workload.
    config.update(epochs=1, batch_size=1, patch_size=5, patches_per_window=1,
                  validation_patch_size=5, validation_crops_per_window=1, cpu_threads=1)
    path = output/'train-smoke.json'
    path.write_text(json.dumps(config))
    run = output/'smoke'
    assert main(['train-adaptive', str(path), str(run), '--device', 'cpu']) == 0
    initial, _ = load_joint_checkpoint(run/'initial.pt')
    trained, payload = load_joint_checkpoint(run/'last.pt')
    history = json.loads((run/'history.jsonl').read_text().splitlines()[0])
    assert payload['global_step'] > 0
    for name in ('sampler', 'denoiser'):
        norm = history['train'][name+'_gradient_norm']
        assert math.isfinite(norm) and norm > 0
        before, after = getattr(initial, name).state_dict(), getattr(trained, name).state_dict()
        assert any(not torch.equal(before[key], after[key]) for key in before)
        assert all(torch.isfinite(value).all() for value in after.values())
    for metrics in (history['train'], history['validation']):
        assert all(math.isfinite(metrics[key]) for key in ('loss', 'spatial_l1', 'temporal_l1'))
        assert metrics['actual_spp'] == 4
        assert metrics['replayed_paths'] == 4*metrics['replayed_pixels']
        assert metrics['zero_fraction'] == metrics['unavailable_fraction'] == 0
