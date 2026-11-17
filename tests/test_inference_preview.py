"""Checkpoint snapshots, scene isolation and exported allocation evidence."""
import hashlib
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset
from test_r2_streaming import decoded_cloud, inline_cloud, cloud, r2_api
from test_final_handoff import model_bundle, heldout_bundle, tool
from wavefront_denoiser.adaptive_data import read_v5_dataset
from wavefront_denoiser.inference_preview import (allocation_images, select_scene_groups,
                                                 separate_directories)
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
from wavefront_denoiser.model_artifacts import build_bundle, build_best_bundle, verify_bundle


def test_export_pins_weights_resume_and_hash_during_live_replacement(model_bundle, tmp_path, monkeypatch):
    bundle, _, metadata, checkpoint = model_bundle
    original_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original_load = load_joint_checkpoint
    def load_then_trainer_saves(path, *args, **kwargs):
        model, payload = original_load(path, *args, **kwargs)
        save_joint_checkpoint(checkpoint, model, config=payload['config'],
                              dataset_split=payload['dataset_split'], epoch=99)
        return model, payload
    monkeypatch.setattr('wavefront_denoiser.model_artifacts.load_joint_checkpoint', load_then_trainer_saves)
    output = tmp_path/'snapshot'
    manifest = build_bundle(checkpoint, metadata, output, include_resume=True)
    assert manifest['source_checkpoint']['sha256'] == original_sha
    assert hashlib.sha256((output/'resume.pt').read_bytes()).hexdigest() == original_sha
    assert original_load(output/'weights.pt')[1]['epoch'] == 3
    assert original_load(checkpoint)[1]['epoch'] == 99
    verify_bundle(output)


def test_export_best_does_not_need_finished_summary(model_bundle, tmp_path):
    _, _, metadata, checkpoint = model_bundle
    run = tmp_path/'ongoing'
    (run/'train').mkdir(parents=True)
    shutil.copytree(metadata, run/'data')
    shutil.copyfile(checkpoint, run/'train/model.pt')
    (run/'train/history.jsonl').write_text('{"epoch": 3, "validation": {"loss": 0.25}}\n{"epoch":')
    manifest = build_best_bundle(run, tmp_path/'best-export')
    assert manifest['epoch'] == 3
    assert manifest['selection']['validation'] == {'loss': .25}
    assert not manifest['selection']['initialization_selected']
    assert not (run/'train/summary.json').exists()
    verify_bundle(tmp_path/'best-export')


def test_reexport_picks_up_later_best_and_preserves_earlier_bundle(model_bundle, tmp_path):
    _, _, metadata, checkpoint = model_bundle
    run = tmp_path/'ongoing'
    (run/'train').mkdir(parents=True)
    shutil.copytree(metadata, run/'data')
    shutil.copyfile(checkpoint, run/'train/model.pt')
    (run/'train/history.jsonl').write_text('{"epoch": 3, "validation": {"loss": 0.25}}\n')
    first = build_best_bundle(run, tmp_path/'early-export')
    original_bytes = (tmp_path/'early-export/weights.pt').read_bytes()
    model, payload = load_joint_checkpoint(checkpoint)
    with torch.no_grad():
        next(model.denoiser.parameters()).add_(.001)
    save_joint_checkpoint(run/'train/model.pt', model, config=payload['config'],
                          dataset_split=payload['dataset_split'], epoch=4, global_step=16)
    with (run/'train/history.jsonl').open('a') as stream:
        stream.write('{"epoch": 4, "validation": {"loss": 0.20}}\n')
    later = build_best_bundle(run, tmp_path/'later-export')
    assert first['epoch'] == 3 and later['epoch'] == 4
    assert later['selection']['validation']['loss'] == .2
    early_model, _ = load_joint_checkpoint(tmp_path/'early-export/weights.pt')
    later_model, _ = load_joint_checkpoint(tmp_path/'later-export/weights.pt')
    assert not torch.equal(next(early_model.denoiser.parameters()), next(later_model.denoiser.parameters()))
    assert (tmp_path/'early-export/weights.pt').read_bytes() == original_bytes
    verify_bundle(tmp_path/'early-export'); verify_bundle(tmp_path/'later-export')


def scene_descriptor():
    return {'groups': {'bathroom_b': {'scene': 'bathroom'}, 'bathroom_a': {'scene': 'bathroom'},
                       'classroom_a': {'scene': 'classroom'},
                       'sponza_view_00_circle_slow': {'scene': 'sponza'},
                       'sponza_other': {'scene': 'sponza'}},
            'partitions': {'test': ['sponza_other', 'sponza_view_00_circle_slow']}}


@pytest.mark.parametrize('scenes,expected', [(['bathroom'], ['bathroom_a']),
    (['classroom'], ['classroom_a']), (['sponza'], ['sponza_view_00_circle_slow']),
    (['bathroom', 'classroom', 'sponza'], ['bathroom_a', 'classroom_a', 'sponza_view_00_circle_slow'])])
def test_scenes_select_only_one_frozen_clip(scenes, expected):
    assert select_scene_groups(scene_descriptor(), ['bathroom_b', 'classroom_a', 'bathroom_a'], scenes) == expected


def test_missing_or_duplicate_scenes_rejected():
    with pytest.raises(ValueError, match='No compatible'):
        select_scene_groups(scene_descriptor(), ['classroom_a'], ['bathroom'])
    with pytest.raises(ValueError, match='unique'):
        select_scene_groups(scene_descriptor(), ['classroom_a'], ['classroom', 'classroom'])


def test_maps_use_actual_counts_and_fixed_scale():
    counts = np.array([[0, 1, 4, 8, 63]], dtype=np.uint8)
    images = allocation_images(counts, 63)
    blue = np.asarray(images['allocation-heatmap'])
    assert blue[0, 0].tolist() == [247, 251, 255]
    assert blue[0, -1].tolist() == [8, 48, 107]
    assert np.all(np.diff(blue.astype(int), axis=1) <= 0)
    gray = np.asarray(images['sample-count'])
    assert gray[0, 0] == 0 and gray[0, -1] == 255
    np.testing.assert_array_equal(blue[:, 2:3], allocation_images(counts[:, 2:3], 63)['allocation-heatmap'])


def test_directories_reject_overlap_and_symlink_alias(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        separate_directories(tmp_path/'downloads', tmp_path/'downloads/previews')
    (tmp_path/'actual').mkdir()
    (tmp_path/'alias').symlink_to(tmp_path/'actual', target_is_directory=True)
    with pytest.raises(ValueError, match='separate'):
        separate_directories(tmp_path/'actual', tmp_path/'alias')


def test_preview_keeps_downloads_outside_retained_images(heldout_bundle, tmp_path):
    bundle, api, metadata, _ = heldout_bundle
    descriptor = read_v5_dataset(metadata)
    groups = descriptor['partitions']['validation'][:1]+descriptor['partitions']['test'][:1]
    output, downloads = tmp_path/'previews', tmp_path/'downloads'
    report = tool('generate_presentation').generate(bundle, output, groups=groups, include_test=True,
                                                   preview=True, download_directory=downloads, crop_size=3)
    assert report['complete'] and report['purpose'] == 'visual_preview'
    assert (downloads/'test-metadata').is_dir()
    assert not list((downloads/'spool').iterdir())
    assert not (output/'spool').exists() and not (output/'test-metadata').exists()
    for clip in report['clips']:
        folder = output/clip['group']
        counts = np.load(folder/'sample-count.npy')
        expected = allocation_images(counts, 63)
        for name, image in expected.items():
            np.testing.assert_array_equal(Image.open(folder/(name+'.png')), image)
        for i, (x, y, w, h) in enumerate(clip['crops_xywh'], 1):
            np.testing.assert_array_equal(np.load(folder/f'crop_{i:02d}/sample-count.npy'), counts[y:y+h, x:x+w])
            np.testing.assert_array_equal(Image.open(folder/f'crop_{i:02d}/allocation-heatmap.png'),
                                          expected['allocation-heatmap'].crop((x, y, x+w, y+h)))
        assert clip['elapsed_seconds'] > 0
        assert all(row['timing']['total_seconds'] > 0 for row in clip['frames'])
        assert clip['quality']['adaptive-denoised']['spatial_frames'] == len(clip['frames'])-1
        assert clip['quality']['adaptive-denoised']['temporal_pairs'] == len(clip['frames'])-2
        for i, box in enumerate(clip['crops_xywh'], 1):
            metrics = json.loads((folder/f'crop_{i:02d}/metrics.json').read_text())
            assert metrics['crops_xywh'] == box and metrics['frame'] == len(clip['frames'])-1
            assert metrics['quality']['adaptive-denoised']['psnr_db'] is not None
        frames = list(csv.DictReader((folder/'quality-frames.csv').open()))
        assert len(frames) == len(clip['frames'])*4
        assert frames[0]['included_in_spatial_summary'] == 'False'
        assert frames[0]['tpsnr_db'] == ''
    summary = list(csv.DictReader((output/'quality-summary.csv').open()))
    assert len(summary) == len(groups)*4
    assert {row['partition'] for row in summary} == {'validation', 'test'}
    assert all(row['checkpoint_sha256'] == report['checkpoint_sha256'] for row in summary)
    assert 'PSNR (dB)' in (output/'index.md').read_text()
    assert (output/'metric-protocol.json').is_file()
    shutil.rmtree(downloads)
    assert (output/'presentation.json').is_file()
    assert all((output/g/'allocation-heatmap.png').is_file() for g in groups)


def test_preview_cli_checks_paths_before_download(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    module = tool('preview_inference')
    def unexpected(*args, **kwargs): pytest.fail('No download should occur with invalid paths')
    monkeypatch.setattr(module, 'download_bundle', unexpected)
    with pytest.raises(ValueError, match='separate'):
        module.preview(model_r2='r2://bucket/model-runs/snapshot', scenes=['bathroom'], device='cpu',
                       download_dir=tmp_path/'downloads', output=tmp_path/'downloads/previews')


def test_full_image_crop_matches_last_frame_quality_and_temporal_pair(model_bundle, tmp_path):
    bundle, _, metadata, _ = model_bundle
    descriptor = read_v5_dataset(metadata)
    group = descriptor['partitions']['validation'][0]
    config = descriptor['groups'][group]['source_config']
    crops = tmp_path/'crops.json'
    crops.write_text(json.dumps({group: [[0, 0, config['width'], config['height']]]}))
    output = tmp_path/'quality-preview'
    report = tool('generate_presentation').generate(bundle, output, groups=[group], crops=crops, preview=True)
    clip = report['clips'][0]
    crop = json.loads((output/group/'crop_01/metrics.json').read_text())
    assert crop['quality'] == clip['frames'][-1]['quality']
    assert crop['previous_frame'] == len(clip['frames'])-2


def test_preview_cli_downloads_and_runs_selected_clip(model_bundle, tmp_path, monkeypatch, capsys):
    from wavefront_denoiser.inference_preview import SCENE_PARTITIONS
    from wavefront_denoiser.model_artifacts import upload_bundle
    bundle, api, metadata, _ = model_bundle
    # The captured fixture has a tiny validation_scene rather than Bathroom.
    monkeypatch.setitem(SCENE_PARTITIONS, 'validation_scene', 'validation')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    upload_bundle(bundle, 'r2://bucket/model-runs/preview')
    output, downloads = tmp_path/'retained', tmp_path/'disposable'
    tool('preview_inference').main(['--model-r2', 'r2://bucket/model-runs/preview',
        '--download-dir', str(downloads), '--output', str(output),
        '--scenes', 'validation_scene', '--device', 'cpu', '--crop-size', '3'])
    summary = json.loads((output/'preview.json').read_text())
    assert summary['complete'] and summary['epoch'] == 3
    report = json.loads((output/'presentation.json').read_text())
    assert len(report['clips']) == 1 and not report['test_pixels_loaded']
    assert (downloads/'model/weights.pt').is_file()
    assert not list((downloads/'work/spool').iterdir())
    assert (output/'index.md').is_file()
    verify_bundle(downloads/'model')
