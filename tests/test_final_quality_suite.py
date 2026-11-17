"""Budget, sparse coverage, matched training, native scoring, and kill-path checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
import run_final_quality_suite as suite
from final_quality_model import ScreenModel, allocate, relaxed_allocation
from final_quality_native import native, aggregate, frame_metrics
from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from wavefront_denoiser.allocation import allocate_samples
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_training import training_config


@pytest.mark.parametrize('quota', [0., .01, .05])
def test_quota_protects_missing_history_and_redistributes_exact_budget(quota):
    torch.manual_seed(21)
    logits = (torch.randn(2, 1, 10, 20)*4).requires_grad_()
    eligible = torch.ones_like(logits, dtype=torch.bool)
    eligible[:, :, :4] = False
    expected, counts = allocate(logits, 4, quota, eligible)
    assert torch.all(counts.flatten(1).sum(1) == 800)
    assert (counts[~eligible] >= 1).all()
    assert torch.all((counts == 0).flatten(1).sum(1) == int(quota*200))
    assert counts.max() <= 63
    (expected*torch.arange(200).reshape(1, 1, 10, 20)).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    if quota == 0:
        baseline = allocate_samples(logits, 3, policy='capped_softmax_v1', max_spp=62).expected+1
        torch.testing.assert_close(expected, baseline, rtol=0, atol=0)


def test_no_eligible_history_keeps_dense_and_nonfinite_rejected():
    logits = torch.zeros(1, 1, 10, 10)
    _, counts = allocate(logits, 4, .05, torch.zeros_like(logits, dtype=torch.bool))
    assert torch.all(counts == 4)
    with pytest.raises(ValueError): allocate(logits*float('nan'), 4)
    with pytest.raises(ValueError): allocate(logits, 4, .1)


def test_relaxation_allows_but_does_not_force_zeros():
    logits = torch.zeros(1, 1, 10, 20, requires_grad=True)
    eligible = torch.ones_like(logits, dtype=torch.bool)
    _, counts = relaxed_allocation(logits, 4, .05, eligible, 0.)
    assert counts.min() >= 1 and counts.sum() == 800
    with torch.no_grad():
        logits[..., :2, :] = -20
        logits[..., 5, 5] = 1000  # exercise a saturated active-set iteration during backward
    expected, counts = relaxed_allocation(logits, 4, .05, eligible, 0.)
    assert (counts == 0).sum() == 10 and counts.sum() == 800 and counts.max() == 63
    expected.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_backward_probe_exposes_zero_chain_and_native_metric_uses_all_frames():
    probe = suite.backward_probe()
    assert probe['count_0_linear'] == 3
    assert probe['count_0_encoded'] == probe['count_1_encoded'] == 0
    frames = [{'frame': i, 'learned': {'spatial_l1': float(i), 'temporal_l1': float(2*i),
                                    'unexpected_black_fraction': 0.}} for i in range(4)]
    m = aggregate(frames, 'learned')
    assert m['spatial_l1'] == 2 and m['temporal_l1'] == 5 and m['loss'] == 7
    a, target = frame_metrics(torch.zeros(3, 2, 2), torch.ones(3, 2, 2))
    assert a['unexpected_black_fraction'] == 1 and a['temporal_l1'] is None


def setup_fixture(root, tmp_path):
    torch.set_num_threads(1)
    path, _ = config_for(root, learning_rate=.0003, radiance_gradient_floor=.001)
    config = training_config(path)
    descriptor = suite.h.read_v5_dataset(root)
    for split in ('train', 'validation'): config[split+'_groups'] = descriptor['partitions'][split]
    config['reconstruction_version'] = 'pixel_mean_v2'
    model = JointNTASAD(**{k: config[k] for k in suite.h.MODEL_KEYS})
    checkpoint = tmp_path/'warm.pt'
    torch.save({'format': 'wavefront-training-readiness', 'model_spec': model.specification(),
                'state_dict': model.state_dict(), 'config': config, 'inventory_sha256': 'fixture'}, checkpoint)
    return config, checkpoint, model


def test_all_modes_training_native_and_matching(joint_dataset, tmp_path, monkeypatch):
    config, checkpoint, warm = setup_fixture(joint_dataset, tmp_path)
    base = tmp_path/'screen'; base.mkdir()
    policies = {k: dict(v, crop=7 if v['crop'] == 256 else 5) for k,v in suite.ARMS.items()}
    monkeypatch.setattr(suite, 'ARMS', policies)
    plan = {'checkpoint': str(checkpoint), 'epochs': 3}
    rows = {}
    with suite.h.execution_policy(config):
        for name in policies:
            s = suite.train_arm(base, name, config, plan, 'fixture', time.monotonic()+180, 'cpu')
            assert s['completed_epochs'] == 3 and s['controls']['learned']['complete']
            saved = torch.load(base/name/'last.pt', weights_only=True)
            assert saved['epoch_boundary'] and saved['global_step'] > 0
            if policies[name]['mode'] == 'fixed':
                assert all(torch.equal(v, saved['state_dict'][k]) for k,v in warm.state_dict().items() if k.startswith('sampler.'))
            rows[name] = [json.loads(x) for x in (base/name/'steps.jsonl').read_text().splitlines()]
            assert all(r['actual_spp'] == 4 for r in rows[name])
        assert [r['crop_sha256'] for r in rows['fixed-128']] == [r['crop_sha256'] for r in rows['joint-128']]
        assert [r['window_augmentation_sha256'] for r in rows['fixed-128']] == [r['window_augmentation_sha256'] for r in rows['fixed-256']]
        last = base/'zero-curriculum-128/last.pt'
        result = native(base, 'sparse', last, config, time.monotonic()+60, 'cpu')
        assert result['complete']
        assert result['zero_fraction_policy'] == .05
        for clip in result['clips'].values():
            assert len(clip['frames']) == clip['expected_frames'] == 8
            for frame in clip['frames']:
                assert frame['learned']['paths'] == frame['uniform']['paths']
                assert frame['learned']['zero_fraction'] <= .05
        decision = suite.make_report(base, 'completed')
        assert not decision['suite_complete'] and decision['selected_arm'] is None


@pytest.mark.parametrize('reason', ['deadline', 'memory', 'monitor_error'])
def test_watchdog_kills_child_on_limits_or_monitor_failure(tmp_path, monkeypatch, reason):
    def sample():
        if reason == 'monitor_error': raise RuntimeError('monitor unavailable')
        return {'host_total_bytes': 100, 'host_used_bytes': 79 if reason == 'memory' else 1,
                'gpu_total_bytes': 100, 'gpu_used_bytes': 1}
    monkeypatch.setattr(suite.h, 'memory_sample', sample)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    if reason == 'monitor_error':
        with pytest.raises(RuntimeError): suite.monitor(process, tmp_path, time.monotonic()+10)
    else:
        status = suite.monitor(process, tmp_path, time.monotonic()-1 if reason == 'deadline' else time.monotonic()+10)
        assert status == reason+'_guard_stopped' if reason == 'memory' else status == 'deadline_stopped'
    assert process.poll() is not None


def test_refuses_cpu_launch_before_writing_and_incomplete_native_cannot_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, 'WORKSPACE', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        suite.launch(tmp_path/'runs/new', {}, {})
    assert not (tmp_path/'runs/new').exists()
    assert suite.native_reasons({'complete': False})


def test_native_gate_requires_improvement_and_rejects_missing_outputs():
    metric = lambda x: {'loss': x, 'spatial_l1': x*.6, 'temporal_l1': x*.4}
    frame = {'paths': 400, 'zero_fraction': 0., 'unavailable_fraction': 0., 'unexpected_black_fraction': 0.}
    result = {'complete': True, 'zero_fraction_policy': 0., 'clips': {'clip': {
        'learned': metric(.99), 'uniform': metric(1.),
        'frames': [{'frame': 1, 'learned': dict(frame), 'uniform': dict(frame)}]}}}
    assert suite.native_reasons(result) == []
    changed = copy.deepcopy(result); changed['clips']['clip']['learned'] = metric(1.)
    assert suite.native_reasons(changed)
    changed = copy.deepcopy(result)
    changed['clips']['clip']['frames'][0]['learned']['unavailable_fraction'] = .0001
    assert suite.native_reasons(changed)
