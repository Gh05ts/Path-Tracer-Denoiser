#!/usr/bin/env python3
"""Bounded final development screen. Preparation/plan inspection never trains."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import numpy as np
import torch
import diagnose_quality_collapse as h
from final_quality_model import ScreenModel
from final_quality_native import native
from wavefront_denoiser.checkpoint import _cpu_tree, _rng_state
from wavefront_denoiser.joint_data import JointPatchDataset
from wavefront_denoiser.ntasad_radiance import encode_radiance, radiance_gradient_policy
from wavefront_denoiser.renderer_gradient import differentiable_replay

WORKSPACE = Path(os.environ.get('QUALITY_WORKSPACE', '/home/ubuntu/quality_workspace'))
PILOT = WORKSPACE/'runs/bmw-living-quality-2h'
ARMS = {
    'fixed-128': {'crop': 128, 'mode': 'fixed'},
    'fixed-256': {'crop': 256, 'mode': 'fixed'},
    'joint-128': {'crop': 128, 'mode': 'joint'},
    'staged-128': {'crop': 128, 'mode': 'staged'},
    'zero-curriculum-128': {'crop': 128, 'mode': 'curriculum'},
    'zero-relaxation-128': {'crop': 128, 'mode': 'relaxation'},
}


class MatchedCrops(JointPatchDataset):
    """Identical window/augmentation draws; 128 crops center inside 256 crops."""
    def plan(self, index):
        if not self.augment:
            return super().plan(index)
        group, _, _ = self.sequences.windows[index//self.patches_per_window]
        spec = self.descriptor['groups'][group]['source_config']
        anchor = min(256, spec['height'], spec['width'])
        if self.patch_size > anchor:
            raise ValueError('Crop exceeds common anchor')
        offset = (anchor-self.patch_size)//2
        top = int(torch.randint(spec['height']-anchor+1, (1,)))+offset
        left = int(torch.randint(spec['width']-anchor+1, (1,)))+offset
        return (index, top, left, bool(torch.randint(2, (1,))), bool(torch.randint(2, (1,))), int(torch.randint(4, (1,))))

    def load_plan(self, plan):
        result = super().load_plan(plan)
        result['crop_plan'] = torch.tensor(plan, dtype=torch.int64)
        result['window_augmentation'] = torch.tensor([plan[i] for i in (0, 3, 4, 5)], dtype=torch.int64)
        return result


def dataset(config, partition):
    validation = partition == 'validation'
    return MatchedCrops(config['data_root'], partition=partition, sequence_ids=config[partition+'_groups'],
        length=config['window_length'], patch_size=config['validation_patch_size'] if validation else config['patch_size'],
        patches_per_window=config['validation_crops_per_window'] if validation else config['patches_per_window'],
        augment=not validation, cache_maps=config.get('loader_policy') is not None,
        cache_frames=config.get('decoded_cache_frames', 16), fast_crops=config.get('loader_policy') == 'prefetch_fused_v1')


def phase(mode, epoch, epochs):
    warm = max(1, epochs//3)
    if mode == 'fixed' or (mode in ('staged', 'curriculum', 'relaxation') and epoch <= warm):
        return False, 0.
    if mode in ('curriculum', 'relaxation'):
        # Two checked stages, each following dense warm-up.
        return True, .01 if epoch <= warm + max(1, (epochs-warm)//2) else .05
    return True, 0.


def load_model(path, config, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') not in ('wavefront-training-readiness', 'wavefront-final-quality-suite'):
        raise ValueError('Expected the reviewed readiness checkpoint or a suite checkpoint')
    spec = {k: config[k] for k in h.MODEL_KEYS}
    if payload['model_spec'] != spec:
        raise ValueError('Checkpoint architecture or rendering policies differ')
    model = ScreenModel(spec, payload.get('zero_fraction', 0.), payload.get('sparse_policy', 'forced'), payload.get('reserve_floor', 1.))
    model.load_state_dict(payload['state_dict']); model.to(device)
    return model, payload


def save_checkpoint(path, model, optimizer, config, arm, epoch, step, generator, inventory_sha, selection):
    payload = {'format': 'wavefront-final-quality-suite', 'version': 1, 'diagnostic_only': True,
        'model_spec': model.specification(), 'state_dict': _cpu_tree(model.state_dict()),
        'optimizer': _cpu_tree(optimizer.state_dict()), 'rng': _cpu_tree(_rng_state(generator)),
        'config': config, 'arm': arm, 'completed_epochs': epoch, 'global_step': step,
        'zero_fraction': model.zero_fraction, 'inventory_sha256': inventory_sha,
        'sparse_policy': model.sparse_policy, 'reserve_floor': model.reserve_floor,
        'selection': selection, 'epoch_boundary': True, 'torch_version': str(torch.__version__),
        'resume_note': 'Audit state retained; this short screen does not expose automatic resume.'}
    temp = path.with_suffix('.partial'); torch.save(payload, temp); temp.replace(path)


@torch.inference_mode()
def evaluate(model, data, config, deadline, variant='learned', device='cuda'):
    totals, per_scene, windows, paths, pixels = {}, {}, 0, 0, 0
    model.eval()
    with h.batches(data, config, torch.Generator().manual_seed(0)) as batches:
        for batch in batches:
            if time.monotonic() >= deadline:
                break
            buffers, layers, reference = h._batch(batch, device)
            pred = model.forward_sequence(buffers, layers, transformed=True, variant=variant)
            metrics = h._metrics(pred, reference, model.max_spp)
            if not torch.all(pred.counts.flatten(3).sum(-1) == model.average_spp*pred.counts.shape[-2]*pred.counts.shape[-1]):
                raise ValueError('Crop integer budget mismatch')
            values = {k: float(v) for k, v in metrics.items()}
            n = len(reference)
            if n != 1: raise ValueError('Per-clip validation requires batch one')
            windows += n; paths += int(pred.counts[:, 1:].sum()); pixels += pred.counts[:, 1:].numel()
            scene = data.descriptor['groups'][batch['sequence_id'][0]]['scene']
            record = per_scene.setdefault(scene, {'windows': 0, 'totals': {}}); record['windows'] += 1
            for k, v in values.items():
                totals[k] = totals.get(k, 0.)+v*n
                record['totals'][k] = record['totals'].get(k, 0.)+v
    return {'complete': windows == len(data), 'windows': windows, 'expected_windows': len(data),
            'metrics': {k: v/windows for k, v in totals.items()} if windows else None,
            'per_scene': {k: {m: v/r['windows'] for m, v in r['totals'].items()} for k, r in per_scene.items()},
            'paths': paths, 'pixels': pixels}


def backward_probe():
    """Expose both the renderer surrogate and its blocked exact-black chain."""
    result = {}
    with radiance_gradient_policy(1e-3):
        for count in (0, 1, 4):
            for encoded in (False, True):
                expected = torch.full((1, 1, 1, 1), float(count), requires_grad=True)
                rendered = differentiable_replay(torch.zeros(1, 6, 3, 1, 1), expected,
                                                torch.ones(1, 3, 1, 1), zero_policy='unit_denominator')
                value = encode_radiance(rendered.radiance) if encoded else rendered.radiance
                value.sum().backward()
                result[f'count_{count}_'+('encoded' if encoded else 'linear')] = float(expected.grad)
    result['interpretation'] = 'Finite DASR denominator does not remove exact-black or availability-mask gradient blocking. Forced zero quota does not differentiate hole selection.'
    return result


def train_arm(base, name, original_config, plan, inventory_sha, deadline, device='cuda'):
    arm = dict(ARMS[name], name=name)
    config = dict(original_config, patch_size=arm['crop'], radiance_gradient_floor=1e-3)
    out = base/name; out.mkdir()
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    model, payload = load_model(plan['checkpoint'], config, device); del payload
    optimizer = torch.optim.Adam([{'params': model.denoiser.parameters(), 'lr': 1e-4, 'name': 'denoiser'},
                                  {'params': model.sampler.parameters(), 'lr': 1e-5, 'name': 'sampler'}])
    train, validation = dataset(config, 'train'), dataset(config, 'validation')
    history, stages, step, completed, bad_epochs = [], [], 0, 0, 0
    selected, best_loss, stop = {'epoch': 0, 'zero_fraction': 0.}, float('inf'), 'epoch_target'
    started = time.monotonic(); last_phase = (False, 0.)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    h.write(out/'config.json', config); h.write(out/'arm.json', arm)
    try:
        initial = evaluate(model, validation, config, deadline, device=device)
        if not initial['complete']: raise TimeoutError('Initial validation incomplete')
        best_loss = initial['metrics']['loss']
        save_checkpoint(out/'best.pt', model, optimizer, config, arm, 0, 0, generator, inventory_sha, selected)
        h.write(out/'initial-validation.json', initial)
        for epoch in range(1, plan['epochs']+1):
            # Reserve one full validation and final controls within this arm.
            if time.monotonic() >= deadline-60:
                stop = 'arm_deadline'; break
            active, quota = phase(arm['mode'], epoch, plan['epochs'])
            if (active, quota) != last_phase and epoch > 1:
                # Restore the best weights+Adam state before each new stage.
                best = torch.load(out/'best.pt', map_location=device, weights_only=True)
                model.load_state_dict(best['state_dict']); optimizer.load_state_dict(best['optimizer'])
                stages.append({'epoch': epoch, 'restored_best_epoch': best['completed_epochs'], 'new_quota': quota})
                del best
                bad_epochs = 0
            model.sampler.requires_grad_(active); model.zero_fraction = quota
            model.sparse_policy = 'relaxed' if arm['mode'] == 'relaxation' else 'forced'
            model.reserve_floor = 1. if quota == 0 else .5 if quota == .01 else 0.
            last_phase = (active, quota)
            model.train(); totals = {}; updates = 0; train_started = time.monotonic()
            epoch_complete = True; anomaly = None
            with h.batches(train, config, generator, shuffle=True) as batches:
                for batch in batches:
                    if time.monotonic() >= deadline-45:
                        epoch_complete = False; stop = 'arm_deadline'; break
                    buffers, layers, reference = h._batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model.forward_sequence(buffers, layers, training_reference=reference, transformed=True)
                    metrics = h._metrics(prediction, reference, model.max_spp)
                    metrics['loss'].backward()
                    grads = [[p.grad for p in net.parameters() if p.grad is not None] for net in (model.sampler, model.denoiser)]
                    if not all(h.finite_group_flags(grads, device)):
                        raise RuntimeError('Nonfinite gradients; arm stopped before optimizer update')
                    norms = [float(torch.stack([g.detach().norm() for g in values]).norm()) if values else 0. for values in grads]
                    row = {'epoch': epoch, 'step': step+1, 'quota': quota, 'sampler_gradient_norm': norms[0],
                           'denoiser_gradient_norm': norms[1], **{k: float(v) for k, v in metrics.items()},
                           'crop_sha256': hashlib.sha256(batch['crop_plan'].numpy().tobytes()).hexdigest(),
                           'window_augmentation_sha256': hashlib.sha256(batch['window_augmentation'].numpy().tobytes()).hexdigest()}
                    h.append(out/'steps.jsonl', row)
                    if max(norms) > 100 or row['unavailable_fraction'] > 0 or row['saturated_fraction'] > .01:
                        anomaly = row; stop = 'anomaly_rollback'; epoch_complete = False
                        h.write(out/'anomaly.json', row); break
                    optimizer.step()
                    if not all(h.finite_group_flags([list(model.parameters())], device)):
                        raise RuntimeError('Nonfinite weights')
                    step += 1; updates += 1
                    for k, v in row.items():
                        if isinstance(v, (int, float)): totals[k] = totals.get(k, 0.)+v
            seconds = time.monotonic()-train_started
            if not epoch_complete:
                # Preserve previous epoch boundary last.pt. Never label a partial
                # update stream as an exact-resume checkpoint.
                break
            completed = epoch
            validation_result = evaluate(model, validation, config, deadline-25, device=device)
            row = {'epoch': epoch, 'complete': validation_result['complete'], 'updates': updates, 'quota': quota,
                   'train_seconds': seconds, 'windows_per_second': updates*config['batch_size']/max(seconds, 1e-9),
                   'pixel_frames_per_second': updates*config['batch_size']*config['window_length']*config['patch_size']**2/max(seconds, 1e-9),
                   'train': {k: v/max(updates, 1) for k, v in totals.items()}, 'validation': validation_result,
                   'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups}}
            history.append(row); h.append(out/'history.jsonl', row)
            h.write(base/'progress.json', {'stage': 'training', 'arm': name, 'epoch': epoch, 'step': step})
            if not validation_result['complete']:
                stop = 'validation_incomplete'; break
            loss = validation_result['metrics']['loss']
            if loss < best_loss:
                best_loss = loss; bad_epochs = 0; selected = {'epoch': epoch, 'zero_fraction': quota}
                save_checkpoint(out/'best.pt', model, optimizer, config, arm, epoch, step, generator, inventory_sha, selected)
            else:
                bad_epochs += 1
                for group in optimizer.param_groups: group['lr'] *= .5
            save_checkpoint(out/'last.pt', model, optimizer, config, arm, epoch, step, generator, inventory_sha, selected)
            print(f'{name} epoch={epoch} loss={loss:.6f} best={best_loss:.6f} quota={quota}', flush=True)
            if bad_epochs >= 3:
                stop = 'validation_early_stop'; break
        model = None; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
        model, best = load_model(out/'best.pt', config, device)
        controls = {}
        for variant in ('learned', 'uniform', 'shuffled'):
            if variant == 'shuffled' and model.zero_fraction:
                controls[variant] = {'complete': False, 'reason': 'Sparse shuffle violates protected-hole contract'}; continue
            controls[variant] = evaluate(model, validation, config, deadline, variant, device)
        summary = {'arm': arm, 'completed_epochs': completed, 'global_step': step, 'stop_reason': stop,
                   'best': selected, 'best_loss': best_loss, 'controls': controls, 'stages': stages,
                   'elapsed_seconds': time.monotonic()-started, 'maximum_tested_quota': max([0.]+[r['quota'] for r in history]),
                   'maximum_observed_zero_fraction': max([0.]+[r['train']['zero_fraction'] for r in history]),
                   'training_complete': completed == plan['epochs'], 'test_pixels_loaded': False,
                   'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                   'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else None}
        h.write(out/'summary.json', summary)
        return summary
    finally:
        train.close(); validation.close()
        del model, optimizer
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()


def native_reasons(result, baseline=None):
    reasons = []
    if not result or not result.get('complete'): return ['native coverage incomplete']
    learned_total = sum(c['learned']['loss'] for c in result['clips'].values())
    uniform_total = sum(c['uniform']['loss'] for c in result['clips'].values())
    if learned_total > uniform_total*.999:
        reasons.append('native allocation improvement below 0.1%')
    for group, clip in result['clips'].items():
        for metric in ('loss', 'spatial_l1', 'temporal_l1'):
            if clip['learned'][metric] > clip['uniform'][metric] * 1.001:
                reasons.append(f'{group}: {metric} regresses >0.1% against uniform')
            if baseline and baseline.get('complete') and clip['learned'][metric] > baseline['clips'][group]['learned'][metric]*1.005:
                reasons.append(f'{group}: {metric} regresses >0.5% against warm checkpoint')
        for frame in clip['frames']:
            a, u = frame['learned'], frame['uniform']
            if a['paths'] != u['paths'] or a['unavailable_fraction']:
                reasons.append(f'{group}: missing output or unequal budget'); break
            if a['zero_fraction'] > result['zero_fraction_policy']+1e-6:
                reasons.append(f'{group}: zero quota exceeded'); break
            if frame['frame'] and a['unexpected_black_fraction'] > u['unexpected_black_fraction']+.001:
                reasons.append(f'{group}: excess unexpected black'); break
    return reasons


def make_report(base, status):
    native_results, summaries = {}, {}
    for p in (base/'native').glob('*/summary.json') if (base/'native').exists() else []:
        native_results[p.parent.name] = h.read(p)
    for name in ARMS:
        p = base/name/'summary.json'
        if p.exists(): summaries[name] = h.read(p)
    baseline = native_results.get('warm-start')
    candidates = []
    reasons_by_arm = {}
    for name in ARMS:
        s = summaries.get(name); n = native_results.get(name+'-best')
        reasons = native_reasons(n, baseline)
        if not s: reasons.append('training summary missing')
        elif s['stop_reason'] not in ('epoch_target', 'validation_early_stop'):
            reasons.append('training interrupted or anomaly captured')
        if s:
            controls = s['controls']
            if name.startswith('zero-'):
                if s['maximum_observed_zero_fraction'] == 0:
                    reasons.append('zero-sample behavior was not exercised in training')
                if s['best']['zero_fraction'] == 0:
                    reasons.append('selected best remains a dense policy; no sparse candidate established')
            if not all(controls.get(k, {}).get('complete') for k in ('learned', 'uniform')):
                reasons.append('crop control coverage incomplete')
            elif controls['learned']['metrics']['loss'] > controls['uniform']['metrics']['loss']*.999:
                reasons.append('crop allocation improvement below 0.1%')
            if s['best']['zero_fraction'] == 0 and (not controls['shuffled'].get('complete') or
                    controls['learned']['metrics']['loss'] >= controls['shuffled']['metrics']['loss']):
                reasons.append('placement does not beat shuffled counts or audit incomplete')
            if s['best']['epoch'] == 0: reasons.append('best remains warm initialization')
        reasons_by_arm[name] = reasons
        if not reasons:
            loss = sum(c['learned']['loss'] for c in n['clips'].values())/len(n['clips'])
            candidates.append((loss, name))
    complete = (len(summaries) == len(ARMS) and baseline is not None and baseline.get('complete')
                and all(native_results.get(n+'-best', {}).get('complete') for n in ARMS)
                and native_results.get('zero-curriculum-last', {}).get('complete', False)
                and native_results.get('zero-relaxation-last', {}).get('complete', False)
                and summaries.get('zero-curriculum-128', {}).get('maximum_tested_quota', 0) >= .05
                and summaries.get('zero-relaxation-128', {}).get('maximum_tested_quota', 0) >= .05
                and all(s['stop_reason'] in ('epoch_target', 'validation_early_stop')
                        and all(s['controls'].get(k, {}).get('complete') for k in ('learned', 'uniform'))
                        for s in summaries.values()))
    selected = min(candidates)[1] if candidates and complete and status == 'completed' else None
    # Verify the recorded matching claims rather than relying only on seeds.
    streams = {}
    for name in ARMS:
        path = base/name/'steps.jsonl'
        if path.exists():
            rows = []
            for line in path.read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: break  # a killed worker can leave one partial trailing row
            streams[name] = rows
    shared = min((len(v) for v in streams.values()), default=0)
    matched_windows = len(streams) == len(ARMS) and shared > 0 and all(
        len({rows[i]['window_augmentation_sha256'] for rows in streams.values()}) == 1 for i in range(shared))
    same_crops = [v for k, v in streams.items() if ARMS[k]['crop'] == 128]
    matched_crops = bool(same_crops) and shared > 0 and all(
        len({rows[i]['crop_sha256'] for rows in same_crops}) == 1 for i in range(shared))
    if not (matched_windows and matched_crops): selected = None; complete = False
    decision = {'suite_complete': bool(complete), 'selected_arm': selected,
                'selected_checkpoint': str(base/selected/'best.pt') if selected else None,
                'matched_window_augmentation_prefix': matched_windows, 'matched_128_crop_prefix': matched_crops,
                'shared_recorded_steps': shared,
                'relaxation_produced_training_zeros': summaries.get('zero-relaxation-128', {}).get('maximum_observed_zero_fraction', 0) > 0,
                'reasons_by_arm': reasons_by_arm, 'paper_quality_established': False,
                'full_data_run_prepared': False, 'test_pixels_loaded': False,
                'scope': 'Single-seed pilot development screen; no automatic final launch.'}
    h.write(base/'decision.json', decision)
    rows = ['# Final quality development screen', '', f'Launcher status: **{status}**. Complete coverage: **{bool(complete)}**.', '',
            '| Arm | Epochs | Best epoch / zero quota | Crop loss | Native learned/uniform delta | Stop |',
            '|---|---:|---|---:|---:|---|']
    for name in ARMS:
        s = summaries.get(name); n = native_results.get(name+'-best')
        delta = 'incomplete'
        if n and n.get('complete'):
            a = sum(c['learned']['loss'] for c in n['clips'].values())
            u = sum(c['uniform']['loss'] for c in n['clips'].values())
            delta = f'{100*(a/u-1):+.3f}%'
        rows.append(f"| {name} | {s['completed_epochs'] if s else '—'} | {str(s['best']) if s else '—'} | {s['best_loss'] if s else '—'} | {delta} | {s['stop_reason'] if s else 'missing'} |")
    rows += ['', f'Selected candidate: **{selected or "none; inspect decision.json"}**.', '',
             'Native scores cover all learned frames and all pairs of learned frames in all 12 selected development clips. Frame zero and the initialization pair are recorded separately. Lower loss is better. Native learned/uniform delta is 100 * (learned_loss / uniform_loss - 1); positive values mean learned allocation is worse than uniform.', '',
             'Zero-relaxation reduces the reserve floor 1 -> 0.5 -> 0 in protected eligible pixels, with at most 1%, then 5% eligible for zeros. Zero-curriculum is a separate forced-hole stress test at those quotas. Neither differentiates the discrete selector or repairs exact-black gradients. No observed zeros means relaxation did not exercise sparse behavior; see decision.json and backward-probe.json.', '',
             'Crop comparisons use the same windows, augmentations, batch size, seed and maximum optimizer-update budget; 128 crops share centers with 256 crops. Larger crops process four times as many pixels. Early stopping and rollbacks are explicit in histories.', '',
             'Source, input hashes, per-step gradients, phase rollbacks, exact budgets, best/last checkpoints, per-frame native metrics, fixed-exposure PNGs and GIFs are retained. No paper-quality equivalence or full-data readiness follows from this screen.']
    (base/'report.md').write_text('\n'.join(rows)+'\n')
    return decision


def worker(base, plan, config):
    start = time.monotonic(); deadline = start+plan['minutes']*60-10
    # Cap allocator under external GPU usage with additional nonallocator margin.
    free, total = torch.cuda.mem_get_info(0)
    allocator_budget = min(.75*total, .79*total-(total-free)-512*2**20)
    if allocator_budget <= 0: raise RuntimeError('Insufficient memory headroom')
    torch.cuda.set_per_process_memory_fraction(allocator_budget/total, 0)
    with h.execution_policy(dict(config, radiance_gradient_floor=1e-3)), h.joint_dataset_read_lock(config['data_root']):
        provenance = h.read(base/'provenance.json')
        for relative, expected in provenance['source_sha256'].items():
            if h.digest(base/'source'/relative) != expected: raise ValueError('Snapshot changed')
        if h.digest(Path(plan['checkpoint'])) != provenance['checkpoint_sha256']: raise ValueError('Warm checkpoint changed')
        h.write(base/'progress.json', {'stage': 'verify_data'})
        inventory = h.joint_inventory(config['data_root'], {p: config[p+'_groups'] for p in ('train', 'validation')})
        if inventory != h.read(plan['inventory']): raise ValueError('Selected data inventory changed')
        h.write(base/'inventory.json', inventory)
        inventory_sha = h.digest(base/'inventory.json')
        h.write(base/'backward-probe.json', backward_probe())
        train_deadline = start+plan['minutes']*60*.5
        for index, name in enumerate(ARMS):
            names_left = list(ARMS)[index:]
            weights = {n: 2. if ARMS[n]['crop'] == 256 else 1. for n in names_left}
            arm_deadline = min(train_deadline, time.monotonic()+max(0, train_deadline-time.monotonic())*weights[name]/sum(weights.values()))
            if arm_deadline-time.monotonic() < 90:
                h.write(base/f'{name}-skipped.json', {'reason': 'training budget exhausted'}); continue
            try:
                train_arm(base, name, config, plan, inventory_sha, arm_deadline)
            except (ValueError, RuntimeError, OSError, TimeoutError) as error:
                folder = base/name; folder.mkdir(exist_ok=True)
                h.write(folder/'failure.json', {'error': str(error)})
                print(f'{name}: {error}', flush=True)
                if isinstance(error, torch.cuda.OutOfMemoryError): raise
            finally:
                gc.collect(); torch.cuda.empty_cache()
        cases = [('warm-start', Path(plan['checkpoint']))]
        cases += [(name+'-best', base/name/'best.pt') for name in ARMS]
        cases += [('zero-curriculum-last', base/'zero-curriculum-128/last.pt')]
        cases += [('zero-relaxation-last', base/'zero-relaxation-128/last.pt')]
        for i, (label, path) in enumerate(cases):
            case_deadline = min(deadline, time.monotonic()+max(0, deadline-time.monotonic())/(len(cases)-i))
            if not path.is_file() or case_deadline <= time.monotonic(): continue
            try:
                native(base, label, path, config, case_deadline)
            except (ValueError, RuntimeError, OSError) as error:
                h.write(base/f'native-{label}-failure.json', {'error': str(error)})
                if isinstance(error, torch.cuda.OutOfMemoryError): raise
        make_report(base, 'completed')


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def monitor(process, base, deadline):
    status = 'completed'
    try:
        while process.poll() is None:
            sample = {'time': time.time(), **h.memory_sample()}
            h.append(base/'resources.jsonl', sample)
            if any(sample[k+'_used_bytes'] >= .79*sample[k+'_total_bytes'] for k in ('host', 'gpu')):
                status = 'memory_guard_stopped'; break
            if time.monotonic() >= deadline:
                status = 'deadline_stopped'; break
            time.sleep(.5)
        if status == 'completed' and process.returncode != 0: status = 'worker_failed'
    except BaseException:
        stop_process(process)
        raise
    finally:
        stop_process(process)
    return status


def launch(base, plan, config):
    if base.exists(): raise FileExistsError('Output already exists; choose a fresh directory')
    if not base.is_relative_to((WORKSPACE/'runs').resolve()): raise ValueError('Outputs must be beneath workspace/runs')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; run on the GPU host')
    sample = h.memory_sample()
    if any(sample[k+'_used_bytes'] >= .79*sample[k+'_total_bytes'] for k in ('host', 'gpu')):
        raise RuntimeError('Memory already exceeds guard')
    base.mkdir(parents=True)
    h.write(base/'plan.json', plan); h.write(base/'config.json', config)
    sources = list((REPO/'src').rglob('*.py')) + [REPO/'tools'/n for n in (
        'diagnose_quality_collapse.py', 'run_final_quality_suite.py', 'final_quality_model.py', 'final_quality_native.py')]
    hashes = {}
    for source in sources:
        relative = source.relative_to(REPO); destination = base/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        hashes[str(relative)] = h.digest(destination)
    h.write(base/'provenance.json', {'source_sha256': hashes, 'checkpoint_sha256': h.digest(Path(plan['checkpoint'])),
                                  'inventory_sha256': h.digest(Path(plan['inventory'])), 'python': sys.executable,
                                  'torch': str(torch.__version__), 'data_copy': False})
    env = dict(os.environ, PYTHONPATH=str(base/'source/src')+':'+str(WORKSPACE/'.test-deps'), PYTHONUNBUFFERED='1')
    start = time.monotonic(); status = 'launcher_failed'
    h.write(base/'status.json', {'status': 'running'})
    try:
        with (base/'run.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(base/'source/tools/run_final_quality_suite.py'),
                                        '--worker', '--output', str(base)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status = monitor(process, base, start+plan['minutes']*60)
    except BaseException as error:
        h.write(base/'launcher-error.json', {'error': str(error)})
        raise
    finally:
        h.write(base/'status.json', {'status': status, 'wall_seconds': time.monotonic()-start})
        decision = make_report(base, status)
        print(f'{status}; suite_complete={decision["suite_complete"]}; report={base/"report.md"}', flush=True)
    return 0 if status == 'completed' and decision['suite_complete'] else 1


def prepare(args):
    if not math.isfinite(args.minutes) or args.minutes < 10: raise ValueError('Use at least 10 minutes; default 60')
    if args.epochs < 3: raise ValueError('At least three epochs are needed to exercise all curriculum stages')
    config = h.training_config(args.config)
    if ('r2_streaming' in config or config['average_spp'] != 4 or config['max_spp'] != 63
            or config['reconstruction_version'] != 'pixel_mean_v2' or config['allocation_policy'] != 'capped_softmax_v1'
            or config.get('validation_batch_size', 1) != 1 or config['zero_policy'] != 'unit_denominator'):
        raise ValueError('Unexpected data, architecture, count, or validation policy')
    descriptor = h.read_v5_dataset(config['data_root'])
    for partition in ('train', 'validation'):
        if not config[partition+'_groups'] or not set(config[partition+'_groups']) <= set(descriptor['partitions'][partition]):
            raise ValueError('Selected groups cross partition or are empty')
    if set(config['train_groups']) & set(config['validation_groups']): raise ValueError('Split overlap')
    model, payload = load_model(args.checkpoint, config)
    inventory = Path(args.inventory).resolve()
    if payload['inventory_sha256'] != h.digest(inventory): raise ValueError('Checkpoint/inventory mismatch')
    # The only permitted training configuration change here is crop size. The
    # caller may choose a different explicitly prepared inventory in a later run.
    if any(payload['config'].get(k) != config.get(k) for k in ('data_root', 'train_groups', 'validation_groups')):
        raise ValueError('Source checkpoint uses a different dataset selection')
    del model, payload
    plan = {'version': 1, 'minutes': args.minutes, 'epochs': args.epochs, 'checkpoint': str(Path(args.checkpoint).resolve()),
            'inventory': str(inventory), 'arms': ARMS, 'native_groups': config['validation_groups'],
            'training_fraction_of_budget': .5, 'torch_allocator_fraction_ceiling': .75,
            'whole_host_gpu_stop_fraction': .79, 'user_memory_limit': .8, 'test_pixels_loaded': False,
            'training_scope': f"{len(config['train_groups'])} pilot training clips; {len(config['validation_groups'])} development clips",
            'zero_policy_note': 'Relaxed reserve 1 -> 0.5 -> 0 with protected zero eligibility 0 -> 1% -> 5%; separate forced-hole stress control. Hard selection remains nondifferentiable.'}
    return plan, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=WORKSPACE/'configs/quality-bmw-living-2h.json')
    parser.add_argument('--checkpoint', type=Path, default=PILOT/'training-readiness/results/fixed-sampler/best.pt')
    parser.add_argument('--inventory', type=Path, default=PILOT/'training-readiness/results/inventory.json')
    parser.add_argument('--output', type=Path, default=WORKSPACE/'runs/final-quality-screen-20261117')
    parser.add_argument('--minutes', type=float, default=60.)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv); base = args.output.resolve()
    if args.report_only:
        make_report(base, h.read(base/'status.json')['status']); return 0
    if args.worker:
        worker(base, h.read(base/'plan.json'), h.read(base/'config.json')); return 0
    plan, config = prepare(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return 0
    return launch(base, plan, config)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Final quality screen: {error}', file=sys.stderr); raise SystemExit(1)
