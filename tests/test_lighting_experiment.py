import copy
from pathlib import Path

import numpy as np
import pytest

from wavefront_denoiser import lighting_experiment as study


@pytest.fixture
def protocol():
    import json
    return json.loads((Path(__file__).parents[1] / 'configs/lighting_experiment_v1.json').read_text())


def write_config(tmp_path, protocol):
    import json
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    return path


def test_matched_camera_and_optimizer_budgets(protocol):
    groups = study.training_groups(protocol)
    train_images = {arm: len(ids) * len(protocol['cameras']['train']) for arm, ids in groups.items()}
    assert train_images == {'control': 16, 'diverse': 16}
    assert set(groups['control']) & set(groups['diverse']) == {'train_shared_sky'}
    # 64 random patches = 16 batches per epoch, for both interventions.
    assert train_images['control'] * protocol['training']['patches_per_frame'] == 64
    assert protocol['lighting']['large_area']['size'] ** 2 * protocol['lighting']['large_area']['emission'][0] == protocol['lighting']['small_area']['size'] ** 2 * protocol['lighting']['small_area']['emission'][0]


@pytest.mark.parametrize('problem', ['camera_overlap', 'seed_duplicate', 'negative_count', 'patch_too_large', 'unknown_option'])
def test_invalid_protocol_rejected(protocol, tmp_path, problem):
    if problem == 'camera_overlap':
        protocol['cameras']['probe'][0] = protocol['cameras']['train'][0]
    elif problem == 'seed_duplicate':
        protocol['seeds'] = [7, 7]
    elif problem == 'negative_count':
        protocol['reference_counts'] = [-1, 2048]
    elif problem == 'patch_too_large':
        protocol['training']['patch_size'] = 256
    else:
        protocol['test_scene_selection'] = 'sponza'
    with pytest.raises(ValueError):
        study.experiment_config(write_config(tmp_path, protocol))


def test_shadow_or_material_changes_cannot_pass_as_fixed_geometry(monkeypatch):
    from types import SimpleNamespace
    original = SimpleNamespace(metadata={'camera': {'position': [0, 0, 0]}, 'geometry_fingerprint': '123'},
                               buffers={k: np.ones((2, 2, 3), dtype=np.float32) for k in ['albedo', 'normal', 'geometric_normal', 'depth', 'guide_valid']})
    changed = copy.deepcopy(original)
    changed.buffers['depth'][0, 0] = 2
    monkeypatch.setattr(study, 'load_frame', lambda p: original if p == 'a' else changed)
    with pytest.raises(ValueError, match='visible depth'):
        study.assert_fixed_guides([['a'], ['b']])
    changed = copy.deepcopy(original)
    changed.metadata['camera']['position'][0] = 1
    with pytest.raises(ValueError, match='changed a camera'):
        study.assert_fixed_guides([['a'], ['b']])


def test_paired_analysis_keeps_seeds_matched():
    records = []
    for seed, multiplier in [(7, 1), (19, 10)]:
        for arm, factor, psnr in [('control', 1, 20), ('diverse', 0.8, 22)]:
            records.append({'seed': seed, 'arm': arm, 'probe': {'small': {'denoised_mean': {'linear_rmse': multiplier * factor, 'display_psnr_db': psnr}}}})
    paired = study.paired_comparison(records, [7, 19], ['small'])['small']
    assert paired['mean_hdr_rmse_change_fraction'] == pytest.approx(-0.2)
    assert paired['mean_display_psnr_gain_db'] == pytest.approx(2)
    with pytest.raises(ValueError, match='Missing matched'):
        study.paired_comparison(records[:-1], [7, 19], ['small'])


def test_reference_reuse_rejects_changed_calibration_protocol(tmp_path, protocol):
    old = tmp_path/'pilot'
    (old/'reference_convergence/captures').mkdir(parents=True)
    altered = copy.deepcopy(protocol)
    altered['width'] += 1
    study.write_json(old/'protocol.json', altered)
    protocol['reuse_reference_captures'] = 'pilot/reference_convergence/captures'
    path = write_config(tmp_path, protocol)
    with pytest.raises(ValueError, match='Reusable reference protocol changed'):
        study.prepare_experiment(path, tmp_path/'output', '/nonexistent-capture')
    assert not list((tmp_path/'output/reference_convergence').rglob('frame.exr'))


def test_reference_extension_must_increase_budget(tmp_path, protocol):
    protocol['extra_reference_counts'] = {'side_area_sky': [4096]}
    with pytest.raises(ValueError, match='extend the original ladder'):
        study.experiment_config(write_config(tmp_path, protocol))
