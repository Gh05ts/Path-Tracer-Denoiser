"""Numerical quality checks independent of a network or downloaded dataset."""
import json

import numpy as np
import pytest

from wavefront_denoiser.preview import display_rgb, tonemap_rgb
from wavefront_denoiser.preview_metrics import aggregate_quality, frame_quality, metric_protocol


def rgb(value):
    return np.full((2, 3, 3), value, dtype=np.float64)


def test_known_psnr_and_signed_temporal_error():
    first, residual = frame_quality(rgb(.2), rgb(.1))
    second, _ = frame_quality(rgb(.4), rgb(.2), residual)
    assert first['mse'] == pytest.approx(.01)
    assert first['rmse'] == pytest.approx(.1)
    assert first['psnr_db'] == pytest.approx(20)
    assert first['tpsnr_db'] is None and first['temporal_exact_match'] is None
    assert second['psnr_db'] == pytest.approx(-10*np.log10(.04))
    assert second['temporal_mse'] == pytest.approx(.01)
    assert second['tpsnr_db'] == pytest.approx(20)


def test_constant_bias_cancels_temporally_but_is_not_spatially_exact():
    values, _ = frame_quality(rgb(.25), rgb(0), rgb(.25))
    assert values['psnr_db'] == pytest.approx(-10*np.log10(.0625))
    assert not values['exact_match']
    assert values['temporal_mse'] == 0 and values['temporal_exact_match']
    assert values['tpsnr_db'] is None
    json.dumps(values, allow_nan=False)


def test_perfect_match_and_missing_history_are_distinct_json_safe_states():
    values, residual = frame_quality(rgb(.5), rgb(.5))
    assert values['mse'] == 0 and values['exact_match']
    assert values['psnr_db'] is None
    assert values['temporal_mse'] is None and values['temporal_exact_match'] is None
    following, _ = frame_quality(rgb(.7), rgb(.7), residual)
    assert following['temporal_mse'] == 0 and following['temporal_exact_match']
    assert following['tpsnr_db'] is None
    json.dumps(following, allow_nan=False)


def test_aggregation_excludes_seed_and_first_pair_and_pools_mse():
    rows, previous = [], None
    for value in (1., .1, .2):
        measured, previous = frame_quality(rgb(value), rgb(0), previous)
        rows.append(measured)
    result = aggregate_quality(rows)
    assert result['spatial_frames'] == 2 and result['temporal_pairs'] == 1
    assert result['mse'] == pytest.approx((.01+.04)/2)
    assert result['psnr_db'] == pytest.approx(-10*np.log10(.025))
    assert result['psnr_db'] != pytest.approx(np.mean([r['psnr_db'] for r in rows[1:]]))
    assert result['temporal_mse'] == pytest.approx(.01)


def test_short_clips_do_not_invent_quality_scores():
    one, _ = frame_quality(rgb(.1), rgb(0))
    result = aggregate_quality([one])
    assert result['spatial_frames'] == result['temporal_pairs'] == 0
    assert result['psnr_db'] is result['tpsnr_db'] is result['exact_match'] is None


def test_tone_map_precedes_temporal_difference_and_matches_png_transform():
    exposure = 1.7
    p0, r0, p1, r1 = (rgb(x) for x in (.4, .2, .8, .6))
    _, previous = frame_quality(tonemap_rgb(p0, exposure), tonemap_rgb(r0, exposure))
    values, _ = frame_quality(tonemap_rgb(p1, exposure), tonemap_rgb(r1, exposure), previous)
    expected = ((tonemap_rgb(p1, exposure)-tonemap_rgb(p0, exposure))
                -(tonemap_rgb(r1, exposure)-tonemap_rgb(r0, exposure)))
    assert values['temporal_mse'] == pytest.approx(np.square(expected).mean())
    assert values['temporal_mse'] > 0  # Constant linear-HDR bias need not cancel after tone mapping.
    linear = p1*exposure
    expected_png = np.uint8(np.clip((linear/(1+linear))**(1/2.2)*255, 0, 255))
    np.testing.assert_array_equal(display_rgb(p1, exposure), expected_png)
    assert metric_protocol(exposure)['peak'] == 1


@pytest.mark.parametrize('prediction,reference', [
    (np.zeros((2, 3)), rgb(0)), (rgb(-.1), rgb(0)), (rgb(float('nan')), rgb(0)),
    (rgb(1.1), rgb(0)), (rgb(0), np.zeros((3, 2, 3)))])
def test_invalid_metric_inputs_rejected(prediction, reference):
    with pytest.raises(ValueError, match='matching finite'):
        frame_quality(prediction, reference)


def test_invalid_history_rejected():
    with pytest.raises(ValueError, match='residual'):
        frame_quality(rgb(0), rgb(0), np.zeros((3, 2, 3)))
