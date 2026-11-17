"""PSNR and temporal PSNR on the preview's unquantized tone-mapped RGB."""
import math

import numpy as np


METHODS = ('uniform-4spp', 'adaptive-input', 'uniform-denoised', 'adaptive-denoised')
QUALITY_KEYS = ('mse', 'rmse', 'psnr_db', 'exact_match', 'temporal_mse',
                'temporal_rmse', 'tpsnr_db', 'temporal_exact_match')
PAPER = 'https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf'


def metric_protocol(exposure):
    return {'version': 'ntasad-preview-quality-v1', 'paper': PAPER,
            'paper_quality_metrics': ['psnr_db', 'tpsnr_db'],
            'space': 'unquantized float64 RGB after shared display transform, before PNG/GIF encoding',
            'transform': '(exposure*x/(1+exposure*x))**(1/2.2)', 'exposure': exposure,
            'peak': 1.0, 'channels': 'mean over all RGB channels and pixels',
            'psnr': '-10*log10(mean((T(prediction)-T(reference))**2))',
            'tpsnr': '-10*log10(mean(((T(pred[t])-T(pred[t-1]))-(T(ref[t])-T(ref[t-1])))**2))',
            'temporal': 'adjacent frames; no motion warp, absolute value or clipping of differences',
            'clip_spatial': 'frames 1 onward; frame 0 is initialization only',
            'clip_temporal': 'pairs 1->2 onward; pair 0->1 remains a per-frame diagnostic',
            'aggregation': 'mean frame MSE, then convert to RMSE/dB; not mean frame dB',
            'zero_error': 'MSE 0, exact_match true, dB null (mathematically +infinity)',
            'missing': 'null with zero contributing frames/pairs',
            'comparability': 'paper metric definitions with the documented preview tone map; '
                             'not an exact reproduction of paper scores or its evaluation pipeline'}


def _quality(mse, temporal_mse):
    def psnr(value):
        return -10*math.log10(value) if value is not None and value > 0 else None
    return {'mse': mse, 'rmse': math.sqrt(mse) if mse is not None else None,
            'psnr_db': psnr(mse), 'exact_match': mse == 0 if mse is not None else None,
            'temporal_mse': temporal_mse,
            'temporal_rmse': math.sqrt(temporal_mse) if temporal_mse is not None else None,
            'tpsnr_db': psnr(temporal_mse),
            'temporal_exact_match': temporal_mse == 0 if temporal_mse is not None else None}


def frame_quality(prediction, reference, previous_error=None):
    """Return metrics and signed RGB residual for the next temporal difference.

    Inputs are already tone-mapped HWC float RGB. Temporal MSE is the mean
    squared change in residual, equivalent to comparing image differences.
    """
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if (prediction.ndim != 3 or prediction.shape[-1] != 3 or min(prediction.shape) < 1
            or prediction.shape != reference.shape
            or any(not np.isfinite(x).all() or (x < 0).any() or (x > 1).any()
                   for x in (prediction, reference))):
        raise ValueError('Quality metrics require matching finite tone-mapped HWC RGB in [0, 1]')
    error = prediction.astype(np.float64)-reference.astype(np.float64)
    temporal_mse = None
    if previous_error is not None:
        previous_error = np.asarray(previous_error)
        if previous_error.shape != error.shape or not np.isfinite(previous_error).all():
            raise ValueError('Temporal residual must match the current finite RGB shape')
        temporal_mse = float(np.square(error-previous_error).mean())
    return _quality(float(np.square(error).mean()), temporal_mse), error


def aggregate_quality(rows):
    """Use identical post-initialization frame/pair sets for every method."""
    spatial = [row['mse'] for row in rows[1:]]
    temporal = [row['temporal_mse'] for row in rows[2:] if row['temporal_mse'] is not None]
    values = _quality(float(np.mean(spatial)) if spatial else None,
                      float(np.mean(temporal)) if temporal else None)
    return dict(values, spatial_frames=len(spatial), temporal_pairs=len(temporal))


def format_db(value, exact):
    return '+inf (exact)' if exact else 'n/a' if value is None else f'{value:.3f}'
