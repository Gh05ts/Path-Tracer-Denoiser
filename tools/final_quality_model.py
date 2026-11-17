"""Explicit diagnostic allocation policies; production model defaults are untouched."""
import math

import torch

from wavefront_denoiser.allocation import allocate_samples
from wavefront_denoiser.joint import JointNTASAD, JointPrediction
from wavefront_denoiser.model import INPUT_CHANNELS
from wavefront_denoiser.ntasad_radiance import decode_radiance
from wavefront_denoiser.renderer_gradient import _RendererGradient
from wavefront_denoiser.sample_layers import replay_sample_layers


def exact_counts(expected, average_spp, active, minimum=None):
    """Largest remainder with stable index ties; masked pixels stay exactly zero."""
    if not torch.isfinite(expected).all() or (expected < 0).any() or (expected > 63.00001).any():
        raise ValueError('Invalid allocation')
    values = expected.detach().flatten(1).double().clamp(0, 63)
    enabled = active.flatten(1)
    lower = enabled.long() if minimum is None else minimum.flatten(1)
    counts = values.floor().long()
    fractions = values - counts
    for row in range(len(values)):
        difference = average_spp * values.shape[1] - int(counts[row].sum())
        eligible = enabled[row] & ((counts[row] < 63) if difference > 0 else (counts[row] > lower[row]))
        if abs(difference) > int(eligible.sum()):
            raise ValueError('Allocation cannot be repaired by largest remainder')
        if difference:
            score = fractions[row] if difference > 0 else -fractions[row]
            order = torch.argsort(score.masked_fill(~eligible, -torch.inf), descending=True, stable=True)
            counts[row, order[:abs(difference)]] += 1 if difference > 0 else -1
    if ((counts[~enabled] != 0).any() or (counts < lower).any()
            or not torch.all(counts.sum(1) == average_spp * values.shape[1])):
        raise RuntimeError('Exact budget or coverage contract failed')
    return counts.reshape_as(expected)


def allocate(logits, average_spp, zero_fraction=0., eligible=None):
    """Dense baseline, or a protected forced-zero quota for sparse robustness.

    Quota selection is discrete, based on the lowest logits in pixels with valid
    warped history and valid geometry. It is NOT a learned differentiable zero
    selector. This deliberately exercises holes even when warm logits are mild.
    Remaining pixels retain minimum one; all freed paths are redistributed.
    """
    if not math.isfinite(zero_fraction) or not 0 <= zero_fraction <= .05:
        raise ValueError('Zero quota must be in [0, 0.05]')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite sampler logits')
    active = torch.ones_like(logits, dtype=torch.bool)
    if zero_fraction:
        if eligible is None or eligible.shape != logits.shape:
            raise ValueError('Sparse allocation needs matching history eligibility')
        with torch.no_grad():
            scores, allowed, mask = logits.flatten(1), eligible.bool().flatten(1), active.flatten(1)
            for row in range(len(scores)):
                count = min(math.floor(zero_fraction * scores.shape[1]), int(allowed[row].sum()))
                order = torch.argsort(scores[row].masked_fill(~allowed[row], torch.inf), stable=True)
                mask[row, order[:count]] = False
    if active.all():
        expected = allocate_samples(logits, average_spp-1, policy='capped_softmax_v1', max_spp=62).expected + 1
    else:
        scores = logits.flatten(1).float()
        enabled = active.flatten(1)
        budget = average_spp * scores.shape[1] - enabled.sum(1, keepdim=True)
        if (budget > 62 * enabled.sum(1, keepdim=True)).any():
            raise ValueError('Insufficient sample-bank capacity')
        fixed = ~enabled
        saturated = torch.zeros_like(enabled)
        for _ in range(scores.shape[1]+1):
            remaining = budget - saturated.sum(1, keepdim=True) * 62
            free = torch.softmax(scores.masked_fill(fixed, -torch.inf), 1) * remaining
            overflow = (free > 62) & ~fixed
            if not overflow.any():
                extra = torch.where(saturated, 62., free)
                expected = torch.where(enabled, extra + 1, 0.).reshape_as(logits)
                break
            saturated |= overflow
            fixed = ~enabled | saturated
        else:
            raise RuntimeError('Capped allocation did not converge')
    return expected, exact_counts(expected, average_spp, active)


def relaxed_allocation(logits, average_spp, quota, eligible, floor):
    """Relax a bounded set of eligible pixels from a floor of one toward zero.

    Unlike forced holes, weak logits may yield no zeros. That outcome is reported
    as unexercised sparse behavior, not evidence of successful zero-count learning.
    """
    if not 0 <= floor <= 1 or not 0 <= quota <= .05: raise ValueError('Invalid relaxation')
    if not quota or floor == 1: return allocate(logits, average_spp)
    permitted = torch.zeros_like(logits, dtype=torch.bool)
    with torch.no_grad():
        for row in range(len(logits)):
            allowed = eligible[row].bool().flatten()
            k = min(math.floor(quota*allowed.numel()), int(allowed.sum()))
            order = torch.argsort(logits[row].flatten().masked_fill(~allowed, torch.inf), stable=True)
            permitted[row].view(-1)[order[:k]] = True
    scores = logits.flatten(1).float()
    lower = torch.where(permitted, floor, 1.).flatten(1)
    capacity = 63-lower
    budget = average_spp*scores.shape[1]-lower.sum(1, keepdim=True)
    fixed = torch.zeros_like(scores, dtype=torch.bool)
    for _ in range(scores.shape[1]+1):
        remaining = budget-(fixed*capacity).sum(1, keepdim=True)
        free = torch.softmax(scores.masked_fill(fixed, -torch.inf), 1)*remaining
        overflow = (free > capacity) & ~fixed
        if not overflow.any():
            expected = (lower+torch.where(fixed, capacity, free)).reshape_as(logits)
            break
        fixed = fixed | overflow
    else: raise RuntimeError('Relaxed allocation failed to converge')
    return expected, exact_counts(expected, average_spp, torch.ones_like(permitted), (~permitted).long())


class ScreenModel(JointNTASAD):
    def __init__(self, spec, zero_fraction=0., sparse_policy='forced', reserve_floor=1.):
        super().__init__(**spec)
        self.zero_fraction = zero_fraction
        self.sparse_policy = sparse_policy
        self.reserve_floor = reserve_floor

    def step(self, frame, layer, state, reference=None, variant='learned', shuffle_seed=0):
        rgb = frame['radiance']
        if state is None or variant == 'uniform':
            counts = torch.full_like(frame['sample_count'], self.average_spp, dtype=torch.int64)
            expected = counts.to(rgb.dtype)
        else:
            history = decode_radiance(state.radiance)
            logits = self.sampler.predict_logits(frame, history, history_available=state.valid, history_backend='gather')
            eligible = None
            if self.zero_fraction:
                _, valid = self.denoiser._warp(state, frame, rgb, backend='gather')
                eligible = valid & frame['guide_valid'].bool()
            if self.sparse_policy == 'relaxed':
                expected, counts = relaxed_allocation(logits, self.average_spp, self.zero_fraction, eligible, self.reserve_floor)
            else:
                expected, counts = allocate(logits, self.average_spp, self.zero_fraction, eligible)
            if variant == 'shuffled':
                if self.zero_fraction:
                    raise ValueError('Shuffling sparse maps would violate protected coverage')
                generator = torch.Generator(device=counts.device).manual_seed(shuffle_seed)
                counts = counts.flatten(2)
                order = torch.randperm(counts.shape[-1], device=counts.device, generator=generator)
                counts = counts[:, :, order].reshape_as(expected)
        image = replay_sample_layers(layer, counts)
        if reference is not None and state is not None and variant == 'learned':
            image = _RendererGradient.apply(image, expected, reference, counts, self.zero_policy)
        output = self.denoiser._step(dict(frame, radiance=image, sample_count=counts), state)
        return output, expected, counts

    def forward_sequence(self, buffers, layers, *, training_reference=None, transformed=False, variant='learned'):
        state = None
        outputs, validity, expected, counts = [], [], [], []
        for index in range(layers.shape[1]):
            frame = {k: buffers[k][:, index] for k in INPUT_CHANNELS}
            reference = training_reference[:, index] if training_reference is not None else None
            state, a, s = self.step(frame, layers[:, index], state, reference, variant, 704+index)
            outputs.append(state.radiance if transformed else decode_radiance(state.radiance))
            validity.append(state.valid); expected.append(a); counts.append(s)
        return JointPrediction(torch.stack(outputs, 1), torch.stack(validity, 1),
                               torch.stack(expected, 1), torch.stack(counts, 1))
