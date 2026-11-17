"""Explicit raw and bounded sample-map policies for the finite v5 sample bank."""

import math

import torch

from .adaptive import SampleMap, normalize_sample_map
from .sample_layers import MAX_SPP


POLICIES = ("paper_softmax_v1", "capped_softmax_v1", "minimum_one_exact_v1")


def allocate_samples(logits: torch.Tensor, average_spp: float, *, policy: str, max_spp: int = MAX_SPP) -> SampleMap:
    """Normalize per image, with explicitly selected finite-capacity behavior.

    paper_softmax_v1 uses NTASAD equation (2), rejecting rounded overflow.
    capped_softmax_v1 redistributes overflow proportionally among free pixels:
    y_i=min(max_spp, lambda*exp(logit_i)), sum_i y_i=H*W*average_spp.
    This bounded policy is an engineering variant, not the published equation.

    minimum_one_exact_v1 reserves one path per pixel and distributes the rest
    with capped softmax, then uses stable largest-remainder integer rounding.
    This is the dense collapse-prevention policy from the development runs.

    The active-set softmax stays stable even for very large finite logits.
    Ordinary autograd differentiates the final free-pixel softmax; saturated
    allocations have zero derivative. At exact saturation equality a pixel is
    considered free. Independent round-to-even can change the INTEGER budget.
    """
    if policy not in POLICIES:
        raise ValueError(f"allocation_policy must be one of {POLICIES}")
    if type(max_spp) is not int or not 1 <= max_spp <= MAX_SPP:
        raise ValueError(f"max_spp must be an integer in [1,{MAX_SPP}]")
    if (isinstance(average_spp, bool) or not isinstance(average_spp,(int,float))
            or not math.isfinite(average_spp) or not 0 < average_spp <= max_spp):
        raise ValueError("average_spp must be positive and no larger than max_spp")
    if (logits.ndim != 4 or logits.shape[1] != 1 or min(logits.shape) < 1
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating N1HW")
    if policy == "minimum_one_exact_v1":
        if type(average_spp) is not int or average_spp < 1:
            raise ValueError("Minimum-one allocation requires integer average_spp >= 1")
        if average_spp == 1:
            expected = logits * 0 + 1
        else:
            free = allocate_samples(logits, average_spp-1,
                                    policy="capped_softmax_v1", max_spp=max_spp-1)
            expected = free.expected + 1
        return SampleMap(expected, exact_counts(expected, average_spp, maximum=max_spp))
    if policy == "paper_softmax_v1":
        result = normalize_sample_map(logits,average_spp)
        if (result.counts > max_spp).any():
            raise ValueError("Paper allocation exceeds sample-bank capacity; choose an explicit bounded policy")
        return result
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    scores = logits.flatten(1).to(dtype)
    pixels = scores.shape[1]
    budget = average_spp*pixels
    if average_spp == max_spp:
        expected = logits.to(dtype)*0 + max_spp
    else:
        fixed = torch.zeros_like(scores,dtype=torch.bool)
        # Every unfinished row fixes at least one pixel per iteration. Since
        # each fixed pixel consumes max_spp, this is a finite upper bound.
        for _ in range(min(pixels, math.floor(budget/max_spp)+1)):
            remaining = budget - fixed.sum(1,keepdim=True).to(dtype)*max_spp
            free = torch.softmax(scores.masked_fill(fixed,-torch.inf),dim=1)*remaining
            overflow = (free > max_spp) & ~fixed
            if not overflow.any():
                expected = torch.where(fixed, scores.new_tensor(float(max_spp)), free).reshape(logits.shape)
                break
            fixed = fixed | overflow
        else:
            raise RuntimeError("Bounded allocation failed to converge")
    return SampleMap(expected,expected.round().to(torch.int64))


@torch.no_grad()
def exact_counts(expected, average_spp, *, minimum=1, maximum=63):
    """Largest fractional remainder, stable flat-index ties, exact image sum.

    Typical allocations need additions to floor(expected). A defensive removal
    branch handles floating-point budget overshoot. This does not differentiate
    the discrete allocation or claim the published independent rounding rule.
    """
    if (expected.ndim != 4 or expected.shape[1] != 1 or min(expected.shape) < 1
            or not expected.is_floating_point() or not torch.isfinite(expected).all()):
        raise ValueError("Expected finite floating N1HW allocations")
    if not (type(average_spp) is int and type(minimum) is int and type(maximum) is int
            and 0 <= minimum <= average_spp <= maximum):
        raise ValueError("Invalid integer budget or bounds")
    if (expected < minimum-1e-5).any() or (expected > maximum+1e-5).any():
        raise ValueError("Expected allocations outside declared bounds")
    values = expected.detach().flatten(1).double().clamp(minimum, maximum)
    floors = values.floor()
    counts = floors.long()
    target = average_spp*values.shape[1]
    for row in range(values.shape[0]):
        difference = target-int(counts[row].sum())
        if not difference:
            continue
        fractions = values[row]-floors[row]
        eligible = counts[row] < maximum if difference > 0 else counts[row] > minimum
        if abs(difference) > int(eligible.sum()):
            raise ValueError("Expected map is incompatible with a one-sample remainder repair")
        scores = fractions if difference > 0 else -fractions
        order = torch.argsort(scores.masked_fill(~eligible, -torch.inf), descending=True, stable=True)
        counts[row, order[:abs(difference)]] += 1 if difference > 0 else -1
    if not torch.all(counts.sum(1) == target):
        raise RuntimeError("Integer budget repair failed")
    return counts.reshape_as(expected)
