"""Forward-only replay of NTASAD's independent power-of-two render layers."""

import torch


LAYER_SPP = (1, 2, 4, 8, 16, 32)
MAX_SPP = sum(LAYER_SPP)  # Six binary layers represent 0..63, not 64.


def replay_sample_layers(layers: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Return linear RGB means from N,6,3,H,W independently rendered means.

    Select layers by the bits of integer N1HW counts, weight each layer mean
    by its SPP, then divide by actual count. Zero counts return zero radiance;
    the caller must retain the count mask for reconstruction. Captures must
    use disjoint sample ranges, not nested prefix averages. Range independence
    cannot be verified from RGB tensors and belongs in the capture metadata.

    This function differentiates layer values but has no derivative through
    integer counts. Training code can use the separate
    renderer_gradient.differentiable_replay operator for the DASR approximation;
    this inference/replay primitive requires no reference.
    """
    if (layers.ndim != 5 or min(layers.shape) < 1 or layers.shape[1:3] != (6, 3)
            or not layers.is_floating_point() or not torch.isfinite(layers).all()
            or (layers < 0).any()):
        raise ValueError("layers must be finite nonnegative floating N6x3HW means")
    n, _, _, h, w = layers.shape
    if (counts.shape != (n, 1, h, w) or counts.device != layers.device
            or counts.dtype not in (torch.int32, torch.int64)
            or (counts < 0).any() or (counts > MAX_SPP).any()):
        raise ValueError(f"counts must be matching integer N1HW in [0, {MAX_SPP}]")
    # Accumulate low-precision storage in float32, preserving doubles for audits.
    dtype = torch.float64 if layers.dtype == torch.float64 else torch.float32
    result = torch.zeros((n, 3, h, w), dtype=dtype, device=layers.device)
    for index, spp in enumerate(LAYER_SPP):
        selected = (counts & spp) != 0
        result = result + torch.where(selected, layers[:, index].to(dtype) * spp, 0)
    return result / counts.clamp_min(1).to(dtype)
