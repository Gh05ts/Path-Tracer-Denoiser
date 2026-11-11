"""NTASAD equation (4): equal spatial and temporal L1 in gamma/log space."""

import torch
from torch.nn import functional as F

from .ntasad_radiance import encode_radiance


def sequence_loss(prediction: torch.Tensor, reference: torch.Tensor, *,
                  transformed: bool = False) -> dict[str, torch.Tensor]:
    """Use the final frame and final temporal difference, with no motion warp.

    Reference is linear HDR NT3HW; prediction may already be gamma/log encoded.
    At least three frames keep initialization out of the final temporal pair.
    """
    if (prediction.ndim != 5 or prediction.shape != reference.shape or prediction.shape[1] < 3
            or prediction.shape[2] != 3 or min(prediction.shape) < 1):
        raise ValueError("Loss requires matching NT3HW sequences with at least three frames")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("Sequence loss inputs must be finite")
    if not transformed:
        prediction = encode_radiance(prediction)
    reference = encode_radiance(reference)
    spatial = F.l1_loss(prediction[:, -1], reference[:, -1])
    temporal = F.l1_loss(prediction[:, -1] - prediction[:, -2],
                         reference[:, -1] - reference[:, -2])
    return {"loss": spatial + temporal, "spatial_l1": spatial, "temporal_l1": temporal}
