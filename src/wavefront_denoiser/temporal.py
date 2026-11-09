"""Differentiable NTASAD building blocks; tensors use NCHW layout.

These operators preserve the supplied radiance space. The recurrent model uses
the NTASAD gamma/log space; linear HDR is also supported by the primitives.
They do not detach history: callers control truncated back-propagation boundaries.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def reproject_history(history: torch.Tensor, motion: torch.Tensor,
                      history_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward bilinear warp using previous-minus-current pixel motion.

    Return warped history and a boolean N1HW mask. Out-of-frame coordinates are
    invalid even if the supplied renderer mask is set. Invalid motion is ignored.
    """
    if history.ndim != 4 or min(history.shape) < 1 or not history.is_floating_point():
        raise ValueError("history must be a nonempty floating NCHW tensor")
    n, _, h, w = history.shape
    if motion.shape != (n, 2, h, w) or history_valid.shape != (n, 1, h, w):
        raise ValueError("motion/mask must have matching N2HW/N1HW shapes")
    if motion.device != history.device or history_valid.device != history.device:
        raise ValueError("history, motion and mask must share a device")
    if motion.dtype != history.dtype:
        raise ValueError("history and motion must share a floating dtype")
    if not torch.all((history_valid == 0) | (history_valid == 1)):
        raise ValueError("history_valid must be binary")
    y, x = torch.meshgrid(torch.arange(h, device=history.device, dtype=history.dtype),
                          torch.arange(w, device=history.device, dtype=history.dtype), indexing="ij")
    px, py = x + motion[:, 0], y + motion[:, 1]
    valid = ((history_valid[:, 0] == 1) & torch.isfinite(px) & torch.isfinite(py)
             & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1))
    # align_corners=False maps pixel centers correctly, including singleton axes.
    grid = torch.stack((2 * (px + .5) / w - 1, 2 * (py + .5) / h - 1), dim=-1)
    grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
    warped = F.grid_sample(history, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False)
    return torch.where(valid[:, None], warped, torch.zeros_like(warped)), valid[:, None]


def apply_kernel(image: torch.Tensor, logits: torch.Tensor, *,
                 valid: torch.Tensor | None = None) -> torch.Tensor:
    """Apply per-pixel normalized kernels, excluding borders and invalid taps.

    Logits have N x K² x H x W shape in row-major offset order. The default
    NTASAD kernel is 5x5. All-invalid neighborhoods return zero. This unfold
    reference implementation prioritizes correctness over full-resolution speed.
    """
    if image.ndim != 4 or logits.ndim != 4 or min(image.shape) < 1:
        raise ValueError("Expected nonempty NCHW image and kernel logits")
    n, c, h, w = image.shape
    k = int(logits.shape[1] ** .5)
    if k < 1 or k % 2 != 1 or k * k != logits.shape[1] or logits.shape != (n, k*k, h, w):
        raise ValueError("Expected matching NK²HW logits with an odd square kernel")
    if not image.is_floating_point() or image.dtype != logits.dtype or image.device != logits.device:
        raise ValueError("image and logits must share floating dtype and device")
    if valid is None:
        valid = torch.ones_like(image[:, :1], dtype=torch.bool)
    if valid.shape != (n, 1, h, w) or valid.device != image.device:
        raise ValueError("valid must have matching N1HW shape and device")
    if not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("valid must be binary")
    taps = F.unfold(valid.to(image.dtype), k, padding=k//2).reshape(n, k*k, h, w) > 0
    any_valid = taps.any(dim=1, keepdim=True)
    masked = logits.masked_fill(~taps, -torch.inf)
    masked = torch.where(any_valid, masked, torch.zeros_like(masked))
    weights = torch.softmax(masked, dim=1).masked_fill(~taps, 0)
    clean = torch.where(valid.bool(), image, torch.zeros_like(image))
    patches = F.unfold(clean, k, padding=k//2).reshape(n, c, k*k, h, w)
    return (patches * weights[:, None]).sum(dim=2)


def combine_scales(fine: torch.Tensor, coarse: torch.Tensor,
                   alpha_logits: torch.Tensor) -> torch.Tensor:
    """NTASAD equation (3): fine - alpha*U(D(fine)) + alpha*U(coarse).

    Levels must be exactly 2:1; pad odd image sizes before building the pyramid.
    This replaces low frequencies while retaining fine-scale detail.
    """
    if (fine.ndim != 4 or coarse.ndim != 4 or
            coarse.shape != (*fine.shape[:2], fine.shape[2]//2, fine.shape[3]//2) or
            fine.shape[2] % 2 or fine.shape[3] % 2 or min(fine.shape[2:]) < 2 or
            alpha_logits.shape != (fine.shape[0], 1, *fine.shape[2:])):
        raise ValueError("Expected 2:1 fine/coarse NCHW levels and N1HW alpha logits")
    alpha = torch.sigmoid(alpha_logits)
    low = F.interpolate(F.avg_pool2d(fine, 2), scale_factor=2, mode="nearest")
    up = F.interpolate(coarse, scale_factor=2, mode="nearest")
    return fine + alpha * (up - low)
