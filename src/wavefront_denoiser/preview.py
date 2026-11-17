"""Display transforms only; captured HDR buffers stay in renderer units."""
import math

import numpy as np


def tonemap_rgb(rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Unquantized float64 display RGB in [0, 1], shared by metrics and PNGs."""
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be finite and positive")
    if not np.isfinite(rgb).all():
        raise ValueError("Non-finite values in EXR RGB layer")
    linear = np.maximum(rgb.astype(np.float64), 0) * exposure
    return (linear / (1 + linear)) ** (1 / 2.2)


def display_rgb(rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    return np.uint8(np.clip(tonemap_rgb(rgb, exposure) * 255, 0, 255))
