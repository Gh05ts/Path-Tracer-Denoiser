"""Fixed-exposure metrics; display transforms never affect HDR output files."""

import numpy as np


def display_rgb(radiance: np.ndarray) -> np.ndarray:
    linear = np.maximum(radiance.astype(np.float64), 0)
    mapped = linear / (1 + linear)
    return np.where(mapped <= 0.0031308, 12.92 * mapped,
                    1.055 * np.power(mapped, 1 / 2.4) - 0.055)


def image_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if prediction.shape != reference.shape or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Metrics need matching finite RGB images")
    if (prediction < 0).any() or (reference < 0).any():
        raise ValueError("Metrics expect nonnegative linear radiance")
    prediction, reference = prediction.astype(np.float64), reference.astype(np.float64)
    mse = float(np.square(prediction - reference).mean())
    display_mse = float(np.square(display_rgb(prediction) - display_rgb(reference)).mean())
    return {"linear_rmse": float(np.sqrt(mse)),
            "relative_mse": float((np.square(prediction - reference) / (np.square(reference) + 0.01)).mean()),
            "log_l1": float(np.abs(np.log1p(prediction) - np.log1p(reference)).mean()),
            "display_psnr_db": float(-10 * np.log10(max(display_mse, 1e-12)))}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No images to evaluate")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
