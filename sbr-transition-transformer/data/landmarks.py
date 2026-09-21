from __future__ import annotations

import numpy as np


def smooth_piecewise_linear(
    t: np.ndarray,
    breakpoint: float,
    slope_before: float,
    slope_after: float,
    sharpness: float,
) -> np.ndarray:
    """Two linear segments blended by a logistic weight centered at `breakpoint`.

    Used for landmarks that mark a change in trend rather than a bump: the
    DO elbow (slope rises once ammonia oxidation demand drops) and the
    nitrate knee (ORP's steep decline flattens once denitrification finishes).
    `sharpness` is the logistic length scale in minutes; smaller means a
    sharper, more step-like bend.
    """
    w = 1.0 / (1.0 + np.exp(-(t - breakpoint) / sharpness))
    before = slope_before * (t - breakpoint)
    after = slope_after * (t - breakpoint)
    return (1 - w) * before + w * after


def gaussian_dip(t: np.ndarray, center: float, depth: float, sigma: float) -> np.ndarray:
    """Negative bump: the ammonia valley in the pH curve."""
    return -depth * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def gaussian_bump(t: np.ndarray, center: float, height: float, sigma: float) -> np.ndarray:
    """Positive bump: the nitrate apex, layered on top of the ORP knee trend."""
    return height * np.exp(-0.5 * ((t - center) / sigma) ** 2)
