"""Frozen numerical definitions for Gate-C tile validation."""

from __future__ import annotations

import numpy as np

SN_LABELS = ("00-10", "10-20", "20-30", "30-50", "50+")
SN_EDGES = (0.0, 10.0, 20.0, 30.0, 50.0, np.inf)


def sn_bin(value: float) -> str:
    """Return the frozen local per-pixel S/N bin."""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"invalid local S/N: {value!r}")
    return SN_LABELS[int(np.digitize(value, SN_EDGES[1:-1]))]


def robust_width(values: np.ndarray, min_pixels: int = 30) -> float | None:
    """Half of the 16th-to-84th percentile span, frozen in Gate C2/C6-B."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < min_pixels:
        return None
    lower, upper = np.percentile(finite, [16, 84])
    return float((upper - lower) / 2.0)


def normalized_difference(
    flux_a: np.ndarray,
    flux_b: np.ndarray,
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    k_a: float = 1.0,
    k_b: float = 1.0,
) -> np.ndarray:
    r"""Return the cross-bin validation residual.

    .. math::
       z = (f_a-f_b) / \sqrt{k_a^2\sigma_a^2+k_b^2\sigma_b^2}
    """
    denominator = np.sqrt((float(k_a) * sigma_a) ** 2 + (float(k_b) * sigma_b) ** 2)
    return np.divide(
        np.asarray(flux_a) - np.asarray(flux_b),
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0),
    )
