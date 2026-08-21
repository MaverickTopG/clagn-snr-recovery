"""Frozen coordinates for the bounded J082 T2-B calibration closure."""

from __future__ import annotations

import numpy as np

T2_INTERVAL = (5400.0, 5600.0)
SCIENCE_INTERVAL = (50.0, 55.0)


def closure_sn_bin(value: float) -> str:
    """Return the bounded T2 closure bin without recreating an open 50+ cell."""
    if not np.isfinite(value):
        return "NA"
    if value < 0:
        return "NA"
    if value < 10:
        return "00-10"
    if value < 20:
        return "10-20"
    if value < 30:
        return "20-30"
    if value < 50:
        return "30-50"
    if value < 55:
        return "50-55"
    return "55+_OUTSIDE_CLOSURE"


def local_snr(flux: np.ndarray, ivar: np.ndarray) -> float:
    """Frozen per-pixel coordinate on already-selected valid T2 pixels."""
    if flux.size == 0 or flux.shape != ivar.shape:
        return float("nan")
    return float(np.median(np.abs(flux) * np.sqrt(ivar)))
