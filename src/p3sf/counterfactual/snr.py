"""Counterfactual experiment 1 — signal-to-noise degradation.

Given a real spectrum observed at some S/N, produce what the *same* observation
would have looked like at a lower S/N.

Two rules govern this module, both from the preregistration:

**Degradation is one-directional.** Noise is added in quadrature,

.. math::
    \\sigma_{\\rm added}^2(\\lambda) = \\sigma_{\\rm target}^2(\\lambda) - \\sigma_{\\rm original}^2(\\lambda)

which requires :math:`\\sigma_{\\rm target} \\geq \\sigma_{\\rm original}` pixel by pixel.
Requesting a *higher* S/N than observed is an error, not a rescaling. We never
claim to upgrade an observed spectrum.

**Noise keeps its survey structure.** The added noise inherits the wavelength
dependence of the original error array, so skyline residuals, detector edges,
and per-pixel sensitivity survive the degradation. Adding flat white noise
would make low-S/N simulations systematically easier to fit than real low-S/N
observations, which would bias every completeness estimate upward.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

from p3sf.access.spectra import Spectrum


@dataclass(frozen=True)
class DegradationResult:
    """A degraded spectrum plus what was actually done to it."""

    spectrum: Spectrum
    requested_snr: float
    original_snr: float
    achieved_snr: float
    scale_factor: float
    seed: int

    @property
    def snr_error(self) -> float:
        return self.achieved_snr - self.requested_snr


def measure_snr(
    spectrum: Spectrum,
    window_rest: tuple[float, float],
) -> float:
    """Median per-pixel S/N in a rest-frame wavelength window.

    Median rather than mean: a handful of bad pixels or an unmasked skyline
    should not set the S/N of the whole window.
    """
    rest = spectrum.rest_wavelength()
    inside = (rest >= window_rest[0]) & (rest <= window_rest[1]) & spectrum.good
    if inside.sum() < 5:
        raise ValueError(
            f"only {int(inside.sum())} good pixels in rest window {window_rest}; "
            "cannot measure S/N"
        )
    return float(np.median(spectrum.flux[inside] / spectrum.error[inside]))


def degrade_to_snr(
    spectrum: Spectrum,
    target_snr: float,
    *,
    window_rest: tuple[float, float],
    seed: int,
    tolerance: float = 1e-9,
) -> DegradationResult:
    """Degrade ``spectrum`` so its S/N in ``window_rest`` becomes ``target_snr``.

    Parameters
    ----------
    target_snr
        Desired median per-pixel S/N. Must not exceed the original.
    window_rest
        Rest-frame window in which S/N is defined, e.g. the 5100 A continuum.
    seed
        Explicit seed. Every stochastic step in this project is seeded; the
        same seed must reproduce the same spectrum bit for bit.
    tolerance
        Slack allowed when comparing target to original S/N, so that asking for
        exactly the observed S/N is a no-op rather than a floating-point error.

    Raises
    ------
    ValueError
        If ``target_snr`` exceeds the original S/N. Degradation only degrades.
    """
    if target_snr <= 0:
        raise ValueError(f"target_snr must be positive, got {target_snr}")

    original_snr = measure_snr(spectrum, window_rest)
    if not np.isfinite(original_snr) or original_snr <= 0:
        raise ValueError(f"original S/N is {original_snr}; spectrum is unusable")

    if target_snr > original_snr * (1.0 + tolerance):
        raise ValueError(
            f"cannot degrade to S/N {target_snr:.2f}: spectrum is already at "
            f"{original_snr:.2f}. Degradation is one-directional; a spectrum "
            "cannot be upgraded. Exclude this object with "
            "REFERENCE_SNR_TOO_LOW_TO_DEGRADE."
        )

    # sigma scales as 1/SNR, so the whole error array scales by this factor.
    # Applying one scalar to sigma(lambda) preserves its wavelength structure.
    scale = original_snr / target_snr
    scale = max(scale, 1.0)  # guards the no-op case against tiny negatives

    good = spectrum.good
    sigma_original = np.where(good, spectrum.error, np.nan)
    sigma_target = sigma_original * scale

    # sigma_added^2 = sigma_target^2 - sigma_original^2, clipped at 0 for safety.
    added_variance = np.clip(sigma_target**2 - sigma_original**2, 0.0, None)
    sigma_added = np.sqrt(added_variance)

    rng = np.random.default_rng(seed)
    noise = np.zeros_like(spectrum.flux, dtype=float)
    noise[good] = rng.normal(0.0, 1.0, size=int(good.sum())) * sigma_added[good]

    new_flux = spectrum.flux + noise
    new_error = np.where(good, sigma_target, spectrum.error)

    degraded = replace(
        spectrum,
        flux=new_flux,
        error=new_error,
        meta={
            **dict(spectrum.meta),
            "counterfactual_snr_target": float(target_snr),
            "counterfactual_snr_original": float(original_snr),
            "counterfactual_snr_scale": float(scale),
            "counterfactual_snr_seed": int(seed),
        },
    )

    return DegradationResult(
        spectrum=degraded,
        requested_snr=float(target_snr),
        original_snr=float(original_snr),
        achieved_snr=measure_snr(degraded, window_rest),
        scale_factor=float(scale),
        seed=int(seed),
    )


def degradation_ladder(
    spectrum: Spectrum,
    snr_grid: list[float],
    *,
    window_rest: tuple[float, float],
    base_seed: int,
    realization: int = 0,
) -> dict[float, DegradationResult]:
    """Degrade one spectrum to every reachable S/N on the grid.

    Grid points above the spectrum's own S/N are skipped rather than raising,
    because a mixed-quality reference sample will legitimately have objects
    that cannot reach the top of the ladder. The caller is responsible for
    recording those as exclusions where it matters.

    The seed for each rung is derived deterministically from ``base_seed``,
    the realization index, and the target S/N, so rungs are independent of one
    another and reproducible in any order.
    """
    original = measure_snr(spectrum, window_rest)
    out: dict[float, DegradationResult] = {}
    for target in snr_grid:
        if target > original:
            continue
        rung_seed = _derive_seed(base_seed, realization, target)
        out[target] = degrade_to_snr(
            spectrum, target, window_rest=window_rest, seed=rung_seed
        )
    return out


def _derive_seed(base_seed: int, realization: int, target: float) -> int:
    """Stable child seed for one rung of the ladder.

    Uses SHA-256 rather than :func:`hash`, which is salted per process for
    strings and would make runs irreproducible across invocations.
    """
    payload = f"{int(base_seed)}|{int(realization)}|{float(target):.6f}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


__all__ = [
    "DegradationResult",
    "degradation_ladder",
    "degrade_to_snr",
    "measure_snr",
]
