"""Continuous transition statistics, retained alongside every binary label.

Binary labels change abruptly at a threshold while the underlying astrophysics
changes smoothly. Keeping the continuous statistics lets us show where a
classification flips without any physical change having occurred, which is the
mechanism the paper is measuring.

Every statistic returns ``None`` rather than a sentinel number when it cannot
be computed, so that "not measurable" never silently becomes "zero change".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from p3sf.criteria.base import EpochPair


@dataclass(frozen=True)
class TransitionMetrics:
    """Continuous descriptors of one epoch pair."""

    object_id: str
    line: str
    direction: str
    baseline_days: float

    delta_log_flux: float | None
    flux_ratio: float | None
    flux_ratio_is_limit: bool
    signed_significance: float | None
    delta_ew: float | None
    delta_log_luminosity: float | None
    delta_fwhm_kms: float | None
    mean_host_fraction: float | None
    min_continuum_snr: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: float | None) -> float | None:
    """Return ``value`` when it is a finite number, else None."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _difference(late: float | None, early: float | None) -> float | None:
    """``late - early`` when both are finite, else None (never zero)."""
    a, b = _finite(late), _finite(early)
    if a is None or b is None:
        return None
    return a - b


def signed_significance(pair: EpochPair) -> float | None:
    r"""S = (F_late - F_early) / sqrt(sigma_late^2 + sigma_early^2).

    Signed, unlike the published ``N_sigma``, so turn-on and turn-off remain
    distinguishable in the continuous statistic.
    """
    numerator = _difference(pair.late.flux, pair.early.flux)
    early_error, late_error = _finite(pair.early.flux_error), _finite(pair.late.flux_error)
    if numerator is None or early_error is None or late_error is None:
        return None

    denominator = float(np.hypot(early_error, late_error))
    if denominator <= 0:
        return None

    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def delta_log_flux(pair: EpochPair) -> float | None:
    """log10(F_late / F_early), defined only when both epochs are positive detections."""
    early, late = pair.early, pair.late
    if not (early.detected and late.detected):
        return None

    early_flux, late_flux = _finite(early.flux), _finite(late.flux)
    if early_flux is None or late_flux is None:
        return None
    if early_flux <= 0 or late_flux <= 0:
        return None

    return float(np.log10(late_flux / early_flux))


def compute_metrics(pair: EpochPair) -> TransitionMetrics:
    """All continuous statistics for one epoch pair."""
    from p3sf.criteria.published import flux_ratio_statistic

    e, ll = pair.early, pair.late
    ratio, is_limit = flux_ratio_statistic(pair)

    host_values: list[float] = [
        finite
        for finite in (_finite(e.host_fraction_5100), _finite(ll.host_fraction_5100))
        if finite is not None
    ]
    mean_host = float(np.mean(host_values)) if host_values else None

    return TransitionMetrics(
        object_id=pair.object_id,
        line=pair.line,
        direction=pair.direction,
        baseline_days=pair.baseline_days,
        delta_log_flux=delta_log_flux(pair),
        flux_ratio=ratio,
        flux_ratio_is_limit=is_limit,
        signed_significance=signed_significance(pair),
        delta_ew=_difference(ll.ew, e.ew),
        delta_log_luminosity=_difference(ll.continuum_luminosity, e.continuum_luminosity),
        delta_fwhm_kms=_difference(ll.fwhm_kms, e.fwhm_kms),
        mean_host_fraction=mean_host,
        min_continuum_snr=float(min(e.continuum_snr, ll.continuum_snr)),
    )


__all__ = ["TransitionMetrics", "compute_metrics", "delta_log_flux", "signed_significance"]
