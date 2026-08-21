"""Legacy generic CLAGN criterion scaffolds.

These generic functions predate the source-verification gate and remain locked.
In particular, ``apply_nsigma`` uses integrated fitted-line flux and is not the
MacLeod/Green pixel statistic, while ``apply_difference_spectrum`` is not the
Potts--Villforth search protocol. Source-specific implementations live in
``p3sf.criteria.source_verified``.

Threshold sweeps are supported (``threshold=`` override) but are *robustness*,
never primary. The published value is what appears in the main results.
"""

from __future__ import annotations

import numpy as np

from p3sf.config import Config, load_config
from p3sf.criteria.base import (
    CriterionResult,
    EpochPair,
    Label,
    require_verified,
    unclassifiable,
)

# ---------------------------------------------------------------------------
# Family A — significance
# ---------------------------------------------------------------------------


def nsigma_statistic(pair: EpochPair) -> float | None:
    r"""N_sigma = |F_b - F_d| / sqrt(sigma_b^2 + sigma_d^2).

    Returns None when either epoch lacks a flux *and* uncertainty. A
    non-detection with only an upper limit has no usable sigma for this
    statistic as published, which is itself a finding about the criterion's
    domain of applicability.
    """
    early, late = pair.early, pair.late
    early_flux, late_flux = early.flux, late.flux
    early_error, late_error = early.flux_error, late.flux_error
    if early_flux is None or late_flux is None or early_error is None or late_error is None:
        return None

    denominator = float(np.hypot(early_error, late_error))
    if not np.isfinite(denominator) or denominator <= 0:
        return None

    value = abs(late_flux - early_flux) / denominator
    return float(value) if np.isfinite(value) else None


def apply_nsigma(
    pair: EpochPair,
    *,
    threshold: float | None = None,
    config: Config | None = None,
    enforce_verified: bool = True,
) -> CriterionResult:
    """Family A. Published threshold N_sigma > 3 (MacLeod/Green family)."""
    name = "nsigma"
    if enforce_verified:
        require_verified("A", name)

    cfg = config or load_config()
    limit = cfg.criteria.nsigma.threshold if threshold is None else threshold

    if not pair.usable:
        return unclassifiable(name, "one or both epochs are unusable", pair.direction)

    statistic = nsigma_statistic(pair)
    if statistic is None:
        return unclassifiable(
            name,
            "flux and uncertainty not available in both epochs; "
            "N_sigma is undefined against an upper limit",
            pair.direction,
        )

    return CriterionResult(
        criterion=name,
        label=Label.CL if statistic > limit else Label.NON_CL,
        statistic=statistic,
        threshold=limit,
        direction=pair.direction,
    )


# ---------------------------------------------------------------------------
# Family B — broad-line flux ratio
# ---------------------------------------------------------------------------


def flux_ratio_statistic(pair: EpochPair) -> tuple[float | None, bool]:
    """Return ``(ratio, is_lower_limit)``.

    When the faint state is a non-detection the ratio is a *lower* limit,
    because the true faint flux is somewhere below the upper limit. Reporting
    it as a plain number would overstate precision; how the source paper
    handles this case determines whether the criterion is even computable on
    dim states, and is flagged as unresolved in selection_definitions.md.
    """
    bright, faint = pair.bright, pair.faint

    if not bright.detected or bright.flux is None or bright.flux <= 0:
        return None, False

    if faint.detected and faint.flux is not None and faint.flux > 0:
        return float(bright.flux / faint.flux), False

    if faint.upper_limit is not None and faint.upper_limit > 0:
        return float(bright.flux / faint.upper_limit), True

    return None, False


def apply_flux_ratio(
    pair: EpochPair,
    *,
    threshold: float | None = None,
    config: Config | None = None,
    enforce_verified: bool = True,
) -> CriterionResult:
    """Family B. Threshold is deliberately null in the frozen config until verified."""
    name = "flux_ratio"
    if enforce_verified:
        require_verified("B", name)

    cfg = config or load_config()
    limit = cfg.criteria.flux_ratio.threshold if threshold is None else threshold
    if limit is None:
        raise ValueError(
            "flux_ratio threshold is null in frozen_config.yaml. Transcribe the published "
            "value from the source paper into 01_literature/selection_definitions.md and "
            "set criteria.flux_ratio.threshold before running this criterion."
        )

    if not pair.usable:
        return unclassifiable(name, "one or both epochs are unusable", pair.direction)

    ratio, is_lower_limit = flux_ratio_statistic(pair)
    if ratio is None:
        return unclassifiable(
            name, "no detected bright state; ratio undefined", pair.direction
        )

    if is_lower_limit and ratio <= limit:
        # A lower limit below the threshold is genuinely uninformative: the true
        # ratio could be anywhere above this value, including above the threshold.
        return unclassifiable(
            name,
            f"faint state is an upper limit; ratio > {ratio:.3g} does not resolve "
            f"the threshold {limit:.3g}",
            pair.direction,
        )

    return CriterionResult(
        criterion=name,
        label=Label.CL if ratio > limit else Label.NON_CL,
        statistic=ratio,
        threshold=limit,
        direction=pair.direction,
        reason="lower limit" if is_lower_limit else "",
    )


# ---------------------------------------------------------------------------
# Family C — appearance / disappearance / type change
# ---------------------------------------------------------------------------


def apply_type_change(
    pair: EpochPair,
    *,
    config: Config | None = None,
    enforce_verified: bool = True,
) -> CriterionResult:
    """Family C. A change in broad-line *detectability* between epochs.

    This is the most S/N-sensitive family, and the one where the counterintuitive
    effect lives: raising S/N can make a weak residual broad line detectable and
    thereby *remove* the CL label from a genuine transition. That is only
    representable because non-detections carry upper limits rather than zeros.
    """
    name = "type_change"
    if enforce_verified:
        require_verified("C", name)
    del config  # detectability is set by fitting.detection_sigma upstream

    if not pair.usable:
        return unclassifiable(name, "one or both epochs are unusable", pair.direction)

    changed = pair.early.detected != pair.late.detected
    return CriterionResult(
        criterion=name,
        label=Label.CL if changed else Label.NON_CL,
        statistic=float(changed),
        threshold=None,
        direction=pair.direction,
        reason=(
            f"detected {pair.early.detected} -> {pair.late.detected}"
        ),
    )


# ---------------------------------------------------------------------------
# Family D — difference spectrum
# ---------------------------------------------------------------------------


def apply_difference_spectrum(
    pair: EpochPair,
    *,
    difference_significance: float | None,
    threshold: float | None = None,
    config: Config | None = None,
    enforce_verified: bool = True,
) -> CriterionResult:
    """Family D. Broad-line detection in the epoch difference.

    ``difference_significance`` is computed upstream by fitting a broad
    component to ``F_bright - F_faint``; it is passed in rather than recomputed
    here so that this module stays a pure decision layer over fit products.
    """
    name = "difference_spectrum"
    if enforce_verified:
        require_verified("D", name)

    cfg = config or load_config()
    limit = cfg.fitting.detection_sigma if threshold is None else threshold

    if not pair.usable:
        return unclassifiable(name, "one or both epochs are unusable", pair.direction)
    if difference_significance is None or not np.isfinite(difference_significance):
        return unclassifiable(
            name, "difference-spectrum fit unavailable or non-finite", pair.direction
        )

    return CriterionResult(
        criterion=name,
        label=Label.CL if difference_significance > limit else Label.NON_CL,
        statistic=float(difference_significance),
        threshold=limit,
        direction=pair.direction,
    )


__all__ = [
    "apply_difference_spectrum",
    "apply_flux_ratio",
    "apply_nsigma",
    "apply_type_change",
    "flux_ratio_statistic",
    "nsigma_statistic",
]
