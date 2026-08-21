"""Wavelength-geometry validation: rest frame and stellar-feature alignment.

Two *independent* checks, deliberately not one:

``check_rest_frame``
    Fits a local continuum plus Gaussian to [O III] 5007 and measures the
    centroid offset. This validates the **galaxy** rest frame — it catches a
    redshift applied twice, not at all, or a wavelength-solution offset.

``check_stellar_alignment``
    Compares observed stellar absorption features against their laboratory
    wavelengths. This validates that the **stellar template system** shares the
    galaxy's wavelength system, which [O III] alone cannot establish: an
    emission-line rest frame can be perfect while the template grid is shifted.

Neither defaults to PASS. Insufficient coverage, a failed fit, or an
unmeasurable feature yields an explicit non-PASS status, because the earlier
argmax diagnostic returned a confident wrong answer by taking the maximum of a
rising continuum at a window edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy.optimize import curve_fit

C_KMS = 299792.458
OIII_5007 = 5006.843

#: Stellar absorption features used for template-system alignment.
#:
#: **Hbeta absorption is deliberately excluded.** In an AGN the Hbeta region is
#: dominated by broad *emission*, so an absorption probe there does not find a
#: stellar line: it latches onto emission-wing structure and reports a spurious
#: displacement. Measured on the four diagnostic objects, Hbeta_abs returned
#: -31.3 and +28.1 A where every genuine stellar feature in the same spectrum
#: agreed to within ~2 A, and the only object that passed was the one where
#: Hbeta_abs happened not to be measured.
#:
#: It remains available via :data:`HOST_DOMINATED_FEATURES` for spectra with no
#: appreciable broad emission, where stellar Hbeta absorption is real.
STELLAR_FEATURES: dict[str, tuple[float, tuple[float, float]]] = {
    "CaII_K": (3933.66, (3900.0, 3970.0)),
    "CaII_H": (3968.47, (3940.0, 4000.0)),
    "G_band": (4304.40, (4270.0, 4340.0)),
    "Mgb": (5175.00, (5140.0, 5210.0)),
    "NaD": (5892.50, (5860.0, 5925.0)),
}

#: Only for spectra without broad Balmer emission.
HOST_DOMINATED_FEATURES: dict[str, tuple[float, tuple[float, float]]] = {
    **STELLAR_FEATURES,
    "Hbeta_abs": (4861.33, (4830.0, 4890.0)),
}


class AlignmentStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    FIT_FAILED = "FIT_FAILED"
    NOT_MEASURABLE = "NOT_MEASURABLE"

    @property
    def is_pass(self) -> bool:
        return self is AlignmentStatus.PASS


@dataclass(frozen=True)
class AlignmentResult:
    name: str
    rest_wavelength: float
    status: AlignmentStatus
    fitted_centroid: float | None = None
    offset_angstrom: float | None = None
    offset_kms: float | None = None
    amplitude: float | None = None
    significance: float | None = None
    n_pixels: int = 0
    detail: str = ""


def _gauss_plus_line(lam, offset, slope, amplitude, centre, sigma):
    return offset + slope * (lam - centre) + amplitude * np.exp(
        -0.5 * ((lam - centre) / sigma) ** 2
    )


def _fit_feature(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray | None,
    rest: float,
    window: tuple[float, float],
    *,
    name: str,
    emission: bool,
    max_offset_angstrom: float,
    min_significance: float = 3.0,
) -> AlignmentResult:
    """Local continuum + Gaussian, returning an explicit status.

    The centroid is searched across the whole window, not merely within the
    tolerance. Constraining the search to the pass threshold would make a badly
    displaced feature unfittable, and it would then be reported as *absent*
    rather than as *misplaced* — hiding the exact error this check exists to
    catch. Search width and verdict threshold are therefore separate.
    """
    inside = (wavelength >= window[0]) & (wavelength <= window[1]) & np.isfinite(flux)
    count = int(inside.sum())
    if count < 12:
        return AlignmentResult(name, rest, AlignmentStatus.INSUFFICIENT_COVERAGE,
                               n_pixels=count,
                               detail=f"{count} pixels in {window}")

    lam, values = wavelength[inside], flux[inside]
    sigma_values = error[inside] if error is not None else None

    # Seed the centroid from the continuum-SUBTRACTED extremum. A wide search
    # range is needed to measure a displaced feature, but a wide range with the
    # seed pinned at `rest` lets the optimizer wander to the wrong side. Fitting
    # and removing a local baseline first is what makes this safe: the discarded
    # argmax diagnostic failed precisely because it read a rising *raw*
    # continuum. Here the extremum is only a starting point for the fit.
    edge = max(3, count // 4)
    edge_lam = np.concatenate([lam[:edge], lam[-edge:]])
    edge_flux = np.concatenate([values[:edge], values[-edge:]])
    slope_guess, intercept_guess = np.polyfit(edge_lam, edge_flux, 1)
    residual = values - (intercept_guess + slope_guess * lam)

    peak_index = int(np.argmax(residual)) if emission else int(np.argmin(residual))
    centre_guess = float(lam[peak_index])
    guess_amplitude = float(residual[peak_index])
    if (emission and guess_amplitude <= 0) or (not emission and guess_amplitude >= 0):
        guess_amplitude = float(np.ptp(values)) * (1.0 if emission else -1.0)

    baseline = float(intercept_guess + slope_guess * centre_guess)
    search = max(float(window[1] - rest), float(rest - window[0]))
    guess = [baseline, float(slope_guess), guess_amplitude, centre_guess, 3.0]
    bounds = (
        [-np.inf, -np.inf, 0.0 if emission else -np.inf, rest - search, 0.5],
        [np.inf, np.inf, np.inf if emission else 0.0, rest + search, 30.0],
    )

    try:
        popt, pcov = curve_fit(
            _gauss_plus_line, lam, values, p0=guess, bounds=bounds,
            sigma=sigma_values, absolute_sigma=sigma_values is not None, maxfev=20000,
        )
    except Exception as error_:  # noqa: BLE001
        return AlignmentResult(name, rest, AlignmentStatus.FIT_FAILED, n_pixels=count,
                               detail=f"{type(error_).__name__}: {str(error_)[:80]}")

    amplitude, centre = float(popt[2]), float(popt[3])
    amplitude_error = float(np.sqrt(abs(pcov[2, 2]))) if np.all(np.isfinite(pcov)) else np.nan
    significance = abs(amplitude) / amplitude_error if amplitude_error > 0 else np.nan

    if not np.isfinite(significance) or significance < min_significance:
        return AlignmentResult(
            name, rest, AlignmentStatus.NOT_MEASURABLE, fitted_centroid=centre,
            amplitude=amplitude, significance=None if not np.isfinite(significance) else significance,
            n_pixels=count,
            detail=f"feature not detected above {min_significance} sigma",
        )

    offset = centre - rest
    status = (
        AlignmentStatus.PASS if abs(offset) <= max_offset_angstrom
        else AlignmentStatus.FAIL
    )
    return AlignmentResult(
        name, rest, status, fitted_centroid=centre, offset_angstrom=offset,
        offset_kms=C_KMS * offset / rest, amplitude=amplitude,
        significance=significance, n_pixels=count,
        detail="" if status.is_pass else f"centroid offset {offset:+.2f} A",
    )


def check_rest_frame(
    wavelength_rest: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray | None = None,
    *,
    max_offset_angstrom: float = 8.0,
) -> AlignmentResult:
    """Galaxy rest frame, from a fitted [O III] 5007 centroid.

    Replaces an argmax diagnostic that reported +37 A offsets by locating the
    maximum of a rising continuum at the edge of its search window. A fitted
    centroid over a local linear continuum cannot do that, and reports
    NOT_MEASURABLE when the line is genuinely absent rather than inventing a
    position.
    """
    return _fit_feature(
        wavelength_rest, flux, error, OIII_5007, (OIII_5007 - 45.0, OIII_5007 + 45.0),
        name="OIII_5007", emission=True, max_offset_angstrom=max_offset_angstrom,
    )


@dataclass(frozen=True)
class StellarAlignment:
    """Aggregate verdict over stellar absorption features."""

    features: dict[str, AlignmentResult] = field(default_factory=dict)

    @property
    def measured(self) -> dict[str, AlignmentResult]:
        return {
            name: result for name, result in self.features.items()
            if result.status in {AlignmentStatus.PASS, AlignmentStatus.FAIL}
        }

    @property
    def n_measured(self) -> int:
        return len(self.measured)

    @property
    def median_offset_kms(self) -> float | None:
        offsets = [r.offset_kms for r in self.measured.values() if r.offset_kms is not None]
        return float(np.median(offsets)) if offsets else None

    @property
    def status(self) -> AlignmentStatus:
        """PASS only when at least two features were measured and all agree.

        One feature is not enough: a single absorption line can be mimicked by a
        continuum inflection, so two independent features are required before
        the template wavelength system is called validated.
        """
        measured = self.measured
        if len(measured) < 2:
            return AlignmentStatus.INSUFFICIENT_COVERAGE
        if any(r.status is AlignmentStatus.FAIL for r in measured.values()):
            return AlignmentStatus.FAIL
        return AlignmentStatus.PASS


def check_stellar_alignment(
    wavelength_rest: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray | None = None,
    *,
    features: dict[str, tuple[float, tuple[float, float]]] | None = None,
    max_offset_angstrom: float = 8.0,
) -> StellarAlignment:
    """Stellar-feature alignment, independent of any emission line."""
    catalogue = STELLAR_FEATURES if features is None else features
    results = {
        name: _fit_feature(
            wavelength_rest, flux, error, rest, window,
            name=name, emission=False, max_offset_angstrom=max_offset_angstrom,
        )
        for name, (rest, window) in catalogue.items()
    }
    return StellarAlignment(results)


__all__ = [
    "C_KMS",
    "OIII_5007",
    "HOST_DOMINATED_FEATURES",
    "STELLAR_FEATURES",
    "AlignmentResult",
    "AlignmentStatus",
    "StellarAlignment",
    "check_rest_frame",
    "check_stellar_alignment",
]
