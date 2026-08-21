"""Domain-required spectral summaries.

Every function here **demands an explicit wavelength domain**. None will silently
summarise a whole backing array.

This exists because the same class of mistake has now occurred twice in this
project, each time producing a technically valid but scientifically meaningless
scalar that was believed for a while:

1. A median of E-MILES' ``fwhm`` over its full 1680-50000 A extent returned
   4.94 A, where the fitted optical value is 2.51 A. That manufactured a
   template/data resolution incompatibility that did not exist and blocked the
   analysis until it was retracted (D-044, D-045).
2. A host component was read through a generic attribute whose zero value meant
   "decomposition declined", not "no host" (D-037).

Both were caught, but only after being acted on. More tests would not have
prevented either, because in both cases the code did exactly what it said. The
fix is a semantic API: make the meaningless call impossible to write by
accident.

Rules enforced here:

* a domain is mandatory, never defaulted;
* an empty or too-sparse domain returns ``None``, never a number computed from
  nothing;
* the number of contributing pixels is always available alongside the value, so
  a caller can tell a well-sampled result from a two-pixel accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

#: Minimum contributing pixels before a windowed summary means anything.
MIN_PIXELS = 5


class DomainError(ValueError):
    """Raised when a requested domain cannot support a summary."""


@dataclass(frozen=True)
class WindowedValue:
    """A summary plus the evidence for how well it was sampled.

    Carries both pixel count and wavelength span, because 20 oversampled pixels
    spanning 2 A and 20 pixels spread across 200 A support very different
    claims while reporting the same count.
    """

    value: float | None
    n_pixels: int
    window: tuple[float, float]
    wavelength_span: float | None = None

    @property
    def well_sampled(self) -> bool:
        return self.value is not None and self.n_pixels >= MIN_PIXELS

    @property
    def window_coverage(self) -> float | None:
        """Fraction of the requested window the contributing pixels actually span."""
        if self.wavelength_span is None:
            return None
        requested = self.window[1] - self.window[0]
        return float(self.wavelength_span / requested) if requested > 0 else None

    def __float__(self) -> float:
        if self.value is None:
            raise DomainError(
                f"no value: only {self.n_pixels} pixels in {self.window}"
            )
        return float(self.value)


def _span(wavelength: np.ndarray, mask: np.ndarray) -> float | None:
    """Wavelength range actually covered by the contributing pixels."""
    if not mask.any():
        return None
    selected = wavelength[mask]
    return float(selected.max() - selected.min())


def _validate(window: tuple[float, float]) -> tuple[float, float]:
    low, high = float(window[0]), float(window[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise DomainError(f"window bounds must be finite, got {window}")
    if low >= high:
        raise DomainError(f"window must be increasing, got {window}")
    return low, high


def in_window(
    wavelength: np.ndarray,
    window: tuple[float, float],
    *,
    good: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask for a wavelength window, optionally intersected with `good`."""
    low, high = _validate(window)
    mask = (wavelength >= low) & (wavelength <= high)
    if good is not None:
        mask = mask & good
    return mask


def median_in_window(
    wavelength: np.ndarray,
    values: np.ndarray,
    window: tuple[float, float],
    *,
    good: np.ndarray | None = None,
) -> WindowedValue:
    """Median of `values` inside an explicit window.

    Use this instead of ``np.median(values)`` whenever `values` is a
    wavelength-dependent instrument or model property. The full-array median of
    such a quantity is almost never the number anyone wants.
    """
    _validate(window)
    mask = in_window(wavelength, window, good=good)
    finite = mask & np.isfinite(values)
    count = int(finite.sum())
    span = _span(wavelength, finite)
    if count < MIN_PIXELS:
        return WindowedValue(None, count, (window[0], window[1]), span)
    return WindowedValue(
        float(np.median(values[finite])), count, (window[0], window[1]), span
    )


def fwhm_in_fit_region(
    lam_template: np.ndarray,
    fwhm_template: np.ndarray,
    fit_range: tuple[float, float],
) -> WindowedValue:
    """Template resolution over the range actually fitted.

    Instrument and library resolutions are wavelength dependent. Summarising one
    over its full backing array is the exact error that produced the retracted
    D-044 conclusion, so this function refuses to run without `fit_range`.
    """
    fwhm = np.atleast_1d(fwhm_template)
    if fwhm.size == 1:
        fwhm = np.full(lam_template.size, float(fwhm[0]))
    if fwhm.size != lam_template.size:
        raise DomainError(
            f"fwhm and lam must match: {fwhm.size} vs {lam_template.size}"
        )
    return median_in_window(lam_template, fwhm, fit_range)


class CoverageStatus(StrEnum):
    """Why a sub-domain verdict says what it says.

    ``clipped_fraction = 0.0`` is dangerously ambiguous on its own: it can mean
    every pixel was compatible, or that there were no pixels to evaluate. These
    statuses separate the two, and the fraction is ``None`` rather than zero
    whenever coverage cannot support a verdict.
    """

    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    NO_COVERAGE = "NO_COVERAGE"


@dataclass(frozen=True)
class SubdomainVerdict:
    """LSF verdict for one named sub-domain, with its coverage evidence."""

    name: str
    window: tuple[float, float]
    n_pixels: int
    wavelength_span: float | None
    clipped_fraction: float | None
    status: CoverageStatus

    @property
    def window_coverage(self) -> float | None:
        if self.wavelength_span is None:
            return None
        requested = self.window[1] - self.window[0]
        return float(self.wavelength_span / requested) if requested > 0 else None

    @property
    def verdict_available(self) -> bool:
        return self.status in {CoverageStatus.COMPATIBLE, CoverageStatus.INCOMPATIBLE}


def coverage_mask(
    lam_template: np.ndarray,
    galaxy_lam: np.ndarray,
    galaxy_good: np.ndarray,
    *,
    tolerance: float | None = None,
) -> np.ndarray:
    """Template pixels backed by a usable galaxy pixel.

    Masked or absent galaxy pixels must not contribute implicitly to an LSF
    verdict: a region with no data is not a region that passed.
    """
    good_lam = np.asarray(galaxy_lam)[np.asarray(galaxy_good, dtype=bool)]
    if good_lam.size == 0:
        return np.zeros(lam_template.size, dtype=bool)
    if tolerance is None:
        spacing = np.diff(np.sort(good_lam))
        tolerance = float(np.median(spacing)) * 2.0 if spacing.size else np.inf
    order = np.sort(good_lam)
    index = np.clip(np.searchsorted(order, lam_template), 1, order.size - 1)
    nearest = np.minimum(
        np.abs(lam_template - order[index - 1]), np.abs(lam_template - order[index])
    )
    return nearest <= tolerance


#: Sub-domains where an LSF mismatch matters most for this project. A small
#: incompatible fraction sitting on the Hbeta core is not equivalent to the same
#: fraction at a band edge.
CRITICAL_SUBDOMAINS: dict[str, tuple[float, float]] = {
    "hbeta_core": (4830.0, 4890.0),
    "hbeta_region": (4700.0, 5100.0),
    "oiii_5007": (4995.0, 5020.0),
}


def lsf_compatible(
    fwhm_galaxy: float,
    lam_template: np.ndarray,
    fwhm_template: np.ndarray,
    fit_range: tuple[float, float],
    *,
    critical_subdomains: dict[str, tuple[float, float]] | None = None,
    galaxy_lam: np.ndarray | None = None,
    galaxy_good: np.ndarray | None = None,
) -> dict[str, object]:
    """Whether templates can be convolved up to the data resolution.

    Reports **severity as well as extent**. A bare incompatible fraction can
    look harmless at 1% while that 1% sits directly across the Hbeta core, which
    is exactly where this project's measurements live. So the result also
    carries the worst deficit, where it occurs, and the incompatible fraction
    inside each named critical sub-domain.
    """
    fwhm = np.atleast_1d(fwhm_template)
    if fwhm.size == 1:
        fwhm = np.full(lam_template.size, float(fwhm[0]))
    mask = in_window(lam_template, fit_range)
    # Evaluate over fit domain AND valid galaxy pixels AND template coverage, so
    # masked or absent data cannot contribute implicitly to a passing verdict.
    if galaxy_lam is not None and galaxy_good is not None:
        mask = mask & coverage_mask(lam_template, galaxy_lam, galaxy_good)
    if mask.sum() < MIN_PIXELS:
        raise DomainError(
            f"fewer than {MIN_PIXELS} usable pixels in {fit_range} after "
            "intersecting fit domain, galaxy validity and template coverage"
        )

    lam_fit = lam_template[mask]
    difference = float(fwhm_galaxy) ** 2 - fwhm[mask] ** 2
    incompatible = difference < 0
    clipped = float(np.mean(incompatible))

    worst_index = int(np.argmin(difference))
    result: dict[str, object] = {
        "fwhm_galaxy": float(fwhm_galaxy),
        "fwhm_template_in_fit": float(np.median(fwhm[mask])),
        "clipped_fraction": clipped,
        "compatible": clipped == 0.0,
        "min_difference_sq": float(difference[worst_index]),
        "worst_wavelength": float(lam_fit[worst_index]),
        "worst_fwhm_template": float(fwhm[mask][worst_index]),
        "n_pixels": int(mask.sum()),
        "fit_range": fit_range,
    }

    subdomains = CRITICAL_SUBDOMAINS if critical_subdomains is None else critical_subdomains
    verdicts: dict[str, SubdomainVerdict] = {}
    for name, window in subdomains.items():
        inside = in_window(lam_fit, window)
        count = int(inside.sum())
        span = _span(lam_fit, inside)
        if count == 0:
            status, fraction = CoverageStatus.NO_COVERAGE, None
        elif count < MIN_PIXELS:
            status, fraction = CoverageStatus.INSUFFICIENT_COVERAGE, None
        else:
            fraction = float(np.mean(incompatible[inside]))
            status = (
                CoverageStatus.INCOMPATIBLE if fraction > 0 else CoverageStatus.COMPATIBLE
            )
        verdicts[name] = SubdomainVerdict(name, window, count, span, fraction, status)

    result["critical"] = verdicts
    result["critical_clipped_fraction"] = {n: v.clipped_fraction for n, v in verdicts.items()}
    result["critical_status"] = {n: str(v.status) for n, v in verdicts.items()}
    result["critical_any_incompatible"] = any(
        v.status is CoverageStatus.INCOMPATIBLE for v in verdicts.values()
    )
    # A sub-domain we could not evaluate is not a sub-domain that passed.
    result["critical_all_evaluated"] = all(v.verdict_available for v in verdicts.values())
    return result


def host_fraction_in_window(
    wavelength: np.ndarray,
    host: np.ndarray,
    agn_continuum: np.ndarray,
    window: tuple[float, float],
    *,
    good: np.ndarray | None = None,
) -> WindowedValue:
    """The Paper-3 host fraction, from reconstructed component spectra.

    Never from coefficient sums: template families are conditioned differently,
    so weight sums are not flux ratios.
    """
    _validate(window)
    mask = in_window(wavelength, window, good=good)
    finite = mask & np.isfinite(host) & np.isfinite(agn_continuum)
    count = int(finite.sum())
    if count < MIN_PIXELS:
        return WindowedValue(None, count, (window[0], window[1]), _span(wavelength, finite))

    span = _span(wavelength, finite)
    host_level = float(np.median(host[finite]))
    agn_level = float(np.median(agn_continuum[finite]))
    total = host_level + agn_level
    if total <= 0:
        return WindowedValue(None, count, (window[0], window[1]), span)
    return WindowedValue(host_level / total, count, (window[0], window[1]), span)


def snr_in_window(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    window: tuple[float, float],
    *,
    good: np.ndarray | None = None,
) -> WindowedValue:
    """Median per-pixel S/N inside an explicit window."""
    _validate(window)
    mask = in_window(wavelength, window, good=good)
    finite = mask & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    count = int(finite.sum())
    span = _span(wavelength, finite)
    if count < MIN_PIXELS:
        return WindowedValue(None, count, (window[0], window[1]), span)
    return WindowedValue(
        float(np.median(flux[finite] / error[finite])), count, (window[0], window[1]), span
    )


__all__ = [
    "CRITICAL_SUBDOMAINS",
    "CoverageStatus",
    "SubdomainVerdict",
    "coverage_mask",
    "MIN_PIXELS",
    "DomainError",
    "WindowedValue",
    "fwhm_in_fit_region",
    "host_fraction_in_window",
    "in_window",
    "lsf_compatible",
    "median_in_window",
    "snr_in_window",
]
