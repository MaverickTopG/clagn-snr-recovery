"""Result contracts for spectral fitting.

``BroadHbetaEW`` is reproduced from ``clagn_apj.spectral`` (upstream rev
c3bd1d4) so the vendored PyQSOFit and pPXF drivers keep their original
contract. ``BroadLineMeasurement`` is the Paper 3 contract: it carries the
line *flux* as well as the EW, and it is the type that the classification
criteria consume.

The invariant that matters for this project: a non-detection is never stored
as a flux of zero. It is stored as ``detected=False`` together with a finite
``upper_limit``, so that "the broad line disappeared" and "the broad line was
not measurable" remain distinguishable downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

FitStatus = Literal[
    "ok",
    "unusable_coverage",
    "low_continuum_snr",
    "fit_failed",
    "masked",
]


@dataclass(frozen=True)
class BroadHbetaEW:
    """Upstream contract — host-subtracted, continuum-normalised broad-Hbeta EW."""

    status: str
    broad_hbeta_ew: float | None
    broad_hbeta_ew_error: float | None
    broad_hbeta_detected: bool
    upper_limit_95: float | None
    continuum_snr: float


@dataclass(frozen=True)
class BroadLineMeasurement:
    """One broad line, one epoch — the unit the CL criteria are computed from.

    Attributes
    ----------
    flux, flux_error
        Integrated broad-line flux and its 1-sigma uncertainty, in the flux
        units of the input spectrum. ``flux`` may be small or formally
        negative; it is never coerced to zero.
    detected
        ``flux / flux_error >= detection_sigma`` from the frozen config.
    significance
        ``flux / flux_error``. Retained even when not detected.
    upper_limit
        N-sigma upper limit on the flux, always finite when ``detected`` is
        False. This is what "disappearance" criteria must be evaluated against.
    ew, ew_error
        Rest-frame equivalent width against the fitted AGN continuum.
    fwhm_kms
        Broad-component FWHM.
    host_fraction_5100
        Fitted host fraction of the total continuum at 5100 A.
    continuum_luminosity
        log10 L_5100 if computable, else None.
    continuum_snr
        Median continuum S/N per pixel in the fitting window.
    """

    status: FitStatus
    line: str
    flux: float | None
    flux_error: float | None
    detected: bool
    significance: float | None
    upper_limit: float | None
    ew: float | None
    ew_error: float | None
    fwhm_kms: float | None
    host_fraction_5100: float | None
    continuum_luminosity: float | None
    continuum_snr: float
    reduced_chi2: float | None = None

    @property
    def usable(self) -> bool:
        """True when this epoch can enter a classification decision at all.

        A measurement that is not usable makes the *pair* unclassifiable — it
        must not silently become a negative.
        """
        return self.status == "ok" and (self.detected or self.upper_limit is not None)

    def with_status(self, status: FitStatus) -> BroadLineMeasurement:
        return replace(self, status=status)

    @classmethod
    def failed(cls, line: str, status: FitStatus, continuum_snr: float = 0.0) -> BroadLineMeasurement:
        """Construct an explicit failure. Never returns a zero-flux detection."""
        return cls(
            status=status,
            line=line,
            flux=None,
            flux_error=None,
            detected=False,
            significance=None,
            upper_limit=None,
            ew=None,
            ew_error=None,
            fwhm_kms=None,
            host_fraction_5100=None,
            continuum_luminosity=None,
            continuum_snr=continuum_snr,
        )


__all__ = ["BroadHbetaEW", "BroadLineMeasurement", "FitStatus"]
