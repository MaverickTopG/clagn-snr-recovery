"""Builders for measurements with explicit, hand-checkable values."""

from __future__ import annotations

from p3sf.fitting.results import BroadLineMeasurement


def measurement(
    *,
    flux: float | None,
    error: float | None,
    detected: bool,
    upper_limit: float | None = None,
    ew: float | None = 50.0,
    fwhm: float | None = 5000.0,
    host_fraction: float | None = 0.5,
    snr: float = 20.0,
    line: str = "Hbeta",
    status: str = "ok",
) -> BroadLineMeasurement:
    """Build a BroadLineMeasurement whose every field is known to the caller."""
    significance = (flux / error) if (flux is not None and error) else None
    return BroadLineMeasurement(
        status=status,  # type: ignore[arg-type]
        line=line,
        flux=flux,
        flux_error=error,
        detected=detected,
        significance=significance,
        upper_limit=upper_limit,
        ew=ew,
        ew_error=5.0 if ew is not None else None,
        fwhm_kms=fwhm,
        host_fraction_5100=host_fraction,
        continuum_luminosity=44.0,
        continuum_snr=snr,
    )


__all__ = ["measurement"]
