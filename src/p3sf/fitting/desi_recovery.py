"""Small, frozen helpers for the corrected-DESI 17G-R regeneration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DesiSpectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    resolution: np.ndarray
    good: np.ndarray


def read_corrected_desi(path: Path) -> DesiSpectrum:
    """Read the operational corrected product, honoring native IVAR and MASK."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        wavelength = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
        flux = np.asarray(hdul["FLUX"].data, dtype=float)
        ivar = np.asarray(hdul["IVAR"].data, dtype=float)
        mask = np.asarray(hdul["MASK"].data)
        resolution = np.asarray(hdul["RESOLUTION"].data, dtype=float)
    error = np.full(ivar.shape, np.nan, dtype=float)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    good = (
        np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error)
        & (error > 0) & (mask == 0)
    )
    return DesiSpectrum(wavelength, flux, error, resolution, good)


def resolution_fwhm_angstrom(
    wavelength: np.ndarray, resolution: np.ndarray
) -> np.ndarray:
    """Convert a DESI resolution matrix to a wavelength-aware Gaussian FWHM.

    The matrix rows are diagonals centred on zero pixel offset.  Its normalized
    second moment gives sigma in pixels; the local wavelength spacing converts
    this to Angstrom.  No scalar full-spectrum resolution is substituted.
    """
    matrix = np.asarray(resolution, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != wavelength.size:
        raise ValueError(
            f"RESOLUTION must be (ndiag, npix), got {matrix.shape} for {wavelength.size}"
        )
    offsets = np.arange(matrix.shape[0], dtype=float) - matrix.shape[0] // 2
    weights = np.clip(matrix, 0.0, None)
    norm = weights.sum(axis=0)
    variance = np.divide(
        (weights * offsets[:, None] ** 2).sum(axis=0), norm,
        out=np.full(norm.shape, np.nan), where=norm > 0,
    )
    spacing = np.gradient(np.asarray(wavelength, dtype=float))
    return 2.354820045 * np.sqrt(np.clip(variance, 0.0, None)) * spacing


def residual_summary(
    wavelength: np.ndarray, observed: np.ndarray, model: np.ndarray
) -> dict[str, float | int | None]:
    """Unweighted descriptive residuals in the frozen diagnostic domains."""
    from p3sf.qc.alignment import STELLAR_FEATURES

    residual = np.asarray(observed) - np.asarray(model)

    def rms(mask: np.ndarray) -> float | None:
        finite = mask & np.isfinite(residual)
        return float(np.sqrt(np.mean(residual[finite] ** 2))) if finite.sum() > 5 else None

    stellar = np.zeros(wavelength.size, dtype=bool)
    for _centre, window in STELLAR_FEATURES.values():
        stellar |= (wavelength >= window[0]) & (wavelength <= window[1])
    hbeta = (wavelength >= 4700.0) & (wavelength <= 5100.0)
    return {
        "rms_global": rms(np.ones(wavelength.size, dtype=bool)),
        "rms_stellar_features": rms(stellar),
        "rms_hbeta": rms(hbeta),
        "n_stellar_feature_pixels": int(stellar.sum()),
    }


def run_pyqsofit_host_audit(
    spectrum: DesiSpectrum, redshift: float, vendor: Path
) -> tuple[Any | None, dict[str, object]]:
    """Run the exact legacy host-audit configuration and name decline branches."""
    from pyqsofit import HostDecomp
    from pyqsofit.PyQSOFit import QSOFit

    good = spectrum.good
    captured: dict[str, object] = {}
    original = HostDecomp.Linear_decomp.auto_decomp

    def capturing(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["cube"] = result[0]
        return result

    HostDecomp.Linear_decomp.auto_decomp = capturing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = QSOFit(
                spectrum.wavelength[good], spectrum.flux[good], spectrum.error[good],
                redshift, path=str(vendor),
            )
            fit.Fit(
                name="17G-R", deredden=True, decompose_host=True, Fe_uv_op=True,
                linefit=True, MC=False, save_result=False, plot_fig=False,
                save_fig=False, verbose=False,
            )
    except Exception as error:  # noqa: BLE001
        return None, {
            "host_output_state": "FIT_EXCEPTION",
            "decline_reason": f"FIT_EXCEPTION: {type(error).__name__}: {str(error)[:120]}",
        }
    finally:
        HostDecomp.Linear_decomp.auto_decomp = original

    decomposed = bool(getattr(fit, "decomposed", False))
    state = "OUTPUT_VALID_NONZERO" if decomposed else "DECOMPOSITION_FAILED"
    reason = "ACCEPTED_BY_CONDITIONS" if decomposed else "UNKNOWN"
    detail: dict[str, object] = {
        "host_output_state": state,
        "pyqsofit_decomposed": decomposed,
        "decline_reason": reason,
    }
    if "cube" not in captured:
        detail["decline_reason"] = "TEMPLATE_COVERAGE_BELOW_50_PERCENT"
        return fit, detail

    cube = np.asarray(captured["cube"])
    data, host_model, qso_model = cube[1], cube[3], cube[4]
    level = float(np.median(np.abs(data)))
    negative_fraction = float(np.mean((host_model < 0) | (qso_model < 0)))
    host_ratio = float(np.median(host_model) / level) if level > 0 else np.nan
    host_spec_median = float(np.median(data - qso_model))
    reasons: list[str] = []
    if negative_fraction > 0.1:
        reasons.append("NEGATIVE_MODEL_PIXELS_OVER_10PC")
    if host_ratio < 0.01:
        reasons.append("LOW_HOST_CONTRAST")
    if host_spec_median < 0:
        reasons.append("NEGATIVE_HOST_RESIDUAL")
    detail.update(
        neg_pixel_fraction=negative_fraction,
        host_median_over_flux_level=host_ratio,
        median_host_spec=host_spec_median,
        decline_reason="+".join(reasons) if reasons else "ACCEPTED_BY_CONDITIONS",
    )
    return fit, detail


__all__ = [
    "DesiSpectrum", "read_corrected_desi", "resolution_fwhm_angstrom",
    "residual_summary", "run_pyqsofit_host_audit",
]
