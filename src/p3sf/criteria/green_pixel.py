"""Green et al. (2022) H-beta pixel statistic (D-089).

Two parts, with different claims attached to each.

The **statistic** is reproduced from the source: signed bright-minus-faint over
the combined error, 2 Angstrom rest-frame rebinning, a centered 16-pixel median
with clipped end windows, subtraction of the 4750 Angstrom reference, the
maximum over 4750-4940 Angstrom, and the inclusive threshold at 3.

The **preprocessing** is not. Green et al. did not use PyQSOFit; we reproduce
their described decomposition rather than their implementation of it, so the
line spectrum entering the statistic is our construction and any difference
propagates into the result. This is the same standing as the Yang-ratio
measurement in ``p3sf.fitting.pyqsofit_driver`` and is stated symmetrically in
the manuscript. Do not describe the pipeline as a whole as source-faithful.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from p3sf.criteria.base import CriterionResult
from p3sf.criteria.source_verified import apply_green2022

GREEN_HBETA_INTERVAL_ANGSTROM = (4750.0, 4940.0)
GREEN_REBIN_ANGSTROM = 2.0
GREEN_MEDIAN_PIXELS = 16


class VarianceProvenance(StrEnum):
    """Whether a pixel variance is admissible for the exact measurement domain."""

    NATIVE_SDSS_SUPPORTED = "NATIVE_SDSS_SUPPORTED"
    NATIVE_LAMOST_SUPPORTED = "NATIVE_LAMOST_SUPPORTED"
    NATIVE_SDSSV_SUPPORTED = "NATIVE_SDSSV_SUPPORTED"
    NATIVE_DESI_EDR_SUPPORTED = "NATIVE_DESI_EDR_SUPPORTED"
    GATE_C_CORRECTED_SUPPORTED = "GATE_C_CORRECTED_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class GreenEpochSpectrum:
    """One continuum/host-subtracted rest-frame line spectrum."""

    wavelength_rest: np.ndarray
    line_flux: np.ndarray
    variance: np.ndarray
    variance_provenance: np.ndarray
    preprocessing_valid: bool = True
    invalid_reason: str = ""
    scale_factor: float = 1.0
    scale_provenance: str = "NO_RESCALE_SOURCE_RULE"

    def __post_init__(self) -> None:
        lengths = {
            len(self.wavelength_rest),
            len(self.line_flux),
            len(self.variance),
            len(self.variance_provenance),
        }
        if len(lengths) != 1:
            raise ValueError("Green epoch arrays must have identical lengths")


@dataclass(frozen=True)
class GreenPixelMeasurement:
    """Measured source statistic plus fail-closed provenance."""

    nsigma_hbeta: float | None
    measurement_available: bool
    invalid_reason: str
    max_wavelength_rest: float | None
    reference_nsigma_4750: float | None
    n_rebinned_pixels: int
    all_variance_supported: bool


def _input_pixel_edges(wavelength: np.ndarray) -> np.ndarray:
    if len(wavelength) < 2 or not np.all(np.diff(wavelength) > 0):
        raise ValueError("wavelength must be strictly increasing")
    midpoint = 0.5 * (wavelength[:-1] + wavelength[1:])
    first = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    last = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return np.concatenate(([first], midpoint, [last]))


def _rebin_two_angstrom(
    epoch: GreenEpochSpectrum,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Flux-conserving 2-A bin means with exact variance propagation."""
    order = np.argsort(epoch.wavelength_rest)
    wave = np.asarray(epoch.wavelength_rest, dtype=float)[order]
    flux = np.asarray(epoch.line_flux, dtype=float)[order]
    variance = np.asarray(epoch.variance, dtype=float)[order]
    provenance = np.asarray(epoch.variance_provenance, dtype=str)[order]
    finite = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
    if not epoch.preprocessing_valid or finite.sum() < 2:
        return None
    wave, flux, variance, provenance = (
        wave[finite], flux[finite], variance[finite], provenance[finite]
    )
    local_for_gap = (
        (wave >= GREEN_HBETA_INTERVAL_ANGSTROM[0] - GREEN_REBIN_ANGSTROM)
        & (wave <= GREEN_HBETA_INTERVAL_ANGSTROM[1] + GREEN_REBIN_ANGSTROM)
    )
    if np.any(np.diff(wave[local_for_gap]) > GREEN_REBIN_ANGSTROM):
        return None
    edges_in = _input_pixel_edges(wave)
    edges_out = np.arange(
        GREEN_HBETA_INTERVAL_ANGSTROM[0],
        GREEN_HBETA_INTERVAL_ANGSTROM[1] + GREEN_REBIN_ANGSTROM,
        GREEN_REBIN_ANGSTROM,
    )
    output_wave = edges_out[:-1]
    output_flux = np.empty(len(output_wave), dtype=float)
    output_variance = np.empty(len(output_wave), dtype=float)
    supported_values = {
        VarianceProvenance.NATIVE_SDSS_SUPPORTED.value,
        VarianceProvenance.NATIVE_LAMOST_SUPPORTED.value,
        VarianceProvenance.NATIVE_SDSSV_SUPPORTED.value,
        VarianceProvenance.NATIVE_DESI_EDR_SUPPORTED.value,
        VarianceProvenance.GATE_C_CORRECTED_SUPPORTED.value,
    }
    for index, (left, right) in enumerate(zip(edges_out[:-1], edges_out[1:], strict=True)):
        overlap = np.maximum(
            0.0, np.minimum(edges_in[1:], right) - np.maximum(edges_in[:-1], left)
        )
        used = overlap > 0
        total = float(overlap.sum())
        if (
            not used.any()
            or not np.isclose(total, right - left, rtol=0.0, atol=1.0e-8)
            or not set(provenance[used]) <= supported_values
        ):
            return None
        weights = overlap[used] / total
        output_flux[index] = float(np.sum(weights * flux[used]))
        output_variance[index] = float(np.sum(weights**2 * variance[used]))
    return output_wave, output_flux, output_variance


def _running_median_16(values: np.ndarray) -> np.ndarray:
    """Centered 16-pixel median; clipped windows define endpoint behavior."""
    output = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - 7)
        stop = min(len(values), index + 9)
        output[index] = float(np.median(values[start:stop]))
    return output


def measure_green_pixel_nsigma(
    bright: GreenEpochSpectrum,
    faint: GreenEpochSpectrum,
) -> GreenPixelMeasurement:
    """Measure max[median16(Nsigma)-median16(Nsigma)[4750]] over H-beta."""
    bright_rebinned = _rebin_two_angstrom(bright)
    faint_rebinned = _rebin_two_angstrom(faint)
    if bright_rebinned is None or faint_rebinned is None:
        reasons = [
            reason
            for reason in (bright.invalid_reason, faint.invalid_reason)
            if reason
        ]
        reason = "|".join(reasons) or "MASK_COVERAGE_OR_VARIANCE_PROVENANCE_UNSUPPORTED"
        return GreenPixelMeasurement(None, False, reason, None, None, 0, False)
    wave_b, flux_b, var_b = bright_rebinned
    wave_f, flux_f, var_f = faint_rebinned
    if not np.array_equal(wave_b, wave_f):
        return GreenPixelMeasurement(
            None, False, "COMMON_REST_GRID_MISMATCH", None, None, 0, False
        )
    denominator = np.sqrt(var_b + var_f)
    if not np.all(np.isfinite(denominator) & (denominator > 0)):
        return GreenPixelMeasurement(
            None, False, "COMBINED_VARIANCE_INVALID", None, None, len(wave_b), False
        )
    # Green equation (1): bright minus dim. It is not an absolute difference.
    pixel_nsigma = (flux_b - flux_f) / denominator
    smoothed = _running_median_16(pixel_nsigma)
    relative = smoothed - smoothed[0]
    maximum = int(np.argmax(relative))
    statistic = float(relative[maximum])
    return GreenPixelMeasurement(
        nsigma_hbeta=statistic,
        measurement_available=True,
        invalid_reason="",
        max_wavelength_rest=float(wave_b[maximum]),
        reference_nsigma_4750=float(smoothed[0]),
        n_rebinned_pixels=len(wave_b),
        all_variance_supported=True,
    )


def classify_green_pixel_measurement(
    measurement: GreenPixelMeasurement,
) -> CriterionResult:
    return apply_green2022(
        nsigma_hbeta=measurement.nsigma_hbeta,
        measurement_available=measurement.measurement_available,
    )


def fit_green_epoch_line_spectrum_pyqsofit(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    redshift: float,
    path: str | Path,
    variance_provenance: VarianceProvenance,
    ra: float,
    dec: float,
    scale_factor: float = 1.0,
    scale_provenance: str = "NO_RESCALE_SOURCE_RULE",
) -> GreenEpochSpectrum:
    """Create Green's rest-frame continuum/host-subtracted line spectrum.

    This Green-specific fit follows the paper: Galactic dereddening, separate
    host decomposition per epoch, power law plus optical/UV Fe II and Balmer
    continuum, and no polynomial continuum. The supplied scalar rescaling must
    be determined before realization outcomes are inspected.
    """
    from pyqsofit.PyQSOFit import QSOFit

    wave = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(error, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if valid.sum() < 200 or not np.isfinite(scale_factor) or scale_factor <= 0:
        return GreenEpochSpectrum(
            np.array([]), np.array([]), np.array([]), np.array([], dtype=str),
            False, "INPUT_OR_SCALE_INVALID", scale_factor, scale_provenance,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = QSOFit(
                wave[valid], values[valid], errors[valid], redshift,
                ra=ra, dec=dec, path=str(path),
            )
            q.Fit(
                name=None, nsmooth=1, deredden=True, reject_badpix=False,
                decompose_host=True, host_prior=False, npca_gal=5, npca_qso=10,
                Fe_uv_op=True, poly=False, BC=True, linefit=False, MCMC=False,
                plot_fig=False, save_fig=False, save_result=False,
            )
        continuum_valid = bool(
            getattr(q.conti_fit, "success", False)
            and np.all(np.isfinite(q.line_flux))
            and np.all(np.isfinite(q.err) & (q.err > 0))
        )
    except Exception:  # noqa: BLE001
        return GreenEpochSpectrum(
            np.array([]), np.array([]), np.array([]), np.array([], dtype=str),
            False, "GREEN_PYQSOFIT_PREPROCESSING_FAILED", scale_factor, scale_provenance,
        )
    provenance = np.full(len(q.wave), variance_provenance.value, dtype=object)
    return GreenEpochSpectrum(
        wavelength_rest=np.asarray(q.wave, dtype=float),
        line_flux=np.asarray(q.line_flux, dtype=float) * scale_factor,
        variance=(np.asarray(q.err, dtype=float) * scale_factor) ** 2,
        variance_provenance=provenance,
        preprocessing_valid=continuum_valid,
        invalid_reason="" if continuum_valid else "GREEN_CONTINUUM_PREPROCESSING_INVALID",
        scale_factor=scale_factor,
        scale_provenance=scale_provenance,
    )


__all__ = [
    "GREEN_HBETA_INTERVAL_ANGSTROM",
    "GREEN_MEDIAN_PIXELS",
    "GREEN_REBIN_ANGSTROM",
    "GreenEpochSpectrum",
    "GreenPixelMeasurement",
    "VarianceProvenance",
    "classify_green_pixel_measurement",
    "fit_green_epoch_line_spectrum_pyqsofit",
    "measure_green_pixel_nsigma",
]
