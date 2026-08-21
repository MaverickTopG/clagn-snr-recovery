"""Frozen XSL-to-DESI stellar-shape transform for the D-083/D-084 pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p3sf.fitting.desi_recovery import resolution_fwhm_angstrom

C_KMS = 299792.458
GAUSSIAN_FWHM = 2.354820045


@dataclass(frozen=True)
class StellarTransformDiagnostics:
    losvd_sigma_kms: float
    n_target_pixels: int
    n_resolution_interpolated: int
    min_convolution_fwhm_angstrom: float
    median_convolution_fwhm_angstrom: float
    max_convolution_fwhm_angstrom: float
    convolution_clipped: bool


def _fill_resolution(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & (values > 0)
    if finite.sum() < 2:
        raise ValueError("target RESOLUTION has fewer than two finite positive widths")
    index = np.arange(values.size)
    filled = np.interp(index, index[finite], values[finite])
    return filled, int((~finite).sum())


def _variable_gaussian_sample(
    source_wave: np.ndarray,
    source_flux: np.ndarray,
    target_wave: np.ndarray,
    sigma_angstrom: np.ndarray,
) -> np.ndarray:
    """Integrate a locally Gaussian-broadened source at exact target wavelengths."""
    wave = np.asarray(source_wave, dtype=float)
    flux = np.asarray(source_flux, dtype=float)
    target = np.asarray(target_wave, dtype=float)
    sigma = np.asarray(sigma_angstrom, dtype=float)
    if wave.ndim != 1 or not np.all(np.diff(wave) > 0):
        raise ValueError("source wavelength must be one-dimensional and strictly increasing")
    if wave.shape != flux.shape or target.shape != sigma.shape:
        raise ValueError("stellar transform array shapes are inconsistent")
    widths = np.gradient(wave)
    out = np.empty(target.shape, dtype=float)
    for i, (centre, local_sigma) in enumerate(zip(target, sigma, strict=True)):
        if not np.isfinite(local_sigma) or local_sigma <= 0:
            raise ValueError(f"non-positive convolution sigma at target pixel {i}")
        lo = int(np.searchsorted(wave, centre - 5.0 * local_sigma, side="left"))
        hi = int(np.searchsorted(wave, centre + 5.0 * local_sigma, side="right"))
        if hi - lo < 5:
            raise ValueError(f"insufficient XSL support at target wavelength {centre:.3f}")
        offset = (wave[lo:hi] - centre) / local_sigma
        weight = np.exp(-0.5 * offset**2) * widths[lo:hi]
        out[i] = float(np.sum(flux[lo:hi] * weight) / np.sum(weight))
    return out


def transform_xsl_to_desi(
    template_wave_rest: np.ndarray,
    template_flux: np.ndarray,
    template_fwhm_angstrom: np.ndarray,
    target_wave_observed: np.ndarray,
    target_resolution: np.ndarray,
    *,
    redshift: float,
    losvd_sigma_kms: float = 150.0,
) -> tuple[np.ndarray, StellarTransformDiagnostics]:
    """Broaden an intrinsic-rest XSL shape, then redshift/resample to a DESI grid.

    XSL already carries its wavelength-dependent library LSF. The additional
    Gaussian width is the positive quadrature difference needed to reach the
    target DESI instrumental LSF plus the frozen LOSVD convention. Any negative
    difference is a hard failure; no clipping is permitted.
    """
    target_wave = np.asarray(target_wave_observed, dtype=float)
    target_rest = target_wave / (1.0 + float(redshift))
    observed_fwhm = resolution_fwhm_angstrom(target_wave, target_resolution)
    observed_fwhm, n_interpolated = _fill_resolution(observed_fwhm)
    instrumental_rest = observed_fwhm / (1.0 + float(redshift))
    losvd_fwhm = GAUSSIAN_FWHM * float(losvd_sigma_kms) / C_KMS * target_rest
    target_total = np.hypot(instrumental_rest, losvd_fwhm)
    intrinsic = np.interp(
        target_rest, template_wave_rest, template_fwhm_angstrom,
        left=np.nan, right=np.nan,
    )
    transformable = np.isfinite(intrinsic)
    required = (target_rest >= 3700.0) & (target_rest <= 7000.0)
    if not np.all(transformable[required]):
        raise ValueError("XSL LSF does not cover the required rest 3700-7000 A domain")
    difference_squared = target_total[transformable] ** 2 - intrinsic[transformable] ** 2
    clipped = bool(np.any(difference_squared <= 0))
    if clipped:
        raise ValueError("XSL-to-target convolution would require LSF clipping")
    convolution_fwhm = np.sqrt(difference_squared)
    transformed = np.zeros(target_rest.shape, dtype=float)
    transformed[transformable] = _variable_gaussian_sample(
        np.asarray(template_wave_rest, dtype=float),
        np.asarray(template_flux, dtype=float),
        target_rest[transformable],
        convolution_fwhm / GAUSSIAN_FWHM,
    )
    diagnostic = StellarTransformDiagnostics(
        losvd_sigma_kms=float(losvd_sigma_kms),
        n_target_pixels=int(transformable.sum()),
        n_resolution_interpolated=n_interpolated,
        min_convolution_fwhm_angstrom=float(np.min(convolution_fwhm)),
        median_convolution_fwhm_angstrom=float(np.median(convolution_fwhm)),
        max_convolution_fwhm_angstrom=float(np.max(convolution_fwhm)),
        convolution_clipped=clipped,
    )
    return transformed, diagnostic


__all__ = ["StellarTransformDiagnostics", "transform_xsl_to_desi"]
