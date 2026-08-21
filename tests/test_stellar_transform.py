from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from p3sf.counterfactual.stellar import GAUSSIAN_FWHM, transform_xsl_to_desi


def test_xsl_to_desi_analytic_line_width_and_no_clipping() -> None:
    source_wave = np.arange(4950.0, 5250.0, 0.02)
    intrinsic_fwhm = np.full(source_wave.shape, 0.8)
    source = np.exp(-0.5 * ((source_wave - 5100.0) / (0.8 / GAUSSIAN_FWHM)) ** 2)
    target_wave = np.arange(5000.0, 5200.0, 0.2)
    instrumental_fwhm = 2.0
    sigma_pixel = instrumental_fwhm / GAUSSIAN_FWHM / 0.2
    offsets = np.arange(-25, 26, dtype=float)
    profile = np.exp(-0.5 * (offsets / sigma_pixel) ** 2)
    resolution = np.repeat(profile[:, None], target_wave.size, axis=1)
    transformed, diagnostic = transform_xsl_to_desi(
        source_wave, source, intrinsic_fwhm, target_wave, resolution,
        redshift=0.0, losvd_sigma_kms=150.0,
    )
    around = np.abs(target_wave - 5100.0) < 20.0

    def gaussian(x, amplitude, centre, sigma):
        return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)

    fitted, _ = curve_fit(
        gaussian, target_wave[around], transformed[around],
        p0=(1.0, 5100.0, 3.0),
    )
    expected = np.hypot(
        instrumental_fwhm,
        GAUSSIAN_FWHM * 150.0 / 299792.458 * 5100.0,
    )
    measured = abs(float(fitted[2])) * GAUSSIAN_FWHM
    assert abs(measured / expected - 1.0) < 1e-3
    assert diagnostic.convolution_clipped is False
