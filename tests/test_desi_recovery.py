import numpy as np

from p3sf.fitting.desi_recovery import resolution_fwhm_angstrom


def test_resolution_second_moment_recovers_gaussian_fwhm() -> None:
    offsets = np.arange(-5, 6, dtype=float)
    sigma_pixels = 1.25
    profile = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
    matrix = np.repeat(profile[:, None], 100, axis=1)
    wavelength = 4000.0 + np.arange(100) * 0.8
    measured = resolution_fwhm_angstrom(wavelength, matrix)
    assert np.allclose(measured, 2.354820045 * sigma_pixels * 0.8, rtol=2e-3)


def test_resolution_shape_mismatch_fails_closed() -> None:
    with np.testing.assert_raises_regex(ValueError, "RESOLUTION must be"):
        resolution_fwhm_angstrom(np.arange(10.0), np.ones((11, 9)))
