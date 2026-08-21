"""Synthetic spectra and measurements with exactly known truth.

Fixtures here are analytic: the host fraction, line flux, and S/N are set by
construction, so a test can assert that the engine recovers what was injected
without any dependence on the fitter.
"""

from __future__ import annotations

import numpy as np
import pytest
from factories import measurement  # noqa: F401  (re-exported for tests)

from p3sf.access.spectra import Spectrum
from p3sf.counterfactual.host import Decomposition

HBETA = 4861.33


@pytest.fixture
def wavelength() -> np.ndarray:
    """Observed-frame grid covering rest 4400-5600 A at z = 0.2."""
    rest = np.arange(4400.0, 5600.0, 0.8)
    return rest * 1.2


@pytest.fixture
def redshift() -> float:
    return 0.2


def _gaussian(x: np.ndarray, center: float, fwhm: float, flux: float) -> np.ndarray:
    sigma = fwhm / 2.3548200450309493
    profile = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    area = np.trapezoid(profile, x)
    return profile * (flux / area) if area > 0 else profile * 0.0


@pytest.fixture
def decomposition(wavelength: np.ndarray, redshift: float) -> Decomposition:
    """AGN power law + flat host + one broad Hbeta, with a known host fraction."""
    rest = wavelength / (1.0 + redshift)

    agn = 10.0 * (rest / 5100.0) ** -1.5
    host = 10.0 * np.ones_like(rest)  # equal to AGN at 5100 A -> f_host = 0.5
    lines = _gaussian(rest, HBETA, fwhm=80.0, flux=500.0)

    return Decomposition(
        wavelength=wavelength,
        agn_continuum=agn,
        host=host,
        lines=lines,
        redshift=redshift,
    )


@pytest.fixture
def clean_spectrum(decomposition: Decomposition) -> Spectrum:
    """Noiseless spectrum exactly equal to the model, with a known S/N of 50."""
    model = decomposition.agn_continuum + decomposition.host + decomposition.lines
    error = model / 50.0
    return Spectrum(
        wavelength=decomposition.wavelength,
        flux=model,
        error=error,
        redshift=decomposition.redshift,
        mask=np.zeros_like(model, dtype=int),
    )


@pytest.fixture
def noisy_spectrum(decomposition: Decomposition) -> Spectrum:
    """Same model with one realization of noise at S/N = 50."""
    model = decomposition.agn_continuum + decomposition.host + decomposition.lines
    error = model / 50.0
    rng = np.random.default_rng(20260814)
    return Spectrum(
        wavelength=decomposition.wavelength,
        flux=model + rng.normal(0.0, 1.0, model.size) * error,
        error=error,
        redshift=decomposition.redshift,
        mask=np.zeros_like(model, dtype=int),
    )


@pytest.fixture
def continuum_window() -> tuple[float, float]:
    return (5080.0, 5130.0)
