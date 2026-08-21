# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/spectral_ppxf.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 53bc427ca67d75f01071a17aeb2858b69437e567c1b81b26582cd1fd93f9411a
# =====================================================================
"""Production pPXF broad-Hbeta EW fitter (host-subtracted, aperture-robust).

Decomposes a spectrum into stellar host (E-MILES SPS templates), an AGN power-law
continuum, narrow emission lines, and a broad-Hbeta component, then measures the
broad-Hbeta equivalent width against the **AGN power-law continuum only** (host
removed) — the quantity that is robust to the SDSS->DESI aperture change because
both numerator and denominator are compact-AGN emission.

Drop-in behind the `BroadHbetaEW` contract used by `outcome.build_outcome`; a
production run swaps this in for the constrained fixture in `spectral.py`.

STATUS: the engine RUNS correctly (host+gas decomposition validated on real SDSS
data, chi2/dof=1.0), but is **NOT yet calibrated for confirmatory use**. The
synthetic inject-and-recover (`scripts/calibrate_ppxf.py`) exposed the classic
broad-line <-> continuum degeneracy: a broad Hbeta (sigma~45 A) has wings that
mimic the AGN power-law, so the host/AGN continuum split destabilizes -- EW
recovery is not yet stable (ratio ~1.3 at EW 25 with a multiplicative polynomial,
and the power-law weight collapses at EW >~ 50). Reliable recovery needs the
careful Phase-2 decomposition work (line-free continuum windows, power-law priors,
a versioned Fe II template) and the aperture-invariance cross-instrument null.
`EW_CALIBRATION` stays 1.0 until that work lands. Until then the constrained
fixture in `spectral.py` remains the outcome path for feasibility.
"""

from __future__ import annotations

import os

import numpy as np

from p3sf.fitting.results import BroadHbetaEW

_C_KMS = 299792.458
_HBETA = 4861.33
_PL_PIVOT = 5100.0

# Empirical EW zero-point (recovered = raw * EW_CALIBRATION); set from
# scripts/calibrate_ppxf.py. 1.0 until the calibration run refines it.
EW_CALIBRATION = 1.0


def _default_sps_filename() -> str:
    import ppxf  # type: ignore[import-untyped]

    return os.path.join(os.path.dirname(ppxf.__file__), "sps_models", "spectra_emiles_9.0.npz")


def fit_broad_hbeta_ew_ppxf(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    redshift: float,
    fwhm_angstrom: float = 2.0,
    broad_sigma_kms: float = 2000.0,
    powerlaw_slope: float = -1.5,
    sps_filename: str | None = None,
) -> BroadHbetaEW:
    """Fit host + AGN power-law + narrow/broad Hbeta; return host-subtracted broad-Hbeta EW."""
    from ppxf import ppxf_util as util  # type: ignore[import-untyped]
    from ppxf import sps_util as lib  # type: ignore[import-untyped]
    from ppxf.ppxf import ppxf  # type: ignore[import-untyped]

    wave = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(error, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    wave, values, errors = wave[valid], values[valid], errors[valid]
    # Hbeta+[OIII] region in the rest frame -> observed
    lo, hi = 4400.0 * (1 + redshift), 5350.0 * (1 + redshift)
    region = (wave > lo) & (wave < hi)
    wave, values, errors = wave[region], values[region], errors[region]
    if len(wave) < 150 or wave.min() / (1 + redshift) > 4750 or wave.max() / (1 + redshift) < 5050:
        return BroadHbetaEW("unusable_coverage", None, None, False, None, 0.0)

    norm = float(np.median(values))
    if not np.isfinite(norm) or norm <= 0:
        return BroadHbetaEW("unusable_snr", None, None, False, None, 0.0)
    galaxy = values / norm
    noise = errors / norm
    continuum_snr = float(np.median(values / errors))

    ln_lam = np.log(wave)
    velscale = _C_KMS * (ln_lam[-1] - ln_lam[0]) / (ln_lam.size - 1)
    fwhm_gal = {"lam": wave, "fwhm": np.full_like(wave, fwhm_angstrom)}
    sps = lib.sps_lib(sps_filename or _default_sps_filename(), velscale, fwhm_gal,
                      norm_range=[5070, 5950])
    stars = sps.templates.reshape(sps.templates.shape[0], -1)
    lam_temp = np.exp(sps.ln_lam_temp)
    powerlaw = (lam_temp / _PL_PIVOT) ** powerlaw_slope  # =1 at pivot
    stars = np.column_stack([stars, powerlaw])
    pl_index = stars.shape[1] - 1

    lam_range = np.array([wave.min(), wave.max()]) / (1 + redshift)
    gas, gas_names, _ = util.emission_lines(sps.ln_lam_temp, lam_range, fwhm_gal)
    broad_sigma_A = broad_sigma_kms / _C_KMS * _HBETA
    broad = np.exp(-0.5 * ((lam_temp - _HBETA) / broad_sigma_A) ** 2)
    broad /= np.trapezoid(broad, lam_temp)  # unit-integral, like the narrow gas templates

    templates = np.column_stack([stars, gas, broad.reshape(-1, 1)])
    n_star = stars.shape[1]
    n_gas = gas.shape[1]
    component = [0] * n_star + [1] * n_gas + [2]
    gas_component = np.array(component) > 0
    names = list(gas_names) + ["Hb_broad"]
    vel = _C_KMS * np.log(1 + redshift)
    start = [[vel, 200.0], [vel, 200.0], [vel, broad_sigma_kms]]

    fit = ppxf(
        templates, galaxy, noise, velscale, start, moments=[2, 2, 2],
        degree=-1, mdegree=10, lam=wave, lam_temp=lam_temp,
        component=component, gas_component=gas_component, gas_names=names, quiet=True,
    )
    flux_by_name = dict(zip(names, fit.gas_flux, strict=True))
    err_by_name = dict(zip(names, fit.gas_flux_error, strict=True))
    broad_flux = float(flux_by_name["Hb_broad"])
    broad_error = float(err_by_name["Hb_broad"])
    # AGN continuum (host-subtracted) at Hbeta = power-law contribution only
    agn_continuum = float(fit.weights[pl_index] * (_HBETA / _PL_PIVOT) ** powerlaw_slope)
    if agn_continuum <= 0:
        return BroadHbetaEW("unusable_snr", None, None, False, None, continuum_snr)

    ew = EW_CALIBRATION * broad_flux / agn_continuum
    ew_error = EW_CALIBRATION * broad_error / agn_continuum
    detected = bool(broad_flux > 3 * broad_error)
    upper_limit = None if detected else EW_CALIBRATION * 3 * broad_error / agn_continuum
    return BroadHbetaEW("ok", ew, ew_error, detected, upper_limit, continuum_snr)
