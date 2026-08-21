# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/spectral_pyqsofit.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 6ff1e055d67061955617b0f030e32dd1de66cf8b0193e5e1e3386f865bc9a0ab
# =====================================================================
"""Production broad-Hbeta EW fitter via PyQSOFit (host + Fe II + AGN decomposition).

PyQSOFit (Guo, Shen & Wang) is purpose-built for AGN spectral decomposition:
host-galaxy PCA subtraction + AGN power-law + Fe II template + Balmer continuum,
then multi-Gaussian broad/narrow emission lines. It measures the broad-Hbeta
equivalent width against the *host-subtracted AGN continuum* — the aperture-robust
quantity our outcome needs — and handles the broad-line<->continuum degeneracy
(via line-free continuum windows) that destabilized the naive pPXF fit.

Drop-in behind the `BroadHbetaEW` contract used by `outcome.build_outcome`.
Detection uses PyQSOFit's own broad-Hbeta SNR. Errors are approximated from
EW/SNR without MCMC (set MCMC on for rigorous errors in the production run).

STATUS: installed + wrapped; runs the full decomposition on real SDSS data and
synthetics. PyQSOFit is the literature-standard AGN fitter (validated across many
published quasar-property catalogs), so it supersedes the hand-rolled
`spectral_ppxf.py`. It is NOT re-validated by the pPXF synthetic (that synthetic
is built on a different host/continuum model, so it isn't a fair test). The
remaining Phase-2 validation is the standard one: compare recovered broad-Hbeta
EW/FWHM against a published catalog (e.g. Shen et al.) on real spectra, then run
the aperture-invariance cross-instrument null. Requires PyQSOFit installed from
github.com/legolason/PyQSOFit (not on PyPI); the `spectral.py` fixture remains
the feasibility outcome until the catalog validation lands.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from p3sf.fitting.results import BroadHbetaEW

# Broad-Hbeta rest-frame EW rarely exceeds a few hundred A even in extreme AGN;
# 2000 A is a generous physical ceiling above which the value is a continuum-
# collapse artifact of the decomposition, not a real line measurement.
EW_PHYSICAL_MAX_ANGSTROM = 2000.0
YANG_LOCAL_HBETA_RANGE_ANGSTROM = (4640.0, 5100.0)
YANG_BROAD_FWHM_RANGE_KMS = (1200.0, 20_000.0)
_BOUND_RTOL = 1.0e-5


@dataclass(frozen=True)
class BroadHbetaDiagnostics:
    """Persisted PyQSOFit quantities used for external-catalog validation."""

    status: str
    broad_hbeta_flux: float | None
    broad_hbeta_flux_error: float | None
    broad_hbeta_fwhm_kms: float | None
    broad_hbeta_fwhm_error_kms: float | None
    broad_hbeta_ew: float | None
    broad_hbeta_ew_error: float | None
    broad_hbeta_snr: float | None
    broad_hbeta_detected: bool
    continuum_at_hbeta: float | None
    continuum_snr: float
    broad_halpha_flux: float | None = None
    broad_halpha_flux_error: float | None = None
    broad_halpha_fwhm_kms: float | None = None
    broad_halpha_fwhm_error_kms: float | None = None
    broad_halpha_ew: float | None = None
    broad_halpha_ew_error: float | None = None
    broad_halpha_snr: float | None = None
    broad_halpha_detected: bool = False
    continuum_at_halpha: float | None = None
    l5100: float | None = None
    frac_host_5100: float | None = None
    oiii5007_flux: float | None = None
    oiii5007_snr: float | None = None


@dataclass(frozen=True)
class YangHbetaFitAudit:
    """Prospective local validity contract for one Yang H-beta measurement."""

    fit_completed: bool
    optimizer_success: bool
    parameter_at_bound: bool
    local_hbeta_coverage: bool
    local_mask_fraction: float
    continuum_valid: bool
    broad_component_present: bool
    broad_component_amplitude: float | None
    broad_component_width: float | None
    broad_flux: float | None
    residual_rms_local: float | None
    finite_measurement: bool
    fit_valid: bool
    invalid_reason: str
    valid_nondetection: bool


def _complete_two_angstrom_coverage(
    rest: np.ndarray, valid: np.ndarray, interval: tuple[float, float]
) -> bool:
    """Require at least one usable input pixel in every structural 2-A bin."""
    lo, hi = interval
    edges = np.arange(lo, hi + 2.0, 2.0)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    counts, _ = np.histogram(rest[valid], bins=edges)
    return bool(len(counts) > 0 and np.all(counts > 0))


def _at_bound(value: float, lower: float, upper: float) -> bool:
    return bool(
        np.isclose(value, lower, rtol=_BOUND_RTOL, atol=0.0)
        or np.isclose(value, upper, rtol=_BOUND_RTOL, atol=0.0)
    )


def _failed_yang_audit(
    *, local_mask_fraction: float, coverage: bool, reason: str
) -> YangHbetaFitAudit:
    return YangHbetaFitAudit(
        fit_completed=False,
        optimizer_success=False,
        parameter_at_bound=False,
        local_hbeta_coverage=coverage,
        local_mask_fraction=local_mask_fraction,
        continuum_valid=False,
        broad_component_present=False,
        broad_component_amplitude=None,
        broad_component_width=None,
        broad_flux=None,
        residual_rms_local=None,
        finite_measurement=False,
        fit_valid=False,
        invalid_reason=reason,
        valid_nondetection=False,
    )


def _oiii5007_from_fit(fit: Any) -> tuple[float | None, float | None]:
    """Return the frozen narrow OIII5007c area and S/N from a PyQSOFit object."""
    try:
        properties = fit.line_prop_from_name("OIII5007c", "narrow")
        area = float(properties[4])
        snr = float(properties[5])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None, None
    return (
        area if np.isfinite(area) and area > 0 else None,
        snr if np.isfinite(snr) and snr > 0 else None,
    )


def _pyqsofit_path() -> str:
    import pyqsofit  # type: ignore[import-not-found]

    return os.path.dirname(pyqsofit.__file__)


def fit_continuum_luminosity_pyqsofit(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    redshift: float,
    path: str | None = None,
) -> tuple[float | None, float | None]:
    """Return (log L5100, host fraction at 5100 A) from a PyQSOFit continuum fit.

    L5100 is the standard optical AGN continuum luminosity -- the luminosity
    covariate for the confirmatory fit -- and is independent of the WISE
    predictor (no shared-instrument dependency). Returns (None, None) when the
    5100 A continuum is not measurable (PyQSOFit reports L5100 <= 0 / -1).
    """
    from pyqsofit.PyQSOFit import QSOFit  # type: ignore[import-not-found]

    wave = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(error, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    wave, values, errors = wave[valid], values[valid], errors[valid]
    rest = wave / (1 + redshift)
    if len(wave) < 200 or rest.min() > 5100 or rest.max() < 5100:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = QSOFit(wave, values, errors, redshift, path=path or _pyqsofit_path())
            q.Fit(
                name=None, nsmooth=1, deredden=False, reject_badpix=False,
                decompose_host=True, host_prior=False, Fe_uv_op=True, poly=True, BC=False,
                linefit=False, MCMC=False, plot_fig=False, save_fig=False, save_result=False,
            )
        conti = dict(zip(q.conti_result_name, q.conti_result, strict=True))
    except Exception:  # noqa: BLE001
        return None, None
    l5100 = float(conti.get("L5100", -1.0))
    if not np.isfinite(l5100) or l5100 <= 0:
        return None, None
    frac_host = conti.get("frac_host_5100")
    return l5100, (float(frac_host) if frac_host is not None else None)


def fit_broad_hbeta_ew_pyqsofit(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    redshift: float,
    path: str | None = None,
    npca_gal: int = 5,
    npca_qso: int = 10,
    decompose_host: bool = True,
) -> BroadHbetaEW:
    """Fit with PyQSOFit; return the host-subtracted broad-Hbeta EW (BroadHbetaEW).

    ``decompose_host`` toggles the host-galaxy PCA subtraction. It defaults to
    True (the aperture-robust, host-subtracted EW the outcome needs); setting it
    False measures the EW against the total AGN+host continuum, used only for the
    calibration diagnostic against total-continuum catalogues like Shen 2011.
    """
    from pyqsofit.PyQSOFit import QSOFit  # type: ignore[import-not-found]

    wave = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(error, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    wave, values, errors = wave[valid], values[valid], errors[valid]
    rest = wave / (1 + redshift)
    if len(wave) < 200 or rest.min() > 4750 or rest.max() < 5050:
        return BroadHbetaEW("unusable_coverage", None, None, False, None, 0.0)
    continuum_snr = float(np.median(values / errors))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = QSOFit(wave, values, errors, redshift, path=path or _pyqsofit_path())
            q.Fit(
                name=None, nsmooth=1, deredden=False, reject_badpix=False,
                decompose_host=decompose_host, host_prior=False,
                npca_gal=npca_gal, npca_qso=npca_qso,
                Fe_uv_op=True, poly=True, BC=False, linefit=True, MCMC=False,
                plot_fig=False, save_fig=False, save_result=False,
            )
        result = dict(zip(q.line_result_name, q.line_result, strict=True))
    except Exception:  # noqa: BLE001  -- PyQSOFit raises on pathological spectra
        return BroadHbetaEW("unusable_snr", None, None, False, None, continuum_snr)

    ew = float(result.get("Hb_whole_br_ew", 0.0))
    snr = float(result.get("Hb_whole_br_snr", 0.0))
    if not np.isfinite(ew) or snr <= 0:
        return BroadHbetaEW("ok", 0.0, None, False, 0.0, continuum_snr)
    # Physical sanity: broad-Hbeta EW is at most a few hundred A. Values far above
    # that come from the decomposition driving the AGN continuum toward zero
    # (EW = line flux / continuum -> blows up), not a real line. Such a fit is
    # untrustworthy, so mark it unusable rather than emitting a ~1e11 A "detection"
    # that would wreck the outcome scale downstream.
    if abs(ew) > EW_PHYSICAL_MAX_ANGSTROM:
        return BroadHbetaEW("unusable_snr", None, None, False, None, continuum_snr)
    ew_error = abs(ew) / snr
    detected = bool(snr > 3.0)
    upper_limit = None if detected else 3.0 * ew_error
    return BroadHbetaEW("ok", ew, ew_error, detected, upper_limit, continuum_snr)


def _diagnostics_from_result(
    result: dict[str, Any], *, continuum_snr: float
) -> BroadHbetaDiagnostics:
    """Map the installed PyQSOFit result contract without component substitution."""

    def finite(name: str) -> float | None:
        try:
            value = float(result[name])
        except (KeyError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    flux = finite("Hb_whole_br_area")
    flux_error = finite("Hb_whole_br_area_err")
    fwhm = finite("Hb_whole_br_fwhm")
    fwhm_error = finite("Hb_whole_br_fwhm_err")
    ew = finite("Hb_whole_br_ew")
    ew_error = finite("Hb_whole_br_ew_err")
    snr = finite("Hb_whole_br_snr")
    continuum = (
        flux / ew if flux is not None and ew is not None and ew > 0 else None
    )
    ha_flux = finite("Ha_whole_br_area")
    ha_flux_error = finite("Ha_whole_br_area_err")
    ha_fwhm = finite("Ha_whole_br_fwhm")
    ha_fwhm_error = finite("Ha_whole_br_fwhm_err")
    ha_ew = finite("Ha_whole_br_ew")
    ha_ew_error = finite("Ha_whole_br_ew_err")
    ha_snr = finite("Ha_whole_br_snr")
    ha_continuum = (
        ha_flux / ha_ew
        if ha_flux is not None and ha_ew is not None and ha_ew > 0
        else None
    )
    # PyQSOFit's non-MCMC production mode can omit formal errors while still
    # returning the fitted line, EW, and S/N. Preserve each available diagnostic
    # independently; production usability is the same EW/SNR contract as the
    # original fitter, not the availability of optional comparison errors.
    sane = ew is not None and abs(ew) <= EW_PHYSICAL_MAX_ANGSTROM
    status = "ok" if ew is not None and snr is not None and snr > 0 and sane else "unusable"
    return BroadHbetaDiagnostics(
        status=status,
        broad_hbeta_flux=flux,
        broad_hbeta_flux_error=flux_error,
        broad_hbeta_fwhm_kms=fwhm,
        broad_hbeta_fwhm_error_kms=fwhm_error,
        broad_hbeta_ew=ew if status == "ok" else None,
        broad_hbeta_ew_error=ew_error if status == "ok" else None,
        broad_hbeta_snr=snr if status == "ok" else None,
        broad_hbeta_detected=bool(status == "ok" and snr is not None and snr > 3.0),
        continuum_at_hbeta=continuum if status == "ok" else None,
        continuum_snr=float(continuum_snr),
        broad_halpha_flux=ha_flux,
        broad_halpha_flux_error=ha_flux_error,
        broad_halpha_fwhm_kms=ha_fwhm,
        broad_halpha_fwhm_error_kms=ha_fwhm_error,
        broad_halpha_ew=ha_ew,
        broad_halpha_ew_error=ha_ew_error,
        broad_halpha_snr=ha_snr,
        broad_halpha_detected=bool(ha_snr is not None and ha_snr > 3.0),
        continuum_at_halpha=ha_continuum,
    )


def fit_broad_hbeta_diagnostics_pyqsofit(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    redshift: float,
    path: str | None = None,
) -> BroadHbetaDiagnostics:
    """Rerun the frozen production fit and expose flux/width/EW diagnostics.

    This uses the same host decomposition and line settings as
    :func:`fit_broad_hbeta_ew_pyqsofit`; it exists only because the original outcome
    cache persisted the EW contract rather than PyQSOFit's full result vector.
    """

    from pyqsofit.PyQSOFit import QSOFit  # type: ignore[import-not-found]

    wave = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(error, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    wave, values, errors = wave[valid], values[valid], errors[valid]
    rest = wave / (1 + redshift)
    if len(wave) < 200 or rest.min() > 4750 or rest.max() < 5050:
        return BroadHbetaDiagnostics(
            status="unusable", broad_hbeta_flux=None, broad_hbeta_flux_error=None,
            broad_hbeta_fwhm_kms=None, broad_hbeta_fwhm_error_kms=None,
            broad_hbeta_ew=None, broad_hbeta_ew_error=None, broad_hbeta_snr=None,
            broad_hbeta_detected=False, continuum_at_hbeta=None, continuum_snr=0.0,
        )
    continuum_snr = float(np.median(values / errors))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = QSOFit(wave, values, errors, redshift, path=path or _pyqsofit_path())
            q.Fit(
                name=None, nsmooth=1, deredden=False, reject_badpix=False,
                decompose_host=True, host_prior=False, npca_gal=5, npca_qso=10,
                Fe_uv_op=True, poly=True, BC=False, linefit=True, MCMC=False,
                plot_fig=False, save_fig=False, save_result=False,
            )
        result = dict(zip(q.line_result_name, q.line_result, strict=True))
        continuum_result = dict(zip(q.conti_result_name, q.conti_result, strict=True))
    except Exception:  # noqa: BLE001
        return BroadHbetaDiagnostics(
            status="unusable", broad_hbeta_flux=None, broad_hbeta_flux_error=None,
            broad_hbeta_fwhm_kms=None, broad_hbeta_fwhm_error_kms=None,
            broad_hbeta_ew=None, broad_hbeta_ew_error=None, broad_hbeta_snr=None,
            broad_hbeta_detected=False, continuum_at_hbeta=None,
            continuum_snr=continuum_snr,
        )
    diagnostics = _diagnostics_from_result(result, continuum_snr=continuum_snr)

    def finite_continuum(name: str, *, positive: bool) -> float | None:
        try:
            value = float(continuum_result[name])
        except (KeyError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) and (value > 0 or not positive) else None

    oiii5007_flux, oiii5007_snr = _oiii5007_from_fit(q)
    return replace(
        diagnostics,
        l5100=finite_continuum("L5100", positive=True),
        frac_host_5100=finite_continuum("frac_host_5100", positive=False),
        oiii5007_flux=oiii5007_flux,
        oiii5007_snr=oiii5007_snr,
    )


def fit_yang_hbeta_audit_pyqsofit(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    redshift: float,
    path: str | None = None,
) -> tuple[BroadHbetaDiagnostics, YangHbetaFitAudit]:
    """Run the production fit and expose the frozen D-089 Yang validity audit.

    The Yang source supplies a flux-ratio rule and permits a zero faint-state
    flux for a valid fitted nondetection. It does not turn optimizer failures or
    bound-pinned active components into zero. Residual RMS is descriptive only.
    """
    from pyqsofit.PyQSOFit import QSOFit  # type: ignore[import-not-found]

    wave_all = np.asarray(wavelength, dtype=float)
    values_all = np.asarray(flux, dtype=float)
    errors_all = np.asarray(error, dtype=float)
    rest_all = wave_all / (1 + redshift)
    valid_all = (
        np.isfinite(wave_all)
        & np.isfinite(values_all)
        & np.isfinite(errors_all)
        & (errors_all > 0)
    )
    local_all = (
        (rest_all >= YANG_LOCAL_HBETA_RANGE_ANGSTROM[0])
        & (rest_all <= YANG_LOCAL_HBETA_RANGE_ANGSTROM[1])
    )
    local_mask_fraction = (
        float(1.0 - valid_all[local_all].mean()) if local_all.any() else 1.0
    )
    coverage = _complete_two_angstrom_coverage(
        rest_all, valid_all, YANG_LOCAL_HBETA_RANGE_ANGSTROM
    )
    wave = wave_all[valid_all]
    values = values_all[valid_all]
    errors = errors_all[valid_all]
    continuum_snr = float(np.median(values / errors)) if len(values) else 0.0

    empty = BroadHbetaDiagnostics(
        status="unusable",
        broad_hbeta_flux=None,
        broad_hbeta_flux_error=None,
        broad_hbeta_fwhm_kms=None,
        broad_hbeta_fwhm_error_kms=None,
        broad_hbeta_ew=None,
        broad_hbeta_ew_error=None,
        broad_hbeta_snr=None,
        broad_hbeta_detected=False,
        continuum_at_hbeta=None,
        continuum_snr=continuum_snr,
    )
    if len(wave) < 200 or not coverage:
        return empty, _failed_yang_audit(
            local_mask_fraction=local_mask_fraction,
            coverage=coverage,
            reason="LOCAL_HBETA_COVERAGE_INCOMPLETE",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = QSOFit(wave, values, errors, redshift, path=path or _pyqsofit_path())
            q.Fit(
                name=None,
                nsmooth=1,
                deredden=False,
                reject_badpix=False,
                decompose_host=True,
                host_prior=False,
                npca_gal=5,
                npca_qso=10,
                Fe_uv_op=True,
                poly=True,
                BC=False,
                linefit=True,
                MCMC=False,
                plot_fig=False,
                save_fig=False,
                save_result=False,
            )
        result = dict(zip(q.line_result_name, q.line_result, strict=True))
    except Exception:  # noqa: BLE001
        return empty, _failed_yang_audit(
            local_mask_fraction=local_mask_fraction,
            coverage=coverage,
            reason="FIT_DID_NOT_COMPLETE",
        )

    diagnostics = _diagnostics_from_result(result, continuum_snr=continuum_snr)
    optimizer_success = False
    for name, value in result.items():
        if name.endswith("_complex_name") and str(value) == "Hb":
            status_name = name.replace("_complex_name", "_line_status")
            try:
                optimizer_success = bool(int(float(result[status_name])))
            except (KeyError, TypeError, ValueError):
                optimizer_success = False
            break

    broad_prior = next(row for row in q.linelist if str(row["linename"]) == "Hb_br")
    active_parameters_at_bound = False
    active_scales: list[float] = []
    all_scales_at_zero = True
    for component in range(1, int(broad_prior["ngauss"]) + 1):
        try:
            scale = float(result[f"Hb_br_{component}_scale"])
            center = float(result[f"Hb_br_{component}_centerwave"])
            sigma = float(result[f"Hb_br_{component}_sigma"])
        except (KeyError, TypeError, ValueError):
            continue
        scale_zero = np.isclose(
            scale, float(broad_prior["minsca"]), rtol=_BOUND_RTOL, atol=0.0
        )
        all_scales_at_zero = all_scales_at_zero and bool(scale_zero)
        if scale_zero:
            continue
        active_scales.append(scale)
        center0 = float(np.log(float(broad_prior["lambda"])))
        active_parameters_at_bound = active_parameters_at_bound or any(
            (
                np.isclose(scale, float(broad_prior["maxsca"]), rtol=_BOUND_RTOL),
                _at_bound(
                    sigma,
                    float(broad_prior["minsig"]),
                    float(broad_prior["maxsig"]),
                ),
                _at_bound(
                    center,
                    center0 - float(broad_prior["voff"]),
                    center0 + float(broad_prior["voff"]),
                ),
            )
        )

    component_present = bool(active_scales)
    amplitude = float(sum(active_scales)) if component_present else 0.0
    width = diagnostics.broad_hbeta_fwhm_kms
    broad_flux = diagnostics.broad_hbeta_flux
    finite_measurement = bool(
        broad_flux is not None
        and np.isfinite(broad_flux)
        and broad_flux >= 0
        and (
            (component_present and width is not None and np.isfinite(width) and width > 0)
            or (not component_present and all_scales_at_zero and broad_flux == 0)
        )
    )
    continuum_valid = bool(
        getattr(q.conti_fit, "success", False)
        and np.all(np.isfinite(q.f_conti_model))
    )
    local = (
        (q.wave >= YANG_LOCAL_HBETA_RANGE_ANGSTROM[0])
        & (q.wave <= YANG_LOCAL_HBETA_RANGE_ANGSTROM[1])
    )
    residual_rms = (
        float(np.sqrt(np.mean((q.line_flux[local] - q.f_line_model[local]) ** 2)))
        if local.any()
        else None
    )
    reasons: list[str] = []
    if not optimizer_success:
        reasons.append("HBETA_OPTIMIZER_FAILED")
    if not coverage:
        reasons.append("LOCAL_HBETA_COVERAGE_INCOMPLETE")
    if not continuum_valid:
        reasons.append("CONTINUUM_INVALID")
    if active_parameters_at_bound:
        reasons.append("ACTIVE_BROAD_COMPONENT_PARAMETER_AT_BOUND")
    if (
        component_present
        and width is not None
        and not (
            width >= YANG_BROAD_FWHM_RANGE_KMS[0]
            and (
                width <= YANG_BROAD_FWHM_RANGE_KMS[1]
                or np.isclose(
                    width,
                    YANG_BROAD_FWHM_RANGE_KMS[1],
                    rtol=_BOUND_RTOL,
                    atol=0.0,
                )
            )
        )
    ):
        reasons.append("BROAD_COMPONENT_WIDTH_OUTSIDE_YANG_SOURCE_RANGE")
    if not finite_measurement:
        reasons.append("BROAD_MEASUREMENT_NOT_FINITE_OR_WELL_DEFINED")
    fit_valid = not reasons
    valid_nondetection = bool(fit_valid and not component_present and broad_flux == 0)
    return diagnostics, YangHbetaFitAudit(
        fit_completed=True,
        optimizer_success=optimizer_success,
        parameter_at_bound=active_parameters_at_bound,
        local_hbeta_coverage=coverage,
        local_mask_fraction=local_mask_fraction,
        continuum_valid=continuum_valid,
        broad_component_present=component_present,
        broad_component_amplitude=amplitude,
        broad_component_width=width,
        broad_flux=broad_flux,
        residual_rms_local=residual_rms,
        finite_measurement=finite_measurement,
        fit_valid=fit_valid,
        invalid_reason="|".join(reasons),
        valid_nondetection=valid_nondetection,
    )
