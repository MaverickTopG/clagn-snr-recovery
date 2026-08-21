# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/censored.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 7c732c919e285de69dd737dfbef8e7d68871f50c54bfa70e7889bf18be69724c
# =====================================================================
"""Censored errors-in-variables backend (maximum likelihood).

The confirmatory estimand is the standardized slope of the primary outcome (a
change in host-subtracted broad Hβ) on the primary predictor (a signed joint
W1/W2 mid-infrared state change), controlling for covariates, with faint-epoch
broad Hβ retained as left-censored measurements and with measurement error on
the predictor.

Because the structural model is linear-Gaussian, the latent true predictor can
be marginalized analytically: each object's observed (x, y) is bivariate normal
with cov(x, y) = β·τ². Detected outcomes contribute the bivariate density;
left-censored outcomes contribute φ(x)·Φ((L − E[y|x])/sd). The negative
log-likelihood is minimized with scipy, and the slope's standard error comes
from a numeric observed-information Hessian at the optimum. This corrects the
attenuation that ordinary least squares suffers under predictor measurement
error, needs no sampling, and is validated on synthetic recovery.

This is the ``censored_measurement_error`` backend the freeze requires. A PyMC
hierarchical sampler can later replace the engine behind the same interface and
the same standardized estimand and decision rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats  # type: ignore[import-untyped]

_CI_Z = 1.959963984540054  # two-sided 95%

# Column convention for the assembled confirmatory table.
_PREDICTOR = "signed_log_flux_change"
_OUTCOME = "broad_line_change"
_PREDICTOR_ERROR = "signed_log_flux_change_error"
_OUTCOME_ERROR = "broad_line_change_error"
_CENSORING = "censoring"


@dataclass(frozen=True)
class CensoredFit:
    n: int
    n_censored: int
    standardized_slope: float
    slope_se: float
    ci_low: float
    ci_high: float
    passed: bool


def _standardize(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (values - center) / scale


def _negative_log_likelihood(
    params: np.ndarray,
    *,
    zx: np.ndarray,
    zy: np.ndarray,
    sx2: np.ndarray,
    sy2: np.ndarray,
    covariates: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    n_cov = covariates.shape[1]
    mu_x = params[0]
    tau2 = np.exp(params[1]) ** 2
    alpha = params[2]
    beta = params[3]
    gamma = params[4 : 4 + n_cov]
    sigma2 = np.exp(params[4 + n_cov]) ** 2

    contribution = covariates @ gamma if n_cov else np.zeros_like(zx)
    mean_y = alpha + beta * mu_x + contribution
    var_x = tau2 + sx2
    var_y = beta**2 * tau2 + sigma2 + sy2
    cov = beta * tau2

    detected = ~(left | right)
    total = 0.0

    if np.any(detected):
        dx = zx[detected] - mu_x
        dy = zy[detected] - mean_y[detected]
        vx, vy = var_x[detected], var_y[detected]
        det = vx * vy - cov**2
        if np.any(det <= 0):
            return 1e12
        quad = (vy * dx**2 - 2 * cov * dx * dy + vx * dy**2) / det
        total += float(np.sum(-np.log(2 * np.pi) - 0.5 * np.log(det) - 0.5 * quad))

    # Censored contributions: marginal in x, tail in y | x. Left-censored (y <=
    # limit, e.g. turn-offs) use the lower tail; right-censored (y >= limit,
    # e.g. turn-ons) use the upper tail. Both directions are retained so
    # appearance events are not silently discarded.
    for mask, upper_tail in ((left, False), (right, True)):
        if not np.any(mask):
            continue
        dx = zx[mask] - mu_x
        vx, vy = var_x[mask], var_y[mask]
        if np.any(vx <= 0):
            return 1e12
        log_phi_x = stats.norm.logpdf(zx[mask], loc=mu_x, scale=np.sqrt(vx))
        cond_mean = mean_y[mask] + (cov / vx) * dx
        cond_var = vy - cov**2 / vx
        if np.any(cond_var <= 0):
            return 1e12
        cond_sd = np.sqrt(cond_var)
        tail = (
            stats.norm.logsf(zy[mask], loc=cond_mean, scale=cond_sd)
            if upper_tail
            else stats.norm.logcdf(zy[mask], loc=cond_mean, scale=cond_sd)
        )
        total += float(np.sum(log_phi_x + tail))

    if not np.isfinite(total):
        return 1e12
    return -total


def _numeric_hessian(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-4
) -> np.ndarray:
    dim = len(point)
    hessian = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            base_i = np.zeros(dim)
            base_i[i] = step
            base_j = np.zeros(dim)
            base_j[j] = step
            plus = function(point + base_i + base_j)
            plus_minus = function(point + base_i - base_j)
            minus_plus = function(point - base_i + base_j)
            minus = function(point - base_i - base_j)
            value = (plus - plus_minus - minus_plus + minus) / (4 * step**2)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def fit_censored_slope(
    frame: pd.DataFrame,
    *,
    outcome: str,
    predictor: str,
    covariates: tuple[str, ...],
    outcome_error: str,
    predictor_error: str,
    censoring: str,
    smallest_effect: float,
) -> CensoredFit:
    """Fit the standardized MIR slope under two-sided censoring and predictor error.

    ``outcome`` holds the measured value for detected rows and the limit for
    censored rows. ``censoring`` is a per-row label in {"none", "left", "right"}:
    "left" = the outcome is an upper limit (y <= limit, e.g. turn-offs), "right"
    = a lower limit (y >= limit, e.g. turn-ons). Everything is standardized to
    the detected outcome's scale so the fitted slope is on the frozen scale.
    """
    required = {outcome, predictor, outcome_error, predictor_error, censoring, *covariates}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"censored fit missing columns: {sorted(missing)}")

    labels = frame[censoring].astype(str).to_numpy()
    allowed = {"none", "left", "right"}
    if not set(labels) <= allowed:
        raise ValueError(f"censoring must be one of {sorted(allowed)}; got {sorted(set(labels))}")
    left = labels == "left"
    right = labels == "right"
    is_censored = left | right
    x = frame[predictor].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    sx = frame[predictor_error].to_numpy(dtype=float)
    sy = frame[outcome_error].to_numpy(dtype=float)
    n = len(frame)
    if n <= len(covariates) + 5:
        raise ValueError("insufficient rows for a censored fit")

    x_center, x_scale = float(np.mean(x)), float(np.std(x, ddof=0))
    detected = ~is_censored
    y_detected = y[detected]
    y_center = float(np.mean(y_detected)) if len(y_detected) else float(np.mean(y))
    y_scale = float(np.std(y_detected, ddof=0)) if len(y_detected) > 1 else float(np.std(y, ddof=0))
    if x_scale == 0 or y_scale == 0:
        raise ValueError("predictor or outcome has zero spread")

    zx = _standardize(x, x_center, x_scale)
    zy = _standardize(y, y_center, y_scale)
    sx2 = (sx / x_scale) ** 2
    sy2 = (sy / y_scale) ** 2
    if covariates:
        raw = frame.loc[:, list(covariates)].to_numpy(dtype=float)
        cov_matrix = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    else:
        cov_matrix = np.empty((n, 0))

    def objective(params: np.ndarray) -> float:
        return _negative_log_likelihood(
            params,
            zx=zx,
            zy=zy,
            sx2=sx2,
            sy2=sy2,
            covariates=cov_matrix,
            left=left,
            right=right,
        )

    mean_measurement_variance = float(np.mean(sx2))
    tau0 = np.sqrt(max(0.1, 1.0 - mean_measurement_variance))
    start = np.array(
        [0.0, np.log(tau0), 0.0, 0.2, *([0.0] * len(covariates)), np.log(0.8)]
    )
    result = optimize.minimize(objective, start, method="BFGS")
    estimate = result.x
    n_cov = len(covariates)

    def standardized_slope_of(params: np.ndarray) -> float:
        beta = params[3]
        tau2 = np.exp(params[1]) ** 2
        sigma2 = np.exp(params[4 + n_cov]) ** 2
        # Standardized slope on the disattenuated (true) predictor, in units of
        # the intrinsic outcome scatter. Scale-invariant in the y pre-scaling,
        # so truncation of the detected sample cannot bias it.
        return float(beta * np.sqrt(tau2) / np.sqrt(beta**2 * tau2 + sigma2))

    standardized_slope = standardized_slope_of(estimate)

    # numerical delta-method gradient of the standardized slope wrt parameters
    gradient = np.zeros(len(estimate))
    step = 1e-5
    for index in range(len(estimate)):
        shift = np.zeros(len(estimate))
        shift[index] = step
        gradient[index] = (
            standardized_slope_of(estimate + shift) - standardized_slope_of(estimate - shift)
        ) / (2 * step)

    hessian = _numeric_hessian(objective, estimate)
    try:
        covariance = np.linalg.inv(hessian)
        slope_variance = float(gradient @ covariance @ gradient)
        slope_se = float(np.sqrt(slope_variance)) if slope_variance > 0 else float("nan")
    except np.linalg.LinAlgError:
        slope_se = float("nan")

    ci_low = standardized_slope - _CI_Z * slope_se
    ci_high = standardized_slope + _CI_Z * slope_se
    passed = bool(
        np.isfinite(slope_se) and standardized_slope >= smallest_effect and ci_low > 0.0
    )

    return CensoredFit(
        n=n,
        n_censored=int(np.sum(is_censored)),
        standardized_slope=standardized_slope,
        slope_se=slope_se,
        ci_low=ci_low,
        ci_high=ci_high,
        passed=passed,
    )


def run_confirmatory_fit(
    frame: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
    smallest_effect: float,
) -> CensoredFit:
    """Fit the frozen confirmatory model on an assembled table.

    Uses the standard assembled-table column convention so the freeze/confirm
    CLI path stays a thin wrapper: ``signed_log_flux_change`` (+ ``_error``),
    ``broad_line_change`` (+ ``_error``), and a boolean ``censored`` column.
    """
    return fit_censored_slope(
        frame,
        outcome=_OUTCOME,
        predictor=_PREDICTOR,
        covariates=covariates,
        outcome_error=_OUTCOME_ERROR,
        predictor_error=_PREDICTOR_ERROR,
        censoring=_CENSORING,
        smallest_effect=smallest_effect,
    )


def censored_fit_dict(fit: CensoredFit) -> dict[str, int | float | bool]:
    return asdict(fit)
