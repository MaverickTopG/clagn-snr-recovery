# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/latent_ew.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 7095b21c33cf2e3f2d767c5ed520f886a4818c07df73c6eeb4bbb3d878692e86
# =====================================================================
"""Interval-censored *change* model for the broad-Hbeta EW change vs. the
mid-infrared state change.

We model the change dEW = EW2 - EW1 directly through per-source interval bounds
[L_i, U_i]; this is an interval-censored change model, not a joint two-epoch
latent-EW model (we do not carry and integrate out separate latent EW1, EW2). It
replaces the difference-level, one-sided Tobit of :mod:`clagn_apj.censored`, which
mis-modelled the censoring by permitting dEW > y2 (a negative early EW). Using
EW* >= 0, the bounds set by the two epochs' detection states are *intervals*:

* both detected           -> a measured point dEW = y2 - y1 (Gaussian error);
* turn-on  (ep.1 limit U1)-> EW*_1 in [0, U1] with y2 measured -> dEW in [y2-U1, y2];
* turn-off (ep.2 limit U2)-> EW*_2 in [0, U2] with y1 measured -> dEW in [-y1, U2-y1];
* double non-detection    -> EW*_1 in [0,U1], EW*_2 in [0,U2] -> dEW in [-U1, U2].

Nonnegativity of EW (EW* >= 0) is what makes turn-ons/turn-offs and even double
non-detections weakly informative rather than one-sided or vacuous. The model
regresses the standardized latent dEW on the standardized predictor with an
errors-in-variables term for the (noisy, selected-maximum) mid-infrared
predictor, and reports the standardized slope with a Hessian interval. All
sources are standardized on the *detected*-point scale so the slope is on a fixed
reference; sensitivity subsets are standardized with the primary-sample scale
supplied by the caller.

Three properties of the likelihood are worth stating explicitly:

*Intercept.* The mean is ``mu_i = alpha + beta z_i + gamma^T c_i``. The outcome is
centered on the *detected points*, so those have mean zero by construction, but the
censored intervals do not: their midpoints sit systematically below the detected
mean, and the turn-on/turn-off split is not balanced. Without a free ``alpha`` that
offset has nowhere to go but ``beta`` and ``gamma``, which biases the slope whenever
the censored/detected offset is at all correlated with the predictor.

*The censored contribution is a derived marginal, not an ansatz.* For a turn-on the
observables are the noisy epoch-2 measurement ``y2 = EW2 + e2`` (``e2 ~ N(0,
sigma_2^2)``) and the epoch-1 non-detection, approximated as ``EW1 in [0, U1]``.
Since ``d = EW2 - EW1``,

    d + e2 = y2 - EW1  in  [y2 - U1, y2]        (a FIXED interval),

and under the model ``d ~ N(mu, v)`` the shifted variable ``d + e2 ~ N(mu, v +
sigma_2^2)``. Marginalizing the endpoint noise therefore gives *exactly*

    P = Phi((y2 - mu)/s) - Phi((y2 - U1 - mu)/s),   s^2 = v + sigma_2^2,

with common variance on both tails and deterministic bounds (verified against
brute-force Monte Carlo to ~1e-5). Both bounds contain the same noisy ``y2``, so
their errors are perfectly correlated: the noise *shifts* the interval without
widening it, which is why no per-bound variance and no covariance term appears. The
turn-off case is symmetric with ``d - e1 in [-y1, U2 - y1]`` and ``s^2 = v +
sigma_1^2``. The detection limit ``U = 3 sigma`` is a deterministic threshold given
the (known) noise estimate, so no ``sigma_U`` term arises; an earlier revision of
this module added per-bound variances including ``sigma_U = U/3``, which the Monte
Carlo check shows is not the correct marginal. The remaining approximation is the
hard interval ``EW1 in [0, U1]`` for the non-detection event itself -- relaxing that
requires the soft per-epoch detection likelihood discussed in the paper.

*Degenerate intervals.* A non-detection whose reported limit is exactly zero (the
``snr <= 0`` branch of the spectral fitter) yields ``lo == hi``, hence ``P == 0``.
Clipping that to a floor makes it a parameter-free constant with zero gradient, so
such rows silently contribute nothing to the fit while still counting towards ``n``.
They are dropped instead, and the count is reported on the fit.

For the estimator comparison in the paper, ``support="one_sided"`` refits the
*identical* likelihood -- same intercept, scaling, EIV term, covariates, optimizer,
and the same rows -- with the single change that the physical bound of each censored
interval is removed (turn-on loses its upper bound ``y2``; turn-off loses its lower
bound ``-y1``), reproducing the superseded one-sided Tobit's support. On detected
pairs the two supports are the same code path and agree to numerical precision, so
any slope difference between them is attributable to the censored rows' support and
nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]
from scipy.stats import t as student_t  # type: ignore[import-untyped]


@dataclass(frozen=True)
class LatentFit:
    n: int
    n_censored: int
    standardized_slope: float
    slope_se: float
    ci_low: float
    ci_high: float
    intrinsic_scatter: float
    covariate_slopes: tuple[float, ...]
    intercept: float = 0.0
    n_dropped_degenerate: int = 0


def _log_subtract(log_large: np.ndarray, log_small: np.ndarray) -> np.ndarray:
    """Stable ``log(exp(log_large) - exp(log_small))`` for ordered inputs."""
    if np.any(log_small > log_large + 1e-12):
        raise ValueError("log-subtraction operands are out of order")
    delta = np.minimum(log_small - log_large, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return log_large + np.log(-np.expm1(delta))


def _log_interval_probability(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    distribution: str = "normal",
    df: float = 4.0,
) -> np.ndarray:
    """Stable log probability for a standardized interval.

    Direct CDF subtraction becomes exactly zero when both endpoints lie far into the
    same tail.  The likelihood previously floored that zero at ``1e-300``, creating a
    flat, parameter-free contribution.  Work in log space and use survival functions in
    the right tail so narrow extreme intervals retain their gradient.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("interval endpoints must have the same shape")
    if np.any(hi < lo):
        raise ValueError("interval upper endpoint is below lower endpoint")
    if distribution == "normal":
        logcdf = norm.logcdf
        logsf = norm.logsf
    elif distribution == "student_t":
        logcdf = lambda value: student_t.logcdf(value, df)  # noqa: E731
        logsf = lambda value: student_t.logsf(value, df)  # noqa: E731
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    out = np.full(lo.shape, -np.inf, dtype=float)
    full = np.isneginf(lo) & np.isposinf(hi)
    out[full] = 0.0
    left = np.isneginf(lo) & np.isfinite(hi)
    out[left] = logcdf(hi[left])
    right = np.isfinite(lo) & np.isposinf(hi)
    out[right] = logsf(lo[right])
    finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    right_tail = finite & (lo > 0)
    if np.any(right_tail):
        out[right_tail] = _log_subtract(
            logsf(lo[right_tail]), logsf(hi[right_tail])
        )
    other = finite & ~right_tail
    if np.any(other):
        out[other] = _log_subtract(logcdf(hi[other]), logcdf(lo[other]))
    return out


def build_latent_intervals(
    early_ew: np.ndarray,
    early_err: np.ndarray,
    early_detected: np.ndarray,
    early_limit: np.ndarray,
    late_ew: np.ndarray,
    late_err: np.ndarray,
    late_detected: np.ndarray,
    late_limit: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-source outcome bounds for dEW = late - early, in raw EW units.

    Returns arrays ``lo``, ``hi`` (interval bounds; equal for a measured point),
    ``is_point`` (both epochs detected), ``point``, ``point_err``, ``shift_err``
    (the measurement error of the detected endpoint shared by both bounds -- see
    the module docstring's derivation), ``kind`` (0 point, 1 turn-on, 2 turn-off,
    3 double non-detection), and ``degenerate`` (a censored row whose bounds
    coincide because its reported limit is zero).
    """
    early_ew = np.asarray(early_ew, dtype=float)
    early_err = np.asarray(early_err, dtype=float)
    early_limit = np.asarray(early_limit, dtype=float)
    late_ew = np.asarray(late_ew, dtype=float)
    late_err = np.asarray(late_err, dtype=float)
    late_limit = np.asarray(late_limit, dtype=float)
    e_det = np.asarray(early_detected).astype(bool)
    l_det = np.asarray(late_detected).astype(bool)
    n = len(e_det)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    point = np.full(n, np.nan)
    point_err = np.full(n, np.nan)
    # measurement error of the detected endpoint, which shifts BOTH bounds together
    # (the exact marginal has a common variance and fixed bounds -- module docstring)
    shift_err = np.zeros(n)
    kind = np.zeros(n, dtype=int)
    is_point = e_det & l_det

    # both detected: measured change
    point[is_point] = late_ew[is_point] - early_ew[is_point]
    point_err[is_point] = np.hypot(early_err[is_point], late_err[is_point])
    lo[is_point] = point[is_point]
    hi[is_point] = point[is_point]

    # turn-on: early is a limit, late measured -> d + e2 in [y2 - U1, y2] exactly
    ton = (~e_det) & l_det
    lo[ton] = late_ew[ton] - early_limit[ton]
    hi[ton] = late_ew[ton]
    shift_err[ton] = late_err[ton]
    kind[ton] = 1

    # turn-off: late is a limit, early measured -> d - e1 in [-y1, U2 - y1] exactly
    toff = e_det & (~l_det)
    lo[toff] = -early_ew[toff]
    hi[toff] = late_limit[toff] - early_ew[toff]
    shift_err[toff] = early_err[toff]
    kind[toff] = 2

    # double non-detection: both bounds are deterministic limits -> [-U1, U2]
    dnd = (~e_det) & (~l_det)
    lo[dnd] = -early_limit[dnd]
    hi[dnd] = late_limit[dnd]
    kind[dnd] = 3

    # a censored row whose bounds coincide carries no information and would otherwise
    # contribute a clipped, parameter-free constant to the log-likelihood
    degenerate = (~is_point) & np.isfinite(lo) & np.isfinite(hi) & (hi <= lo)

    return {"lo": lo, "hi": hi, "is_point": is_point, "point": point, "point_err": point_err,
            "shift_err": shift_err, "kind": kind, "degenerate": degenerate}


def fit_latent_ew_slope(
    predictor: np.ndarray,
    predictor_error: np.ndarray,
    intervals: dict[str, np.ndarray],
    covariates: np.ndarray | None = None,
    *,
    outcome_center: float | None = None,
    outcome_scale: float | None = None,
    predictor_center: float | None = None,
    predictor_scale: float | None = None,
    support: str = "interval",
    outcome_dist: str = "normal",
    t_df: float = 4.0,
    compute_se: bool = True,
) -> LatentFit:
    """Fit the standardized latent-dEW slope under interval censoring + EIV.

    ``intervals`` is the output of :func:`build_latent_intervals`. The outcome is
    standardized on the detected points, and the predictor on its own spread, unless
    the corresponding ``*_center``/``*_scale`` are supplied.

    Supply the primary sample's values whenever the result must be comparable across
    fits: the standardized slope scales as ``predictor_scale / outcome_scale``, so
    cohorts (or bootstrap replicates) that each re-derive their own scaling are not
    reporting the same quantity. Covariate scaling needs no such pinning -- a linear
    rescaling of a covariate is absorbed by its coefficient and the intercept.

    ``support="one_sided"`` removes each censored interval's physical bound (turn-on
    upper, turn-off lower; a double non-detection becomes vacuous), reproducing the
    superseded one-sided Tobit's support inside an otherwise identical fit. Detected
    points, degenerate-row handling, scaling, EIV, covariates and optimizer are all
    shared, so the difference between the two supports isolates exactly the
    censored-row support change.
    """
    if support not in {"interval", "one_sided"}:
        raise ValueError(f"unknown support {support!r}")
    if outcome_dist not in {"normal", "student_t"}:
        raise ValueError(f"unknown outcome_dist {outcome_dist!r}")
    x = np.asarray(predictor, dtype=float)
    sx = np.asarray(predictor_error, dtype=float)
    lo = np.asarray(intervals["lo"], dtype=float)
    hi = np.asarray(intervals["hi"], dtype=float)
    is_point = np.asarray(intervals["is_point"], dtype=bool)
    point = np.asarray(intervals["point"], dtype=float)
    point_err = np.asarray(intervals["point_err"], dtype=float)
    n_all = len(x)
    shift_err = np.asarray(intervals.get("shift_err", np.zeros(n_all)), dtype=float)
    kind = np.asarray(intervals.get("kind", np.zeros(n_all, dtype=int)), dtype=int)
    degenerate = np.asarray(intervals.get("degenerate", np.zeros(n_all, dtype=bool)), dtype=bool)

    # Drop zero-width censored intervals: they carry no information and, once the
    # interval probability is floored, contribute a constant with zero gradient.
    # Applied identically for both supports so matched fits share the same rows.
    keep = ~degenerate
    n_dropped = int(np.count_nonzero(degenerate))
    if n_dropped:
        x, sx = x[keep], sx[keep]
        lo, hi, is_point = lo[keep], hi[keep], is_point[keep]
        point, point_err = point[keep], point_err[keep]
        shift_err, kind = shift_err[keep], kind[keep]
        if covariates is not None:
            covariates = np.asarray(covariates, dtype=float)[keep]
    n = len(x)

    if support == "one_sided":
        # remove the physical bound only: turn-on loses its upper bound (y2),
        # turn-off its lower bound (-y1); a double non-detection becomes vacuous
        hi = np.where(kind == 1, np.inf, hi)
        lo = np.where(kind == 2, -np.inf, lo)
        hi = np.where(kind == 3, np.inf, hi)
        lo = np.where(kind == 3, -np.inf, lo)

    # standardize predictor; standardize outcome on the detected-point scale
    xm = predictor_center if predictor_center is not None else float(np.mean(x))
    xs = predictor_scale if predictor_scale is not None else float(np.std(x, ddof=0))
    if not np.isfinite(xs) or xs <= 0:
        raise ValueError("predictor scale must be finite and positive")
    zx = (x - xm) / xs
    sxs = sx / xs
    det = is_point & np.isfinite(point)
    yc = outcome_center if outcome_center is not None else float(np.mean(point[det]))
    ys = outcome_scale if outcome_scale is not None else float(np.std(point[det], ddof=0))
    if not np.isfinite(ys) or ys <= 0:
        raise ValueError("outcome scale undefined (need >1 detected point)")

    zlo = (lo - yc) / ys
    zhi = (hi - yc) / ys
    zpt = (point - yc) / ys
    spt = point_err / ys
    # the detected endpoint's noise shifts both bounds together, so it enters as a
    # common variance widening, not per-bound errors (module docstring derivation)
    s_shift = shift_err / ys

    if covariates is None:
        cov = np.empty((n, 0))
    else:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != n:
            raise ValueError("covariates must be an n-by-p matrix")
        center = cov.mean(axis=0)
        scale = cov.std(axis=0, ddof=0)
        if not np.isfinite(cov).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("covariates must be finite with positive spread")
        cov = (cov - center) / scale
    n_cov = cov.shape[1]

    def negloglik(params: np.ndarray) -> float:
        alpha = params[0]
        beta = params[1]
        gamma = params[2 : 2 + n_cov]
        log_sigma = params[2 + n_cov]
        sigma2 = np.exp(2 * log_sigma)
        mu = alpha + beta * zx + (cov @ gamma if n_cov else 0.0)
        # errors-in-variables: predictor noise inflates variance by beta^2 sx^2
        base_var = sigma2 + (beta**2) * sxs**2
        # detected points: Gaussian with measurement error
        v_pt = base_var + spt**2
        if outcome_dist == "student_t":
            # heavy-tailed outcome: `sigma` is the t SCALE, not the sd, so the fitted
            # value is not comparable to the Gaussian one -- the slope is
            sd_pt = np.sqrt(v_pt)
            ll_pt = student_t.logpdf((zpt - mu) / sd_pt, t_df) - np.log(sd_pt)
        else:
            ll_pt = norm.logpdf(zpt, loc=mu, scale=np.sqrt(v_pt))
        # interval-censored: the exact marginal over the shared endpoint noise is a
        # CDF difference with COMMON variance and fixed bounds (module docstring):
        #   P = Phi((hi-mu)/s) - Phi((lo-mu)/s),  s^2 = base_var + s_shift^2
        s = np.sqrt(base_var + s_shift**2)
        ll_int = _log_interval_probability(
            (zlo - mu) / s,
            (zhi - mu) / s,
            distribution=outcome_dist,
            df=t_df,
        )
        ll = np.where(is_point, ll_pt, ll_int)
        value = -float(np.sum(ll))
        # Numerical differentiation probes extreme trial parameters.  A mathematically
        # zero interval probability has infinite deviance; return a large finite barrier
        # so SciPy's finite differences do not produce inf-inf warnings or NaN gradients.
        return value if np.isfinite(value) else 1e100

    p0 = np.concatenate([[0.0, 0.0], np.zeros(n_cov), [0.0]])
    res = minimize(negloglik, p0, method="L-BFGS-B")
    if not res.success or not np.isfinite(res.fun) or not np.isfinite(res.x).all():
        raise RuntimeError(
            f"interval-likelihood optimization failed: status={res.status}; {res.message}"
        )
    alpha_hat = float(res.x[0])
    beta_hat = float(res.x[1])
    gamma_hat = tuple(float(v) for v in res.x[2 : 2 + n_cov])
    sigma_hat = float(np.exp(res.x[2 + n_cov]))

    # Hessian-based SE on beta via finite differences of the gradient (skippable: the
    # numerical Hessian dominates the fit cost, and point-slope-only callers -- e.g. the
    # partition diagnostic -- do not need it).
    if compute_se:
        se = _slope_se(negloglik, res.x, index=1)
        ci_low, ci_high = beta_hat - 1.96 * se, beta_hat + 1.96 * se
    else:
        se = ci_low = ci_high = float("nan")
    n_cens = int(np.sum(~is_point))
    return LatentFit(n, n_cens, beta_hat, se, ci_low, ci_high, sigma_hat, gamma_hat,
                     alpha_hat, n_dropped)


def _slope_se(fn: object, x: np.ndarray, eps: float = 1e-4, *, index: int = 0) -> float:
    """Standard error of parameter ``index`` from a numerical Hessian."""
    from collections.abc import Callable
    from typing import cast

    fn = cast(Callable[[np.ndarray], float], fn)
    m = len(x)
    hess = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            xi = x.copy()
            xi[i] += eps
            xi[j] += eps
            fpp = fn(xi)
            xi = x.copy()
            xi[i] += eps
            xi[j] -= eps
            fpm = fn(xi)
            xi = x.copy()
            xi[i] -= eps
            xi[j] += eps
            fmp = fn(xi)
            xi = x.copy()
            xi[i] -= eps
            xi[j] -= eps
            fmm = fn(xi)
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
    if not np.isfinite(hess).all():
        raise RuntimeError("likelihood Hessian contains non-finite values")
    try:
        np.linalg.cholesky(hess)
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("likelihood Hessian is not positive definite") from exc
    variance = float(cov[index, index])
    if not np.isfinite(variance) or variance <= 0:
        raise RuntimeError("slope variance from Hessian is not finite and positive")
    return float(np.sqrt(variance))


def latent_fit_dict(fit: LatentFit) -> dict[str, object]:
    return asdict(fit)


# ---------------------------------------------------------------------------
# Soft per-epoch detection likelihood (referee point #1)
#
# The interval model above treats a non-detection as a *hard* latent interval
# ``EW in [0, U]``.  But ``U = 3 sigma_EW`` is a detection threshold, not a bound on
# the latent EW, so a latent line can sit above ``U`` and still be missed.  The soft
# model instead carries a latent baseline ``EW1 >= 0`` with a prior, a latent change
# ``dEW | x ~ N(mu, v)``, a latent late EW ``EW2 = EW1 + dEW``, and an explicit
# per-epoch detection probability.  An epoch with true level ``L`` and known noise
# ``sigma`` is *reported detected* when its noisy measurement exceeds ``3 sigma``:
#
#     P(detected | L) = Phi(L/sigma - 3),   reported limit U = 3 sigma.
#
# A detected epoch contributes the Gaussian measurement density ``N(y | L, sigma^2)``;
# a non-detected epoch contributes ``Phi((3 sigma - L)/sigma)``.  The per-source
# marginal integrates over the latent EW1 and the latent change.  The change integral
# is done analytically for every case:
#
#   * detected late epoch:      int N(dEW|mu,v) N(y2 | EW1+dEW, s2^2) d(dEW)
#                               = N(y2 | EW1 + mu, v + s2^2)      (Gaussian convolution)
#   * non-detected late epoch:  int N(dEW|mu,v) Phi((U2 - EW1 - dEW)/s2) d(dEW)
#                               = Phi((U2 - EW1 - mu)/sqrt(v + s2^2))
#                               (the identity E[Phi(a - bZ)] = Phi(a/sqrt(1+b^2))).
#
# so only a single 1-D quadrature over the latent EW1 remains.  The predictor and
# covariates are standardized exactly as in ``fit_latent_ew_slope`` and the intrinsic
# scatter is on the same detected-point outcome scale, so the returned ``beta`` is
# directly comparable to the hard-interval standardized slope; the intercept absorbs
# the latent drift.  ``EW1`` is given the generator's fitted per-source lognormal prior
# (mean ``prior_log_loc``, sd ``prior_log_sd``) so no new latent-population machinery is
# introduced.


@dataclass(frozen=True)
class SoftDetectionFit:
    n: int
    n_censored: int
    standardized_slope: float
    slope_se: float
    ci_low: float
    ci_high: float
    intrinsic_scatter: float
    covariate_slopes: tuple[float, ...]
    intercept: float
    n_quadrature_nodes: int
    max_abs_log_likelihood: float
    mean_clamp_probability: float = 0.0
    max_clamp_probability: float = 0.0
    turnoff_mean_clamp_probability: float = 0.0
    physical_slope: float = 0.0


def _lognormal_logpdf(x: np.ndarray, log_loc: np.ndarray, log_sd: float) -> np.ndarray:
    """Log density of a lognormal at ``x > 0`` with log-location and log-sd."""
    log_x = np.log(x)
    return (
        -log_x
        - np.log(log_sd)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * ((log_x - log_loc) / log_sd) ** 2
    )


# Fixed Gauss--Legendre grid for the Drezner--Wesolowsky bivariate-normal CDF.
_BVN_NODES, _BVN_WEIGHTS = np.polynomial.legendre.leggauss(20)


def _bvn_cdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate-normal CDF ``P(X<=h, Y<=k)`` with correlation ``rho``.

    Drezner--Wesolowsky form: ``Phi(h)Phi(k)`` plus the integral of the bivariate density
    along the correlation path ``t in [0, rho]``.  The integrand is smooth for ``|rho|<1``,
    so a fixed Gauss--Legendre grid is accurate and free of the ``h=0``/``k=0`` singularities
    of the Owen's-T decomposition.  All inputs broadcast together.
    """
    h, k, rho = np.broadcast_arrays(
        np.asarray(h, dtype=float),
        np.asarray(k, dtype=float),
        np.asarray(rho, dtype=float),
    )
    base = norm.cdf(h) * norm.cdf(k)
    half = (rho / 2.0)[..., None]
    t = half * (_BVN_NODES + 1.0)  # map [-1,1] -> [0, rho]
    weights = _BVN_WEIGHTS * half  # includes the interval Jacobian
    one_minus = 1.0 - t**2
    hh = h[..., None]
    kk = k[..., None]
    integrand = np.exp(
        -(hh**2 - 2.0 * t * hh * kk + kk**2) / (2.0 * one_minus)
    ) / np.sqrt(one_minus)
    return base + (weights * integrand).sum(axis=-1) / (2.0 * np.pi)


def _soft_source_loglik(
    params: np.ndarray,
    *,
    zx: np.ndarray,
    sxs: np.ndarray,
    cov: np.ndarray,
    ys: float,
    ew1: np.ndarray,
    sig1: np.ndarray,
    det1: np.ndarray,
    lim1: np.ndarray,
    ew2: np.ndarray,
    sig2: np.ndarray,
    det2: np.ndarray,
    lim2: np.ndarray,
    prior_log_loc: np.ndarray,
    prior_log_sd: float,
    nodes: np.ndarray,
    log_weights: np.ndarray,
    enforce_nonnegativity: bool = True,
) -> np.ndarray:
    """Per-source marginal log-likelihood vector under the soft-detection model.

    ``nodes`` and ``log_weights`` are the per-source ``(n, k)`` latent-EW1 quadrature
    grid and log quadrature weights (Gauss--Legendre weight times interval Jacobian).
    All EW quantities (``ew*``, ``sig*``, ``lim*``, ``nodes``) are in raw angstrom.

    With ``enforce_nonnegativity`` (the default) the latent late EW is clamped,
    ``EW2 = max(EW1 + d, 0)``.  Setting it ``False`` removes that endpoint and uses the
    half-line marginals instead (a detected late epoch is the plain change+measurement
    Gaussian; a turn-off is ``P(EW2 < U2)`` with ``EW2`` unbounded below).  The two are
    otherwise identical, so a matched pair of fits is the soft-detection analog of the
    hard bounded-versus-one-sided support comparison.
    """
    n_cov = cov.shape[1]
    alpha = params[0]
    beta = params[1]
    gamma = params[2 : 2 + n_cov]
    log_sigma = params[2 + n_cov]
    sigma2 = np.exp(2.0 * log_sigma)
    mu_std = alpha + beta * zx + (cov @ gamma if n_cov else 0.0)
    mu_raw = ys * mu_std
    v_raw = (ys**2) * (sigma2 + (beta**2) * sxs**2)
    s2_det = np.sqrt(v_raw + sig2**2)

    node = nodes  # (n, k) latent EW1
    col_loc = prior_log_loc[:, None]
    log_pi = _lognormal_logpdf(node, col_loc, prior_log_sd)

    # early-epoch contribution g1(EW1)
    det1_col = det1[:, None]
    log_g1 = np.where(
        det1_col,
        norm.logpdf(ew1[:, None], loc=node, scale=sig1[:, None]),
        norm.logcdf((lim1[:, None] - node) / sig1[:, None]),
    )
    # late-epoch contribution h2(EW1), change integrated out analytically with the latent
    # late EW held nonnegative: EW2 = max(EW1 + d, 0).  ``a`` is the mean of the unclamped
    # late EW at this early-EW node.
    det2_col = det2[:, None]
    a = node + mu_raw[:, None]
    sd_change = np.sqrt(v_raw)[:, None]
    s2 = s2_det[:, None]

    k_arg = (lim2[:, None] - a) / s2
    if enforce_nonnegativity:
        # Late detected: the clamped latent late EW = max(EW1+d, 0) makes the detected-epoch
        # contribution the sum of (i) the positive-EW2 integral -- the product of the change
        # and measurement Gaussians factored as N(y2 | a, v+sig2^2) times the mass of the
        # combined EW2 Gaussian above zero -- and (ii) the clamped-at-zero atom P(EW1+d<=0)
        # N(y2 | 0, sig2^2), a latent line at zero detected through a positive noise excursion
        # (negligible for y2 >> 3 sig2, but retained for a consistent max(.,0) marginal).
        sig2_col = sig2[:, None]
        sig2_sq = sig2_col**2
        inv_var = 1.0 / v_raw[:, None] + 1.0 / sig2_sq
        m_star = (a / v_raw[:, None] + ew2[:, None] / sig2_sq) / inv_var
        s_star = np.sqrt(1.0 / inv_var)
        log_h2_det_pos = norm.logpdf(ew2[:, None], loc=a, scale=s2) + norm.logcdf(
            m_star / s_star
        )
        log_h2_det_atom = norm.logcdf(-a / sd_change) + norm.logpdf(
            ew2[:, None], loc=0.0, scale=sig2_col
        )
        log_h2_det = np.logaddexp(log_h2_det_pos, log_h2_det_atom)

        # Late nondetected (turn-off): the clamped latent EW2 < 0 region contributes the
        # physical Phi(3) floor, not a probability approaching one.  With the bivariate-normal
        # term for the EW2 >= 0 part:
        #   h2 = Phi(3) Phi(-a/sqrt v) + Phi((U2-a)/s2) - Phi2(-a/sqrt v, (U2-a)/s2; rho).
        h_arg = -a / sd_change
        rho = (np.sqrt(v_raw) / s2_det)[:, None]
        bvn = np.zeros_like(a)
        nondet = ~det2
        if np.any(nondet):
            bvn[nondet] = _bvn_cdf(
                h_arg[nondet],
                k_arg[nondet],
                np.broadcast_to(rho, a.shape)[nondet],
            )
        phi3 = norm.cdf(3.0)
        term_nondet = phi3 * norm.cdf(h_arg) + norm.cdf(k_arg) - bvn
        log_h2_nondet = np.log(np.clip(term_nondet, 1e-300, None))
    else:
        # Nonnegativity removed: the half-line marginal that treats the derived change
        # support as one-sided.  A detected late epoch is the plain change+measurement
        # Gaussian N(y2 | a, v+sig2^2); a turn-off is P(EW2 < U2) = Phi((U2-a)/s2) with
        # EW2 = EW1 + d unbounded below.  This is the soft parallel of the hard one-sided
        # (limit-only) support: the endpoint that nonnegativity supplies is discarded.
        log_h2_det = norm.logpdf(ew2[:, None], loc=a, scale=s2)
        log_h2_nondet = norm.logcdf(k_arg)

    log_h2 = np.where(det2_col, log_h2_det, log_h2_nondet)
    log_integrand = log_pi + log_g1 + log_h2 + log_weights
    return logsumexp(log_integrand, axis=1)


def fit_soft_detection_slope(
    predictor: np.ndarray,
    predictor_error: np.ndarray,
    *,
    early_ew: np.ndarray,
    early_err: np.ndarray,
    early_detected: np.ndarray,
    early_limit: np.ndarray,
    late_ew: np.ndarray,
    late_err: np.ndarray,
    late_detected: np.ndarray,
    late_limit: np.ndarray,
    prior_log_loc: np.ndarray,
    prior_log_sd: float,
    outcome_scale: float,
    predictor_center: float,
    predictor_scale: float,
    covariates: np.ndarray | None = None,
    threshold: float = 3.0,
    n_nodes: int = 512,
    window_buffer: float = 400.0,
    enforce_nonnegativity: bool = True,
) -> SoftDetectionFit:
    """Fit the standardized slope under the soft per-epoch detection likelihood.

    Parameters mirror :func:`fit_latent_ew_slope`; the standardized ``beta`` is
    comparable to the hard-interval fit because the predictor is standardized with the
    supplied center/scale and the intrinsic scatter is on the ``outcome_scale`` (raw
    angstrom per unit) detected-point scale.  ``prior_log_loc`` is the per-source
    lognormal log-location for the latent EW1 prior; ``threshold`` is the detection
    S/N (``U = threshold * sigma``).  Non-detected epochs take ``sigma = limit /
    threshold``.  The latent-change integral is analytic, leaving one Gauss--Legendre
    quadrature over EW1 on a per-source, parameter-independent window.
    """
    x = np.asarray(predictor, dtype=float)
    sx = np.asarray(predictor_error, dtype=float)
    ew1 = np.asarray(early_ew, dtype=float)
    ew2 = np.asarray(late_ew, dtype=float)
    det1 = np.asarray(early_detected).astype(bool)
    det2 = np.asarray(late_detected).astype(bool)
    lim1 = np.asarray(early_limit, dtype=float)
    lim2 = np.asarray(late_limit, dtype=float)
    prior_log_loc = np.asarray(prior_log_loc, dtype=float)
    n = len(x)

    # Per-epoch known noise: the detected error, or limit/threshold for a non-detection.
    sig1 = np.where(det1, np.asarray(early_err, dtype=float), lim1 / threshold)
    sig2 = np.where(det2, np.asarray(late_err, dtype=float), lim2 / threshold)
    if not (np.all(sig1 > 0) and np.all(sig2 > 0)):
        raise ValueError("every epoch needs a positive noise estimate")

    zx = (x - predictor_center) / predictor_scale
    sxs = sx / predictor_scale
    ys = float(outcome_scale)

    if covariates is None:
        cov = np.empty((n, 0))
    else:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != n:
            raise ValueError("covariates must be an n-by-p matrix")
        center = cov.mean(axis=0)
        scale = cov.std(axis=0, ddof=0)
        if not np.isfinite(cov).all() or np.any(scale <= 0):
            raise ValueError("covariates must be finite with positive spread")
        cov = (cov - center) / scale
    n_cov = cov.shape[1]

    # Parameter-independent per-source EW1 quadrature window in raw angstrom.  The upper
    # edge covers the prior tail, both epochs' detected values and limits, and a fixed
    # buffer wider than a few change standard deviations so a turn-on's EW1 ~ y2 - dEW
    # stays inside the window across the optimizer's slope range.
    prior_hi = np.exp(prior_log_loc + 4.0 * prior_log_sd)
    data_hi = np.maximum.reduce([
        np.where(det1, ew1, 0.0),
        np.where(det2, ew2, 0.0),
        lim1,
        lim2,
    ])
    hi = np.maximum(prior_hi, data_hi) + window_buffer
    lo = np.full(n, 1e-2)

    std_nodes, std_weights = np.polynomial.legendre.leggauss(n_nodes)
    half = (hi - lo) / 2.0
    mid = (hi + lo) / 2.0
    nodes = mid[:, None] + half[:, None] * std_nodes[None, :]
    log_weights = np.log(std_weights)[None, :] + np.log(half)[:, None]

    shared = {
        "zx": zx, "sxs": sxs, "cov": cov, "ys": ys,
        "ew1": ew1, "sig1": sig1, "det1": det1, "lim1": lim1,
        "ew2": ew2, "sig2": sig2, "det2": det2, "lim2": lim2,
        "prior_log_loc": prior_log_loc, "prior_log_sd": float(prior_log_sd),
        "nodes": nodes, "log_weights": log_weights,
        "enforce_nonnegativity": bool(enforce_nonnegativity),
    }

    def negloglik(params: np.ndarray) -> float:
        ll = _soft_source_loglik(params, **shared)
        value = -float(np.sum(ll))
        return value if np.isfinite(value) else 1e100

    p0 = np.concatenate([[0.0, 0.0], np.zeros(n_cov), [0.0]])
    res = minimize(negloglik, p0, method="L-BFGS-B")
    if not res.success or not np.isfinite(res.fun) or not np.isfinite(res.x).all():
        raise RuntimeError(
            f"soft-detection optimization failed: status={res.status}; {res.message}"
        )
    beta_hat = float(res.x[1])
    se = _slope_se(negloglik, res.x, index=1)
    per_source = _soft_source_loglik(res.x, **shared)
    n_cens = int(np.sum(~(det1 & det2)))

    # Fitted clamping probability P(EW1 + d < 0): the prior-weighted fraction of latent late
    # EW that the max(.,0) floor is active for.  Small -> the latent-step slope is numerically
    # the physical-change slope.  Expectation over the EW1 prior on the quadrature window.
    mu_std_hat = res.x[0] + res.x[1] * zx + (cov @ res.x[2 : 2 + n_cov] if n_cov else 0.0)
    mu_raw_hat = ys * mu_std_hat
    v_raw_hat = (ys**2) * (np.exp(2.0 * res.x[2 + n_cov]) + (res.x[1] ** 2) * sxs**2)
    prior_w = np.exp(log_weights + _lognormal_logpdf(nodes, prior_log_loc[:, None],
                                                     float(prior_log_sd)))
    clamp_arg = -(nodes + mu_raw_hat[:, None]) / np.sqrt(v_raw_hat)[:, None]
    p_clamp = (prior_w * norm.cdf(clamp_arg)).sum(1) / prior_w.sum(1)
    turnoff = ~det2
    turnoff_clamp = float(p_clamp[turnoff].mean()) if np.any(turnoff) else 0.0

    # Implied physical-change slope: the association of the realized nonnegative change
    # E[max(EW1+D,0) - EW1] with the predictor, on the same standardized scale as the
    # primary (physical-change) slope -- directly comparable to the hard-interval beta,
    # unlike the pre-clamp latent-step beta_hat.  E[max(Z,0)] = a Phi(a/s) + s phi(a/s)
    # for Z ~ N(a, V), a = EW1 + M, s = sqrt(V); averaged over the EW1 prior.
    a_grid = nodes + mu_raw_hat[:, None]
    sd_change = np.sqrt(v_raw_hat)[:, None]
    z = a_grid / sd_change
    e_max = a_grid * norm.cdf(z) + sd_change * norm.pdf(z)  # E[max(EW1+D,0) | EW1]
    exp_phys = ((prior_w * (e_max - nodes)).sum(1) / prior_w.sum(1)) / ys  # standardized
    # FWL: coefficient of the standardized predictor after projecting out intercept+covariates
    design_cov = np.column_stack([np.ones(n), cov]) if n_cov else np.ones((n, 1))
    proj, *_ = np.linalg.lstsq(design_cov, np.column_stack([zx, exp_phys]), rcond=None)
    zx_res = zx - design_cov @ proj[:, 0]
    y_res = exp_phys - design_cov @ proj[:, 1]
    denom = float(zx_res @ zx_res)
    physical_slope = float(zx_res @ y_res / denom) if denom > 0 else float("nan")

    return SoftDetectionFit(
        n=n,
        n_censored=n_cens,
        standardized_slope=beta_hat,
        slope_se=se,
        ci_low=beta_hat - 1.96 * se,
        ci_high=beta_hat + 1.96 * se,
        intrinsic_scatter=float(np.exp(res.x[2 + n_cov])),
        covariate_slopes=tuple(float(v) for v in res.x[2 : 2 + n_cov]),
        intercept=float(res.x[0]),
        n_quadrature_nodes=n_nodes,
        max_abs_log_likelihood=float(np.max(np.abs(per_source))),
        mean_clamp_probability=float(p_clamp.mean()),
        max_clamp_probability=float(p_clamp.max()),
        turnoff_mean_clamp_probability=turnoff_clamp,
        physical_slope=physical_slope,
    )
