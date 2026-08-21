"""Independent stellar-host decomposition with pPXF.

**Objective, deliberately narrow.** Recover an independently modelled
stellar-only spectral component well enough to validate the shape and host
fraction used for counterfactual host-dilution experiments. This is *not* a
stellar-population science pipeline: stellar mass, star-formation history,
metallicity and velocity dispersion are not the goal, and chasing them would
turn a validation into a side project.

**Independence is the whole point.** No PyQSOFit decomposition output — host
spectrum, host coefficients, host fraction — and no PyQSOFit fitted line
parameters enter here. Only the observed spectrum and survey metadata are
shared, because otherwise the comparison would measure preprocessing
differences rather than decomposition differences.

One shared *empirical template* is used: the optical Fe II table distributed
with PyQSOFit. That is published template data, not a fitted result of the
other method, so it does not compromise independence — but it is recorded
explicitly rather than left implicit.

**Why stars alone will not do.** Given only stellar SSPs, pPXF will explain AGN
power-law continuum, Fe II pseudocontinuum and broad Balmer wings with stellar
populations, producing a badly contaminated "independent" host. The template
set therefore carries AGN nuisance components so the fit has somewhere else to
put that flux. The nuisance components are machinery, not measurements: the
broad-line amplitudes recovered here are not claimed as astrophysical results.

**Polynomials.** Additive polynomials are disabled by default. An additive term
absorbs physical continuum flux that this experiment is trying to attribute
between AGN and host, which would make ``f_host`` a model convention rather
than a measurement. A low-order multiplicative polynomial is optional and its
effect is a basis systematic to be tested, never tuned per object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

C_KMS = 299792.458

#: Rest-frame lines given nuisance components so stellar templates are not
#: forced to absorb emission. Amplitudes here are machinery, not measurements.
NARROW_LINES = {
    "Hbeta_n": 4861.33, "OIII4959": 4958.92, "OIII5007": 5006.84,
    "Halpha_n": 6562.80, "NII6548": 6548.03, "NII6583": 6583.41,
    "SII6716": 6716.47, "SII6731": 6730.85, "OII3727": 3727.09,
}
BROAD_LINES = {"Hbeta_b": 4861.33, "Halpha_b": 6562.80, "Hgamma_b": 4340.47}

#: Featureless AGN continuum slopes, spanning the range used in current AGN/host
#: decomposition work. Overridable so a single-slope diagnostic can be run.
DEFAULT_POWERLAW_SLOPES: tuple[float, ...] = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0)


@dataclass
class PpxfHostResult:
    """Stellar-only model and its host fraction, on the fitted grid."""

    status: str
    wavelength: np.ndarray | None = None          # rest-frame
    stellar: np.ndarray | None = None             # stellar-only model
    nonstellar: np.ndarray | None = None          # AGN + FeII + lines
    observed: np.ndarray | None = None
    f_host_5100: float | None = None
    chi2: float | None = None
    n_stellar_templates: int = 0
    nonstellar_names: list[str] = field(default_factory=list)
    fit: object | None = None          # the raw pPXF solution, for component extraction
    sps_library: str = ""
    detail: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.stellar is not None


def _gaussian(lam: np.ndarray, centre: float, fwhm_kms: float) -> np.ndarray:
    sigma = centre * (fwhm_kms / C_KMS) / 2.3548200450309493
    profile = np.exp(-0.5 * ((lam - centre) / sigma) ** 2)
    peak = profile.max()
    return profile / peak if peak > 0 else profile


def balmer_pseudocontinuum(lam: np.ndarray, fwhm_kms: float = 5000.0) -> np.ndarray:
    """High-order Balmer blend, the part of the Balmer pseudocontinuum in range.

    The Balmer continuum edge sits at 3646 A, below the fitted window, so what
    survives inside 3700-7000 A is the blend of high-order Balmer lines piling
    up toward the limit. Approximated as a broadened sum over n = 10..60 with
    amplitudes falling as n^-3, which reproduces the smooth rise blueward of
    ~4000 A without pretending to be a physical recombination calculation.
    """
    rydberg = 3645.07
    profile = np.zeros_like(lam, dtype=float)
    for n in range(10, 60):
        centre = rydberg / (1.0 - 4.0 / n**2)
        if not (lam.min() < centre < lam.max()):
            continue
        sigma = centre * (fwhm_kms / C_KMS) / 2.3548200450309493
        profile += (n ** -3.0) * np.exp(-0.5 * ((lam - centre) / sigma) ** 2)
    peak = profile.max()
    return profile / peak if peak > 0 else profile


def build_nonstellar_templates(
    lam: np.ndarray,
    *,
    feii_path: Path | None = None,
    powerlaw_slopes: tuple[float, ...] = DEFAULT_POWERLAW_SLOPES,
    narrow_fwhm_kms: float = 400.0,
    broad_fwhm_kms: tuple[float, ...] = (2000.0, 5000.0, 10000.0),
    include_powerlaw: bool = True,
    include_feii: bool = True,
    include_lines: bool = True,
    include_balmer: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """AGN nuisance family: power laws, Fe II, narrow and broad lines.

    The power-law grid spans the range used in current AGN/host decomposition
    work. Its purpose is to give featureless blue continuum somewhere to go
    other than young stellar populations — a known degeneracy that bites
    hardest in exactly the AGN-dominated spectra where PyQSOFit declined.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    pivot = 5100.0

    if include_powerlaw:
        for slope in powerlaw_slopes:
            template = (lam / pivot) ** slope
            columns.append(template / np.median(template))
            names.append(f"powerlaw_{slope:+.1f}")

    if include_balmer:
        # Peak-normalised, like the line templates. A median over nonzero
        # entries is wrong for a profile with Gaussian tails: those tails give a
        # median of ~1e-67, and dividing by it produces a column of ~1e66 that
        # destroys the design matrix conditioning.
        balmer = balmer_pseudocontinuum(lam)
        if np.any(balmer > 0):
            columns.append(balmer)          # already peak-normalised to 1
            names.append("balmer_pseudo")

    if include_feii and feii_path is not None and feii_path.exists():
        table = np.genfromtxt(feii_path)
        fe_lam, fe_flux = table[:, 0], table[:, 1]
        for fwhm in (2000.0, 5000.0):
            sigma_pix = (fwhm / C_KMS) * 5100.0 / np.median(np.diff(lam))
            kernel_x = np.arange(-4 * sigma_pix, 4 * sigma_pix + 1)
            kernel = np.exp(-0.5 * (kernel_x / max(sigma_pix, 1e-3)) ** 2)
            kernel /= kernel.sum()
            resampled = np.interp(lam, fe_lam, fe_flux, left=0.0, right=0.0)
            smoothed = np.convolve(resampled, kernel, mode="same")
            scale = np.median(np.abs(smoothed))
            if scale > 0:
                columns.append(smoothed / scale)
                names.append(f"feii_{int(fwhm)}")

    if include_lines:
        for name, centre in NARROW_LINES.items():
            if lam.min() < centre < lam.max():
                columns.append(_gaussian(lam, centre, narrow_fwhm_kms))
                names.append(name)

        for name, centre in BROAD_LINES.items():
            if lam.min() < centre < lam.max():
                for fwhm in broad_fwhm_kms:
                    columns.append(_gaussian(lam, centre, fwhm))
                    names.append(f"{name}_{int(fwhm)}")

    if not columns:
        return np.zeros((lam.size, 0)), []
    return np.column_stack(columns), names


def f_host_from_components(
    lam_rest: np.ndarray,
    stellar: np.ndarray,
    agn_continuum: np.ndarray,
    window: tuple[float, float],
) -> float | None:
    """Host fraction over the same frozen window used everywhere else.

    Only *continuum* components enter the ratio; emission lines are nuclear but
    are not continuum, and including them would make f_host depend on line
    strength rather than on the continuum decomposition.
    """
    inside = (lam_rest >= window[0]) & (lam_rest <= window[1])
    if inside.sum() < 5:
        return None
    host_level = float(np.median(stellar[inside]))
    agn_level = float(np.median(agn_continuum[inside]))
    total = host_level + agn_level
    return float(host_level / total) if total > 0 else None


def decompose(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    redshift: float,
    *,
    sps_file: Path,
    fhost_window: tuple[float, float],
    fwhm_gal: float | dict | None = None,
    feii_path: Path | None = None,
    fit_range: tuple[float, float] = (3700.0, 7000.0),
    degree: int = -1,          # additive polynomial OFF by default
    mdegree: int = 0,          # multiplicative polynomial OFF by default
    age_range: tuple[float, float] | None = None,
    include_powerlaw: bool = True,
    include_feii: bool = True,
    include_lines: bool = True,
    include_balmer: bool = False,
    mask_emission: bool = False,
    powerlaw_slopes: tuple[float, ...] | None = None,
    sps_cache: dict[str, Any] | None = None,
) -> PpxfHostResult:
    """Fit stellar SSPs plus AGN nuisance components; return the stellar model.

    ``fwhm_gal`` is the instrumental resolution in Angstrom. Passing it lets
    pPXF convolve the templates to the data's resolution; omitting it would let
    stellar absorption profiles — Hbeta absorption in particular — differ purely
    through LSF mismatch, which is exactly the ambiguity this comparison exists
    to resolve.
    """
    from ppxf.ppxf import ppxf as run_ppxf
    from ppxf.ppxf_util import log_rebin
    from ppxf.sps_util import sps_lib

    rest = wavelength / (1.0 + redshift)
    inside = (rest >= fit_range[0]) & (rest <= fit_range[1]) & np.isfinite(flux) & (error > 0)
    if inside.sum() < 200:
        return PpxfHostResult("insufficient_coverage",
                              detail=f"{int(inside.sum())} usable pixels in {fit_range}")

    lam = rest[inside]
    spectrum = flux[inside].astype(float)
    noise = error[inside].astype(float)

    lam_range = (lam.min(), lam.max())
    galaxy, ln_lam, velscale = log_rebin(lam_range, spectrum)
    noise_rebinned, _, _ = log_rebin(lam_range, noise)
    noise_rebinned = np.clip(noise_rebinned, 1e-8, None)
    lam_grid = np.exp(ln_lam)

    normalisation = float(np.median(galaxy[galaxy > 0])) if np.any(galaxy > 0) else 1.0
    galaxy = galaxy / normalisation
    noise_rebinned = noise_rebinned / normalisation

    try:
        # norm_range MUST stay None. With a norm_range, sps_lib takes its
        # "different factor for every template" branch and renormalises all 150
        # SSPs independently, destroying the native relative luminosities that
        # make a population grid meaningful. None takes the "single factor for
        # all templates" branch, which is what SSP population fitting requires.
        cache_key = f"{sps_file.resolve()}|{velscale:.12g}|{age_range}"
        if sps_cache is not None and cache_key in sps_cache:
            sps = sps_cache[cache_key]
        else:
            sps = sps_lib(str(sps_file), velscale, fwhm_gal=fwhm_gal,
                          age_range=age_range, norm_range=None)
            if sps_cache is not None:
                # Callers must scope the cache to spectra sharing one LSF.  This
                # avoids reloading/smoothing the same large SPS cube for every
                # frozen model family without changing any fit input.
                sps_cache[cache_key] = sps
    except Exception as error_:  # noqa: BLE001
        return PpxfHostResult("sps_load_failed", detail=f"{type(error_).__name__}: {error_}")

    stars = sps.templates.reshape(sps.templates.shape[0], -1)
    # One scalar for the whole population, computed over the *fitted* optical
    # window. A global median across 1680-50000 A would let far-IR flux, which
    # no pixel in this fit uses, set the numerical scale.
    optical = (sps.lam_temp >= fit_range[0]) & (sps.lam_temp <= fit_range[1])
    stellar_scale = float(np.median(stars[optical][stars[optical] > 0]))
    if stellar_scale > 0:
        stars = stars / stellar_scale
    n_stars = stars.shape[1]
    del stellar_scale  # retained above only for conditioning

    # Nuisance templates must live on the SPS template grid, not the galaxy
    # grid: pPXF resamples the whole template block from lam_temp onto the data,
    # so mixing grids silently misaligns every nonstellar component.
    slopes = (
        DEFAULT_POWERLAW_SLOPES if powerlaw_slopes is None else powerlaw_slopes
    )
    nonstellar, nonstellar_names = build_nonstellar_templates(
        sps.lam_temp, feii_path=feii_path, powerlaw_slopes=slopes,
        include_powerlaw=include_powerlaw, include_feii=include_feii,
        include_lines=include_lines, include_balmer=include_balmer,
    )

    # Condition the nuisance family separately: power laws, Fe II and line
    # profiles carry no population-relative-luminosity meaning, so scaling them
    # independently is harmless. Because the families now use different
    # conventions, f_host must come from reconstructed component spectra, never
    # from raw weight sums.
    # Emission-masked stellar-only fits (M0) legitimately have no nuisance family.
    if nonstellar.shape[1] == 0:
        # Single kinematic component: pPXF infers it, and passing an explicit
        # all-zero `component` array conflicts with its NCOMP bookkeeping.
        templates = stars
        # pPXF wraps `start` itself when ncomp == 1, so it must stay flat here.
        ppxf_kwargs: dict[str, object] = {"moments": 2}
        start: object = [0.0, 150.0]
        nonstellar_scale = 1.0
    else:
        nonstellar_scale = float(np.median(np.abs(nonstellar[nonstellar != 0])))
        if nonstellar_scale > 0:
            nonstellar = nonstellar / nonstellar_scale
        templates = np.column_stack([stars, nonstellar])
        ppxf_kwargs = {
            "component": np.array([0] * n_stars + [1] * nonstellar.shape[1]),
            "moments": [2, 2],
        }
        start = [[0.0, 150.0], [0.0, 500.0]]

    conditioning = {
        "stellar_median": float(np.median(stars[stars > 0])),
        "stellar_max": float(np.max(np.abs(stars))),
        # M0 has no nuisance family at all, so these are undefined rather than zero.
        "nonstellar_median": (
            float(np.median(np.abs(nonstellar[nonstellar != 0])))
            if nonstellar.size and np.any(nonstellar != 0) else float("nan")
        ),
        "nonstellar_max": (
            float(np.max(np.abs(nonstellar))) if nonstellar.size else float("nan")
        ),
        "galaxy_median": float(np.median(galaxy[galaxy > 0])),
    }

    goodpixels = None
    if mask_emission:
        # M0 only: strong emission must not be explained by stellar populations.
        masked = np.zeros(lam_grid.size, dtype=bool)
        for centre, half_width in (
            (4861.33, 120.0), (5006.84, 40.0), (4958.92, 40.0), (6562.80, 150.0),
            (6583.41, 40.0), (6548.03, 40.0), (3727.09, 30.0), (4340.47, 90.0),
        ):
            masked |= np.abs(lam_grid - centre) < half_width
        goodpixels = np.where(~masked)[0]

    try:
        fit = run_ppxf(
            templates, galaxy, noise_rebinned, velscale, start,
            goodpixels=goodpixels, degree=degree, mdegree=mdegree,
            lam=lam_grid, lam_temp=sps.lam_temp, quiet=True, **ppxf_kwargs,
        )
    except Exception as error_:  # noqa: BLE001
        return PpxfHostResult("ppxf_failed", detail=f"{type(error_).__name__}: {error_}")

    weights = fit.weights
    # Use the design matrix pPXF actually fitted, which carries the templates
    # after convolution and resampling onto the galaxy grid.
    matrix = fit.matrix[:, fit.matrix.shape[1] - templates.shape[1]:]
    stellar_model = matrix[:, :n_stars] @ weights[:n_stars]
    nonstellar_model = matrix[:, n_stars:] @ weights[n_stars:]

    # AGN *continuum* only: power laws and Fe II, excluding line components.
    # dtype must be forced: an empty list yields float64, which cannot index.
    # M0 has no nuisance templates at all, so this is the empty case in practice.
    continuum_mask = np.array(
        [n.startswith("powerlaw") or n.startswith("feii") for n in nonstellar_names],
        dtype=bool,
    )
    agn_continuum = matrix[:, n_stars:][:, continuum_mask] @ weights[n_stars:][continuum_mask]

    fhost = f_host_from_components(lam_grid, stellar_model, agn_continuum, fhost_window)

    return PpxfHostResult(
        status="ok",
        nonstellar_names=nonstellar_names,
        fit=fit,
        wavelength=lam_grid,
        stellar=stellar_model,
        nonstellar=nonstellar_model,
        observed=galaxy,
        f_host_5100=fhost,
        chi2=float(fit.chi2),
        n_stellar_templates=n_stars,
        sps_library=Path(sps_file).name,
        diagnostics={
            "stellar_weight_sum": float(np.sum(weights[:n_stars])),
            "nonstellar_weight_sum": float(np.sum(weights[n_stars:])),
            "n_nonstellar_templates": float(nonstellar.shape[1]),
            **conditioning,
        },
    )


__all__ = [
    "BROAD_LINES",
    "NARROW_LINES",
    "PpxfHostResult",
    "build_nonstellar_templates",
    "decompose",
    "f_host_from_components",
]
