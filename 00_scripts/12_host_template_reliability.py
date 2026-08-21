#!/usr/bin/env python
"""Stage 12 / Phase 5B — host-template reliability analysis.

The question is **not** "is the galaxy physically detected". It is:

    can this recovered stellar spectrum be trusted enough to serve as the
    ground-truth host component of a counterfactual experiment?

Those are different questions, and conflating them is how a detection claim
smuggles itself into a simulation input.

Stages
------
``--stage reasons``    exact decline branch for every failed decomposition
``--stage shape``      same-object host spectral shape across epochs
``--stage leakage``    line contamination after scaling to the maximum f_host
``--stage stability``  noise and initialisation refits (slow)
``--stage all``

Provisional states assigned here:

    HOST_TEMPLATE_VALID        stable and clean enough to use as a Q2 input
    HOST_TEMPLATE_UNSTABLE     recovered, but not reproducible under perturbation
    HOST_TEMPLATE_UNVALIDATED  recovered, stability not yet established
    NO_PRIMARY_HOST_TEMPLATE   no same-object template from either epoch

``HOST_CONSISTENT_WITH_ZERO`` is deliberately absent: Phase 5A showed the sample
contains no genuine zero-host solution, only declined decompositions.

Nothing here changes Q2 eligibility. All uncertainties are **provisional**,
computed against the native pixel errors, and must be revisited after Gate C
validates those errors.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import load_config, project_root  # noqa: E402

SCRIPT = "12_host_template_reliability.py"
BANDS = ("B", "R", "Z")

# Frozen before any stability result was inspected: the config's 5100 A
# continuum window, used for every host fraction in this project.
CFG = load_config()
FHOST_WINDOW = tuple(CFG.snr_metrics.continuum_5100)
MAX_INJECTED_FHOST = max(CFG.host_fraction_grid)

LINE_PROBES = {
    "Hbeta_broad_wing": (4790.0, 4930.0),
    "OIII_5007": (4995.0, 5020.0),
    "OIII_4959": (4948.0, 4970.0),
    "FeII_optical": (5150.0, 5350.0),
}


def read_spectrum(path: Path, survey: str, redshift: float):
    from astropy.io import fits

    with fits.open(path) as hdul:
        if survey == "SDSS":
            data = hdul[1].data
            wavelength = 10.0 ** np.asarray(data["loglam"], dtype=float)
            flux = np.asarray(data["flux"], dtype=float)
            ivar = np.asarray(data["ivar"], dtype=float)
        else:
            wavelength = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
            flux = np.asarray(hdul["FLUX"].data, dtype=float)
            ivar = np.asarray(hdul["IVAR"].data, dtype=float)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    good = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    return wavelength[good], flux[good], error[good], float(redshift)


def run_fit(wavelength, flux, error, redshift, vendor, **overrides):
    from pyqsofit.PyQSOFit import QSOFit

    settings = {
        "deredden": True, "decompose_host": True, "Fe_uv_op": True, "linefit": True,
        "MC": False, "save_result": False, "plot_fig": False, "save_fig": False,
        "verbose": False,
    }
    settings.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = QSOFit(wavelength, flux, error, redshift, path=vendor)
        fit.Fit(name="x", **settings)
    return fit


def standardized_fhost(fit, redshift: float) -> float | None:
    """f_host over the frozen rest-frame window, from the model components.

    Computed here rather than taken from PyQSOFit's own ``frac_host_5100`` so
    that every host fraction in this project — measured, injected, recovered —
    uses one identical definition and window.
    """
    host = getattr(fit, "host", None)
    qso = getattr(fit, "qso", None)
    wave = getattr(fit, "wave", None)
    if host is None or qso is None or wave is None:
        return None
    host, qso, wave = (np.asarray(a, dtype=float) for a in (host, qso, wave))
    if not (host.size == qso.size == wave.size) or host.size == 0:
        return None

    rest = wave / (1.0 + redshift)
    window = (rest >= FHOST_WINDOW[0]) & (rest <= FHOST_WINDOW[1])
    if window.sum() < 5:
        return None
    host_level = float(np.median(host[window]))
    qso_level = float(np.median(qso[window]))
    total = host_level + qso_level
    return float(host_level / total) if total > 0 else None


# ---------------------------------------------------------------------------
# stage: decline reasons
# ---------------------------------------------------------------------------


def _template_path() -> str:
    """Where PyQSOFit keeps its PCA/BC03 templates.

    QSOFit passes ``self.install_path`` (the package directory) to
    ``decompose_host_qso``, not the user-supplied ``path``, which locates only
    qsopar.fits. Replicating the decline conditions requires the same directory.
    """
    import pyqsofit.PyQSOFit as module

    return str(Path(module.__file__).resolve().parent)


def decline_reason(wavelength, flux, error, redshift, vendor) -> dict[str, object]:
    """Name the exact decline branch by capturing PyQSOFit's own datacube.

    Re-deriving the decomposition from the raw spectrum does **not** reproduce
    PyQSOFit's decision: ``Fit`` smooths, rejects bad pixels, deredden and trims
    the spectrum before ``decompose_host_qso`` ever sees it, so a replication
    fed raw flux accepts cases the real run declines. Instead we wrap
    ``Linear_decomp.auto_decomp`` for the duration of a genuine fit and evaluate
    the acceptance conditions on the exact array PyQSOFit used.
    """
    from pyqsofit import HostDecomp

    captured: dict[str, object] = {}
    original = HostDecomp.Linear_decomp.auto_decomp

    def capturing(self, *a, **kw):
        result = original(self, *a, **kw)
        captured["cube"] = result[0]
        return result

    out: dict[str, object] = {
        "assertion_ok": None, "neg_pixel_fraction": None,
        "host_median_over_flux_level": None, "median_host_spec": None,
        "cond_negative_pixels": None, "cond_host_too_weak": None,
        "cond_negative_host_spec": None, "decline_reason": "UNKNOWN",
    }

    HostDecomp.Linear_decomp.auto_decomp = capturing
    try:
        fit = run_fit(wavelength, flux, error, redshift, vendor)
        out["assertion_ok"] = "cube" in captured
        out["pyqsofit_decomposed"] = bool(getattr(fit, "decomposed", False))
    except Exception as error_:  # noqa: BLE001
        out["decline_reason"] = f"FIT_EXCEPTION: {type(error_).__name__}"
        return out
    finally:
        HostDecomp.Linear_decomp.auto_decomp = original

    if "cube" not in captured:
        # auto_decomp never ran: the fitter's coverage assertion failed first.
        out["decline_reason"] = "TEMPLATE_COVERAGE_BELOW_50_PERCENT"
        return out

    cube = captured["cube"]
    data, host_model, qso_model = cube[1, :], cube[3, :], cube[4, :]
    flux_level = float(np.median(np.abs(data)))
    host_spec = data - qso_model

    negative_fraction = float(np.mean((host_model < 0) | (qso_model < 0)))
    host_ratio = float(np.median(host_model) / flux_level) if flux_level > 0 else np.nan
    median_host_spec = float(np.median(host_spec))

    cond_negative = negative_fraction > 0.1
    cond_weak = host_ratio < 0.01
    cond_negative_spec = median_host_spec < 0

    reasons = []
    if cond_negative:
        reasons.append("NEGATIVE_MODEL_PIXELS_OVER_10PC")
    if cond_weak:
        reasons.append("LOW_HOST_CONTRAST")
    if cond_negative_spec:
        reasons.append("NEGATIVE_HOST_RESIDUAL")

    out.update(
        neg_pixel_fraction=negative_fraction,
        host_median_over_flux_level=host_ratio,
        median_host_spec=median_host_spec,
        cond_negative_pixels=cond_negative,
        cond_host_too_weak=cond_weak,
        cond_negative_host_spec=cond_negative_spec,
        decline_reason="+".join(reasons) if reasons else "ACCEPTED_BY_CONDITIONS",
    )
    return out


# PyQSOFit defaults, matching the audit run.
CFG_NPCA_GAL = 5
CFG_NPCA_QSO = 10


# ---------------------------------------------------------------------------
# stage: line leakage amplified to the maximum injected host fraction
# ---------------------------------------------------------------------------


def leakage_at_scale(fit, redshift: float, target_fhost: float) -> dict[str, float | None]:
    """Line structure introduced purely by scaling the host template.

    A contamination of 0.5% at a native f_host of 0.20 becomes far larger once
    the template is multiplied to reach 0.85, so leakage must be judged at the
    amplitude the experiment will actually use — not at native amplitude.

    Stellar Hβ *absorption* is physical host dilution and must be preserved;
    what matters is emission-like structure that belongs to the AGN or
    narrow-line model. Sign is therefore retained, not taken in absolute value.
    """
    host = np.asarray(getattr(fit, "host", []), dtype=float)
    qso = np.asarray(getattr(fit, "qso", []), dtype=float)
    wave = np.asarray(getattr(fit, "wave", []), dtype=float)
    if host.size == 0 or host.size != wave.size or qso.size != wave.size:
        return dict.fromkeys([f"{k}_scaled" for k in LINE_PROBES], None)

    rest = wave / (1.0 + redshift)
    window = (rest >= FHOST_WINDOW[0]) & (rest <= FHOST_WINDOW[1])
    host_level = float(np.median(host[window]))
    qso_level = float(np.median(qso[window]))
    if host_level <= 0 or qso_level <= 0:
        return dict.fromkeys([f"{k}_scaled" for k in LINE_PROBES], None)

    scale = (target_fhost / (1.0 - target_fhost)) * (qso_level / host_level)
    scaled = host * scale

    out: dict[str, float | None] = {"host_scale_to_max": scale}
    for name, (low, high) in LINE_PROBES.items():
        probe = (rest >= low) & (rest <= high)
        if probe.sum() < 5:
            out[f"{name}_scaled"] = None
            continue
        edge = max(3, int(probe.sum()) // 6)
        inside = scaled[probe]
        baseline = np.linspace(np.median(inside[:edge]), np.median(inside[-edge:]), inside.size)
        # Absolute flux units, comparable to a line flux.
        out[f"{name}_scaled"] = float(np.trapezoid(inside - baseline, rest[probe]))
    return out


# ---------------------------------------------------------------------------
# stage: stability
# ---------------------------------------------------------------------------


def noise_stability(
    wavelength, flux, error, redshift, vendor, n_mc: int, seed: int
) -> dict[str, object]:
    """Parametric bootstrap: resample pixels from their own errors and refit."""
    rng = np.random.default_rng(seed)
    fractions: list[float] = []
    successes = 0
    for _ in range(n_mc):
        perturbed = flux + rng.normal(0.0, 1.0, flux.size) * error
        try:
            fit = run_fit(wavelength, perturbed, error, redshift, vendor)
        except Exception:  # noqa: BLE001
            continue
        if not bool(getattr(fit, "decomposed", False)):
            continue
        successes += 1
        value = standardized_fhost(fit, redshift)
        if value is not None:
            fractions.append(value)

    if not fractions:
        return {"n_mc": n_mc, "decomp_stability": successes / n_mc if n_mc else None,
                "fhost_p16": None, "fhost_p50": None, "fhost_p84": None, "fhost_sd": None}

    array = np.asarray(fractions)
    p16, p50, p84 = (float(np.percentile(array, q)) for q in (16, 50, 84))
    return {
        "n_mc": n_mc,
        "n_mc_decomposed": successes,
        "decomp_stability": successes / n_mc,
        "fhost_p16": p16, "fhost_p50": p50, "fhost_p84": p84,
        "fhost_sd": float(np.std(array)),
        "fhost_interval_width": p84 - p16,
    }


def basis_stability(
    wavelength, flux, error, redshift, vendor
) -> dict[str, object]:
    """Vary the template basis — a MODEL-CHOICE systematic, not initialisation.

    This changes the number of galaxy and QSO PCA components, i.e. the model
    basis itself, not the optimizer's starting coefficients. The resulting
    spread is therefore ``sigma_basis`` (model-choice systematic uncertainty),
    which is a stronger and more referee-legible statement than initialisation
    sensitivity would be. A separate ``sigma_init`` test, holding the basis
    fixed and perturbing only the starting point, has not been run.

    Pixel-noise resampling alone cannot substitute for this: every noise trial
    restarts in the same solution basin, so it probes photon statistics rather
    than whether the decomposition is determined at all.
    """
    variants = [
        ("default", {"npca_gal": 5, "npca_qso": 10}),
        ("fewer_gal", {"npca_gal": 3, "npca_qso": 10}),
        ("more_gal", {"npca_gal": 10, "npca_qso": 10}),
        ("fewer_qso", {"npca_gal": 5, "npca_qso": 5}),
        ("more_qso", {"npca_gal": 5, "npca_qso": 20}),
    ]
    values: dict[str, float | None] = {}
    for label, overrides in variants:
        try:
            fit = run_fit(wavelength, flux, error, redshift, vendor, **overrides)
            values[label] = (
                standardized_fhost(fit, redshift)
                if bool(getattr(fit, "decomposed", False))
                else None
            )
        except Exception:  # noqa: BLE001
            values[label] = None

    finite = [v for v in values.values() if v is not None]
    return {
        **{f"fhost_basis_{k}": v for k, v in values.items()},
        "n_basis_variants_ok": len(finite),
        "fhost_basis_spread": float(max(finite) - min(finite)) if len(finite) > 1 else None,
    }


# ---------------------------------------------------------------------------
# stage: same-object host shape across epochs
# ---------------------------------------------------------------------------


def host_shape(fit, redshift: float, grid: np.ndarray) -> np.ndarray | None:
    """Host model resampled onto a common rest grid and normalised at 5100 A.

    Normalised because SDSS, BOSS and DESI differ in aperture, seeing and
    calibration, so absolute host flux is not comparable across surveys even
    for an unchanging galaxy. Shape is.
    """
    host = np.asarray(getattr(fit, "host", []), dtype=float)
    wave = np.asarray(getattr(fit, "wave", []), dtype=float)
    if host.size == 0 or host.size != wave.size:
        return None
    rest = wave / (1.0 + redshift)
    window = (rest >= FHOST_WINDOW[0]) & (rest <= FHOST_WINDOW[1])
    if window.sum() < 5:
        return None
    level = float(np.median(host[window]))
    if level <= 0:
        return None
    resampled = np.interp(grid, rest, host / level, left=np.nan, right=np.nan)
    return resampled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    parser.add_argument("--stage", default="all",
                        choices=["all", "reasons", "shape", "leakage", "stability"])
    parser.add_argument("--n-mc", type=int, default=100)
    args = parser.parse_args()

    root = project_root()
    suffix = "_slice" if args.slice else ""
    vendor = str(root / "06_fitting" / "pyqsofit" / "vendor" / "PyQSOFit")
    out_dir = root / "06_fitting" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(out_dir / f"host_output_audit{suffix}.csv")
    qc = pd.read_csv(root / "03_spectra" / "qc" / f"qc_verdicts{suffix}.csv")
    # Join on the full epoch key. Joining on (object_id, spectrum_id) alone fans
    # out when one object has several DESI rows sharing a targetid-derived id.
    key = ["object_id", "survey", "spectrum_id", "mjd"]
    qc_paths = (
        qc[[*key, "path"]].astype({"spectrum_id": str}).drop_duplicates(subset=key)
    )
    audit = audit.astype({"spectrum_id": str}).merge(qc_paths, on=key, how="left")
    missing = int(audit.path.isna().sum())
    if missing:
        raise RuntimeError(f"{missing} audit rows could not be matched to a spectrum path")

    failed = audit[audit.host_output_state == "DECOMPOSITION_FAILED"]
    valid = audit[audit.host_output_state == "OUTPUT_VALID_NONZERO"]

    # -------------------------------------------------- decline reasons
    if args.stage in {"all", "reasons"}:
        print("=" * 78)
        print("5B.1  EXACT DECLINE BRANCH FOR EVERY FAILED DECOMPOSITION")
        print("=" * 78)
        rows = []
        for row in failed.itertuples():
            spec = read_spectrum(root / row.path, row.survey, row.redshift)
            reason = decline_reason(*spec, vendor)
            rows.append({"object_id": row.object_id, "survey": row.survey,
                         "spectrum_id": row.spectrum_id,
                         "snr_hbeta_window": row.snr_hbeta_window, **reason})
            print(f"{row.object_id:26s} {row.survey:5s} "
                  f"S/N {row.snr_hbeta_window:5.1f}  {reason['decline_reason']}")
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"host_decline_reasons{suffix}.csv", index=False)
        print()
        print(frame.decline_reason.value_counts().to_string())
        print()
        print("Condition values (PyQSOFit thresholds: neg>0.10, ratio<0.01, spec<0):")
        print(frame[["object_id", "neg_pixel_fraction", "host_median_over_flux_level",
                     "median_host_spec"]].to_string(index=False,
                                                    float_format=lambda v: f"{v:.4f}"))
        print()

    # -------------------------------------------------- leakage at max scale
    if args.stage in {"all", "leakage"}:
        print("=" * 78)
        print(f"5B.6  LINE LEAKAGE AFTER SCALING TO f_host = {MAX_INJECTED_FHOST}")
        print("=" * 78)
        rows = []
        for row in valid.itertuples():
            spec = read_spectrum(root / row.path, row.survey, row.redshift)
            fit = run_fit(*spec, vendor)
            if not bool(getattr(fit, "decomposed", False)):
                continue
            native = standardized_fhost(fit, spec[3])
            leak = leakage_at_scale(fit, spec[3], MAX_INJECTED_FHOST)
            rows.append({"object_id": row.object_id, "survey": row.survey,
                         "spectrum_id": row.spectrum_id,
                         "fhost_native_standardized": native, **leak})
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"host_leakage_scaled{suffix}.csv", index=False)
        print(frame[["object_id", "survey", "fhost_native_standardized", "host_scale_to_max"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        probes = [c for c in frame.columns if c.endswith("_scaled")]
        print("Induced structure in absolute flux units (sign retained; stellar")
        print("absorption is physical and must survive):")
        print(frame[probes].describe().loc[["count", "mean", "min", "max"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
        print()

    # -------------------------------------------------- epoch shape agreement
    if args.stage in {"all", "shape"}:
        print("=" * 78)
        print("5B.5  SAME-OBJECT HOST SHAPE ACROSS EPOCHS")
        print("=" * 78)
        grid = np.arange(4000.0, 5500.0, 2.0)
        shapes: dict[tuple[str, str], np.ndarray] = {}
        for row in valid.itertuples():
            spec = read_spectrum(root / row.path, row.survey, row.redshift)
            fit = run_fit(*spec, vendor)
            if not bool(getattr(fit, "decomposed", False)):
                continue
            shape = host_shape(fit, spec[3], grid)
            if shape is not None:
                shapes[(row.object_id, str(row.spectrum_id))] = shape

        rows = []
        for object_id in sorted({k[0] for k in shapes}):
            keys = [k for k in shapes if k[0] == object_id]
            if len(keys) < 2:
                continue
            a, b = shapes[keys[0]], shapes[keys[1]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < 50:
                continue
            difference = a[both] - b[both]
            rows.append({
                "object_id": object_id,
                "n_common_pixels": int(both.sum()),
                "median_abs_shape_difference": float(np.median(np.abs(difference))),
                "rms_shape_difference": float(np.sqrt(np.mean(difference**2))),
                "correlation": float(np.corrcoef(a[both], b[both])[0, 1]),
            })
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"host_shape_agreement{suffix}.csv", index=False)
        if frame.empty:
            print("(no object has two valid host templates)")
        else:
            print("Normalised at 5100 A: absolute host flux is not comparable across")
            print("SDSS/BOSS/DESI apertures even for an unchanging galaxy. Shape is.")
            print()
            print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    # -------------------------------------------------- stability
    if args.stage in {"all", "stability"}:
        print("=" * 78)
        print("5B.3 / 5B.4  STABILITY  (provisional: uses native pixel errors)")
        print("=" * 78)
        rows = []
        for index, row in enumerate(valid.itertuples()):
            spec = read_spectrum(root / row.path, row.survey, row.redshift)
            baseline_fit = run_fit(*spec, vendor)
            baseline = standardized_fhost(baseline_fit, spec[3])
            noise = noise_stability(*spec, vendor, args.n_mc, CFG.random_seed + index)
            basis = basis_stability(*spec, vendor)
            bias = (
                noise["fhost_p50"] - baseline
                if noise["fhost_p50"] is not None and baseline is not None
                else None
            )
            rows.append({
                "object_id": row.object_id, "survey": row.survey,
                "spectrum_id": row.spectrum_id,
                "snr_hbeta_window": row.snr_hbeta_window,
                "fhost_baseline": baseline, "fhost_mc_bias": bias,
                **noise, **basis,
            })
            print(f"{row.object_id:26s} {row.survey:5s} "
                  f"f_host {baseline if baseline is None else round(baseline, 3)}  "
                  f"stability {noise['decomp_stability']}  "
                  f"basis spread {basis['fhost_basis_spread']}")
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"host_stability{suffix}.csv", index=False)
        print()
        print(frame[["object_id", "fhost_baseline", "decomp_stability", "fhost_p16",
                     "fhost_p84", "fhost_interval_width", "fhost_basis_spread"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        print("These are PROVISIONAL. They use the native pixel errors, which Gate C")
        print("has not yet validated; DESI DR1 IVAR in particular is known to be")
        print("underestimated above S/N ~ 20-30. Revisit after the error audit.")

    print()
    print("No Q2 eligibility change is made by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
