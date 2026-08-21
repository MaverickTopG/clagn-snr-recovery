#!/usr/bin/env python
"""Stage 15 / 5B.8 — M1a versus M1b: does the collapse need a multi-slope bank?

D-049 localised the stellar collapse to M1, when the AGN continuum family
enters. Two hypotheses remain:

    H1  any reasonable featureless power law makes the host unnecessary
    H2  the *simultaneous bank* of slopes is flexible enough to build continuum
        curvature a single physical power law cannot, mimicking galaxy shape

    M1a   SSP + PL(alpha_k) + BEL/NEL, one slope at a time, every slope saved
    M1b   SSP + sum_k PL(alpha_k) + BEL/NEL, the existing model, unmodified

The preferred M1a slope is chosen by the **global fit objective alone** — never
by which slope yields the largest stellar weight. Every slope's result is
recorded so that choice stays auditable.

Curvature diagnostic: the best-fit summed power-law component from M1b is refit
with a single power law in log space, and the residual scatter reported. Near
zero means the bank is effectively picking one slope and H2 is unlikely;
substantial scatter means it is constructing curvature, and H2 becomes
plausible.

**The production model is not modified.** This is diagnosis.

Usage
-----
    uv run python 00_scripts/15_powerlaw_basis_test.py --slice
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.fitting.components import from_ppxf  # noqa: E402
from p3sf.fitting.ppxf_host import DEFAULT_POWERLAW_SLOPES, decompose  # noqa: E402
from p3sf.qc.alignment import STELLAR_FEATURES  # noqa: E402
from p3sf.spectral_domain import median_in_window  # noqa: E402

FIT_RANGE = (3700.0, 7000.0)
FWHM_SDSS = 2.76
HBETA_REGION = (4700.0, 5100.0)

OBJECTS = [
    ("SDSSJ081319.34+460849.5", "positive control"),
    ("SDSSJ105058.42+241351.18", "intermediate"),
    ("SDSSJ114408.90+424357.5", "low-feature"),
    ("SDSSJ102152.34+464515.6", "paradoxical"),
]


def read_spectrum(path: Path, redshift: float):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul[1].data
        wavelength = 10.0 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        ivar = np.asarray(data["ivar"], dtype=float)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    good = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    return wavelength[good], flux[good], error[good], float(redshift)


def residuals(fit, lam: np.ndarray) -> dict[str, float | None]:
    residual = np.asarray(fit.galaxy) - np.asarray(fit.bestfit)

    def rms(mask: np.ndarray) -> float | None:
        return float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.sum() > 5 else None

    stellar_mask = np.zeros(lam.size, dtype=bool)
    for _rest, window in STELLAR_FEATURES.values():
        stellar_mask |= (lam >= window[0]) & (lam <= window[1])
    hbeta = (lam >= HBETA_REGION[0]) & (lam <= HBETA_REGION[1])
    return {
        "rms_global": rms(np.ones(lam.size, dtype=bool)),
        "rms_stellar_features": rms(stellar_mask),
        "rms_hbeta": rms(hbeta),
    }


def curvature(lam: np.ndarray, powerlaw: np.ndarray) -> dict[str, float | None]:
    """Refit a summed power-law component with ONE power law, in log space.

    Answers directly whether the bank is behaving as a single slope or building
    curvature no physical power law could produce.
    """
    positive = powerlaw > 0
    if positive.sum() < 50:
        return {"alpha_effective": None, "curvature_rms_dex": None}
    log_lam = np.log10(lam[positive])
    log_flux = np.log10(powerlaw[positive])
    slope, intercept = np.polyfit(log_lam, log_flux, 1)
    residual = log_flux - (intercept + slope * log_lam)
    return {
        "alpha_effective": float(slope),
        "curvature_rms_dex": float(np.sqrt(np.mean(residual**2))),
    }


def run(spectrum, sps_file, feii, window, **switches):
    result = decompose(
        *spectrum, sps_file=sps_file, fhost_window=window, fwhm_gal=FWHM_SDSS,
        feii_path=feii, fit_range=FIT_RANGE, include_feii=False,
        include_lines=True, include_balmer=False, **switches,
    )
    if result.status != "ok" or result.fit is None:
        return None, None, result
    model = from_ppxf(
        result.fit, result.wavelength, result.n_stellar_templates,
        result.nonstellar_names, native_f_host=result.f_host_5100,
    )
    return model, model.check_reconstruction(), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    parser.add_argument("--fine", action="store_true",
                        help="31 slopes at 0.1 steps over -3.0..0.0 (D-051)")
    args = parser.parse_args()

    cfg = load_config()
    root = project_root()
    suffix = "_slice" if args.slice else ""
    window = tuple(cfg.snr_metrics.continuum_5100)
    sps_file = root / "06_fitting/alternative_fit/sps_libraries/spectra_xsl_9.0.npz"
    feii = root / "06_fitting/pyqsofit/vendor/PyQSOFit/src/pyqsofit/fe_optical.txt"
    qc = pd.read_csv(root / "03_spectra" / "qc" / f"qc_verdicts{suffix}.csv")

    rows: list[dict[str, object]] = []
    for object_id, role in OBJECTS:
        subset = qc[(qc.object_id == object_id) & qc.qc_pass & (qc.survey == "SDSS")]
        if subset.empty:
            continue
        spectrum = read_spectrum(root / subset.iloc[0].path, subset.iloc[0].redshift)

        print("=" * 78)
        print(f"{object_id}   [{role}]")
        print("=" * 78)
        print(f"{'model':<12}{'w*':>9}{'f_cont':>9}{'chi2':>9}{'RMSglob':>9}"
              f"{'RMSstar':>9}{'RMSHb':>8}{'PL@5100':>9}")

        # -- M1a: every slope, saved individually ----------------------------
        slopes = (
            [round(-3.0 + 0.1 * i, 1) for i in range(31)]
            if args.fine else list(DEFAULT_POWERLAW_SLOPES)
        )
        for slope in slopes:
            model, check, result = run(
                spectrum, sps_file, feii, window,
                include_powerlaw=True, powerlaw_slopes=(slope,),
            )
            if model is None:
                print(f"{'M1a a=' + format(slope, '+.1f'):<12}  FAILED {result.status}")
                continue
            weight = float(result.diagnostics.get("stellar_weight_sum", 0.0))
            fractions = model.canonical_fractions(window) if check.passed else {}
            res = residuals(result.fit, result.wavelength)
            pl = median_in_window(model.wavelength, model.get("agn_smooth"), window).value
            record = {
                "object_id": object_id, "role": role, "model": "M1a",
                "slope": slope, "stellar_weight_sum": weight,
                "f_host_cont": fractions.get("f_host_cont"),
                "chi2": result.chi2, "powerlaw_at_5100": pl,
                "reconstruction_passed": check.passed, **res,
            }
            rows.append(record)
            print(f"{'M1a a=' + format(slope, '+.1f'):<12}{weight:>9.3f}"
                  f"{(fractions.get('f_host_cont') or 0.0):>9.3f}{result.chi2:>9.2f}"
                  f"{res['rms_global']:>9.3f}{res['rms_stellar_features']:>9.3f}"
                  f"{res['rms_hbeta']:>8.3f}{(pl or 0.0):>9.3f}")

        # -- M1b: the existing simultaneous bank, unmodified -----------------
        model, check, result = run(spectrum, sps_file, feii, window, include_powerlaw=True)
        if model is not None:
            weight = float(result.diagnostics.get("stellar_weight_sum", 0.0))
            fractions = model.canonical_fractions(window) if check.passed else {}
            res = residuals(result.fit, result.wavelength)
            pl = median_in_window(model.wavelength, model.get("agn_smooth"), window).value
            curve = curvature(model.wavelength, model.get("agn_smooth"))
            rows.append({
                "object_id": object_id, "role": role, "model": "M1b", "slope": None,
                "stellar_weight_sum": weight,
                "f_host_cont": fractions.get("f_host_cont"),
                "chi2": result.chi2, "powerlaw_at_5100": pl,
                "reconstruction_passed": check.passed, **res, **curve,
            })
            print(f"{'M1b bank':<12}{weight:>9.3f}"
                  f"{(fractions.get('f_host_cont') or 0.0):>9.3f}{result.chi2:>9.2f}"
                  f"{res['rms_global']:>9.3f}{res['rms_stellar_features']:>9.3f}"
                  f"{res['rms_hbeta']:>8.3f}{(pl or 0.0):>9.3f}")
            print(f"  bank curvature: alpha_eff = {curve['alpha_effective']}, "
                  f"residual scatter = {curve['curvature_rms_dex']} dex")
        print()

    frame = pd.DataFrame(rows)
    tag = "_fine" if args.fine else ""
    out = root / "06_fitting" / "validation" / f"powerlaw_basis_test{tag}{suffix}.csv"
    frame.to_csv(out, index=False)

    print("=" * 78)
    print("BEST M1a BY GLOBAL OBJECTIVE (chi2) vs M1b")
    print("=" * 78)
    print(f"{'object':<26}{'best a':>8}{'w* M1a':>9}{'w* M1b':>9}"
          f"{'chi2 M1a':>10}{'chi2 M1b':>10}{'curv dex':>10}")
    for object_id, _role in OBJECTS:
        block = frame[frame.object_id == object_id]
        a = block[block.model == "M1a"]
        b = block[block.model == "M1b"]
        if a.empty or b.empty:
            continue
        best = a.loc[a.chi2.idxmin()]      # global objective only
        print(f"{object_id:<26}{best.slope:>8.1f}{best.stellar_weight_sum:>9.3f}"
              f"{b.iloc[0].stellar_weight_sum:>9.3f}{best.chi2:>10.2f}"
              f"{b.iloc[0].chi2:>10.2f}{b.iloc[0].curvature_rms_dex:>10.4f}")
    print()
    print(f"written -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
