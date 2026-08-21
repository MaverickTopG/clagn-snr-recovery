#!/usr/bin/env python
"""Stage 14 / 5B.8 item 9 — M0 to M3 nuisance-component ablation.

Diagnosis, not tuning. The ladder localises *which* nuisance family causes the
stellar solution to collapse, on four objects chosen for contrast rather than
convenience.

    M0  stars only, strong emission masked          (machinery check)
    M1  M0 + AGN power law + broad/narrow lines     (no Fe II)
    M2  M1 + Fe II
    M3  M2 + Balmer pseudocontinuum

M0's host fraction is deliberately **not** interpreted: forcing stellar
populations to explain nuclear continuum is unphysical for a Type-1 AGN. Its
only job is to answer whether the stellar basis can reproduce the observed
absorption structure at all.

Residuals are tracked in three places, because a global chi-square can improve
while the handful of pixels carrying stellar features get worse — which is
precisely how an optimizer can legitimately choose zero stellar weight despite
visible absorption:

    global            all fitted pixels
    stellar features  windows around independently validated absorption lines
    Hbeta region      the measurement region of this project

Absolute reduced chi-square is not used as a quality threshold: Gate C has not
validated the error model, and pPXF's chi-square minimum location is insensitive
to overall noise scaling.

**Nothing here changes the production model.** No regularisation, no minimum
stellar weight, no PyQSOFit-informed prior, no reweighting of stellar features.

Usage
-----
    uv run python 00_scripts/14_ablation_ladder.py --slice
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
from p3sf.fitting.ppxf_host import decompose  # noqa: E402
from p3sf.qc.alignment import STELLAR_FEATURES  # noqa: E402
from p3sf.spectral_domain import median_in_window  # noqa: E402

SCRIPT = "14_ablation_ladder.py"
FIT_RANGE = (3700.0, 7000.0)
FWHM_SDSS = 2.76
HBETA_REGION = (4700.0, 5100.0)

OBJECTS = [
    ("SDSSJ081319.34+460849.5", "positive control"),
    ("SDSSJ105058.42+241351.18", "intermediate"),
    ("SDSSJ114408.90+424357.5", "low-feature"),
    ("SDSSJ102152.34+464515.6", "paradoxical"),
]

LADDER = {
    "M0": {"include_powerlaw": False, "include_feii": False,
           "include_lines": False, "include_balmer": False, "mask_emission": True},
    "M1": {"include_powerlaw": True, "include_feii": False,
           "include_lines": True, "include_balmer": False, "mask_emission": False},
    "M2": {"include_powerlaw": True, "include_feii": True,
           "include_lines": True, "include_balmer": False, "mask_emission": False},
    "M3": {"include_powerlaw": True, "include_feii": True,
           "include_lines": True, "include_balmer": True, "mask_emission": False},
}


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
    """Unweighted RMS residual, globally and in the regions that matter."""
    observed = np.asarray(fit.galaxy)
    model = np.asarray(fit.bestfit)
    residual = observed - model

    def rms(mask: np.ndarray) -> float | None:
        return float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.sum() > 5 else None

    stellar_mask = np.zeros(lam.size, dtype=bool)
    for _rest, window in STELLAR_FEATURES.values():
        stellar_mask |= (lam >= window[0]) & (lam <= window[1])

    hbeta_mask = (lam >= HBETA_REGION[0]) & (lam <= HBETA_REGION[1])
    return {
        "rms_global": rms(np.ones(lam.size, dtype=bool)),
        "rms_stellar_features": rms(stellar_mask),
        "rms_hbeta": rms(hbeta_mask),
        "n_stellar_feature_pixels": int(stellar_mask.sum()),
    }


def contribution(model, family: str, window: tuple[float, float]) -> float | None:
    """Median flux of one family inside the frozen host-fraction window."""
    value = median_in_window(model.wavelength, model.get(family), window)
    return value.value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    parser.add_argument("--library", default="XSL", choices=["XSL", "E-MILES"])
    args = parser.parse_args()

    cfg = load_config()
    root = project_root()
    suffix = "_slice" if args.slice else ""
    window = tuple(cfg.snr_metrics.continuum_5100)
    sps_file = root / "06_fitting/alternative_fit/sps_libraries" / (
        "spectra_xsl_9.0.npz" if args.library == "XSL" else "spectra_emiles_9.0.npz"
    )
    feii = root / "06_fitting/pyqsofit/vendor/PyQSOFit/src/pyqsofit/fe_optical.txt"
    qc = pd.read_csv(root / "03_spectra" / "qc" / f"qc_verdicts{suffix}.csv")

    rows: list[dict[str, object]] = []
    for object_id, role in OBJECTS:
        subset = qc[(qc.object_id == object_id) & qc.qc_pass & (qc.survey == "SDSS")]
        if subset.empty:
            continue
        row = subset.iloc[0]
        spectrum = read_spectrum(root / row.path, row.redshift)

        print("=" * 78)
        print(f"{object_id}   [{role}]   {args.library}")
        print("=" * 78)
        print(f"{'stage':<5}{'w*':>9}{'f_cont':>9}{'f_pseudo':>10}{'chi2':>8}"
              f"{'RMSglob':>9}{'RMSstar':>9}{'RMSHb':>8}{'PL@5100':>9}"
              f"{'FeII':>8}{'Balmer':>8}  recon")

        for stage, switches in LADDER.items():
            result = decompose(
                *spectrum, sps_file=sps_file, fhost_window=window,
                fwhm_gal=FWHM_SDSS, feii_path=feii, fit_range=FIT_RANGE, **switches,
            )
            record: dict[str, object] = {
                "object_id": object_id, "role": role, "library": args.library,
                "stage": stage, "status": result.status,
            }
            if result.status != "ok" or result.fit is None:
                record["detail"] = result.detail[:120]
                rows.append(record)
                print(f"{stage:<5}  FAILED {result.status}: {result.detail[:50]}")
                continue

            model = from_ppxf(
                result.fit, result.wavelength, result.n_stellar_templates,
                result.nonstellar_names, method=f"ppxf-{args.library}",
                native_f_host=result.f_host_5100,
            )
            check = model.check_reconstruction()
            stellar_weight = float(result.diagnostics.get("stellar_weight_sum", 0.0))

            record.update({
                "reconstruction_passed": check.passed,
                "reconstruction_relative": check.relative_difference,
                "stellar_weight_sum": stellar_weight,
                "chi2": result.chi2,
                **residuals(result.fit, result.wavelength),
                "powerlaw_at_5100": contribution(model, "agn_smooth", window),
                "feii_at_5100": contribution(model, "feii", window),
                "balmer_at_5100": contribution(model, "balmer_cont", window),
            })

            if check.passed:
                fractions = model.canonical_fractions(window)
                record["f_host_cont"] = fractions["f_host_cont"]
                record["f_host_pseudo"] = fractions["f_host_pseudo"]
            else:
                record["f_host_cont"] = None
                record["f_host_pseudo"] = None

            def show(value, width=9, places=3):
                return f"{'--':>{width}}" if value is None else f"{value:>{width}.{places}f}"

            # M0's fraction is not interpreted; it is shown bracketed as a reminder.
            cont = record["f_host_cont"]
            cont_text = (
                f"{'(' + format(cont, '.2f') + ')':>9}" if stage == "M0" and cont is not None
                else show(cont)
            )
            print(
                f"{stage:<5}{stellar_weight:>9.3f}{cont_text}"
                f"{show(record['f_host_pseudo'], 10)}{show(record['chi2'], 8, 2)}"
                f"{show(record['rms_global'])}{show(record['rms_stellar_features'])}"
                f"{show(record['rms_hbeta'], 8)}{show(record['powerlaw_at_5100'])}"
                f"{show(record['feii_at_5100'], 8)}{show(record['balmer_at_5100'], 8)}"
                f"  {'PASS' if check.passed else 'FAIL'}"
            )
            rows.append(record)
        print()

    frame = pd.DataFrame(rows)
    out = root / "06_fitting" / "validation" / (
        f"ablation_ladder_{args.library.lower().replace('-', '')}{suffix}.csv"
    )
    frame.to_csv(out, index=False)

    print("=" * 78)
    print("STELLAR WEIGHT ACROSS THE LADDER")
    print("=" * 78)
    pivot = frame.pivot_table(index="object_id", columns="stage",
                              values="stellar_weight_sum", observed=True)
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("M0 host fractions are bracketed and NOT interpreted: a stars-only fit")
    print("forces stellar populations to explain nuclear continuum.")
    print(f"written -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
