#!/usr/bin/env python
"""Stage 48 -- resampling spread of the transition-level medians, and a
line-free continuum S/N diagnostic for the prespecified H-beta-window metric.

Two separate checks, both of which report a weakness rather than confirming an
absence of one.

MEDIAN SPREAD. The manuscript quotes sample medians of the paired difference
alongside the means. A mean of a bounded quantity over 25-27 units is far better
determined than its median, and quoting a bare median invites the reader to
treat it as a precise summary. This stage resamples transitions with the same
convention the published mean intervals use (5000 draws, seed derived from the
frozen base) and reports the interval for the median so the contrast in
precision is visible.

CONTINUUM S/N. The prespecified target metric is the median per-pixel f/sigma
over rest-frame 4700-5100 A, a window that contains H-beta itself. A spectrum
with a strong broad line therefore scores higher than an otherwise identical
spectrum without one, which affects both rung eligibility and the amount of
noise added to reach a target. This stage recomputes S/N over two conventional
line-free continuum windows on either side of the H-beta complex and reports
how closely the two track, and how often rung support would differ.

The point of the second check is NOT to show the metric choice does not matter.
It shows strong overall agreement together with non-negligible boundary
sensitivity near S/N 10, and both halves of that are reported.

Nothing is refitted and no frozen output is modified. The recomputed H-beta
metric is asserted against the frozen values as a correctness gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3sf.spectral_domain import snr_in_window  # noqa: E402

RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
OUT = ROOT / "05_analysis" / "q1_production" / "d094" / "tables"

BOOTSTRAP_N = 5000
BOOTSTRAP_BASE = 314159
CLASSIFIED = ["CL", "NON_CL"]
HBETA_WINDOW = (4700.0, 5100.0)
# Conventional line-free continuum windows either side of the H-beta complex,
# both strictly outside the analysis window and outside [O III] 4959/5007.
CONTINUUM_WINDOWS = {"blue_4435_4700": (4435.0, 4700.0), "red_5100_5535": (5100.0, 5535.0)}
RUNGS = (5.0, 10.0)


def derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _resample(values: np.ndarray, label: str, statistic) -> tuple[float, float]:
    rng = np.random.default_rng(derived_seed(BOOTSTRAP_BASE, "d095", label))
    drawn = rng.choice(values, size=(BOOTSTRAP_N, len(values)), replace=True)
    stat = statistic(drawn, axis=1)
    return float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))


def _read_outcomes() -> pd.DataFrame:
    """Classifier outcomes ship as CSV here and as parquet in the public tree."""
    csv = RAW / "classifier_outcomes_d094.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")


def median_spread() -> pd.DataFrame:
    """How well determined the quoted medians actually are."""
    outcomes = _read_outcomes()
    gold = set(outcomes.loc[outcomes.reference_tier.eq("GOLD"), "transition_id"])
    green = outcomes[outcomes.criterion_id.eq("GREEN2022_FINAL") & outcomes.transition_id.isin(gold)]
    rows = []
    for arm, label in [("faint_only", "Faint-only"), ("matched", "Matched")]:
        cell = green[green.arm.eq(arm) & green.classification.isin(CLASSIFIED)].copy()
        cell["is_cl"] = cell.classification.eq("CL")
        recovery = cell.groupby(["transition_id", "snr"]).is_cl.mean().unstack()
        delta = (recovery[10] - recovery[5]).dropna().to_numpy(dtype=float)
        mean_lo, mean_hi = _resample(delta, f"{arm}|mean", np.mean)
        med_lo, med_hi = _resample(delta, f"{arm}|median", np.median)
        rows.append({
            "arm": label, "N_tr": len(delta),
            "mean": float(delta.mean()), "mean_ci_low": mean_lo, "mean_ci_high": mean_hi,
            "mean_ci_width": mean_hi - mean_lo,
            "median": float(np.median(delta)), "median_ci_low": med_lo, "median_ci_high": med_hi,
            "median_ci_width": med_hi - med_lo,
        })
    frame = pd.DataFrame(rows)
    # The whole point: the median is the less well determined of the two.
    assert (frame.median_ci_width > frame.mean_ci_width).all()
    return frame


def continuum_snr() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Line-free continuum S/N against the prespecified H-beta-window metric."""
    spec = importlib.util.spec_from_file_location("s34", ROOT / "00_scripts" / "34_execute_full_q1.py")
    stage34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage34)

    manifest = pd.read_csv(RAW / "fit_task_manifest_d094.csv")
    endpoints = manifest.drop_duplicates(subset=["transition_id", "endpoint_role"])
    rows = []
    for record in endpoints.itertuples():
        spectrum = stage34._load_spectrum({
            "local_path": record.local_path, "instrument_survey": record.instrument_survey,
            "redshift": record.redshift, "spectrum_id": record.spectrum_id,
        })
        rest = spectrum.wavelength / (1.0 + spectrum.redshift)
        good = np.asarray(spectrum.mask) == 0
        entry = {"transition_id": record.transition_id, "endpoint_role": record.endpoint_role}
        entry["snr_hbeta_window"] = snr_in_window(
            rest, spectrum.flux, spectrum.error, HBETA_WINDOW, good=good).value
        for name, window in CONTINUUM_WINDOWS.items():
            entry[f"snr_{name}"] = snr_in_window(
                rest, spectrum.flux, spectrum.error, window, good=good).value
        rows.append(entry)
    measured = pd.DataFrame(rows)

    # Correctness gate: the recomputed metric must equal the frozen one.
    fits = pd.read_parquet(
        RAW / "spectrum_fit_results_d094.parquet",
        columns=["transition_id", "task_kind", "original_snr"])
    frozen = fits[fits.task_kind.eq("faint_shared")].groupby("transition_id").original_snr.first()
    mine = measured[measured.endpoint_role.eq("faint")].set_index("transition_id")["snr_hbeta_window"]
    aligned = mine.reindex(frozen.index)
    assert np.nanmax(np.abs(aligned - frozen)) < 1e-9, "recomputed H-beta metric disagrees with frozen"

    summary = []
    for name in CONTINUUM_WINDOWS:
        pair = measured[["snr_hbeta_window", f"snr_{name}"]].dropna()
        row = {
            "continuum_window": name, "n_endpoints": len(pair),
            "pearson_r": float(pair.corr().iloc[0, 1]),
            "spearman_rho": float(pair.corr(method="spearman").iloc[0, 1]),
            "median_ratio_hbeta_over_continuum":
                float((pair.snr_hbeta_window / pair[f"snr_{name}"]).median()),
        }
        for rung in RUNGS:
            by_hbeta = pair.snr_hbeta_window >= rung
            by_continuum = pair[f"snr_{name}"] >= rung
            row[f"support_hbeta_snr{int(rung)}"] = int(by_hbeta.sum())
            row[f"support_continuum_snr{int(rung)}"] = int(by_continuum.sum())
            row[f"support_disagree_snr{int(rung)}"] = int((by_hbeta != by_continuum).sum())
        summary.append(row)
    return measured, pd.DataFrame(summary)


def main() -> int:
    medians = median_spread()
    medians.to_csv(OUT / "paired_median_uncertainty_d094.csv", index=False)
    print("Resampling spread of the paired difference")
    print(medians.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # The continuum diagnostic remeasures the archival spectra, which are not
    # redistributed. Where they are absent the shipped table stands on its own.
    if not (RAW / "fit_task_manifest_d094.csv").exists():
        print("\nSkipping the continuum S/N diagnostic: it remeasures the archival endpoint")
        print("spectra, which are retrieved from the survey archives rather than shipped.")
        print(f"The committed summary is at {(OUT / 'continuum_snr_summary_d094.csv').relative_to(ROOT)}.")
        return 0

    measured, summary = continuum_snr()
    measured.to_csv(OUT / "continuum_snr_endpoints_d094.csv", index=False)
    summary.to_csv(OUT / "continuum_snr_summary_d094.csv", index=False)
    print("\nLine-free continuum S/N against the prespecified H-beta-window metric")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote 3 tables to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
