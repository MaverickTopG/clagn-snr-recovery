#!/usr/bin/env python
"""Stage 45 -- three-state operational outcomes and operational recovery yield.

The manuscript's primary recovery estimand conditions on a classifiable
measurement: p_i = n_CL / n_classifiable. For the Green-statistic protocol that
conditioning is free, because no applicable realization was ever invalid. For a
fitted-line protocol it is not: most applicable realizations return no valid
measurement at all, so a conditional recovery fraction is computed over a
denominator that has already collapsed.

Reporting only the conditional fraction makes the two protocols look closer than
they are, and in the wrong direction. This stage adds the two quantities needed
to read them side by side:

    three-state decomposition   P(CL), P(non-CL), P(invalid) among APPLICABLE
                                realizations, which sum to one by construction

    operational recovery yield  Y_i = n_CL / n_applicable, the per-transition
                                rate a survey would actually observe if it kept
                                invalid measurements in the denominator

Y is not completeness. It is the yield of this protocol on this reference set
under this intervention, and it says nothing about undiscovered transitions.

Two further diagnostics separate baseline fragility from degradation response,
because the distinction matters for how the fitted-line result is described:

    native validity     how often the Yang fit is already invalid on the
                        undegraded archival bright spectrum

    rung validity       how faint-epoch fit validity moves between the rungs

Everything is a re-tabulation of frozen production records. Nothing is refitted
and no frozen output is modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

ROOT = project_root()
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
OUT = ROOT / "05_analysis" / "q1_production" / "d094" / "tables"

CLASSIFIED = ["CL", "NON_CL"]
PROTOCOLS = [("GREEN2022_FINAL", "Green-statistic"), ("YANG2024_FINAL", "Yang-ratio")]
CONDITIONS = [("faint_only", "Faint-only"), ("matched", "Matched")]


def _read_outcomes() -> pd.DataFrame:
    """Classifier outcomes ship as CSV here and as parquet in the public tree."""
    csv = RAW / "classifier_outcomes_d094.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")


def three_state(outcomes: pd.DataFrame, gold: set[str]) -> pd.DataFrame:
    """Outcome shares among applicable realizations, by protocol and condition."""
    rows = []
    for criterion, label in PROTOCOLS:
        for arm, arm_label in CONDITIONS:
            for snr in (5, 10):
                cell = outcomes[
                    outcomes.criterion_id.eq(criterion)
                    & outcomes.transition_id.isin(gold)
                    & outcomes.arm.eq(arm)
                    & outcomes.snr.eq(snr)
                    & outcomes.applicable
                ]
                n_app = len(cell)
                n_cl = int(cell.classification.eq("CL").sum())
                n_noncl = int(cell.classification.eq("NON_CL").sum())
                n_invalid = n_app - n_cl - n_noncl
                rows.append({
                    "protocol": label, "arm": arm_label, "snr": snr,
                    "n_applicable": n_app, "n_cl": n_cl, "n_noncl": n_noncl,
                    "n_invalid": n_invalid,
                    "p_cl": n_cl / n_app, "p_noncl": n_noncl / n_app,
                    "p_invalid": n_invalid / n_app,
                    "operational_yield": n_cl / n_app,
                })
    frame = pd.DataFrame(rows)
    shares = frame[["p_cl", "p_noncl", "p_invalid"]].sum(axis=1)
    assert shares.round(12).eq(1.0).all(), "three-state shares must sum to one"
    assert (frame.n_cl + frame.n_noncl + frame.n_invalid).equals(frame.n_applicable)
    assert frame.loc[frame.protocol.eq("Green-statistic"), "n_invalid"].eq(0).all()
    return frame


def yield_versus_conditional(outcomes: pd.DataFrame, gold: set[str]) -> pd.DataFrame:
    """Equal-transition conditional recovery against equal-transition yield.

    Both are means over transitions, so they differ only in the denominator each
    transition contributes: classifiable realizations for R, applicable ones for
    Y. Where a protocol never returns an invalid measurement the two coincide.
    """
    rows = []
    for criterion, label in PROTOCOLS:
        for arm, arm_label in CONDITIONS:
            for snr in (5, 10):
                cell = outcomes[
                    outcomes.criterion_id.eq(criterion)
                    & outcomes.transition_id.isin(gold)
                    & outcomes.arm.eq(arm)
                    & outcomes.snr.eq(snr)
                    & outcomes.applicable
                ].copy()
                cell["is_cl"] = cell.classification.eq("CL")
                classifiable = cell[cell.classification.isin(CLASSIFIED)]
                p_i = classifiable.groupby("transition_id").is_cl.mean()
                y_i = cell.groupby("transition_id").is_cl.mean()
                rows.append({
                    "protocol": label, "arm": arm_label, "snr": snr,
                    "N_tr_classifiable": len(p_i), "N_tr_applicable": len(y_i),
                    "R_conditional": p_i.mean(), "Y_operational": y_i.mean(),
                    "inflation_factor": p_i.mean() / y_i.mean() if y_i.mean() else float("nan"),
                })
    return pd.DataFrame(rows)


def validity_baseline() -> pd.DataFrame:
    """Fit validity on undegraded spectra against validity at each rung.

    Separates a protocol that is broken by added noise from one that could not
    measure these spectra to begin with.
    """
    columns = ["task_kind", "target_snr", "yang_fit_valid", "green_preprocessing_valid"]
    full = RAW / "spectrum_fit_results_d094.parquet"
    if full.exists():
        fits = pd.read_parquet(full, columns=columns)
    else:
        # `fit_validity_flags_d094.csv` is the same four columns, checked against
        # the full product by stage 47. The full product is 675 MB and records
        # absolute paths, so it is not redistributed.
        fits = pd.read_csv(RAW / "fit_validity_flags_d094.csv")[columns]
    rows = []
    native = fits[fits.task_kind.eq("native_bright")]
    rows.append({
        "spectra": "Native bright (undegraded)", "target_snr": "native",
        "n_fits": len(native),
        "yang_valid_fraction": native.yang_fit_valid.mean(),
        "green_valid_fraction": native.green_preprocessing_valid.mean(),
    })
    for kind, label in [("faint_shared", "Degraded faint"),
                        ("matched_bright", "Degraded bright (matched)")]:
        for snr, group in fits[fits.task_kind.eq(kind)].groupby("target_snr"):
            rows.append({
                "spectra": label, "target_snr": f"{int(snr)}",
                "n_fits": len(group),
                "yang_valid_fraction": group.yang_fit_valid.mean(),
                "green_valid_fraction": group.green_preprocessing_valid.mean(),
            })
    return pd.DataFrame(rows)


def heterogeneity(outcomes: pd.DataFrame, gold: set[str]) -> pd.DataFrame:
    """Shape of the paired Green response, which a mean alone does not convey."""
    rows = []
    green = outcomes[outcomes.criterion_id.eq("GREEN2022_FINAL") & outcomes.transition_id.isin(gold)]
    for arm, arm_label in CONDITIONS:
        cell = green[green.arm.eq(arm) & green.classification.isin(CLASSIFIED)].copy()
        cell["is_cl"] = cell.classification.eq("CL")
        p = cell.groupby(["transition_id", "snr"]).is_cl.mean().unstack()
        delta = (p[10] - p[5]).dropna().sort_values()
        top5 = delta.nlargest(5).sum() / delta.sum() if delta.sum() else float("nan")
        rows.append({
            "arm": arm_label, "N_tr": len(delta),
            "mean": delta.mean(), "median": delta.median(),
            "q25": delta.quantile(0.25), "q75": delta.quantile(0.75),
            "frac_abs_delta_below_0p05": (delta.abs() < 0.05).mean(),
            "frac_delta_at_least_0p2": (delta >= 0.2).mean(),
            "top5_share_of_total": top5,
        })
    return pd.DataFrame(rows)


def main() -> int:
    outcomes = _read_outcomes()
    gold = set(outcomes.loc[outcomes.reference_tier.eq("GOLD"), "transition_id"])

    states = three_state(outcomes, gold)
    yields = yield_versus_conditional(outcomes, gold)
    baseline = validity_baseline()
    spread = heterogeneity(outcomes, gold)

    for frame, name in [
        (states, "three_state_outcomes_d094.csv"),
        (yields, "operational_yield_d094.csv"),
        (baseline, "fit_validity_baseline_d094.csv"),
        (spread, "green_paired_heterogeneity_d094.csv"),
    ]:
        frame.to_csv(OUT / name, index=False)

    print("Three-state shares among applicable realizations")
    print(states[["protocol", "arm", "snr", "n_applicable", "p_cl", "p_noncl", "p_invalid"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nConditional recovery against operational yield")
    print(yields[["protocol", "arm", "snr", "R_conditional", "Y_operational", "inflation_factor"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nFit validity, undegraded against degraded")
    print(baseline.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nShape of the paired Green response")
    print(spread.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote 4 tables to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
