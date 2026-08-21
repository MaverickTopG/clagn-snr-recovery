#!/usr/bin/env python
"""Stage 37 / D-099 — equal-transition Green-Yang protocol disagreement.

The D-094/D-095 contingency tables pool Monte Carlo realization cells. Fifty
realizations of one transition are repeated measurements on one AGN, not fifty
AGN, so a pooled disagreement fraction weights a transition by how many of its
draws happened to be classifiable under both protocols. That is the wrong unit
for a scientific statement about protocol dependence.

This module restates the *same frozen outcomes* with the transition as the unit.
Nothing is simulated, refit or reclassified: the only input is the immutable
D-094 classifier-outcome table, and the first thing this script does is
reproduce the frozen per-transition agreement table exactly before deriving
anything new.

No minimum both-classifiable count is imposed. None was prospectively frozen,
and inventing one after the outcomes are visible would be an outcome-conditioned
threshold. The denominator distribution is reported instead, so the reader can
see that some transitions contribute a single draw and others fifty.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

# D-095 statistical conventions, carried forward unchanged.
BOOTSTRAP_N = 5000
BOOTSTRAP_BASE = 314159
CLASSIFIABLE = ("CL", "NON_CL")
GREEN = "GREEN2022_FINAL"
YANG = "YANG2024_FINAL"


def derived_seed(*parts: object) -> int:
    """Deterministic child seed, identical construction to D-095."""
    payload = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def bootstrap_mean_ci(values: np.ndarray, label: str) -> tuple[float, float]:
    """Percentile CI for an equal-transition mean; transitions are the clusters."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(derived_seed(BOOTSTRAP_BASE, "d099", label))
    sampled = rng.choice(array, size=(BOOTSTRAP_N, array.size), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def both_classifiable_cells(outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per realization cell classifiable under *both* protocols."""
    subset = outcomes[outcomes["criterion_id"].isin([GREEN, YANG])]
    index = [
        "transition_id", "object_id", "reference_tier", "source_family",
        "arm", "snr", "realization",
    ]
    wide = subset.pivot(index=index, columns="criterion_id", values="classification").reset_index()
    both = wide[wide[YANG].isin(CLASSIFIABLE) & wide[GREEN].isin(CLASSIFIABLE)].copy()
    both["disagree"] = both[YANG] != both[GREEN]
    both["yang_cl_green_noncl"] = (both[YANG] == "CL") & (both[GREEN] == "NON_CL")
    both["green_cl_yang_noncl"] = (both[GREEN] == "CL") & (both[YANG] == "NON_CL")
    return both


def transition_disagreement(both: pd.DataFrame) -> pd.DataFrame:
    """d_i(s,a) for every transition with at least one both-classifiable draw."""
    keys = ["transition_id", "object_id", "reference_tier", "source_family", "arm", "snr"]
    grouped = both.groupby(keys, sort=True).agg(
        N_both_classifiable=("disagree", "size"),
        N_disagree=("disagree", "sum"),
        N_YANG_CL_GREEN_NONCL=("yang_cl_green_noncl", "sum"),
        N_YANG_NONCL_GREEN_CL=("green_cl_yang_noncl", "sum"),
    ).reset_index()
    grouped["N_agree"] = grouped["N_both_classifiable"] - grouped["N_disagree"]
    grouped["disagreement_i"] = grouped["N_disagree"] / grouped["N_both_classifiable"]
    return grouped


def verify_against_frozen(derived: pd.DataFrame, frozen_path: Path) -> dict[str, Any]:
    """Refuse to publish a restatement that does not reproduce the frozen table."""
    frozen = pd.read_csv(frozen_path)
    keys = ["transition_id", "arm", "snr"]
    merged = frozen.merge(derived, on=keys, suffixes=("_frozen", "_derived"), how="outer",
                          indicator=True)
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("Derived transition rows do not match the frozen D-094 table")
    checks = {}
    for column in ("N_both_classifiable", "N_agree", "N_disagree",
                   "N_YANG_CL_GREEN_NONCL", "N_YANG_NONCL_GREEN_CL"):
        checks[column] = bool(
            (merged[f"{column}_frozen"].astype(int) == merged[f"{column}_derived"].astype(int)).all()
        )
    checks["disagreement_i"] = bool(
        np.allclose(merged["disagreement_i_frozen"], merged["disagreement_i_derived"], atol=1e-12)
    )
    checks["row_count"] = len(frozen) == len(derived)
    if not all(checks.values()):
        raise RuntimeError(f"Frozen-table reproduction failed: {checks}")
    return checks


def summarise(transitions: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """Equal-transition summary beside the pooled realization fraction."""
    rows: list[dict[str, Any]] = []
    scopes = {
        "PRIMARY_GOLD": transitions["reference_tier"].eq("GOLD"),
        "GOLD_PLUS_SILVER": transitions["reference_tier"].isin(["GOLD", "SILVER"]),
    }
    pooled_scopes = {
        "PRIMARY_GOLD": pooled["reference_tier"].eq("GOLD"),
        "GOLD_PLUS_SILVER": pooled["reference_tier"].isin(["GOLD", "SILVER"]),
    }
    for scope, mask in scopes.items():
        scoped = transitions[mask]
        scoped_pooled = pooled[pooled_scopes[scope]]
        for (arm, snr), group in scoped.groupby(["arm", "snr"], sort=True):
            cells = scoped_pooled[(scoped_pooled["arm"] == arm) & (scoped_pooled["snr"] == snr)]
            d = group["disagreement_i"].to_numpy(dtype=float)
            n = group["N_both_classifiable"].to_numpy(dtype=int)
            low, high = bootstrap_mean_ci(d, f"disagreement|{scope}|{arm}|{snr}")
            rows.append({
                "selection_scope": scope,
                "arm": arm,
                "snr": int(snr),
                "N_transitions_defined": int(len(group)),
                "mean_disagreement_equal_transition": float(d.mean()),
                "ci_low": low,
                "ci_high": high,
                "median_disagreement": float(np.median(d)),
                "q25_disagreement": float(np.quantile(d, 0.25)),
                "q75_disagreement": float(np.quantile(d, 0.75)),
                "min_disagreement": float(d.min()),
                "max_disagreement": float(d.max()),
                "N_transitions_zero_disagreement": int((d == 0).sum()),
                "N_transitions_full_disagreement": int((d == 1).sum()),
                "denominator_min": int(n.min()),
                "denominator_q25": float(np.quantile(n, 0.25)),
                "denominator_median": float(np.median(n)),
                "denominator_q75": float(np.quantile(n, 0.75)),
                "denominator_max": int(n.max()),
                "N_transitions_denominator_lt_10": int((n < 10).sum()),
                "N_transitions_denominator_ge_25": int((n >= 25).sum()),
                "mean_yang_cl_green_noncl_share": float(
                    (group["N_YANG_CL_GREEN_NONCL"] / group["N_both_classifiable"]).mean()
                ),
                "mean_green_cl_yang_noncl_share": float(
                    (group["N_YANG_NONCL_GREEN_CL"] / group["N_both_classifiable"]).mean()
                ),
                "pooled_realization_cells": int(len(cells)),
                "pooled_realization_disagreements": int(cells["disagree"].sum()),
                "pooled_realization_disagreement": float(cells["disagree"].mean()),
                "bootstrap_n": BOOTSTRAP_N,
                "bootstrap_cluster": "transition_id",
                "bootstrap_seed_label": f"{BOOTSTRAP_BASE}|d099|disagreement|{scope}|{arm}|{snr}",
            })
    return pd.DataFrame(rows)


def main() -> int:
    root = project_root()
    d094 = root / "05_analysis" / "q1_production" / "d094"
    out = root / "05_analysis" / "derived"
    out.mkdir(parents=True, exist_ok=True)

    outcomes = pd.read_csv(d094 / "raw" / "classifier_outcomes_d094.csv")
    pooled = both_classifiable_cells(outcomes)
    transitions = transition_disagreement(pooled)
    checks = verify_against_frozen(
        transitions, d094 / "tables" / "yang_green_agreement_transition_d094.csv"
    )
    summary = summarise(transitions, pooled)

    transitions.to_csv(out / "green_yang_transition_disagreement_d099.csv", index=False)
    summary.to_csv(out / "green_yang_equal_transition_summary_d099.csv", index=False)
    (out / "green_yang_disagreement_provenance_d099.json").write_text(
        json.dumps(
            {
                "derivation": "RESTATEMENT_OF_IMMUTABLE_D094_OUTCOMES",
                "source_artifact": "05_analysis/q1_production/d094/raw/classifier_outcomes_d094.csv",
                "frozen_table_reproduced": checks,
                "unit_of_analysis": "transition (one physical AGN)",
                "inclusion_rule": "every transition with >=1 both-classifiable realization",
                "minimum_denominator_threshold": None,
                "minimum_denominator_rationale": (
                    "no minimum was prospectively frozen; imposing one after outcomes are "
                    "visible would be an outcome-conditioned threshold"
                ),
                "bootstrap": {
                    "draws": BOOTSTRAP_N,
                    "cluster": "transition_id",
                    "interval": "percentile 95% CI for the equal-transition mean",
                    "seed_base": BOOTSTRAP_BASE,
                    "seed_construction": "sha256('<base>|d099|disagreement|<scope>|<arm>|<snr>')[:8]",
                },
                "simulation_or_refit_performed": False,
            },
            indent=2,
        )
        + "\n"
    )

    print("=" * 96)
    print("D-099 | equal-transition Green-Yang protocol disagreement (frozen D-094 outcomes)")
    print("=" * 96)
    print(f"frozen per-transition table reproduced exactly: {all(checks.values())}")
    print()
    show = summary[summary.selection_scope.eq("PRIMARY_GOLD")]
    print(f"{'arm':<12}{'S/N':>4}{'N_tr':>6}{'mean':>8}{'95% CI':>18}{'median':>8}"
          f"{'IQR':>16}{'denom med':>11}{'denom rng':>12}{'pooled':>9}")
    for r in show.itertuples():
        print(
            f"{r.arm:<12}{r.snr:>4}{r.N_transitions_defined:>6}"
            f"{r.mean_disagreement_equal_transition:>8.3f}"
            f"{f'[{r.ci_low:.3f}, {r.ci_high:.3f}]':>18}"
            f"{r.median_disagreement:>8.3f}"
            f"{f'[{r.q25_disagreement:.2f}, {r.q75_disagreement:.2f}]':>16}"
            f"{r.denominator_median:>11.0f}"
            f"{f'{r.denominator_min}-{r.denominator_max}':>12}"
            f"{r.pooled_realization_disagreement:>9.3f}"
        )
    print()
    print("Realizations are repeated measurements on one AGN. The pooled column is the")
    print("D-094/D-095 realization-cell fraction, retained only as appendix accounting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
