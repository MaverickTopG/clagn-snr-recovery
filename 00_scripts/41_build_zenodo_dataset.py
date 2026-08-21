#!/usr/bin/env python
"""Assemble the public Zenodo reproducibility DATASET (data only, no code).

Every table here is copied or deterministically extracted from a frozen result
file. Nothing is refit, reclassified or recomputed from spectra.

One correction is applied and is called out in the data dictionary: the D-095
paired-transition files carry `delta_*` columns whose values were serialized by
a defective writer and contain a stringified pandas Series rather than a number.
The `p_i` columns beside them are authoritative, so the differences are
recomputed from those by subtraction. This is arithmetic on frozen values, not a
reanalysis, and it is what the manuscript's own table builder has always done.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p3sf.config import project_root  # noqa: E402

ROOT = project_root()
D094 = ROOT / "05_analysis/q1_production/d094"
D095 = ROOT / "05_analysis/q1_production/d095/tables"
SUB = ROOT / "05_analysis/derived"
OUT = ROOT / "release/zenodo_dataset_v1.0.0"

DOI = "10.5281/zenodo.22022007"


def copy(src: Path, name: str) -> None:
    shutil.copyfile(src, OUT / name)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # -- A. reference sample -------------------------------------------------
    manifest = pd.read_csv(SUB / "reference_transition_manifest.csv")
    assert len(manifest) == 62, len(manifest)
    assert (manifest.reference_tier == "GOLD").sum() == 58
    manifest.to_csv(OUT / "reference_manifest.csv", index=False)

    # -- B. endpoint provenance ---------------------------------------------
    copy(D094 / "raw/endpoint_bindings_d094.csv", "endpoint_bindings.csv")

    # -- C. condition matrix -------------------------------------------------
    conditions = pd.read_csv(D094 / "raw/frozen_condition_matrix_d093.csv")
    conditions.to_csv(OUT / "condition_matrix.csv", index=False)

    # -- D. per-realization classification outcomes --------------------------
    outcomes = pd.read_csv(D094 / "raw/classifier_outcomes_d094.csv")
    assert len(outcomes) == 25050, len(outcomes)
    automated = outcomes[outcomes.criterion_id != "MACLEOD2019_FINAL"]
    assert len(automated) == 16700, len(automated)
    outcomes.to_csv(OUT / "classification_outcomes.csv", index=False)

    # -- E. transition-level estimands ---------------------------------------
    copy(D094 / "tables/transition_level_estimands_d094.csv", "transition_estimands.csv")

    # -- F. primary paired contrasts -----------------------------------------
    snr = pd.read_csv(D095 / "green_common_support_snr_paired_transitions_d095.csv")
    arm = pd.read_csv(D095 / "green_common_support_arm_paired_transitions_d095.csv")
    snr["delta_p_10_minus_5"] = snr.p_i_10.astype(float) - snr.p_i_5.astype(float)
    snr["delta_u_10_minus_5"] = snr.u_i_10.astype(float) - snr.u_i_5.astype(float)
    arm["delta_p_faint_only_minus_matched"] = (
        arm.p_i_faint_only.astype(float) - arm.p_i_matched.astype(float)
    )
    arm["delta_u_faint_only_minus_matched"] = (
        arm.u_i_faint_only.astype(float) - arm.u_i_matched.astype(float)
    )
    snr.insert(0, "contrast", "snr_10_minus_5")
    arm.insert(0, "contrast", "arm_faint_only_minus_matched")
    snr.to_csv(OUT / "paired_green_contrasts_snr.csv", index=False)
    arm.to_csv(OUT / "paired_green_contrasts_arm.csv", index=False)

    # -- G. protocol disagreement --------------------------------------------
    copy(SUB / "tables/green_yang_transition_disagreement_d099.csv", "protocol_disagreement.csv")

    # -- H. robustness / composition -----------------------------------------
    copy(D095 / "green_paired_robustness_d095.csv", "green_robustness.csv")
    copy(D094 / "tables/leave_one_transition_out_d094.csv", "green_leave_one_transition_out.csv")
    copy(D094 / "tables/leave_one_source_family_out_d094.csv", "green_leave_one_family_out.csv")
    copy(D094 / "tables/equal_source_family_weighted_d094.csv", "green_equal_family_weighted.csv")
    copy(D094 / "tables/source_family_estimands_d094.csv", "green_source_family_estimands.csv")
    copy(SUB / "tables/paired_green_strata_d100.csv", "green_instrument_direction_strata.csv")

    # -- I. Yang invalidity accounting ---------------------------------------
    copy(D095 / "yang_failure_decomposition_summary_d095.csv", "yang_invalidity_summary.csv")
    copy(D095 / "yang_failure_decomposition_rows_d095.csv", "yang_invalidity_rows.csv")
    copy(SUB / "tables/yang_failure_strata_d100.csv", "yang_invalidity_strata.csv")
    copy(D094 / "tables/unclassifiable_reasons_d094.csv", "unclassifiable_reasons.csv")

    # -- J. manuscript aggregate tables --------------------------------------
    copy(D094 / "tables/aggregate_estimands_bootstrap_d094.csv", "aggregate_recovery.csv")
    copy(D095 / "green_common_support_snr_summary_d095.csv", "paired_contrasts_summary_snr.csv")
    copy(D095 / "green_common_support_arm_summary_d095.csv", "paired_contrasts_summary_arm.csv")
    copy(SUB / "tables/green_yang_equal_transition_summary_d099.csv",
         "protocol_disagreement_summary.csv")
    copy(D095 / "yang_green_both_classifiable_contingency_d095.csv",
         "protocol_contingency_pooled.csv")
    copy(SUB / "tables/aggregate_monte_carlo_error_d100.csv", "monte_carlo_error.csv")

    # Production accounting, assembled from the frozen counts rather than typed.
    accounting = pd.DataFrame([
        {"quantity": "reference transitions (GOLD, primary)", "value": 58},
        {"quantity": "reference transitions (GOLD+SILVER, sensitivity)", "value": 62},
        {"quantity": "transition-arm-rung conditions", "value": int(len(conditions))},
        {"quantity": "condition-realizations", "value": 8350},
        {"quantity": "automated protocol evaluations", "value": int(len(automated))},
        {"quantity": "MacLeod inapplicability status rows", "value": 8350},
        {"quantity": "total stored classification records", "value": int(len(outcomes))},
        {"quantity": "noise realizations per condition (M)", "value": 50},
        {"quantity": "S/N rungs", "value": "5, 10"},
    ])
    accounting.to_csv(OUT / "production_accounting.csv", index=False)

    # -- K. randomization metadata -------------------------------------------
    (OUT / "randomization_metadata.txt").write_text(
        """Frozen stochastic design
========================

Noise realizations per condition (M): 50
Seed namespace:                       p3sf:q1:full:v1

Random streams are deterministic. A draw is identified by the transition, the
spectrum, the realization index and the TARGET S/N, and the seed is derived from
that key by SHA-256. The same key therefore reproduces the same spectrum
bit-for-bit on any machine.

Two consequences of including the target S/N in the key matter for interpreting
the results:

1. S/N 5 and S/N 10 use INDEPENDENT random streams. No transition-realization
   pair shares a draw between the two rungs. Conditional on the fixed archival
   spectrum, the two rungs are independent, which is why the finite-M variance
   of the paired S/N contrast adds the two binomial terms with no covariance
   term.

2. Corresponding faint draws ARE shared between the faint-only and matched arms
   at the same rung, wherever a transition supports both. This is deliberate:
   it makes the arm comparison paired rather than contaminated by Monte Carlo
   noise. The same formula therefore does NOT apply to the arm contrast, and no
   analogous finite-M figure is published for it.

The degradation adds independent Gaussian noise per native pixel with no
off-diagonal covariance, on the observed-frame native sampling, before any
protocol-specific processing. A target above a spectrum's native S/N is refused;
no spectrum is ever upgraded.

Analysis source code is not part of this deposit and is maintained separately.
"""
    )

    print(f"wrote {len(list(OUT.glob('*')))} files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
