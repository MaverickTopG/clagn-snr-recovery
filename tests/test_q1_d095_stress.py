from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_analysis" / "q1_production" / "d095"
TABLES = OUT / "tables"


def test_d095_independent_accounting_passes_every_identity() -> None:
    accounting = json.loads((OUT / "production_accounting_reverification_d095.json").read_text())
    assert accounting["status"] == "PASS"
    assert all(accounting["checks"].values())
    assert accounting["counts"] == {
        "condition_realizations": 8350,
        "conditions": 167,
        "terminal_classifier_outcomes": 25050,
        "transition_condition_classifier_rows": 501,
    }


def test_d095_green_snr_uses_paired_common_support() -> None:
    summary = pd.read_csv(TABLES / "green_common_support_snr_summary_d095.csv")
    expected = {
        "faint_only": (27, 0.2896296296296296, 18, 8, 1),
        "matched": (25, 0.2592, 18, 6, 1),
    }
    for row in summary.itertuples(index=False):
        count, effect, positive, zero, negative = expected[row.arm]
        assert row.paired_transition_count == count
        assert np.isclose(row.delta_p_mean_10_minus_5, effect)
        assert (row.N_positive, row.N_zero, row.N_negative) == (
            positive,
            zero,
            negative,
        )
        assert row.paired_mean_ci_low > 0
        assert row.paired_mean_ci_high > row.paired_mean_ci_low
        assert np.isclose(
            row.common_support_paired_difference_10_minus_5,
            row.delta_p_mean_10_minus_5,
        )


def test_d095_green_arm_effect_is_pairwise_and_positive() -> None:
    summary = pd.read_csv(TABLES / "green_common_support_arm_summary_d095.csv")
    expected = {5: (43, 0.1967441860465116), 10: (25, 0.2448)}
    for row in summary.itertuples(index=False):
        count, effect = expected[row.snr]
        assert row.paired_transition_count == count
        assert np.isclose(row.delta_p_mean_faint_only_minus_matched, effect)
        assert row.paired_mean_ci_low > 0


def test_d095_yang_failure_categories_and_incident_are_frozen() -> None:
    yang = json.loads((OUT / "yang_failure_decomposition_d095.json").read_text())
    categories = yang["terminal_priority_categories"]
    assert yang["applicable_unclassifiable"] == 4456
    assert yang["inapplicable_by_design"] == 1300
    assert (
        sum(value for key, value in categories.items() if key != "inapplicable_by_design") == 4456
    )
    assert categories["native_boundary_failure"] == 550
    assert categories["degraded_fit_boundary_failure"] == 3387
    assert categories.get("nondetection_semantics", 0) == 0
    assert yang["valid_nondetection_semantics_used_in_classifiable_rows"] == 18

    incident = json.loads((OUT / "mask_contract_incident_audit_d095.json").read_text())
    assert incident["affected_count_reconstructed"] == 6305
    assert incident["affected_count_frozen_audit"] == 6305
    assert incident["unaffected_count_reconstructed"] == 2107
    assert incident["unaffected_instrument_counts"] == {
        "DESI_EDR": 1,
        "LAMOST_DR11": 2106,
    }
    assert incident["all_8350_degraded_task_seeds_match_frozen_sha256_construction"]
    assert incident["faint_crn_task_seed_array_identity_across_arms"]


def test_d095_status_and_claim_freeze() -> None:
    summary = json.loads((OUT / "d095_machine_summary.json").read_text())
    assert summary["decision"] == "Q1_SCIENCE_PARTIAL"
    claims = (OUT / "allowed_scientific_claims_d095.md").read_text()
    assert "STRONGLY_SUPPORTED" in claims
    assert "SUPPORTED_WITH_QUALIFICATION" in claims
    assert "NOT_SUPPORTED" in claims
