"""D-087 Q1 eligibility, matrix, and object-level denominator freeze."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from p3sf.design.q1 import Q1Validity
from p3sf.stats.q1 import Q1Counts, summarise_q1_outcomes, summarise_q1_realizations

ROOT = Path(__file__).resolve().parents[1]


def test_q1_eligibility_sets_are_exact_and_do_not_use_later_substitutes() -> None:
    frame = pd.read_csv(ROOT / "05_analysis/q1_design/q1_eligibility_d087.csv")
    assert len(frame) == 17
    assert frame.q1_primary_eligible.sum() == 4
    assert frame.q1_sensitivity_eligible.sum() == 8
    assert set(frame.loc[frame.q1_primary_eligible, "reference_tier"]) == {"GOLD"}
    assert set(frame.loc[frame.q1_sensitivity_eligible, "reference_tier"]) <= {
        "GOLD", "SILVER"
    }
    missing = frame.loc[~frame.exact_faint_endpoint_local]
    assert not missing.q1_primary_eligible.any()
    assert not missing.q1_sensitivity_eligible.any()
    assert not frame.loc[frame.transition_id.str.contains("J123359"), "q1_sensitivity_eligible"].item()


def test_q1_matrix_is_provisional_unexecuted_and_preserves_arms() -> None:
    matrix = pd.read_csv(ROOT / "05_analysis/q1_design/proposed_q1_matrix_d087_unexecuted.csv")
    assert len(matrix) == 40
    assert matrix.loc[matrix.reference_tier == "GOLD"].shape[0] == 18
    assert set(matrix.arm) == {"faint_only", "matched"}
    assert set(matrix.snr_rung) == {5, 10, 15, 20}
    assert not matrix.execution_authorized.any()
    assert not matrix.spectra_generated.any()
    assert (matrix.grid_status == "DEVELOPMENT_ONLY_PROVISIONAL").all()
    assert (matrix.loc[matrix.arm == "faint_only", "bright_epoch_action"] == "UNCHANGED").all()
    assert (matrix.loc[matrix.arm == "matched", "bright_epoch_action"] == "DEGRADE_TO_RUNG").all()
    assert matrix.object_specific_crn.all()


def test_d088_final_grid_matrix_is_exact_and_still_unexecuted() -> None:
    matrix = pd.read_csv(
        ROOT / "05_analysis/q1_design/proposed_final_q1_matrix_d088_unexecuted.csv"
    )
    assert len(matrix) == 30
    assert len(matrix.loc[matrix.reference_tier == "GOLD"]) == 14
    assert set(matrix.snr_rung) == {5, 10}
    assert set(matrix.arm) == {"faint_only", "matched"}
    assert not matrix.execution_authorized.any()
    assert not matrix.spectra_generated.any()
    assert (matrix.grid_status == "FINAL_FROZEN_D088_UNEXECUTED").all()
    assert (matrix.loc[matrix.arm == "faint_only", "bright_epoch_action"] == "UNCHANGED").all()


def test_d088_support_is_bound_to_hbeta_and_reports_sparse_rungs() -> None:
    support = pd.read_csv(ROOT / "05_analysis/q1_design/rung_support_d088.csv")
    assert support.loc[support.snr_rung == 5, "gold_faint_only"].item() == 4
    assert support.loc[support.snr_rung == 10, "gold_faint_only"].item() == 3
    assert support.loc[support.snr_rung == 15, "gold_faint_only"].item() == 1
    assert support.loc[support.snr_rung == 30, "gold_matched"].item() == 0
    assert support.loc[support.snr_rung == 40, "d088_disposition"].item() == (
        "UNREACHABLE_REMOVED"
    )


def test_d091_expanded_support_grid_and_matrix_are_frozen_unexecuted() -> None:
    support = pd.read_csv(ROOT / "05_analysis/q1_design/expanded_rung_support_d091.csv")
    matrix = pd.read_csv(
        ROOT / "05_analysis/q1_design/proposed_expanded_final_q1_matrix_d091_unexecuted.csv"
    )
    assert support.gold_n.eq(10).all()
    assert support.gold_plus_silver_n.eq(14).all()
    assert support.set_index("snr_rung").gold_matched.to_dict() == {
        5: 10, 10: 8, 15: 3, 20: 1, 30: 0, 40: 0,
    }
    assert set(support.loc[support.d091_disposition == "FINAL_RETAINED", "snr_rung"]) == {5, 10}
    assert len(matrix) == 52
    assert len(matrix[matrix.reference_tier == "GOLD"]) == 36
    assert set(matrix.arm) == {"faint_only", "matched"}
    assert set(matrix.snr_rung) == {5, 10}
    assert matrix.mc_realizations.eq(50).all()
    assert not matrix.execution_authorized.any()
    assert not matrix.spectra_generated.any()


def test_d091_workload_and_robustness_units_preserve_independence() -> None:
    workload = pd.read_csv(ROOT / "05_analysis/q1_design/expanded_q1_workload_d091.csv").iloc[0]
    robustness = pd.read_csv(
        ROOT / "05_analysis/q1_design/expanded_q1_robustness_freeze_d091.csv"
    )
    criteria = pd.read_csv(
        ROOT / "05_analysis/q1_design/expanded_q1_classifier_applicability_d091.csv"
    )
    assert workload.condition_realization_count == 2600
    assert workload.unique_reusable_spectrum_fit_inputs == 2614
    assert workload.unique_reusable_fit_count == 5228
    assert workload.classifier_evaluation_count == 7800
    assert len(robustness[robustness.summary == "LEAVE_ONE_TRANSITION_OUT"]) == 10
    assert len(robustness[robustness.summary == "LEAVE_ONE_SOURCE_FAMILY_OUT"]) == 6
    search = criteria[criteria.criterion.isin({
        "MACLEOD2016_SEARCH", "GUO_QUICK_SEARCH", "POTTS_VILLFORTH_SEARCH"
    })]
    assert search.denominator_semantics.eq("NON_DENOMINATOR").all()


def test_q1_counts_enforce_complete_denominator_accounting() -> None:
    count = Q1Counts(10, "faint_only", "green2022_hbeta", 5, 4, 3, 2, 1, 1, 3)
    assert count.empirical_recovery_fraction == 0.5
    assert count.unclassifiable_fraction == 0.6
    with pytest.raises(ValueError, match="must nest"):
        Q1Counts(10, "matched", "x", 2, 3, 2, 2, 1, 1, 0)
    with pytest.raises(ValueError, match="N_CL"):
        Q1Counts(10, "matched", "x", 2, 2, 2, 2, 2, 1, 0)


def test_q1_repeated_realizations_do_not_inflate_object_counts() -> None:
    rows = []
    for realization in range(3):
        rows.append({
            "object_id": "A", "snr_rung": 10, "arm": "faint_only",
            "criterion": "green", "fit_completed": True, "fit_valid": True,
            "label": "CL", "realization": realization,
        })
    rows.append({
        "object_id": "B", "snr_rung": 10, "arm": "faint_only",
        "criterion": "green", "fit_completed": False, "fit_valid": False,
        "label": "unclassifiable", "realization": 0,
    })
    result = summarise_q1_outcomes(pd.DataFrame(rows)).iloc[0]
    assert result.n_eligible == 2
    assert result.n_fit_completed == 1
    assert result.n_fit_valid == 1
    assert result.n_classifiable == 1
    assert result.n_cl == 1
    assert result.n_unclassifiable == 1


def test_q1_validity_chain_fails_closed() -> None:
    status = Q1Validity(True, True, True, True, True, False, False, False)
    assert status.terminal_status == "REQUIRED_MEASUREMENT_UNAVAILABLE"
    with pytest.raises(ValueError, match="must be monotone"):
        Q1Validity(True, True, True, False, True, True, True, True)


def test_q1_realizations_receive_equal_transition_weight() -> None:
    rows = []
    # A has 4/4 CL; B has 0/2 CL and two unclassifiable draws. Pooling the six
    # classifiable rows would give 2/3, while equal-object weighting must give 1/2.
    for realization in range(4):
        rows.append({"object_id": "A", "snr_rung": 5, "arm": "faint_only",
                     "criterion": "yang", "realization": realization, "label": "CL"})
    for realization, label in enumerate(["non-CL", "non-CL", "unclassifiable", "unclassifiable"]):
        rows.append({"object_id": "B", "snr_rung": 5, "arm": "faint_only",
                     "criterion": "yang", "realization": realization, "label": label})
    transition, aggregate = summarise_q1_realizations(pd.DataFrame(rows))
    assert transition.set_index("object_id").loc["B", "u_i_unclassifiable"] == 0.5
    assert aggregate.iloc[0].R_equal_object_mean == 0.5
    assert aggregate.iloc[0].U_equal_object_mean == 0.25
