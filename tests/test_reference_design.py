"""D-086 reference truth, pair independence, and unexecuted Q2 matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from p3sf.design.reference import (
    assert_noop_identity,
    validate_pair_eligibility,
    validate_reference_tiers,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reference_tiers_are_unique_blind_and_counted_as_frozen() -> None:
    tiers = pd.read_csv(ROOT / "04_reference_sample/reference_tiers_d086.csv")
    validate_reference_tiers(tiers)
    assert len(tiers) == 17
    assert tiers.paper3_reference_tier.value_counts().to_dict() == {
        "SILVER": 9,
        "GOLD": 4,
        "BORDERLINE": 4,
    }


def test_legacy_gold_did_not_automatically_become_paper3_gold() -> None:
    tiers = pd.read_csv(ROOT / "04_reference_sample/reference_tiers_d086.csv")
    assert (tiers.paper3_reference_tier != "GOLD").sum() == 13


def test_pair_eligibility_is_one_row_per_transition_and_only_j082_is_physical() -> None:
    pairs = pd.read_csv(ROOT / "05_analysis/full_q2_design/pair_eligibility_d086.csv")
    validate_pair_eligibility(pairs)
    eligible = pairs[pairs.pair_eligibility == "PAIR_CONSISTENT_ELIGIBLE"]
    assert eligible.object_id.tolist() == ["SDSSJ082942.66+415436.8"]
    assert pairs.pair_eligibility.value_counts().to_dict() == {
        "SPECTRUM_DIAGNOSTIC_ONLY": 10,
        "OUT_OF_DOMAIN": 6,
        "PAIR_CONSISTENT_ELIGIBLE": 1,
    }


def test_gold_set_is_exact_and_not_expanded_to_hit_a_target_n() -> None:
    primary = pd.read_csv(ROOT / "04_reference_sample/primary_reference_set_d086.csv")
    assert set(primary.object_id) == {
        "SDSSJ082942.66+415436.8",
        "SDSSJ102152.34+464515.6",
        "SDSSJ105325.40+302419.34",
        "SDSSJ233602.98+001728.7",
    }
    assert primary.primary_q2_pair_consistent.sum() == 1


def test_execution_matrix_is_exactly_ten_unexecuted_pair_consistent_cells() -> None:
    matrix = pd.read_csv(
        ROOT / "05_analysis/full_q2_design/final_execution_matrix_d086_unexecuted.csv"
    )
    assert len(matrix) == 10
    assert matrix.object_id.nunique() == 1
    assert set(matrix.semantics) == {"PAIR_CONSISTENT"}
    assert matrix.shared_absolute_amplitude_across_pair.all()
    assert not matrix.spectra_generated.any()
    assert not matrix.execution_authorized.any()
    assert not matrix.average_across_H.any()
    assert matrix.host_shape_weight.isna().all()
    assert (matrix.host_shape_id == "NONE").sum() == 1
    nonzero = matrix[matrix.requested_f_added_star_anchor > 0]
    assert set(nonzero.requested_f_added_star_anchor) == {0.1, 0.5, 0.85}
    assert set(nonzero.host_shape_id) == {"XSL_H1", "XSL_H2", "XSL_H3"}
    assert not nonzero.duplicated(["requested_f_added_star_anchor", "host_shape_id"]).any()


def test_noop_identity_is_an_engineering_gate() -> None:
    authoritative = {"wavelength": "a", "flux": "b", "ivar": "c", "mask": "d"}
    assert_noop_identity(authoritative, authoritative.copy())
    changed = authoritative | {"ivar": "different"}
    with pytest.raises(RuntimeError, match="NOOP_IDENTITY_FAIL"):
        assert_noop_identity(authoritative, changed)


def test_reference_validation_rejects_classifier_peeking() -> None:
    frame = pd.DataFrame(
        [{
            "object_id": "X",
            "paper3_reference_tier": "GOLD",
            "adjudication_basis": "raw spectra",
            "d085_classifier_outputs_inspected": True,
        }]
    )
    with pytest.raises(ValueError, match="circular"):
        validate_reference_tiers(frame)


def test_d090_expansion_counts_are_prospective_and_do_not_mutate_d086() -> None:
    candidates = pd.read_csv(
        ROOT / "04_reference_sample/q1_expansion_candidates_d090.csv"
    )
    current = pd.read_csv(ROOT / "04_reference_sample/reference_tiers_d086.csv")
    assert len(candidates) == 51
    assert candidates.object.is_unique
    assert set(candidates.object).isdisjoint(current.object_id)

    additional_gold = candidates[
        (candidates.proposed_reference_tier == "GOLD")
        & candidates.q1_eligible.str.startswith("PROVISIONAL_YES")
    ]
    additional_silver = candidates[
        (candidates.proposed_reference_tier == "SILVER")
        & (candidates.q1_eligible == "SENSITIVITY_CANDIDATE")
    ]
    assert len(additional_gold) == 6
    assert len(additional_silver) == 19
    assert additional_gold.exact_endpoints.eq("YES").all()
    assert additional_gold.publicly_retrievable.eq("YES").all()
    assert additional_gold.hbeta_coverage.eq("YES").all()
    assert current.paper3_reference_tier.value_counts().to_dict()["GOLD"] == 4


def test_d090_never_counts_later_substitutes_as_q1_endpoints() -> None:
    candidates = pd.read_csv(
        ROOT / "04_reference_sample/q1_expansion_candidates_d090.csv"
    )
    known_later_substitutes = candidates[candidates.object.isin({
        "SDSSJ105513.88+242553.69",
        "SDSSJ111329.68+531338.78",
        "SDSSJ143455.30+572345.10",
    })]
    assert len(known_later_substitutes) == 3
    assert known_later_substitutes.q1_eligible.eq("NO").all()
    assert known_later_substitutes.exclusion_reason.str.contains(
        "substitute", case=False
    ).all()


_FULL_TREE = (Path(__file__).resolve().parents[1] / "03_spectra" / "raw_sdss").is_dir()
_needs_full_tree = pytest.mark.skipif(
    not _FULL_TREE,
    reason="requires the full development tree, including raw survey spectra",
)


@_needs_full_tree
def test_d091_exact_endpoint_acquisition_is_bounded_unique_and_identity_valid() -> None:
    manifest = pd.read_csv(
        ROOT / "03_spectra/raw_sdss/manifest_d091_expansion.csv"
    )
    identity = pd.read_csv(
        ROOT / "05_analysis/q1_design/endpoint_identity_validation_d091.csv"
    )
    expected = {
        "403-51871-549", "3609-55201-524", "661-52163-604", "2878-54465-377",
        "945-52652-22", "8181-57073-827", "387-51791-110", "669-52559-306",
        "1670-54553-73", "1670-53438-61", "405-51816-458", "9383-58097-829",
    }
    assert len(manifest) == len(identity) == 12
    assert manifest.object.nunique() == 6
    assert set(manifest.spectrum_id) == expected
    assert manifest.local_filename.is_unique
    assert manifest.sha256.is_unique
    assert manifest.unique_file.all() and manifest.unique_checksum.all()
    assert manifest.byte_size.gt(0).all()
    assert identity.identity_pass.all()
    assert identity.expected_product_structure.all()
    assert identity.coordinate_consistent.all()
    assert identity.redshift_consistent.all()


def test_d091_reference_evidence_and_native_endpoint_validity_are_separate() -> None:
    validity = pd.read_csv(
        ROOT / "05_analysis/q1_design/added_gold_validity_d091.csv"
    )
    qc = pd.read_csv(ROOT / "05_analysis/q1_design/native_qc_d091.csv")
    green = pd.read_csv(
        ROOT / "05_analysis/q1_design/green_native_feasibility_d091.csv"
    )
    assert len(validity) == 6
    assert validity.reference_evidence_status.eq("GOLD_REFERENCE_EVIDENCE_PASS").all()
    assert validity.q1_native_endpoint_valid.all()
    assert validity.q1_eligible.all()
    assert len(qc) == len(green) == 12
    assert qc.qc_pass.all() and qc.degradation_source_eligible.all()
    assert qc.covers_hbeta.all()
    assert green.preprocessing_valid.all()
    assert green.pair_zero_flux_feasibility_available.all()


def test_d092_source_universe_is_bounded_exhausted_and_fully_classified() -> None:
    families = pd.read_csv(
        ROOT / "04_reference_sample/q1_completeness_source_family_screen_d092.csv"
    )
    allowed = {
        "SCREENED_D090",
        "SCREENED_EARLIER",
        "DUPLICATIVE",
        "OUT_OF_Q1_DOMAIN",
        "UNSCREENED_RELEVANT",
    }
    assert len(families) == 31
    assert families.source_family.is_unique
    assert set(families.initial_classification) <= allowed
    assert families.initial_classification.value_counts().to_dict() == {
        "SCREENED_D090": 12,
        "UNSCREENED_RELEVANT": 9,
        "DUPLICATIVE": 5,
        "OUT_OF_Q1_DOMAIN": 4,
        "SCREENED_EARLIER": 1,
    }
    newly_screened = families[
        families.initial_classification == "UNSCREENED_RELEVANT"
    ]
    assert len(newly_screened) == 9
    assert newly_screened.final_audit_state.str.startswith("SCREENED_D092").all()
    out = families[families.initial_classification == "OUT_OF_Q1_DOMAIN"]
    assert out.reason.notna().all() and out.reason.str.len().gt(30).all()
    assert out.out_of_domain_records.sum() == 757


def test_d092_exact_gold_manifest_is_prospective_bounded_and_identifier_complete() -> None:
    manifest = pd.read_csv(
        ROOT / "04_reference_sample/q1_completeness_gold_acquisition_d092.csv"
    )
    families = pd.read_csv(
        ROOT / "04_reference_sample/q1_completeness_source_family_screen_d092.csv"
    )
    assert len(manifest) == manifest.object.nunique() == 54
    assert manifest.source_family.value_counts().to_dict() == {
        "Dong2025_SDSS_LAMOST": 37,
        "Zeltyn2024_SDSSV": 11,
        "Yang2025_turn_on": 6,
    }
    assert manifest.status.eq("ACQUIRE_VALIDATE_ONLY").all()
    assert manifest.hbeta_coverage.eq("YES").all()
    assert manifest.public_retrieval.str.startswith("YES_").all()
    assert manifest.endpoint_1_identifier.str.len().gt(20).all()
    assert manifest.endpoint_2_identifier.str.len().gt(20).all()
    assert set(manifest.endpoint_1_role) == {"bright", "faint"}
    assert set(manifest.endpoint_2_role) == {"bright", "faint"}
    assert (manifest.endpoint_1_role != manifest.endpoint_2_role).all()
    assert families.additional_exact_gold_unique.sum() == 54
    assert families.additional_exact_silver.sum() == 74
    assert families.prospective_gold_morphology_screen.sum() == 64
    assert families.duplicate_gold_events.sum() == 2
    assert families.unrecoverable_gold_endpoints.sum() == 8


def test_d092_does_not_mutate_the_validated_d091_reference_count() -> None:
    validity = pd.read_csv(
        ROOT / "05_analysis/q1_design/added_gold_validity_d091.csv"
    )
    prospective = pd.read_csv(
        ROOT / "04_reference_sample/q1_completeness_gold_acquisition_d092.csv"
    )
    assert len(validity) == 6
    assert validity.q1_eligible.sum() == 6
    assert len(prospective) == 54
    assert prospective.status.ne("VALIDATED_GOLD").all()
    assert 4 + validity.q1_eligible.sum() == 10
    assert 10 + len(prospective) == 64


def test_d093_acquires_exactly_the_frozen_108_endpoints_with_provenance() -> None:
    manifest = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/endpoint_acquisition_manifest_d093.csv"
    )
    assert len(manifest) == 108
    assert manifest.transition_id.nunique() == 54
    assert not manifest.duplicated(["transition_id", "physical_role"]).any()
    assert manifest.byte_size.gt(0).all()
    assert manifest.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert manifest.local_filename.str.len().gt(0).all()
    assert manifest.immutable_source_identifier.str.len().gt(20).all()
    assert set(manifest.instrument_survey) == {
        "SDSS_LEGACY", "LAMOST_DR11", "SDSSV_DR19", "DESI_EDR"
    }


def test_d093_identity_and_native_qc_fail_closed_without_demoting_gold() -> None:
    identity = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/endpoint_identity_validation_d093.csv"
    )
    qc = pd.read_csv(ROOT / "05_analysis/q1_design/d093/native_qc_d093.csv")
    validity = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/new_gold_validity_d093.csv"
    )
    pair_identity = identity.groupby("transition_id").endpoint_identity_pass.all()
    assert pair_identity.sum() == 50
    assert set(pair_identity[~pair_identity].index) == {
        "J151143.41+210104.0", "J1311+0705",
        "J020649.48-041452.7", "J221026.83-001721.1",
    }
    assert qc.native_qc_pass.sum() == 106
    assert set(qc.loc[~qc.native_qc_pass, "transition_id"]) == {
        "J000253.52+210109.9", "J131001.32+143705.2"
    }
    assert len(validity) == 54 and validity.q1_eligible.sum() == 48
    assert validity.gold_reference_evidence.eq("GOLD_REFERENCE_EVIDENCE").all()
    assert set(validity.loc[~validity.q1_eligible, "transition_id"]) == {
        "J000253.52+210109.9", "J131001.32+143705.2",
        "J151143.41+210104.0", "J1311+0705",
        "J020649.48-041452.7", "J221026.83-001721.1",
    }


def test_d093_instrument_domains_and_discovery_bias_are_prospectively_frozen() -> None:
    instruments = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/instrument_domain_validation_d093.csv"
    )
    independence = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/discovery_classifier_independence_d093.csv"
    )
    assert len(instruments) == 4
    assert instruments.noise_degradation_operator_compatible.all()
    assert instruments.yang_contract_compatible.all()
    assert instruments.green_contract_compatible.all()
    assert instruments.empirical_uncertainty_correction.eq("NONE").all()
    assert independence.source_family.nunique() == 10
    assert set(independence.classifier_independence_status) == {
        "DISCOVERY_INDEPENDENT",
        "PARTIALLY_METHOD_OVERLAPPING",
        "DIRECTLY_CRITERION_SELECTED",
    }
    direct = independence[
        independence.classifier_independence_status == "DIRECTLY_CRITERION_SELECTED"
    ]
    assert direct[["source_family", "final_classifier"]].values.tolist() == [
        ["Green2022", "GREEN2022_FINAL"]
    ]
    assert not direct.discovery_independent_sensitivity_included.any()


def test_d093_final_sample_support_and_matrix_are_exact_and_unexecuted() -> None:
    gold = pd.read_csv(ROOT / "05_analysis/q1_design/d093/final_gold_sample_d093.csv")
    sensitivity = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/final_sensitivity_sample_d093.csv"
    )
    support = pd.read_csv(ROOT / "05_analysis/q1_design/d093/final_snr_support_d093.csv")
    matrix = pd.read_csv(
        ROOT / "05_analysis/q1_design/d093/final_q1_matrix_d093_unexecuted.csv"
    )
    workload = pd.read_csv(ROOT / "05_analysis/q1_design/d093/final_q1_workload_d093.csv")
    assert len(gold) == 58 and len(sensitivity) == 62
    assert gold.source_family.value_counts().to_dict()["Dong2025_SDSS_LAMOST"] == 34
    all_gold = support[support.scope == "ALL_FINAL_GOLD"]
    expected = {
        ("faint_only", 5): 48, ("faint_only", 10): 29,
        ("faint_only", 15): 14, ("faint_only", 20): 7,
        ("faint_only", 30): 1, ("faint_only", 40): 0,
        ("matched", 5): 47, ("matched", 10): 27,
        ("matched", 15): 12, ("matched", 20): 7,
        ("matched", 30): 1, ("matched", 40): 0,
    }
    assert {
        (row.arm, row.snr_rung): row.n_supported
        for row in all_gold.itertuples()
    } == expected
    assert len(matrix) == 167
    assert set(matrix.snr_rung) == {5, 10}
    assert not matrix.execution_authorized.any()
    assert not matrix.spectra_generated.any()
    row = workload.iloc[0]
    assert row.identity_valid_exact_pairs == 50
    assert row.endpoint_valid_new_gold == 48
    assert row.final_gold_n == 58 and row.final_sensitivity_n == 62
    assert row.condition_realizations == 8350
    assert row.reusable_spectrum_fit_inputs == 8412
    assert row.unclassifiable_by_design_cells == 205
