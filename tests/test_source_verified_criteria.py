"""Primary-paper semantics for the D-085 literature gate."""

from __future__ import annotations

import pytest

from p3sf.criteria.base import Label
from p3sf.criteria.source_verified import (
    AnalysisRole,
    ProtocolDisposition,
    apply_green2022,
    apply_guo_quick_screen,
    apply_macleod2019,
    guo_final_protocol_status,
    guo_fractional_flux_change,
    potts_villforth_protocol_status,
)


def test_macleod_requires_visual_transition_and_strictly_exceeds_three() -> None:
    assert apply_macleod2019(
        nsigma_hbeta=4.0, visual_broad_hbeta_appearance_or_disappearance=True
    ).label is Label.CL
    assert apply_macleod2019(
        nsigma_hbeta=3.0, visual_broad_hbeta_appearance_or_disappearance=True
    ).label is Label.NON_CL
    assert apply_macleod2019(
        nsigma_hbeta=8.0, visual_broad_hbeta_appearance_or_disappearance=False
    ).label is Label.NON_CL


def test_macleod_missing_visual_or_measurement_is_unclassifiable() -> None:
    assert apply_macleod2019(
        nsigma_hbeta=8.0, visual_broad_hbeta_appearance_or_disappearance=None
    ).label is Label.UNCLASSIFIABLE
    assert apply_macleod2019(
        nsigma_hbeta=None, visual_broad_hbeta_appearance_or_disappearance=True
    ).label is Label.UNCLASSIFIABLE


def test_green_catalogue_note_includes_exactly_three() -> None:
    assert apply_green2022(nsigma_hbeta=3.0).label is Label.CL
    assert apply_green2022(nsigma_hbeta=2.999).label is Label.NON_CL
    assert apply_green2022(nsigma_hbeta=None).label is Label.UNCLASSIFIABLE


def test_guo_r_is_fractional_change_not_bright_to_dim_ratio() -> None:
    assert guo_fractional_flux_change(25.0, 10.0) == pytest.approx(1.5)
    assert guo_fractional_flux_change(10.0, 0.0) is None
    assert guo_fractional_flux_change(10.0, -1.0) is None


def test_guo_quick_screen_requires_both_strict_cuts_on_same_line() -> None:
    split = apply_guo_quick_screen(
        {"Hbeta": (4.0, 1.0), "Halpha": (2.0, 2.0)}
    )
    assert split.disposition is ProtocolDisposition.FAIL
    passed = apply_guo_quick_screen({"Hbeta": (3.01, 1.5001)})
    assert passed.disposition is ProtocolDisposition.PASS
    assert passed.role is AnalysisRole.SEARCH_STAGE
    assert not passed.counts_toward_classifier_denominator


def test_guo_threshold_equality_does_not_pass_and_missing_is_not_fail() -> None:
    assert apply_guo_quick_screen({"Hbeta": (3.0, 2.0)}).disposition is ProtocolDisposition.FAIL
    assert apply_guo_quick_screen({}).disposition is ProtocolDisposition.UNCLASSIFIABLE


def test_guo_quick_screen_cannot_stand_in_for_final_protocol() -> None:
    result = guo_final_protocol_status(all_published_stages_available=False)
    assert result.role is AnalysisRole.CONFIRMATION_PROTOCOL
    assert result.disposition is ProtocolDisposition.UNCLASSIFIABLE
    assert "quick screen alone" in result.reason


def test_potts_is_never_coerced_into_a_final_scalar_label() -> None:
    result = potts_villforth_protocol_status(measurements_available=True)
    assert result.role is AnalysisRole.SEARCH_STAGE
    assert result.disposition is ProtocolDisposition.UNCLASSIFIABLE
    assert not result.counts_toward_classifier_denominator
    assert "not uniquely specified" in result.reason
