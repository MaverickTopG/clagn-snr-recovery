"""Published CLAGN criteria, the three-class outcome, and the verification gate.

Every criterion is checked against a hand-calculated case, per specification
§75: "Known hand-calculated spectrum pair gives expected classification."
"""

from __future__ import annotations

import pytest
from factories import measurement

from p3sf.criteria.base import (
    EpochPair,
    Label,
    UnverifiedCriterionError,
    require_verified,
    verified_criterion_families,
)
from p3sf.criteria.metrics import compute_metrics, signed_significance
from p3sf.criteria.published import (
    apply_difference_spectrum,
    apply_flux_ratio,
    apply_nsigma,
    apply_type_change,
    flux_ratio_statistic,
    nsigma_statistic,
)

OFF = {"enforce_verified": False}


def pair(early, late, *, object_id: str = "OBJ1") -> EpochPair:
    return EpochPair(
        object_id=object_id, early=early, late=late, early_mjd=52000.0, late_mjd=58000.0
    )


# ---------------------------------------------------------------------------
# the literature verification gate
# ---------------------------------------------------------------------------


def test_only_source_specific_criterion_or_protocol_keys_are_verified() -> None:
    """Verification must not unlock the old generic A/D approximations."""
    assert verified_criterion_families() == frozenset(
        {
            "A_PIXEL",
            "B_FLUX_RATIO",
            "GUO_PROTOCOL",
            "POTTS_PROTOCOL",
            "SEARCH_PROTOCOL",
        }
    )


def test_criteria_refuse_to_run_while_unverified() -> None:
    p = pair(
        measurement(flux=100.0, error=5.0, detected=True),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    for apply in (apply_nsigma, apply_type_change):
        with pytest.raises(UnverifiedCriterionError, match="verified_against_pdf"):
            apply(p)


def test_gate_names_the_file_and_the_fix() -> None:
    with pytest.raises(UnverifiedCriterionError) as excinfo:
        require_verified("A", "nsigma")
    message = str(excinfo.value)
    assert "literature_matrix.csv" in message
    assert "selection_definitions.md" in message


# ---------------------------------------------------------------------------
# epoch pairing
# ---------------------------------------------------------------------------


def test_epochs_must_be_time_ordered() -> None:
    m = measurement(flux=100.0, error=5.0, detected=True)
    with pytest.raises(ValueError, match="does not follow"):
        EpochPair(object_id="X", early=m, late=m, early_mjd=58000.0, late_mjd=52000.0)


def test_epochs_must_measure_the_same_line() -> None:
    with pytest.raises(ValueError, match="different lines"):
        pair(
            measurement(flux=100.0, error=5.0, detected=True, line="Hbeta"),
            measurement(flux=100.0, error=5.0, detected=True, line="Halpha"),
        )


def test_direction_is_turn_on_when_the_line_strengthens() -> None:
    p = pair(
        measurement(flux=100.0, error=5.0, detected=True),
        measurement(flux=400.0, error=6.0, detected=True),
    )
    assert p.direction == "turn_on"
    assert p.bright is p.late and p.faint is p.early


def test_direction_is_turn_off_when_the_line_weakens() -> None:
    p = pair(
        measurement(flux=400.0, error=6.0, detected=True),
        measurement(flux=100.0, error=5.0, detected=True),
    )
    assert p.direction == "turn_off"
    assert p.bright is p.early and p.faint is p.late


def test_direction_uses_detection_state_when_one_epoch_is_a_limit() -> None:
    """An undetected early epoch and a detected late epoch is a turn-on,
    even if the numeric upper limit happens to exceed the detected flux."""
    p = pair(
        measurement(flux=5.0, error=30.0, detected=False, upper_limit=900.0),
        measurement(flux=400.0, error=20.0, detected=True),
    )
    assert p.direction == "turn_on"
    assert p.bright is p.late


# ---------------------------------------------------------------------------
# Family A - N_sigma
# ---------------------------------------------------------------------------


def test_nsigma_hand_calculation() -> None:
    """|300 - 100| / sqrt(3^2 + 4^2) = 200 / 5 = 40."""
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=300.0, error=3.0, detected=True),
    )
    assert nsigma_statistic(p) == pytest.approx(40.0)


def test_nsigma_labels_cl_above_the_published_threshold() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=300.0, error=3.0, detected=True),
    )
    result = apply_nsigma(p, **OFF)
    assert result.label is Label.CL
    assert result.threshold == 3.0


def test_nsigma_labels_non_cl_below_the_threshold() -> None:
    """|105 - 100| / sqrt(4^2 + 3^2) = 1.0, below 3."""
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=105.0, error=3.0, detected=True),
    )
    result = apply_nsigma(p, **OFF)
    assert result.label is Label.NON_CL
    assert result.statistic == pytest.approx(1.0)


def test_nsigma_uses_the_published_threshold_exactly() -> None:
    """A statistic of exactly 3.0 is not > 3.0. No 'close enough' rounding."""
    p = pair(
        measurement(flux=100.0, error=3.0, detected=True),
        measurement(flux=115.0, error=4.0, detected=True),
    )
    assert nsigma_statistic(p) == pytest.approx(3.0)
    assert apply_nsigma(p, **OFF).label is Label.NON_CL


def test_nsigma_threshold_sweep_is_supported_for_robustness() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=115.0, error=3.0, detected=True),
    )  # N_sigma = 3.0
    assert apply_nsigma(p, threshold=2.0, **OFF).label is Label.CL
    assert apply_nsigma(p, threshold=5.0, **OFF).label is Label.NON_CL


def test_nsigma_is_unclassifiable_against_an_upper_limit() -> None:
    """The published statistic needs two sigmas; a limit does not supply one."""
    p = pair(
        measurement(flux=None, error=None, detected=False, upper_limit=50.0),
        measurement(flux=300.0, error=3.0, detected=True),
    )
    result = apply_nsigma(p, **OFF)
    assert result.label is Label.UNCLASSIFIABLE
    assert "upper limit" in result.reason


# ---------------------------------------------------------------------------
# Family B - flux ratio
# ---------------------------------------------------------------------------


def test_flux_ratio_hand_calculation() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=350.0, error=6.0, detected=True),
    )
    ratio, is_limit = flux_ratio_statistic(p)
    assert ratio == pytest.approx(3.5)
    assert is_limit is False


def test_flux_ratio_against_a_non_detection_is_a_lower_limit() -> None:
    p = pair(
        measurement(flux=10.0, error=20.0, detected=False, upper_limit=60.0),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    ratio, is_limit = flux_ratio_statistic(p)
    assert ratio == pytest.approx(5.0)
    assert is_limit is True


def test_flux_ratio_requires_a_threshold_from_the_literature() -> None:
    """The frozen config leaves this null on purpose."""
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=350.0, error=6.0, detected=True),
    )
    with pytest.raises(ValueError, match="threshold is null"):
        apply_flux_ratio(p, **OFF)


def test_flux_ratio_classifies_once_a_threshold_is_supplied() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=350.0, error=6.0, detected=True),
    )
    assert apply_flux_ratio(p, threshold=2.0, **OFF).label is Label.CL
    assert apply_flux_ratio(p, threshold=5.0, **OFF).label is Label.NON_CL


def test_unresolved_lower_limit_is_unclassifiable_not_negative() -> None:
    """A lower limit below the threshold cannot rule the transition out."""
    p = pair(
        measurement(flux=10.0, error=20.0, detected=False, upper_limit=200.0),
        measurement(flux=300.0, error=6.0, detected=True),
    )  # ratio > 1.5 only
    result = apply_flux_ratio(p, threshold=3.0, **OFF)
    assert result.label is Label.UNCLASSIFIABLE
    assert "does not resolve" in result.reason


def test_lower_limit_above_the_threshold_still_classifies() -> None:
    p = pair(
        measurement(flux=10.0, error=20.0, detected=False, upper_limit=30.0),
        measurement(flux=300.0, error=6.0, detected=True),
    )  # ratio > 10
    result = apply_flux_ratio(p, threshold=3.0, **OFF)
    assert result.label is Label.CL
    assert result.reason == "lower limit"


# ---------------------------------------------------------------------------
# Family C - type change
# ---------------------------------------------------------------------------


def test_type_change_detects_an_appearance() -> None:
    p = pair(
        measurement(flux=10.0, error=20.0, detected=False, upper_limit=60.0),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    assert apply_type_change(p, **OFF).label is Label.CL


def test_type_change_is_negative_when_both_epochs_are_detected() -> None:
    """A large but always-detected change is not an appearance/disappearance."""
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=900.0, error=6.0, detected=True),
    )
    assert apply_type_change(p, **OFF).label is Label.NON_CL


def test_type_change_label_can_be_lost_as_sensitivity_improves() -> None:
    """The counterintuitive effect this paper exists to quantify.

    Same genuine transition. At low S/N the faint state is undetected and the
    pair is labelled CL. At high S/N the weak residual broad line becomes
    detectable, and the identical physical event stops qualifying.
    """
    low_snr = pair(
        measurement(flux=8.0, error=20.0, detected=False, upper_limit=68.0),
        measurement(flux=300.0, error=20.0, detected=True),
    )
    high_snr = pair(
        measurement(flux=8.0, error=2.0, detected=True),
        measurement(flux=300.0, error=2.0, detected=True),
    )
    assert apply_type_change(low_snr, **OFF).label is Label.CL
    assert apply_type_change(high_snr, **OFF).label is Label.NON_CL


# ---------------------------------------------------------------------------
# Family D - difference spectrum
# ---------------------------------------------------------------------------


def test_difference_spectrum_classifies_on_the_supplied_significance() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    assert apply_difference_spectrum(p, difference_significance=8.0, **OFF).label is Label.CL
    assert apply_difference_spectrum(p, difference_significance=1.0, **OFF).label is Label.NON_CL


def test_difference_spectrum_is_unclassifiable_without_a_fit() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    result = apply_difference_spectrum(p, difference_significance=None, **OFF)
    assert result.label is Label.UNCLASSIFIABLE


# ---------------------------------------------------------------------------
# unusable epochs never become negatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "apply,kwargs",
    [
        (apply_nsigma, {}),
        (apply_flux_ratio, {"threshold": 2.0}),
        (apply_type_change, {}),
        (apply_difference_spectrum, {"difference_significance": 9.0}),
    ],
)
def test_failed_fit_is_unclassifiable_for_every_criterion(apply, kwargs) -> None:
    """Preregistration §8: a fit failure is not evidence of no transition."""
    p = pair(
        measurement(flux=None, error=None, detected=False, status="fit_failed"),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    result = apply(p, **kwargs, **OFF)
    assert result.label is Label.UNCLASSIFIABLE
    assert result.is_unclassifiable


# ---------------------------------------------------------------------------
# continuous metrics
# ---------------------------------------------------------------------------


def test_signed_significance_keeps_the_direction() -> None:
    up = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=300.0, error=3.0, detected=True),
    )
    down = pair(
        measurement(flux=300.0, error=3.0, detected=True),
        measurement(flux=100.0, error=4.0, detected=True),
    )
    assert signed_significance(up) == pytest.approx(40.0)
    assert signed_significance(down) == pytest.approx(-40.0)
    assert nsigma_statistic(up) == nsigma_statistic(down), "published N_sigma is unsigned"


def test_metrics_are_none_rather_than_zero_when_unmeasurable() -> None:
    """A missing measurement must never look like 'no change'."""
    p = pair(
        measurement(flux=None, error=None, detected=False, upper_limit=50.0, ew=None),
        measurement(flux=300.0, error=6.0, detected=True),
    )
    metrics = compute_metrics(p)
    assert metrics.delta_log_flux is None
    assert metrics.signed_significance is None
    assert metrics.delta_ew is None


def test_metrics_round_trip_to_a_row() -> None:
    p = pair(
        measurement(flux=100.0, error=4.0, detected=True),
        measurement(flux=300.0, error=3.0, detected=True),
    )
    row = compute_metrics(p).to_row()
    assert row["object_id"] == "OBJ1"
    assert row["direction"] == "turn_on"
    assert row["baseline_days"] == 6000.0
    assert row["delta_log_flux"] == pytest.approx(0.47712125, rel=1e-6)
