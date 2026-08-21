"""Per-rung attrition accounting.

The point of these tests: no object may vanish silently between eligibility,
fitting, and classification. If the paper reports 78 classifiable at a rung
where 100 were eligible, the other 22 must be accounted for by construction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from p3sf.stats.accounting import (
    RungCounts,
    accounting_table,
    format_attrition,
    summarise_attrition,
)


def test_counts_nest_and_derive_the_gaps() -> None:
    counts = RungCounts(arm="faint_only", rung=5.0, n_eligible=100,
                        n_fit_succeeded=92, n_classifiable=78)
    assert counts.n_fit_failed == 8
    assert counts.n_unclassifiable == 14
    assert counts.n_fit_failed + counts.n_unclassifiable + counts.n_classifiable == 100


def test_non_nesting_counts_are_rejected() -> None:
    """A pair counted at a rung it never reached is a bug, not a datum."""
    with pytest.raises(ValueError, match="must nest"):
        RungCounts(arm="matched", rung=10.0, n_eligible=10,
                   n_fit_succeeded=12, n_classifiable=5)
    with pytest.raises(ValueError, match="must nest"):
        RungCounts(arm="matched", rung=10.0, n_eligible=10,
                   n_fit_succeeded=8, n_classifiable=9)


def test_classifiable_rate_is_against_eligible_not_survivors() -> None:
    """Dividing by survivors would hide fit failures (preregistration §8)."""
    counts = RungCounts(arm="faint_only", rung=5.0, n_eligible=100,
                        n_fit_succeeded=50, n_classifiable=50)
    assert counts.classifiable_rate == pytest.approx(0.50)
    assert counts.fit_success_rate == pytest.approx(0.50)


def test_rates_are_none_rather_than_a_sentinel_when_nothing_was_eligible() -> None:
    counts = RungCounts(arm="matched", rung=40.0, n_eligible=0,
                        n_fit_succeeded=0, n_classifiable=0)
    assert counts.fit_success_rate is None
    assert counts.classifiable_rate is None


def _records() -> pd.DataFrame:
    """Three objects at one rung: one clean, one fit failure, one undecidable.

    Object A appears twice to represent two Monte Carlo realizations.
    """
    return pd.DataFrame(
        [
            {"object_id": "A", "arm": "faint_only", "rung": 10.0,
             "criterion": "nsigma", "fit_succeeded": True, "label": "CL"},
            {"object_id": "A", "arm": "faint_only", "rung": 10.0,
             "criterion": "nsigma", "fit_succeeded": True, "label": "CL"},
            {"object_id": "B", "arm": "faint_only", "rung": 10.0,
             "criterion": "nsigma", "fit_succeeded": False, "label": "unclassifiable"},
            {"object_id": "C", "arm": "faint_only", "rung": 10.0,
             "criterion": "nsigma", "fit_succeeded": True, "label": "unclassifiable"},
        ]
    )


def test_summary_counts_objects_not_realizations() -> None:
    """MC realizations are not independent AGN; counting rows would inflate N."""
    frame = summarise_attrition(_records())
    row = frame.iloc[0]
    assert row.n_eligible == 3  # not 4
    assert row.n_fit_succeeded == 2
    assert row.n_classifiable == 1


def test_summary_separates_fit_failure_from_undecidability() -> None:
    """B failed to fit; C fit but could not be decided. Different facts."""
    row = summarise_attrition(_records()).iloc[0]
    assert row.n_fit_failed == 1
    assert row.n_unclassifiable == 1


def test_unclassifiable_never_counts_as_a_decision() -> None:
    records = _records()
    records["label"] = "unclassifiable"
    assert summarise_attrition(records).iloc[0].n_classifiable == 0


def test_summary_splits_by_arm_rung_and_criterion() -> None:
    records = _records()
    other = records.copy()
    other["arm"] = "matched"
    other["criterion"] = "flux_ratio"
    frame = summarise_attrition(pd.concat([records, other], ignore_index=True))
    assert len(frame) == 2
    assert set(frame.arm) == {"faint_only", "matched"}
    assert set(frame.criterion) == {"nsigma", "flux_ratio"}


def test_summary_works_without_a_criterion_column() -> None:
    records = _records().drop(columns=["criterion"])
    frame = summarise_attrition(records, criterion_column=None)
    assert len(frame) == 1
    assert frame.iloc[0].criterion is None


def test_empty_table_has_the_full_schema() -> None:
    frame = accounting_table([])
    for column in ("n_eligible", "n_fit_succeeded", "n_classifiable",
                   "n_fit_failed", "n_unclassifiable"):
        assert column in frame.columns


def test_format_shows_every_count() -> None:
    text = format_attrition(summarise_attrition(_records()))
    assert "elig" in text and "unclass" in text
    assert "faint_only" in text


def test_search_stages_are_excluded_from_classifier_denominators() -> None:
    records = pd.DataFrame(
        [
            {"object_id": "A", "arm": "q2", "rung": 10.0, "criterion": "green",
             "fit_succeeded": True, "label": "CL", "analysis_role": "FINAL_CLASSIFIER"},
            {"object_id": "B", "arm": "q2", "rung": 10.0, "criterion": "guo_step1",
             "fit_succeeded": True, "label": "pass", "analysis_role": "SEARCH_STAGE"},
        ]
    )
    frame = summarise_attrition(records, role_column="analysis_role")
    assert len(frame) == 1
    assert frame.iloc[0].criterion == "green"
    assert frame.iloc[0].n_eligible == 1


def test_role_filter_fails_closed_when_schema_is_missing() -> None:
    with pytest.raises(KeyError, match="role column"):
        summarise_attrition(_records(), role_column="analysis_role")
