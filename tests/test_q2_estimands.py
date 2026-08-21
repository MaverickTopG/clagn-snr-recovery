"""D-086 object-level estimands and complete denominator accounting."""

from __future__ import annotations

import pandas as pd
import pytest

from p3sf.stats.q2 import Q2Counts, summarise_q2_outcomes


def test_recovery_and_unclassifiable_estimands_use_distinct_denominators() -> None:
    counts = Q2Counts(
        f_added_star=0.5,
        host_shape_id="XSL_H2",
        criterion="green2022",
        n_eligible=10,
        n_fit_completed=9,
        n_fit_valid=8,
        n_classifiable=6,
        n_cl=4,
        n_noncl=2,
        n_unclassifiable=4,
    )
    assert counts.empirical_recovery_fraction == pytest.approx(4 / 6)
    assert counts.unclassifiable_fraction == pytest.approx(4 / 10)


def test_unclassifiable_cannot_disappear_from_eligible_accounting() -> None:
    with pytest.raises(ValueError, match=r"N_classifiable \+ N_unclassifiable"):
        Q2Counts(
            f_added_star=0.1,
            host_shape_id="XSL_H1",
            criterion="k",
            n_eligible=10,
            n_fit_completed=10,
            n_fit_valid=10,
            n_classifiable=8,
            n_cl=4,
            n_noncl=4,
            n_unclassifiable=1,
        )


def test_repeated_realizations_do_not_inflate_transition_count() -> None:
    records = pd.DataFrame(
        [
            {"object_id": "A", "f_added_star": 0.5, "host_shape_id": "XSL_H1",
             "criterion": "green", "fit_completed": True, "fit_valid": True,
             "label": "CL", "realization": 0},
            {"object_id": "A", "f_added_star": 0.5, "host_shape_id": "XSL_H1",
             "criterion": "green", "fit_completed": True, "fit_valid": True,
             "label": "CL", "realization": 1},
            {"object_id": "B", "f_added_star": 0.5, "host_shape_id": "XSL_H1",
             "criterion": "green", "fit_completed": False, "fit_valid": False,
             "label": "unclassifiable", "realization": 0},
        ]
    )
    row = summarise_q2_outcomes(records).iloc[0]
    assert row.n_eligible == 2
    assert row.n_classifiable == 1
    assert row.n_cl == 1
    assert row.n_unclassifiable == 1


def test_one_transition_cannot_have_conflicting_realization_labels() -> None:
    records = pd.DataFrame(
        [
            {"object_id": "A", "f_added_star": 0.5, "host_shape_id": "XSL_H1",
             "criterion": "green", "fit_completed": True, "fit_valid": True,
             "label": "CL"},
            {"object_id": "A", "f_added_star": 0.5, "host_shape_id": "XSL_H1",
             "criterion": "green", "fit_completed": True, "fit_valid": True,
             "label": "non-CL"},
        ]
    )
    with pytest.raises(ValueError, match="inconsistent"):
        summarise_q2_outcomes(records)
