"""Region-specific QC and Hβ adjudication (D-032).

The rule these tests protect: an object-level survey warning must never become
an object-level scientific rejection without someone checking whether it
touches Hβ at all.
"""

from __future__ import annotations

import pytest

from p3sf.qc.region import ADJUDICATION_COLUMNS, HbetaAdjudication, RegionVerdict


def record(**overrides) -> HbetaAdjudication:
    base = {
        "object_id": "SDSSJ000000.00+000000.0",
        "mjd": 53401.0,
        "zwarning": 64,
        "hbeta_pixel_mask_ok": True,
        "hbeta_ivar_ok": True,
        "hbeta_local_residual_ok": True,
        "hbeta_pyqsofit_valid": True,
        "trigger_line": "H_alpha",
    }
    return HbetaAdjudication(**{**base, **overrides})


# ---------------------------------------------------------------------------
# verdict semantics
# ---------------------------------------------------------------------------


def test_usable_verdicts_are_exactly_the_three_pass_kinds() -> None:
    assert RegionVerdict.PASS.usable
    assert RegionVerdict.PASS_WITH_WARNING.usable
    assert RegionVerdict.REVIEW_PASS.usable
    assert not RegionVerdict.FAIL.usable
    assert not RegionVerdict.PENDING.usable


def test_pending_is_not_usable_so_it_cannot_default_either_way() -> None:
    """An unadjudicated epoch must block, not quietly pass or quietly drop."""
    assert not RegionVerdict.PENDING.usable


# ---------------------------------------------------------------------------
# the NEGATIVE_EMISSION question: which line actually triggered?
# ---------------------------------------------------------------------------


def test_trigger_from_another_line_with_clean_hbeta_passes_with_warning() -> None:
    """The real slice case: Hα raised bit 6 while Hβ is a strong detection."""
    assert record(trigger_line="H_alpha").recommended_disposition() is (
        RegionVerdict.PASS_WITH_WARNING
    )


def test_trigger_from_hbeta_itself_gets_review_pass_not_automatic_rejection() -> None:
    """A legacy line fitter failing on an unusual CLAGN state is not bad data.

    If Hβ survives independent pixel, error, residual and decomposition checks,
    the epoch is usable — but it stays out of Gold pending adjudication.
    """
    assert record(trigger_line="H_beta").recommended_disposition() is RegionVerdict.REVIEW_PASS


@pytest.mark.parametrize("name", ["H_beta", "Hbeta", "hb", "H_BETA"])
def test_hbeta_trigger_is_recognised_across_spellings(name: str) -> None:
    assert record(trigger_line=name).trigger_is_hbeta is True


def test_unknown_trigger_line_is_none_not_false() -> None:
    """'We have not established the trigger' differs from 'it was not Hβ'."""
    assert record(trigger_line=None).trigger_is_hbeta is None


def test_unknown_trigger_blocks_a_recommendation() -> None:
    assert record(trigger_line=None).recommended_disposition() is RegionVerdict.PENDING


# ---------------------------------------------------------------------------
# evidence completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    ["hbeta_pixel_mask_ok", "hbeta_ivar_ok", "hbeta_local_residual_ok", "hbeta_pyqsofit_valid"],
)
def test_any_unchecked_box_forces_pending(missing: str) -> None:
    """An unchecked box must not read as a pass."""
    assert record(**{missing: None}).recommended_disposition() is RegionVerdict.PENDING


@pytest.mark.parametrize(
    "failing",
    ["hbeta_pixel_mask_ok", "hbeta_ivar_ok", "hbeta_local_residual_ok", "hbeta_pyqsofit_valid"],
)
def test_any_failed_hbeta_check_fails_the_region(failing: str) -> None:
    assert record(**{failing: False}).recommended_disposition() is RegionVerdict.FAIL


def test_failed_hbeta_fails_even_when_the_trigger_was_elsewhere() -> None:
    """Global cleanliness cannot rescue a corrupted Hβ region."""
    corrupted = record(trigger_line="C_IV 1549", hbeta_pixel_mask_ok=False)
    assert corrupted.recommended_disposition() is RegionVerdict.FAIL


def test_evidence_completeness_is_reported_separately() -> None:
    assert record().hbeta_evidence_complete
    assert not record(hbeta_ivar_ok=None).hbeta_evidence_complete


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------


def test_recommendation_never_silently_becomes_the_final_verdict() -> None:
    """A person sets the final disposition; the code only advises."""
    entry = record()
    assert entry.recommended_disposition() is RegionVerdict.PASS_WITH_WARNING
    assert entry.final_hbeta_disposition is RegionVerdict.PENDING


def test_strict_zwarning_pass_survives_into_the_row() -> None:
    """Needed for the ZWARNING == 0 robustness re-run."""
    assert record(zwarning=64).to_row()["strict_zwarning_pass"] is False
    assert record(zwarning=0).to_row()["strict_zwarning_pass"] is True


def test_row_carries_every_declared_column() -> None:
    row = record().to_row()
    assert set(row) == set(ADJUDICATION_COLUMNS)
    for key in ("trigger_line", "trigger_is_hbeta", "recommended_disposition",
                "final_hbeta_disposition", "reviewer_note", "reviewer"):
        assert key in row


def test_row_is_serialisable_to_plain_values() -> None:
    row = record().to_row()
    assert isinstance(row["recommended_disposition"], str)
    assert isinstance(row["final_hbeta_disposition"], str)
