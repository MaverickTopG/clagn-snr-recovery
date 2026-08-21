"""ZWARNING bit classification (D-016).

The load-bearing case: a blind ``ZWARNING == 0`` cut would discard broad-line
AGN for being broad-line AGN, and would bite hardest on the dim epochs that
carry the transition.
"""

from __future__ import annotations

import pytest

from p3sf.qc.zwarning import (
    BY_NAME,
    ZWARNING_BITS,
    Disposition,
    classify_zwarning,
    describe_bits,
)


def test_zero_passes_and_is_strict() -> None:
    verdict = classify_zwarning(0)
    assert verdict.disposition is Disposition.PASS
    assert verdict.strict_pass
    assert verdict.bits == ()
    assert not verdict.requires_manual_review


def test_many_outliers_passes() -> None:
    """The whole reason for bit-aware logic.

    MANY_OUTLIERS fires on broad-line galaxies and high-S/N spectra because the
    redshift template does not model a broad line well. Excluding on it would
    remove exactly the objects this paper studies.
    """
    verdict = classify_zwarning(BY_NAME["MANY_OUTLIERS"].value)
    assert verdict.disposition is Disposition.PASS
    assert not verdict.threatens_hbeta_measurement()
    assert not verdict.strict_pass  # nonzero, so the strict cut would have dropped it


def test_the_two_flagged_slice_spectra_survive_or_are_reviewed() -> None:
    """ZWARNING 16 and 64 from the development sample, both on dim epochs."""
    assert classify_zwarning(16).disposition is Disposition.PASS
    assert classify_zwarning(64).disposition is Disposition.REVIEW


@pytest.mark.parametrize("name", ["SKY", "UNPLUGGED", "BAD_TARGET", "NODATA"])
def test_no_data_and_identity_failures_fail_automatically(name: str) -> None:
    verdict = classify_zwarning(BY_NAME[name].value)
    assert verdict.disposition is Disposition.FAIL
    assert not verdict.requires_manual_review  # automatic, no human needed


@pytest.mark.parametrize(
    "name", ["LITTLE_COVERAGE", "SMALL_DELTA_CHI2", "NEGATIVE_MODEL", "Z_FITLIMIT",
             "NEGATIVE_EMISSION"]
)
def test_measurement_threatening_bits_require_review(name: str) -> None:
    verdict = classify_zwarning(BY_NAME[name].value)
    assert verdict.disposition is Disposition.REVIEW
    assert verdict.requires_manual_review


def test_redshift_bits_are_flagged_as_threatening_hbeta() -> None:
    """A wrong redshift puts the rest-frame Hbeta window in the wrong place."""
    for name in ("SMALL_DELTA_CHI2", "Z_FITLIMIT"):
        verdict = classify_zwarning(BY_NAME[name].value)
        assert "redshift" in verdict.threatens
        assert verdict.threatens_hbeta_measurement()


def test_fail_dominates_review_which_dominates_pass() -> None:
    fail_and_pass = BY_NAME["NODATA"].value | BY_NAME["MANY_OUTLIERS"].value
    review_and_pass = BY_NAME["Z_FITLIMIT"].value | BY_NAME["MANY_OUTLIERS"].value
    assert classify_zwarning(fail_and_pass).disposition is Disposition.FAIL
    assert classify_zwarning(review_and_pass).disposition is Disposition.REVIEW


def test_unknown_bits_force_review_rather_than_passing() -> None:
    """An unrecognised warning is exactly when a blind pass is dangerous."""
    verdict = classify_zwarning(1 << 20)
    assert verdict.unknown_bits == (20,)
    assert verdict.disposition is Disposition.REVIEW
    assert verdict.requires_manual_review
    assert "UNKNOWN_20" in verdict.bit_names


def test_original_mask_is_always_preserved() -> None:
    for value in (0, 16, 64, 255, 1 << 20):
        assert classify_zwarning(value).zwarning == value


def test_strict_pass_is_available_alongside_the_bit_aware_verdict() -> None:
    """Needed for the robustness re-run using only ZWARNING == 0 spectra."""
    lenient = classify_zwarning(16)
    assert lenient.disposition is Disposition.PASS
    assert not lenient.strict_pass


def test_multiple_bits_are_all_reported() -> None:
    combined = BY_NAME["LITTLE_COVERAGE"].value | BY_NAME["MANY_OUTLIERS"].value
    verdict = classify_zwarning(combined)
    assert set(verdict.bit_names) == {"LITTLE_COVERAGE", "MANY_OUTLIERS"}


def test_negative_zwarning_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        classify_zwarning(-1)


def test_every_bit_documents_why_it_matters_here() -> None:
    """Survey definition alone is not enough; the disposition needs a reason."""
    for bit in ZWARNING_BITS:
        assert len(bit.survey_definition) > 10, f"{bit.name} lacks a survey definition"
        assert len(bit.why_it_matters) > 40, f"{bit.name} lacks an experiment-specific rationale"
        assert bit.threatens, f"{bit.name} does not say what it threatens"


def test_bit_values_are_unique_and_ordered() -> None:
    bits = [b.bit for b in ZWARNING_BITS]
    assert len(bits) == len(set(bits))
    assert bits == sorted(bits)


def test_describe_bits_renders_the_full_table() -> None:
    text = describe_bits()
    for bit in ZWARNING_BITS:
        assert bit.name in text
