"""Reference-tier, pair-eligibility, and no-op contracts for full Q2."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd


class ReferenceTier(StrEnum):
    GOLD = "GOLD"
    SILVER = "SILVER"
    BORDERLINE = "BORDERLINE"


class PairEligibility(StrEnum):
    PAIR_CONSISTENT_ELIGIBLE = "PAIR_CONSISTENT_ELIGIBLE"
    SPECTRUM_DIAGNOSTIC_ONLY = "SPECTRUM_DIAGNOSTIC_ONLY"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


def validate_reference_tiers(frame: pd.DataFrame) -> None:
    """Fail closed on circular, duplicated, or incomplete reference truth."""
    required = {
        "object_id",
        "paper3_reference_tier",
        "adjudication_basis",
        "d085_classifier_outputs_inspected",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reference-tier table missing columns: {sorted(missing)}")
    if frame.object_id.duplicated().any():
        raise ValueError("one object/transition must have exactly one reference-tier row")
    invalid = set(frame.paper3_reference_tier) - {tier.value for tier in ReferenceTier}
    if invalid:
        raise ValueError(f"invalid reference tiers: {sorted(invalid)}")
    if frame.d085_classifier_outputs_inspected.astype(bool).any():
        raise ValueError("reference truth is circular: D-085 outputs were inspected")
    if frame.adjudication_basis.astype(str).str.strip().eq("").any():
        raise ValueError("every reference tier requires an independent evidence basis")


def validate_pair_eligibility(frame: pd.DataFrame) -> None:
    """Enforce pair-level independence and literal shared-amplitude eligibility."""
    required = {
        "object_id",
        "pair_eligibility",
        "bright_science_record_id",
        "faint_science_record_id",
        "flux_basis_comparability",
        "bright_aperture_arcsec",
        "faint_aperture_arcsec",
        "reason",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pair-eligibility table missing columns: {sorted(missing)}")
    if frame.object_id.duplicated().any():
        raise ValueError("one real transition/object is one independent eligibility unit")
    invalid = set(frame.pair_eligibility) - {status.value for status in PairEligibility}
    if invalid:
        raise ValueError(f"invalid pair eligibility values: {sorted(invalid)}")

    eligible = frame.pair_eligibility == PairEligibility.PAIR_CONSISTENT_ELIGIBLE.value
    if frame.loc[eligible, ["bright_science_record_id", "faint_science_record_id"]].isna().any().any():
        raise ValueError("pair-consistent eligibility requires both exact science records")
    if not frame.loc[eligible, "flux_basis_comparability"].eq("DEFENSIBLE_COMPARABLE").all():
        raise ValueError("pair-consistent eligibility requires a defensible comparable flux basis")
    apertures = frame.loc[eligible, ["bright_aperture_arcsec", "faint_aperture_arcsec"]]
    if not apertures.bright_aperture_arcsec.eq(apertures.faint_aperture_arcsec).all():
        raise ValueError("literal shared galaxy flux cannot cross unequal apertures")


def assert_noop_identity(
    authoritative_hashes: Mapping[str, str], noop_hashes: Mapping[str, str]
) -> None:
    """No-op mismatch is an engineering exception, never a science outcome."""
    if authoritative_hashes.keys() != noop_hashes.keys():
        raise RuntimeError("NOOP_IDENTITY_FAIL: compared component sets differ")
    mismatched = [key for key in authoritative_hashes if authoritative_hashes[key] != noop_hashes[key]]
    if mismatched:
        raise RuntimeError(f"NOOP_IDENTITY_FAIL: mismatched components {mismatched}")


__all__ = [
    "PairEligibility",
    "ReferenceTier",
    "assert_noop_identity",
    "validate_pair_eligibility",
    "validate_reference_tiers",
]
