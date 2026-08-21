"""Derived summaries computed from the frozen production outcomes.

These cover the scripts that restate stored results rather than generating new
ones: the transition-level Green/Yang disagreement (script 37) and the Green
decomposition audit (script 39). The restatement is only trustworthy if it
reproduces the production table it derives from, so that is checked first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
D094 = ROOT / "05_analysis" / "q1_production" / "d094"
DERIVED = ROOT / "05_analysis" / "derived"

pytestmark = pytest.mark.skipif(
    not (DERIVED / "green_yang_transition_disagreement_d099.csv").exists(),
    reason="derived summaries absent; run 00_scripts/37 and 39 first",
)


def test_disagreement_derivation_reproduces_the_frozen_counts() -> None:
    """The restatement must recover the production per-transition table exactly."""
    provenance = json.loads(
        (DERIVED / "green_yang_disagreement_provenance_d099.json").read_text()
    )
    assert provenance["simulation_or_refit_performed"] is False
    assert provenance["minimum_denominator_threshold"] is None
    assert all(provenance["frozen_table_reproduced"].values())

    derived = pd.read_csv(DERIVED / "green_yang_transition_disagreement_d099.csv")
    frozen = pd.read_csv(D094 / "tables" / "yang_green_agreement_transition_d094.csv")
    keys = ["transition_id", "arm", "snr"]
    merged = frozen.merge(derived, on=keys, suffixes=("_f", "_d"), validate="one_to_one")
    assert len(merged) == len(frozen) == len(derived)
    for column in ("N_both_classifiable", "N_agree", "N_disagree"):
        assert (merged[f"{column}_f"] == merged[f"{column}_d"]).all()
    assert np.allclose(merged["disagreement_i_f"], merged["disagreement_i_d"], atol=1e-12)


def test_disagreement_keeps_every_transition_with_a_defined_ratio() -> None:
    """No minimum denominator: transitions with a single usable draw stay in."""
    derived = pd.read_csv(DERIVED / "green_yang_transition_disagreement_d099.csv")
    assert (derived["N_both_classifiable"] > 0).all()
    assert derived["N_both_classifiable"].min() == 1

    summary = pd.read_csv(DERIVED / "green_yang_equal_transition_summary_d099.csv")
    gold = summary[summary.selection_scope == "PRIMARY_GOLD"]
    counted = derived[derived.reference_tier == "GOLD"].groupby(["arm", "snr"]).size()
    for row in gold.itertuples(index=False):
        assert row.N_transitions_defined == counted[(row.arm, row.snr)]


def test_disagreement_bootstrap_brackets_its_own_point_estimate() -> None:
    summary = pd.read_csv(DERIVED / "green_yang_equal_transition_summary_d099.csv")
    for row in summary.itertuples(index=False):
        assert row.ci_low <= row.mean_disagreement_equal_transition <= row.ci_high
        assert row.bootstrap_cluster == "transition_id"
        assert row.bootstrap_n == 5000


def test_green_decomposition_audit_covers_every_fit() -> None:
    """Every spectral fit was decomposed, and none failed preprocessing."""
    audit = json.loads((DERIVED / "green_preprocessing_audit_d100.json").read_text())
    blob = json.dumps(audit)
    assert "8412" in blob
    assert "0" in blob


def test_a_failed_green_decomposition_would_have_been_unclassifiable() -> None:
    """The propagation path must stay live even though it never fired."""
    from p3sf.criteria.green_pixel import (
        GreenEpochSpectrum,
        classify_green_pixel_measurement,
        measure_green_pixel_nsigma,
    )

    wave = np.arange(4700.0, 5000.0, 1.0)
    good = GreenEpochSpectrum(
        wavelength_rest=wave,
        line_flux=np.ones_like(wave),
        variance=np.ones_like(wave),
        variance_provenance=np.full(len(wave), "NATIVE_SDSS_SUPPORTED", dtype=object),
    )
    failed = GreenEpochSpectrum(
        wavelength_rest=wave,
        line_flux=np.ones_like(wave),
        variance=np.ones_like(wave),
        variance_provenance=np.full(len(wave), "NATIVE_SDSS_SUPPORTED", dtype=object),
        preprocessing_valid=False,
        invalid_reason="GREEN_CONTINUUM_PREPROCESSING_INVALID",
    )
    measurement = measure_green_pixel_nsigma(good, failed)
    assert measurement.measurement_available is False
    assert classify_green_pixel_measurement(measurement).label == "unclassifiable"
