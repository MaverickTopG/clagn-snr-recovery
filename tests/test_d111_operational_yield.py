"""D-111: the three-state framing, operational yield, and their robustness.

These guard the PASP revision's central claim: conditional recovery and
operational yield diverge when invalidity is common, and that divergence is a
property of the protocols rather than of which transitions each was posed for.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "05_analysis" / "q1_production" / "d094" / "tables"
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"


@pytest.fixture(scope="module")
def three_state() -> pd.DataFrame:
    return pd.read_csv(TABLES / "three_state_outcomes_d094.csv")


@pytest.fixture(scope="module")
def common() -> pd.DataFrame:
    return pd.read_csv(TABLES / "common_applicability_d094.csv")


def test_three_state_shares_partition_the_applicable_realizations(three_state):
    assert (three_state.n_cl + three_state.n_noncl + three_state.n_invalid).equals(
        three_state.n_applicable
    )
    shares = three_state[["p_cl", "p_noncl", "p_invalid"]].sum(axis=1)
    assert shares.round(12).eq(1.0).all()


def test_green_never_returns_an_invalid_applicable_realization(three_state):
    green = three_state[three_state.protocol.eq("Green-statistic")]
    assert len(green) == 4
    assert green.n_invalid.eq(0).all()
    # With no invalid state, the conditional fraction and the yield must coincide.
    assert green.p_cl.equals(green.operational_yield)


def test_yang_conditional_recovery_overstates_its_yield(three_state):
    yields = pd.read_csv(TABLES / "operational_yield_d094.csv")
    yang = yields[yields.protocol.eq("Yang-ratio")]
    assert yang.Y_operational.between(0.09, 0.14).all()
    assert yang.R_conditional.between(0.35, 0.42).all()
    assert yang.inflation_factor.between(2.9, 3.9).all()
    green = yields[yields.protocol.eq("Green-statistic")]
    assert green.inflation_factor.round(12).eq(1.0).all()


def test_outcome_geometry_survives_holding_applicability_fixed(common):
    """The comparison must not be an artifact of different transition membership."""
    for (arm, snr), group in common.groupby(["arm", "snr"]):
        assert group.N_tr_common.nunique() == 1, (arm, snr)
        assert group.n_applicable.nunique() == 1, (arm, snr)
    green = common[common.protocol.eq("Green-statistic")]
    yang = common[common.protocol.eq("Yang-ratio")]
    assert green.n_invalid.eq(0).all()
    assert green.invalid_fraction.round(12).eq(0.0).all()
    assert yang.invalid_fraction.between(0.63, 0.68).all()
    ratio = yang.R_conditional / yang.Y_operational
    assert ratio.between(2.9, 3.9).all()


def test_invalidity_predates_degradation(three_state):
    """19 of 62 native bright-epoch fits already fail, so noise is not the cause."""
    baseline = pd.read_csv(TABLES / "fit_validity_baseline_d094.csv")
    native = baseline[baseline.spectra.str.startswith("Native bright")].iloc[0]
    assert int(native.n_fits) == 62
    assert round(native.yang_valid_fraction * 62) == 43
    assert native.green_valid_fraction == 1.0
    faint = baseline[baseline.spectra.eq("Degraded faint")].set_index("target_snr")
    # Degradation moves faint-epoch validity only a few points across the rungs.
    assert abs(faint.loc["5"].yang_valid_fraction - faint.loc["10"].yang_valid_fraction) < 0.06


def test_public_validity_flags_reproduce_the_native_baseline():
    """The projection shipped for public reruns must give the same 19 of 62."""
    flags = RAW / "fit_validity_flags_d094.csv"
    if not flags.exists():
        pytest.skip("public validity projection not built in this tree")
    frame = pd.read_csv(flags)
    native = frame[frame.task_kind.eq("native_bright")]
    assert len(native) == 62
    assert int((~native.yang_fit_valid).sum()) == 19
    assert bool(frame.green_preprocessing_valid.all())
