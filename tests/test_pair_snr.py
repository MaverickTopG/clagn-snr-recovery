"""The three Q1 arms (D-010).

The invariant that carries the scientific argument: in the primary arm the
bright epoch must come through *bit-for-bit unchanged*. If it were quietly
degraded, the primary and secondary arms would collapse into each other and the
bright/faint asymmetry the paper is built on would be unmeasurable.
"""

from __future__ import annotations

import numpy as np
import pytest

from p3sf.counterfactual.pair_snr import (
    PairCondition,
    arm_support,
    degrade_pair,
    reachable_conditions,
)
from p3sf.counterfactual.snr import measure_snr


@pytest.fixture
def bright(clean_spectrum):
    """Native S/N 50 by construction."""
    return clean_spectrum


@pytest.fixture
def faint(clean_spectrum, continuum_window):
    """A genuinely poorer partner epoch, S/N 20."""
    from p3sf.counterfactual.snr import degrade_to_snr

    return degrade_to_snr(clean_spectrum, 20.0, window_rest=continuum_window, seed=99).spectrum


# ---------------------------------------------------------------------------
# condition validity
# ---------------------------------------------------------------------------


def test_faint_only_arm_forbids_a_bright_target() -> None:
    with pytest.raises(ValueError, match="native quality"):
        PairCondition("faint_only", snr_faint=10.0, snr_bright=30.0)


def test_matched_arm_requires_equal_targets() -> None:
    with pytest.raises(ValueError, match="same target"):
        PairCondition("matched", snr_faint=10.0, snr_bright=20.0)


def test_matched_arm_accepts_equal_targets() -> None:
    condition = PairCondition("matched", snr_faint=10.0, snr_bright=10.0)
    assert condition.snr_bright == condition.snr_faint


def test_joint_arm_allows_independent_targets() -> None:
    condition = PairCondition("joint", snr_faint=5.0, snr_bright=30.0)
    assert condition.snr_bright == 30.0


def test_non_positive_targets_are_rejected() -> None:
    with pytest.raises(ValueError):
        PairCondition("faint_only", snr_faint=0.0)
    with pytest.raises(ValueError):
        PairCondition("joint", snr_faint=5.0, snr_bright=-1.0)


def test_label_distinguishes_native_from_a_target() -> None:
    assert PairCondition("faint_only", snr_faint=10.0).label == "faint_only:b=native,f=10"
    assert PairCondition("matched", snr_faint=10.0, snr_bright=10.0).label == "matched:b=10,f=10"


# ---------------------------------------------------------------------------
# the primary arm leaves the bright epoch alone
# ---------------------------------------------------------------------------


def test_primary_arm_leaves_the_bright_epoch_bit_for_bit_unchanged(
    bright, faint, continuum_window
) -> None:
    result = degrade_pair(
        bright,
        faint,
        PairCondition("faint_only", snr_faint=10.0),
        window_rest=continuum_window,
        base_seed=314159, pair_id="OBJ_A",
    )
    assert result.bright is None
    assert not result.bright_was_degraded
    assert np.array_equal(result.bright_spectrum.flux, bright.flux)
    assert np.array_equal(result.bright_spectrum.error, bright.error)


def test_primary_arm_still_degrades_the_faint_epoch(bright, faint, continuum_window) -> None:
    result = degrade_pair(
        bright,
        faint,
        PairCondition("faint_only", snr_faint=10.0),
        window_rest=continuum_window,
        base_seed=314159, pair_id="OBJ_A",
    )
    assert result.faint.requested_snr == 10.0
    assert measure_snr(result.faint_spectrum, continuum_window) == pytest.approx(10.0, rel=0.15)
    assert not np.array_equal(result.faint_spectrum.flux, faint.flux)


def test_matched_arm_degrades_both(bright, faint, continuum_window) -> None:
    result = degrade_pair(
        bright,
        faint,
        PairCondition("matched", snr_faint=10.0, snr_bright=10.0),
        window_rest=continuum_window,
        base_seed=314159, pair_id="OBJ_A",
    )
    assert result.bright_was_degraded
    assert not np.array_equal(result.bright_spectrum.flux, bright.flux)
    for spectrum in (result.bright_spectrum, result.faint_spectrum):
        assert measure_snr(spectrum, continuum_window) == pytest.approx(10.0, rel=0.20)


def test_the_two_epochs_draw_independent_noise(bright, continuum_window) -> None:
    """Sharing a seed would correlate the epochs and understate every scatter."""
    twin = bright  # identical latent spectrum in both slots
    result = degrade_pair(
        twin,
        twin,
        PairCondition("matched", snr_faint=10.0, snr_bright=10.0),
        window_rest=continuum_window,
        base_seed=314159, pair_id="OBJ_A",
    )
    assert not np.array_equal(result.bright_spectrum.flux, result.faint_spectrum.flux)


def test_same_seed_reproduces_the_whole_pair(bright, faint, continuum_window) -> None:
    condition = PairCondition("matched", snr_faint=10.0, snr_bright=10.0)
    a = degrade_pair(bright, faint, condition, window_rest=continuum_window, base_seed=7, pair_id="OBJ_A")
    b = degrade_pair(bright, faint, condition, window_rest=continuum_window, base_seed=7, pair_id="OBJ_A")
    assert np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)
    assert np.array_equal(a.bright_spectrum.flux, b.bright_spectrum.flux)


def test_realizations_differ(bright, faint, continuum_window) -> None:
    condition = PairCondition("faint_only", snr_faint=10.0)
    a = degrade_pair(bright, faint, condition, window_rest=continuum_window,
                     base_seed=7, pair_id="OBJ_A", realization=0)
    b = degrade_pair(bright, faint, condition, window_rest=continuum_window,
                     base_seed=7, pair_id="OBJ_A", realization=1)
    assert not np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)


def test_seed_namespace_separates_pilot_from_full_execution(
    bright, faint, continuum_window
) -> None:
    condition = PairCondition("faint_only", snr_faint=10.0)
    pilot = degrade_pair(
        bright, faint, condition, window_rest=continuum_window, base_seed=7,
        pair_id="OBJ_A", seed_namespace="p3sf:q1:d088:stability:v1",
    )
    full = degrade_pair(
        bright, faint, condition, window_rest=continuum_window, base_seed=7,
        pair_id="OBJ_A", seed_namespace="p3sf:q1:full:v1",
    )
    assert not np.array_equal(pilot.faint_spectrum.flux, full.faint_spectrum.flux)


def test_upgrading_either_epoch_is_refused(bright, faint, continuum_window) -> None:
    with pytest.raises(ValueError, match="one-directional"):
        degrade_pair(
            bright, faint, PairCondition("faint_only", snr_faint=40.0),
            window_rest=continuum_window, base_seed=1, pair_id="OBJ_A",
        )
    with pytest.raises(ValueError, match="one-directional"):
        degrade_pair(
            bright, faint, PairCondition("joint", snr_faint=10.0, snr_bright=80.0),
            window_rest=continuum_window, base_seed=1, pair_id="OBJ_A",
        )


# ---------------------------------------------------------------------------
# reachability — why the primary arm buys sample support
# ---------------------------------------------------------------------------


def test_faint_only_reaches_more_rungs_than_matched_for_an_asymmetric_pair(
    bright, faint, continuum_window
) -> None:
    """Bright S/N 50, faint S/N 20.

    Both arms are limited by the faint epoch here, so they tie; the asymmetry
    pays off in the opposite case, covered by the next test.
    """
    conditions = reachable_conditions(
        bright, faint, window_rest=continuum_window, faint_rungs=[5, 10, 15, 20]
    )
    faint_only = [c for c in conditions if c.arm == "faint_only"]
    matched = [c for c in conditions if c.arm == "matched"]
    assert len(faint_only) == 4
    assert len(matched) == 4


def test_matched_arm_loses_rungs_when_the_bright_epoch_is_the_weaker_one(
    clean_spectrum, continuum_window
) -> None:
    """A weak bright epoch caps the matched arm but not the faint-only arm."""
    from p3sf.counterfactual.snr import degrade_to_snr

    weak_bright = degrade_to_snr(
        clean_spectrum, 12.0, window_rest=continuum_window, seed=5
    ).spectrum
    strong_faint = clean_spectrum  # S/N 50

    conditions = reachable_conditions(
        weak_bright, strong_faint, window_rest=continuum_window, faint_rungs=[5, 10, 15, 20]
    )
    faint_only = [c for c in conditions if c.arm == "faint_only"]
    matched = [c for c in conditions if c.arm == "matched"]
    assert len(faint_only) == 4
    assert len(matched) == 2  # bright caps matched at 10


def test_unreachable_rungs_are_skipped_not_raised(bright, faint, continuum_window) -> None:
    conditions = reachable_conditions(
        bright, faint, window_rest=continuum_window, faint_rungs=[5, 10, 20, 30, 40]
    )
    assert {c.snr_faint for c in conditions} == {5, 10, 20}


def test_joint_arm_builds_the_two_dimensional_grid(bright, faint, continuum_window) -> None:
    conditions = reachable_conditions(
        bright, faint, window_rest=continuum_window,
        faint_rungs=[5, 10], bright_rungs=[20, 30], arms=("joint",),
    )
    assert len(conditions) == 4
    assert {(c.snr_bright, c.snr_faint) for c in conditions} == {
        (20, 5), (20, 10), (30, 5), (30, 10)
    }


def test_faint_only_support_never_falls_below_matched() -> None:
    """Arithmetic invariant, true for any sample.

    The matched arm needs min(bright, faint) >= rung, which implies faint >= rung,
    so every pair the matched arm can place is one the faint-only arm can also
    place. This is a property of the definitions, not of any dataset.
    """
    rng = np.random.default_rng(20260814)
    rungs = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    for _ in range(200):
        n = int(rng.integers(1, 25))
        native_bright = list(rng.uniform(3.0, 60.0, n))
        native_faint = list(rng.uniform(3.0, 60.0, n))
        support = arm_support(native_bright, native_faint, rungs)
        for rung in rungs:
            assert support["faint_only"][rung] >= support["matched"][rung]


def test_support_counts_are_exactly_the_arithmetic_answer() -> None:
    """Synthetic fixture with hand-chosen S/N, so the expected counts are derived.

    Deliberately NOT the observed census. Whether the real sample ties or
    diverges at a given rung is an empirical question this paper is meant to
    answer, and pinning a preliminary answer here would turn a future scientific
    result into a spurious test failure.

    Constructed pairs:
      A  bright 50, faint 50  -> both arms reach every rung <= 50
      B  bright 50, faint  8  -> faint reaches 5;      matched reaches 5
      C  bright 12, faint 40  -> faint reaches 5..40;  matched capped at 10
    """
    native_bright = [50.0, 50.0, 12.0]
    native_faint = [50.0, 8.0, 40.0]
    rungs = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

    support = arm_support(native_bright, native_faint, rungs)

    # faint_only: count of faint >= rung
    assert support["faint_only"] == {5: 3, 10: 2, 15: 2, 20: 2, 30: 2, 40: 2}
    # matched: count of min(bright, faint) >= rung
    assert support["matched"] == {5: 3, 10: 2, 15: 1, 20: 1, 30: 1, 40: 1}


def test_divergence_occurs_exactly_where_a_bright_epoch_is_the_binding_one() -> None:
    """The arms differ at a rung iff some pair has faint >= rung > bright.

    A structural statement about when the design distinction can matter at all,
    with no commitment to whether it does in any particular sample.
    """
    rng = np.random.default_rng(11)
    rungs = [5.0, 10.0, 20.0, 30.0]
    for _ in range(200):
        n = int(rng.integers(1, 15))
        native_bright = list(rng.uniform(3.0, 50.0, n))
        native_faint = list(rng.uniform(3.0, 50.0, n))
        support = arm_support(native_bright, native_faint, rungs)
        for rung in rungs:
            binding_bright = any(
                f >= rung > b for b, f in zip(native_bright, native_faint, strict=True)
            )
            diverges = support["faint_only"][rung] > support["matched"][rung]
            assert diverges == binding_bright


def test_support_is_monotone_decreasing_in_rung() -> None:
    native_bright = [50.0, 30.0, 12.0, 6.0]
    native_faint = [45.0, 25.0, 40.0, 4.0]
    rungs = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    support = arm_support(native_bright, native_faint, rungs)
    for arm in ("faint_only", "matched"):
        counts = [support[arm][r] for r in rungs]
        assert counts == sorted(counts, reverse=True)


def test_empty_sample_yields_zero_support() -> None:
    support = arm_support([], [], [5.0, 10.0])
    assert support["faint_only"] == {5.0: 0, 10.0: 0}
    assert support["matched"] == {5.0: 0, 10.0: 0}


# ---------------------------------------------------------------------------
# common random numbers across arms
# ---------------------------------------------------------------------------


def test_faint_spectrum_is_identical_across_arms(bright, faint, continuum_window) -> None:
    """Paired experiment: same object, same degraded faint spectrum.

    For a fixed (object, rung, realization) the faint epoch must be bit-for-bit
    identical in every arm, so the only difference between arms is whether the
    bright epoch was also degraded. Independent faint draws would inject Monte
    Carlo noise into exactly the difference the comparison measures.
    """
    kwargs = {"window_rest": continuum_window, "base_seed": 314159, "pair_id": "OBJ_A", "realization": 3}

    faint_only = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=10.0), **kwargs)
    matched = degrade_pair(
        bright, faint, PairCondition("matched", snr_faint=10.0, snr_bright=10.0), **kwargs
    )
    joint = degrade_pair(
        bright, faint, PairCondition("joint", snr_faint=10.0, snr_bright=20.0), **kwargs
    )

    for other in (matched, joint):
        assert np.array_equal(faint_only.faint_spectrum.flux, other.faint_spectrum.flux)
        assert np.array_equal(faint_only.faint_spectrum.error, other.faint_spectrum.error)
        if faint_only.faint_spectrum.mask is None:
            assert other.faint_spectrum.mask is None
        else:
            assert np.array_equal(faint_only.faint_spectrum.mask, other.faint_spectrum.mask)


def test_faint_realization_is_independent_of_the_bright_target(
    bright, faint, continuum_window
) -> None:
    """Changing only the bright target must not perturb the faint draw."""
    kwargs = {"window_rest": continuum_window, "base_seed": 7, "pair_id": "OBJ_A", "realization": 0}
    a = degrade_pair(bright, faint, PairCondition("joint", snr_faint=5.0, snr_bright=20.0), **kwargs)
    b = degrade_pair(bright, faint, PairCondition("joint", snr_faint=5.0, snr_bright=30.0), **kwargs)
    assert np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)
    assert not np.array_equal(a.bright_spectrum.flux, b.bright_spectrum.flux)


def test_faint_realizations_still_differ_across_rungs_and_realizations(
    bright, faint, continuum_window
) -> None:
    """Common random numbers must not collapse distinct conditions together."""
    kwargs = {"window_rest": continuum_window, "base_seed": 7, "pair_id": "OBJ_A"}
    r5 = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=5.0),
                      realization=0, **kwargs)
    r10 = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=10.0),
                       realization=0, **kwargs)
    j1 = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=5.0),
                      realization=1, **kwargs)
    assert not np.array_equal(r5.faint_spectrum.flux, r10.faint_spectrum.flux)
    assert not np.array_equal(r5.faint_spectrum.flux, j1.faint_spectrum.flux)


def test_matched_arm_bright_and_faint_remain_independent(bright, continuum_window) -> None:
    """Same latent spectrum, same target: the two epochs must still differ."""
    result = degrade_pair(
        bright, bright, PairCondition("matched", snr_faint=10.0, snr_bright=10.0),
        window_rest=continuum_window, base_seed=314159, pair_id="OBJ_A",
    )
    assert not np.array_equal(result.bright_spectrum.flux, result.faint_spectrum.flux)


# ---------------------------------------------------------------------------
# seed identity: CRN across arms, independence across objects
# ---------------------------------------------------------------------------


def test_1_same_object_rung_realization_gives_identical_faint_across_arms(
    bright, faint, continuum_window
) -> None:
    kwargs = {"window_rest": continuum_window, "base_seed": 314159,
              "pair_id": "SDSSJ000000.00+000000.0", "realization": 7}
    a = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=10.0), **kwargs)
    b = degrade_pair(bright, faint, PairCondition("matched", snr_faint=10.0, snr_bright=10.0),
                     **kwargs)
    assert np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)


def test_2_different_objects_get_independent_noise(bright, faint, continuum_window) -> None:
    """Objects are independent experimental units and must not share a stream.

    Without the identity in the seed, every AGN at a given rung and realization
    would draw the same Gaussian sequence.
    """
    kwargs = {"window_rest": continuum_window, "base_seed": 314159, "realization": 7}
    condition = PairCondition("faint_only", snr_faint=10.0)
    a = degrade_pair(bright, faint, condition, pair_id="OBJECT_ONE", **kwargs)
    b = degrade_pair(bright, faint, condition, pair_id="OBJECT_TWO", **kwargs)
    assert not np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)


def test_3_different_realizations_differ_for_the_same_object(
    bright, faint, continuum_window
) -> None:
    kwargs = {"window_rest": continuum_window, "base_seed": 314159, "pair_id": "OBJECT_ONE"}
    condition = PairCondition("faint_only", snr_faint=10.0)
    a = degrade_pair(bright, faint, condition, realization=7, **kwargs)
    b = degrade_pair(bright, faint, condition, realization=8, **kwargs)
    assert not np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)


def test_4_different_rungs_differ_for_the_same_object_and_realization(
    bright, faint, continuum_window
) -> None:
    kwargs = {"window_rest": continuum_window, "base_seed": 314159,
              "pair_id": "OBJECT_ONE", "realization": 7}
    a = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=10.0), **kwargs)
    b = degrade_pair(bright, faint, PairCondition("faint_only", snr_faint=15.0), **kwargs)
    assert not np.array_equal(a.faint_spectrum.flux, b.faint_spectrum.flux)


def test_pair_id_is_required(bright, faint, continuum_window) -> None:
    """Defaulting it would silently reintroduce cross-object correlation."""
    with pytest.raises(TypeError):
        degrade_pair(  # type: ignore[call-arg]
            bright, faint, PairCondition("faint_only", snr_faint=10.0),
            window_rest=continuum_window, base_seed=1,
        )
    with pytest.raises(ValueError, match="pair_id is required"):
        degrade_pair(
            bright, faint, PairCondition("faint_only", snr_faint=10.0),
            window_rest=continuum_window, base_seed=1, pair_id="",
        )


def test_cross_object_streams_are_independent_in_bulk(bright, faint, continuum_window) -> None:
    """Twenty objects at one rung and realization: no two share a draw."""
    condition = PairCondition("faint_only", snr_faint=10.0)
    fluxes = [
        degrade_pair(
            bright, faint, condition, window_rest=continuum_window,
            base_seed=314159, pair_id=f"OBJECT_{n:03d}", realization=7,
        ).faint_spectrum.flux
        for n in range(20)
    ]
    for i in range(len(fluxes)):
        for j in range(i + 1, len(fluxes)):
            assert not np.array_equal(fluxes[i], fluxes[j])


def test_spectrum_identity_is_taken_from_metadata_when_present(
    bright, faint, continuum_window
) -> None:
    """Two epochs of one object must not collide even at the same target."""
    from dataclasses import replace

    tagged_bright = replace(bright, meta={"spectrum_id": "2086-53401-0380"})
    tagged_faint = replace(faint, meta={"spectrum_id": "9604-58133-0005"})
    result = degrade_pair(
        tagged_bright, tagged_faint,
        PairCondition("matched", snr_faint=10.0, snr_bright=10.0),
        window_rest=continuum_window, base_seed=1, pair_id="OBJECT_ONE",
    )
    assert not np.array_equal(result.bright_spectrum.flux, result.faint_spectrum.flux)
