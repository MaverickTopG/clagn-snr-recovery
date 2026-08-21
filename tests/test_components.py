"""Component bookkeeping, reconstruction identity, and canonical host fractions.

The gate these lock in: a host fraction whose denominator was never verified is
an implementation assumption, not a measurement. `require()` must refuse to let
one be quoted.
"""

from __future__ import annotations

import numpy as np
import pytest

from p3sf.fitting.components import (
    RECONSTRUCTION_RTOL,
    ComponentModel,
    from_pyqsofit,
)

WINDOW = (5080.0, 5130.0)


def model(**overrides) -> ComponentModel:
    lam = np.linspace(5000.0, 5200.0, 400)
    components = {
        "stellar": np.full(lam.size, 3.0),
        "agn_smooth": np.full(lam.size, 1.0),
        "feii": np.full(lam.size, 0.5),
        "balmer_cont": np.full(lam.size, 0.5),
        "lines": np.zeros(lam.size),
    }
    components.update(overrides.pop("components", {}))
    bestfit = overrides.pop("bestfit", sum(components.values()))
    return ComponentModel(lam, components, bestfit=bestfit, method="test", **overrides)


# ---------------------------------------------------------------------------
# the hard gate
# ---------------------------------------------------------------------------


def test_exact_reconstruction_passes() -> None:
    check = model().check_reconstruction()
    assert check.passed
    assert check.max_abs_difference == 0.0


def test_dropped_family_fails_the_identity() -> None:
    """The failure mode that matters: a nuisance family missing from extraction."""
    intact = model()
    broken = ComponentModel(
        intact.wavelength,
        {k: v for k, v in intact.components.items() if k != "feii"},
        bestfit=intact.bestfit,
        method="test",
    )
    check = broken.check_reconstruction()
    assert not check.passed
    assert "missing" in check.detail


def test_double_counted_family_fails_the_identity() -> None:
    intact = model()
    doubled = dict(intact.components)
    doubled["agn_smooth"] = doubled["agn_smooth"] * 2
    check = ComponentModel(intact.wavelength, doubled, bestfit=intact.bestfit,
                           method="test").check_reconstruction()
    assert not check.passed


def test_absent_bestfit_cannot_pass() -> None:
    """No reference means the identity is untested, not satisfied."""
    check = ComponentModel(
        np.linspace(5000.0, 5200.0, 400),
        {"stellar": np.ones(400)}, bestfit=None, method="test",
    ).check_reconstruction()
    assert not check.passed
    assert "cannot be tested" in check.detail


def test_shape_mismatch_fails_rather_than_broadcasting() -> None:
    lam = np.linspace(5000.0, 5200.0, 400)
    check = ComponentModel(lam, {"stellar": np.ones(400)}, bestfit=np.ones(200),
                           method="test").check_reconstruction()
    assert not check.passed
    assert "shape mismatch" in check.detail


def test_tiny_float_noise_still_passes() -> None:
    intact = model()
    perturbed = intact.bestfit * (1.0 + RECONSTRUCTION_RTOL * 0.1)
    check = ComponentModel(intact.wavelength, intact.components, bestfit=perturbed,
                           method="test").check_reconstruction()
    assert check.passed


def test_require_raises_and_names_the_consequence() -> None:
    intact = model()
    broken = ComponentModel(intact.wavelength, {"stellar": intact.get("stellar")},
                            bestfit=intact.bestfit, method="test")
    with pytest.raises(ValueError, match="No host fraction may be derived"):
        broken.check_reconstruction().require()


def test_canonical_fractions_are_gated_on_the_identity() -> None:
    """The gate is enforced at the point of use, not merely available."""
    intact = model()
    broken = ComponentModel(intact.wavelength, {"stellar": intact.get("stellar")},
                            bestfit=intact.bestfit, method="test")
    with pytest.raises(ValueError, match="reconstruction identity failed"):
        broken.canonical_fractions(WINDOW)


# ---------------------------------------------------------------------------
# canonical semantics
# ---------------------------------------------------------------------------


def test_cont_and_pseudo_differ_by_the_pseudocontinuum() -> None:
    result = model().canonical_fractions(WINDOW)
    assert result["f_host_cont"] == pytest.approx(3.0 / 4.0)      # 3 / (3 + 1)
    assert result["f_host_pseudo"] == pytest.approx(3.0 / 5.0)    # 3 / (3 + 1 + .5 + .5)
    assert result["f_host_pseudo"] < result["f_host_cont"]


def test_emission_lines_never_enter_either_denominator() -> None:
    """A brighter line spectrum must not change a continuum dilution measure."""
    baseline = model().canonical_fractions(WINDOW)
    lam = np.linspace(5000.0, 5200.0, 400)
    components = dict(model().components)
    components["lines"] = np.full(lam.size, 50.0)
    loud = ComponentModel(lam, components, bestfit=sum(components.values()),
                          method="test").canonical_fractions(WINDOW)
    assert loud["f_host_cont"] == pytest.approx(baseline["f_host_cont"])
    assert loud["f_host_pseudo"] == pytest.approx(baseline["f_host_pseudo"])


def test_native_fraction_is_preserved_not_overwritten() -> None:
    """'Algorithms disagree' must stay distinguishable from 'definitions differ'."""
    result = model(native_f_host=0.47).canonical_fractions(WINDOW)
    assert result["native_f_host"] == 0.47
    assert result["f_host_cont"] != 0.47


def test_missing_family_reads_as_zero_not_an_error() -> None:
    lam = np.linspace(5000.0, 5200.0, 400)
    components = {"stellar": np.full(400, 2.0), "agn_smooth": np.full(400, 2.0)}
    result = ComponentModel(lam, components, bestfit=sum(components.values()),
                            method="test").canonical_fractions(WINDOW)
    assert result["f_host_cont"] == pytest.approx(0.5)
    assert result["f_host_pseudo"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PyQSOFit adapter
# ---------------------------------------------------------------------------


class FakePyQSOFit:
    def __init__(self, host, qso, wave):
        self.host, self.qso, self.wave = host, qso, wave


def test_pyqsofit_adapter_builds_a_consistent_model() -> None:
    wave = np.linspace(6000.0, 6300.0, 300)          # observed frame
    fit = FakePyQSOFit(np.full(300, 3.0), np.full(300, 1.0), wave)
    component_model = from_pyqsofit(fit, redshift=0.2, native_f_host=0.75)
    assert component_model.check_reconstruction().passed
    assert component_model.wavelength.max() < wave.max()   # rest frame applied


def test_pyqsofit_pseudo_equals_cont_and_says_so() -> None:
    """Its `qso` is a whole-nucleus PCA, so the two definitions cannot differ."""
    wave = np.linspace(6000.0, 6300.0, 300)
    fit = FakePyQSOFit(np.full(300, 3.0), np.full(300, 1.0), wave)
    component_model = from_pyqsofit(fit, redshift=0.0)
    result = component_model.canonical_fractions((6100.0, 6200.0))
    assert result["f_host_cont"] == pytest.approx(result["f_host_pseudo"])
    assert component_model.meta["pseudo_equals_cont"] is True


def test_pyqsofit_adapter_rejects_mismatched_arrays() -> None:
    fit = FakePyQSOFit(np.ones(300), np.ones(200), np.linspace(6000.0, 6300.0, 300))
    with pytest.raises(ValueError, match="missing or mismatched"):
        from_pyqsofit(fit, redshift=0.2)
