"""Rest-frame and stellar-alignment checks.

These replace an argmax diagnostic that reported +37 A offsets by finding the
maximum of a rising continuum at its window edge. The central requirement is
that nothing here defaults to PASS.
"""

from __future__ import annotations

import numpy as np
import pytest

from p3sf.qc.alignment import (
    OIII_5007,
    AlignmentStatus,
    check_rest_frame,
    check_stellar_alignment,
)


def synthetic(offset: float = 0.0, *, amplitude: float = 50.0, slope: float = 0.0):
    """Continuum plus a Gaussian [O III] 5007 displaced by `offset`."""
    lam = np.linspace(4900.0, 5120.0, 900)
    continuum = 10.0 + slope * (lam - 5000.0)
    line = amplitude * np.exp(-0.5 * ((lam - (OIII_5007 + offset)) / 2.5) ** 2)
    return lam, continuum + line, np.full(lam.size, 0.5)


# ---------------------------------------------------------------------------
# rest frame
# ---------------------------------------------------------------------------


def test_aligned_line_passes_and_recovers_the_centroid() -> None:
    result = check_rest_frame(*synthetic(0.0))
    assert result.status is AlignmentStatus.PASS
    assert result.fitted_centroid == pytest.approx(OIII_5007, abs=0.3)
    assert abs(result.offset_kms) < 30.0


@pytest.mark.parametrize("offset", [-30.0, -12.0, 12.0, 30.0])
def test_displaced_line_is_detected(offset: float) -> None:
    result = check_rest_frame(*synthetic(offset))
    assert result.status is AlignmentStatus.FAIL
    assert result.offset_angstrom == pytest.approx(offset, abs=2.0)


def test_steep_continuum_does_not_fake_a_detection() -> None:
    """The exact argmax failure: a rising continuum with no line at all."""
    lam = np.linspace(4900.0, 5120.0, 900)
    flux = 10.0 + 0.5 * (lam - 4900.0)          # strongly rising, no [O III]
    result = check_rest_frame(lam, flux, np.full(lam.size, 0.5))
    assert result.status is not AlignmentStatus.PASS
    assert result.status is AlignmentStatus.NOT_MEASURABLE


def test_absent_line_reports_not_measurable_rather_than_a_position() -> None:
    result = check_rest_frame(*synthetic(0.0, amplitude=0.0))
    assert result.status is AlignmentStatus.NOT_MEASURABLE
    assert result.offset_angstrom is None


def test_missing_coverage_is_explicit() -> None:
    lam = np.linspace(4000.0, 4200.0, 400)       # [O III] not covered
    result = check_rest_frame(lam, np.full(400, 10.0), np.full(400, 0.5))
    assert result.status is AlignmentStatus.INSUFFICIENT_COVERAGE
    assert result.n_pixels < 12


def test_velocity_offset_is_consistent_with_the_wavelength_offset() -> None:
    result = check_rest_frame(*synthetic(5.0))
    assert result.offset_kms == pytest.approx(
        299792.458 * result.offset_angstrom / OIII_5007, rel=1e-6
    )


# ---------------------------------------------------------------------------
# stellar alignment, independent of the emission line
# ---------------------------------------------------------------------------


def stellar_spectrum(shift: float = 0.0):
    """Continuum with absorption at Ca K, Ca H, G band, Hbeta and Mg b."""
    lam = np.linspace(3850.0, 5250.0, 3000)
    flux = np.full(lam.size, 100.0)
    for centre in (3933.66, 3968.47, 4304.40, 4861.33, 5175.00):
        flux -= 30.0 * np.exp(-0.5 * ((lam - (centre + shift)) / 3.0) ** 2)
    return lam, flux, np.full(lam.size, 0.5)


def test_aligned_stellar_features_pass() -> None:
    alignment = check_stellar_alignment(*stellar_spectrum(0.0))
    assert alignment.status is AlignmentStatus.PASS
    assert alignment.n_measured >= 2
    assert abs(alignment.median_offset_kms) < 60.0


def test_small_stellar_shift_within_tolerance_passes() -> None:
    """6 A is inside the 8 A verdict threshold, so this must NOT fail."""
    assert check_stellar_alignment(*stellar_spectrum(6.0)).status is AlignmentStatus.PASS


def test_shifted_stellar_system_is_detected() -> None:
    """[O III] can be perfect while the template system is displaced."""
    alignment = check_stellar_alignment(*stellar_spectrum(15.0))
    assert alignment.status is AlignmentStatus.FAIL
    assert abs(alignment.median_offset_kms) > 500.0


def test_single_feature_is_not_enough_to_pass() -> None:
    """One absorption line can be mimicked by a continuum inflection."""
    lam = np.linspace(3910.0, 3960.0, 400)       # Ca II K only
    flux = 100.0 - 30.0 * np.exp(-0.5 * ((lam - 3933.66) / 3.0) ** 2)
    alignment = check_stellar_alignment(lam, flux, np.full(lam.size, 0.5))
    assert alignment.n_measured <= 1
    assert alignment.status is AlignmentStatus.INSUFFICIENT_COVERAGE


def test_hbeta_absorption_is_not_a_default_probe() -> None:
    """Broad Hbeta emission makes an absorption probe there meaningless in AGN.

    Measured on the diagnostics, Hbeta_abs reported -31.3 and +28.1 A while
    every genuine stellar feature in the same spectrum agreed to ~2 A.
    """
    from p3sf.qc.alignment import HOST_DOMINATED_FEATURES, STELLAR_FEATURES

    assert "Hbeta_abs" not in STELLAR_FEATURES
    assert "Hbeta_abs" in HOST_DOMINATED_FEATURES


def test_broad_emission_does_not_corrupt_the_default_stellar_verdict() -> None:
    """A strong broad Hbeta emission line must not flip the alignment verdict."""
    lam, flux, error = stellar_spectrum(0.0)
    contaminated = flux + 80.0 * np.exp(-0.5 * ((lam - 4861.33) / 40.0) ** 2)
    assert check_stellar_alignment(lam, contaminated, error).status is AlignmentStatus.PASS


def test_featureless_spectrum_does_not_pass() -> None:
    lam = np.linspace(3850.0, 5250.0, 3000)
    alignment = check_stellar_alignment(lam, np.full(lam.size, 100.0),
                                        np.full(lam.size, 0.5))
    assert alignment.status is not AlignmentStatus.PASS


def test_emission_and_stellar_checks_are_independent() -> None:
    """A spectrum with a correct line but shifted stars must fail only the latter."""
    lam, flux, error = stellar_spectrum(15.0)
    flux = flux + 50.0 * np.exp(-0.5 * ((lam - OIII_5007) / 2.5) ** 2)
    assert check_rest_frame(lam, flux, error).status is AlignmentStatus.PASS
    assert check_stellar_alignment(lam, flux, error).status is AlignmentStatus.FAIL
