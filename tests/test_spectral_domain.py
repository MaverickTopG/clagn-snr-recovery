"""Domain-required spectral summaries.

The regression these lock in: the retracted D-044 conclusion came from summarising
E-MILES' wavelength-dependent FWHM over its full 1680-50000 A extent instead of the
fitted optical range. The full-range median is 4.94 A; the optical value is 2.51 A.
One number blocked the analysis, the other clears it.
"""

from __future__ import annotations

import numpy as np
import pytest

from p3sf.spectral_domain import (
    CoverageStatus,
    DomainError,
    coverage_mask,
    fwhm_in_fit_region,
    host_fraction_in_window,
    lsf_compatible,
    median_in_window,
    snr_in_window,
)


@pytest.fixture
def emiles_like():
    """A library whose resolution degrades into the near-IR, like E-MILES."""
    lam = np.linspace(1680.0, 50000.0, 20000)
    fwhm = np.where(lam < 7500.0, 2.51, 2.51 + (lam - 7500.0) * 0.0004)
    return lam, fwhm


# ---------------------------------------------------------------------------
# the D-044 regression
# ---------------------------------------------------------------------------


def test_full_range_median_disagrees_with_the_fitted_range(emiles_like) -> None:
    """The exact trap: both numbers are 'the median FWHM', only one is relevant."""
    lam, fwhm = emiles_like
    full_range = float(np.median(fwhm))
    fitted = fwhm_in_fit_region(lam, fwhm, (3700.0, 7000.0))
    assert fitted.value == pytest.approx(2.51)
    assert full_range > 4.0
    assert full_range > fitted.value * 1.5


def test_fit_region_is_mandatory() -> None:
    """No default domain exists, so the meaningless call cannot be written."""
    lam = np.linspace(3000.0, 9000.0, 100)
    with pytest.raises(TypeError):
        fwhm_in_fit_region(lam, np.full(100, 2.5))  # type: ignore[call-arg]


def test_lsf_verdict_flips_with_the_domain(emiles_like) -> None:
    """SDSS-like resolution against a library that degrades redward."""
    lam, fwhm = emiles_like
    optical = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))
    infrared = lsf_compatible(2.76, lam, fwhm, (20000.0, 40000.0))
    assert optical["compatible"] is True
    assert optical["clipped_fraction"] == 0.0
    assert infrared["compatible"] is False


def test_desi_resolution_is_genuinely_incompatible(emiles_like) -> None:
    """DESI at ~1.8 A is sharper than a 2.51 A basis; that concern was real."""
    lam, fwhm = emiles_like
    result = lsf_compatible(1.8, lam, fwhm, (3700.0, 7000.0))
    assert result["compatible"] is False
    assert result["clipped_fraction"] == pytest.approx(1.0)


def test_partial_incompatibility_is_reported_as_a_fraction(emiles_like) -> None:
    """A library can be usable over most of a range and fail at one end."""
    lam, fwhm = emiles_like
    result = lsf_compatible(3.5, lam, fwhm, (3700.0, 15000.0))
    assert 0.0 < result["clipped_fraction"] < 1.0


def test_mismatched_fwhm_and_lam_are_rejected() -> None:
    with pytest.raises(DomainError, match="must match"):
        fwhm_in_fit_region(np.linspace(4000, 5000, 100), np.full(50, 2.5), (4000.0, 5000.0))


def test_scalar_fwhm_is_broadcast(emiles_like) -> None:
    lam, _ = emiles_like
    assert fwhm_in_fit_region(lam, np.array([2.3]), (4000.0, 6000.0)).value == pytest.approx(2.3)


# ---------------------------------------------------------------------------
# sampling evidence
# ---------------------------------------------------------------------------


def test_sparse_window_returns_none_not_a_number() -> None:
    lam = np.linspace(4000.0, 9000.0, 50)
    result = median_in_window(lam, np.ones_like(lam), (4000.0, 4005.0))
    assert result.value is None
    assert not result.well_sampled


def test_float_conversion_refuses_an_absent_value() -> None:
    lam = np.linspace(4000.0, 9000.0, 50)
    result = median_in_window(lam, np.ones_like(lam), (4000.0, 4005.0))
    with pytest.raises(DomainError, match="no value"):
        float(result)


def test_pixel_count_travels_with_the_value() -> None:
    lam = np.linspace(4000.0, 6000.0, 1000)
    result = median_in_window(lam, np.full(1000, 7.0), (5000.0, 5200.0))
    assert result.value == pytest.approx(7.0)
    assert result.n_pixels > 50
    assert result.well_sampled


@pytest.mark.parametrize("window", [(5100.0, 5000.0), (5000.0, 5000.0)])
def test_non_increasing_windows_are_rejected(window) -> None:
    lam = np.linspace(4000.0, 6000.0, 100)
    with pytest.raises(DomainError, match="increasing"):
        median_in_window(lam, np.ones_like(lam), window)


def test_non_finite_bounds_are_rejected() -> None:
    lam = np.linspace(4000.0, 6000.0, 100)
    with pytest.raises(DomainError, match="finite"):
        median_in_window(lam, np.ones_like(lam), (np.nan, 5000.0))


def test_masked_pixels_are_excluded() -> None:
    lam = np.linspace(5000.0, 5200.0, 200)
    values = np.where(lam < 5100.0, 1.0, 100.0)
    good = lam < 5100.0
    assert median_in_window(lam, values, (5000.0, 5200.0), good=good).value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# host fraction and S/N
# ---------------------------------------------------------------------------


def test_host_fraction_matches_a_hand_computed_case() -> None:
    lam = np.linspace(5080.0, 5130.0, 100)
    fraction = host_fraction_in_window(lam, np.full(100, 3.0), np.full(100, 1.0),
                                       (5080.0, 5130.0))
    assert fraction.value == pytest.approx(0.75)


def test_host_fraction_none_when_total_flux_is_non_positive() -> None:
    lam = np.linspace(5080.0, 5130.0, 100)
    result = host_fraction_in_window(lam, np.full(100, -1.0), np.full(100, 1.0),
                                     (5080.0, 5130.0))
    assert result.value is None


def test_snr_in_window_is_the_median_ratio() -> None:
    lam = np.linspace(5080.0, 5130.0, 100)
    result = snr_in_window(lam, np.full(100, 20.0), np.full(100, 2.0), (5080.0, 5130.0))
    assert result.value == pytest.approx(10.0)


def test_snr_ignores_non_positive_errors() -> None:
    lam = np.linspace(5080.0, 5130.0, 100)
    error = np.where(lam < 5105.0, 2.0, 0.0)
    result = snr_in_window(lam, np.full(100, 20.0), error, (5080.0, 5130.0))
    assert result.value == pytest.approx(10.0)
    assert result.n_pixels < 100


# ---------------------------------------------------------------------------
# severity, not just extent
# ---------------------------------------------------------------------------


@pytest.fixture
def hbeta_only_incompatible():
    """Template resolution degraded across the Hbeta core alone.

    The case a bare incompatible fraction hides: ~2% of fitted pixels, but all
    of them sitting exactly where this project measures.
    """
    lam = np.linspace(1680.0, 50000.0, 20000)
    fwhm = np.where((lam > 4830.0) & (lam < 4890.0), 3.5, 2.51)
    return lam, fwhm


def test_small_bare_fraction_can_be_total_within_the_hbeta_core(
    hbeta_only_incompatible,
) -> None:
    lam, fwhm = hbeta_only_incompatible
    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))
    assert result["clipped_fraction"] < 0.03          # looks harmless
    assert result["critical_clipped_fraction"]["hbeta_core"] == pytest.approx(1.0)
    assert result["critical_any_incompatible"] is True


def test_worst_incompatibility_is_located(hbeta_only_incompatible) -> None:
    lam, fwhm = hbeta_only_incompatible
    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))
    assert 4830.0 <= result["worst_wavelength"] <= 4890.0
    assert result["min_difference_sq"] < 0
    assert result["worst_fwhm_template"] == pytest.approx(3.5)


def test_incompatibility_away_from_critical_regions_is_flagged_as_such() -> None:
    """Same extent, harmless location: the distinction the fraction alone misses."""
    lam = np.linspace(1680.0, 50000.0, 20000)
    fwhm = np.where((lam > 6800.0) & (lam < 6860.0), 3.5, 2.51)
    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))
    assert result["clipped_fraction"] > 0
    assert result["critical_clipped_fraction"]["hbeta_core"] == 0.0
    assert result["critical_any_incompatible"] is False


def test_fully_compatible_library_reports_positive_worst_case(emiles_like) -> None:
    lam, fwhm = emiles_like
    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))
    assert result["compatible"] is True
    assert result["min_difference_sq"] > 0
    assert result["critical_any_incompatible"] is False


def test_critical_subdomains_can_be_overridden() -> None:
    lam = np.linspace(3000.0, 9000.0, 10000)
    fwhm = np.where((lam > 6540.0) & (lam < 6590.0), 4.0, 2.0)
    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0),
                            critical_subdomains={"halpha": (6540.0, 6590.0)})
    assert result["critical_clipped_fraction"]["halpha"] == pytest.approx(1.0)
    assert "hbeta_core" not in result["critical_clipped_fraction"]


# ---------------------------------------------------------------------------
# wavelength span distinguishes sampling patterns
# ---------------------------------------------------------------------------


def test_span_distinguishes_clustered_from_spread_pixels() -> None:
    """Equal pixel counts, very different support."""
    window = (5000.0, 5200.0)
    clustered = np.linspace(5000.0, 5002.0, 20)
    spread = np.linspace(5000.0, 5200.0, 20)

    a = median_in_window(clustered, np.ones(20), window)
    b = median_in_window(spread, np.ones(20), window)

    assert a.n_pixels == b.n_pixels == 20
    assert a.wavelength_span < 5.0
    assert b.wavelength_span > 190.0
    assert a.window_coverage < 0.05
    assert b.window_coverage > 0.95


def test_span_is_none_when_nothing_contributes() -> None:
    lam = np.linspace(4000.0, 4100.0, 50)
    result = median_in_window(lam, np.ones(50), (5000.0, 5200.0))
    assert result.value is None
    assert result.wavelength_span is None
    assert result.window_coverage is None


def test_span_travels_through_host_fraction_and_snr() -> None:
    lam = np.linspace(5080.0, 5130.0, 100)
    fraction = host_fraction_in_window(lam, np.full(100, 3.0), np.full(100, 1.0),
                                       (5080.0, 5130.0))
    snr = snr_in_window(lam, np.full(100, 20.0), np.full(100, 2.0), (5080.0, 5130.0))
    assert fraction.wavelength_span == pytest.approx(50.0, abs=1.0)
    assert snr.wavelength_span == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# absent coverage must not read as compatibility
# ---------------------------------------------------------------------------


def test_masked_critical_region_is_not_reported_as_compatible() -> None:
    """The last semantic ambiguity: 0.0 could mean 'all fine' or 'nothing there'."""
    lam = np.linspace(3000.0, 9000.0, 12000)
    fwhm = np.full(lam.size, 2.51)
    galaxy_good = ~((lam > 4830.0) & (lam < 4890.0))   # Hbeta core masked out

    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0),
                            galaxy_lam=lam, galaxy_good=galaxy_good)
    core = result["critical"]["hbeta_core"]

    assert core.status is not CoverageStatus.COMPATIBLE
    assert core.clipped_fraction is None       # not 0.0
    assert not core.verdict_available
    assert result["critical_all_evaluated"] is False


def test_region_outside_coverage_reports_no_coverage() -> None:
    lam = np.linspace(5200.0, 9000.0, 8000)     # starts redward of Hbeta
    fwhm = np.full(lam.size, 2.51)
    result = lsf_compatible(2.76, lam, fwhm, (5200.0, 7000.0))
    core = result["critical"]["hbeta_core"]
    assert core.status is CoverageStatus.NO_COVERAGE
    assert core.clipped_fraction is None
    assert core.n_pixels == 0


def test_well_covered_compatible_region_is_positively_evaluated(emiles_like) -> None:
    lam, fwhm = emiles_like
    result = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))
    core = result["critical"]["hbeta_core"]
    assert core.status is CoverageStatus.COMPATIBLE
    assert core.clipped_fraction == 0.0
    assert core.verdict_available
    assert result["critical_all_evaluated"] is True


def test_incompatible_region_is_distinguished_from_uncovered(
    hbeta_only_incompatible,
) -> None:
    lam, fwhm = hbeta_only_incompatible
    core = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))["critical"]["hbeta_core"]
    assert core.status is CoverageStatus.INCOMPATIBLE
    assert core.clipped_fraction == pytest.approx(1.0)
    assert core.verdict_available


def test_subdomain_carries_span_and_coverage(emiles_like) -> None:
    lam, fwhm = emiles_like
    core = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))["critical"]["hbeta_core"]
    assert core.wavelength_span is not None
    assert core.window_coverage > 0.9


def test_global_assessment_excludes_masked_galaxy_pixels() -> None:
    """Masked data must not implicitly contribute to a passing global verdict.

    The assertions here are structural, not magnitude-based. An earlier version
    asserted a tenfold reduction, but that number came from the ratio of the
    coverage tolerance to this fixture's incompatible-region width — an
    incidental property of the sampling, exactly the kind of empirical magnitude
    that should not be frozen into a software test.

    What *is* guaranteed by construction: masking shrinks the evaluated pixel
    set, the masked pixels themselves cannot participate, and any incompatible
    pixel that survives must lie within the proximity tolerance of the mask
    boundary.
    """
    lam = np.linspace(3000.0, 9000.0, 12000)
    incompatible_region = (lam > 4830.0) & (lam < 4890.0)
    fwhm = np.where(incompatible_region, 3.5, 2.51)
    tolerance = 2.0 * float(np.median(np.diff(lam)))

    with_mask = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0),
                               galaxy_lam=lam, galaxy_good=~incompatible_region)
    without_mask = lsf_compatible(2.76, lam, fwhm, (3700.0, 7000.0))

    # 1. Masking strictly shrinks the evaluated set and the incompatible share.
    assert with_mask["n_pixels"] < without_mask["n_pixels"]
    assert with_mask["clipped_fraction"] < without_mask["clipped_fraction"]

    # 2. Every surviving incompatible pixel sits within tolerance of the edge.
    surviving = coverage_mask(lam, lam, ~incompatible_region) & incompatible_region
    if surviving.any():
        distance_to_edge = np.minimum(
            np.abs(lam[surviving] - 4830.0), np.abs(lam[surviving] - 4890.0)
        )
        assert distance_to_edge.max() <= tolerance

    # 3. The critical region is then unevaluated, not passed.
    assert with_mask["critical"]["hbeta_core"].status is not CoverageStatus.COMPATIBLE
    assert with_mask["critical_all_evaluated"] is False


def test_entirely_masked_galaxy_raises_rather_than_passing() -> None:
    lam = np.linspace(3000.0, 9000.0, 12000)
    with pytest.raises(DomainError, match="usable pixels"):
        lsf_compatible(2.76, lam, np.full(lam.size, 2.51), (3700.0, 7000.0),
                       galaxy_lam=lam, galaxy_good=np.zeros(lam.size, dtype=bool))
