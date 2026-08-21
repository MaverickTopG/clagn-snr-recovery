"""Deterministic D-089 source-measurement fixtures; no real-object labels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from p3sf.config import load_config
from p3sf.criteria.base import Label
from p3sf.criteria.green_pixel import (
    GreenEpochSpectrum,
    VarianceProvenance,
    classify_green_pixel_measurement,
    measure_green_pixel_nsigma,
)
from p3sf.criteria.source_verified import apply_yang2024_hbeta
from p3sf.fitting.pyqsofit_driver import YangHbetaFitAudit


def yang_audit(*, flux: float, present: bool = True) -> YangHbetaFitAudit:
    return YangHbetaFitAudit(
        fit_completed=True,
        optimizer_success=True,
        parameter_at_bound=False,
        local_hbeta_coverage=True,
        local_mask_fraction=0.0,
        continuum_valid=True,
        broad_component_present=present,
        broad_component_amplitude=flux,
        broad_component_width=4000.0 if present else None,
        broad_flux=flux,
        residual_rms_local=1.0,
        finite_measurement=True,
        fit_valid=True,
        invalid_reason="",
        valid_nondetection=not present and flux == 0.0,
    )


def green_epoch(
    nsigma: np.ndarray,
    *,
    unsupported_index: int | None = None,
    remove: slice | None = None,
) -> GreenEpochSpectrum:
    wave = np.arange(4749.0, 4942.0, 1.0)
    # With 1-A input sampling, each 2-A output bin has overlap weights
    # (1/4, 1/2, 1/4). Input variance 4/3 therefore gives rebinned variance 1/2
    # in each epoch and a combined bright/faint denominator of exactly one.
    variance = np.full(len(wave), 4.0 / 3.0)
    provenance = np.full(
        len(wave), VarianceProvenance.NATIVE_SDSS_SUPPORTED.value, dtype=object
    )
    if unsupported_index is not None:
        provenance[unsupported_index] = VarianceProvenance.UNSUPPORTED.value
    keep = np.ones(len(wave), dtype=bool)
    if remove is not None:
        keep[remove] = False
    return GreenEpochSpectrum(
        wave[keep], nsigma[keep], variance[keep], provenance[keep]
    )


def test_yang_valid_flux_arithmetic_and_strict_threshold() -> None:
    assert apply_yang2024_hbeta(
        bright=yang_audit(flux=100.0), faint=yang_audit(flux=20.0)
    ).label is Label.CL
    edge = apply_yang2024_hbeta(
        bright=yang_audit(flux=100.0), faint=yang_audit(flux=30.0)
    )
    assert edge.statistic == 0.3
    assert edge.label is Label.NON_CL


def test_yang_only_valid_fitted_nondetection_becomes_zero() -> None:
    valid = apply_yang2024_hbeta(
        bright=yang_audit(flux=100.0), faint=yang_audit(flux=0.0, present=False)
    )
    assert valid.label is Label.CL
    assert valid.statistic == 0.0
    unavailable = replace(
        yang_audit(flux=0.0, present=False), valid_nondetection=False
    )
    assert apply_yang2024_hbeta(
        bright=yang_audit(flux=100.0), faint=unavailable
    ).label is Label.UNCLASSIFIABLE


def test_yang_boundary_or_failure_is_unclassifiable() -> None:
    boundary = replace(
        yang_audit(flux=20.0),
        parameter_at_bound=True,
        fit_valid=False,
        invalid_reason="ACTIVE_BROAD_COMPONENT_PARAMETER_AT_BOUND",
    )
    failed = replace(
        yang_audit(flux=20.0),
        fit_completed=False,
        optimizer_success=False,
        fit_valid=False,
        invalid_reason="FIT_DID_NOT_COMPLETE",
    )
    bright = yang_audit(flux=100.0)
    assert apply_yang2024_hbeta(bright=bright, faint=boundary).label is Label.UNCLASSIFIABLE
    assert apply_yang2024_hbeta(bright=bright, faint=failed).label is Label.UNCLASSIFIABLE


def test_yang_source_width_domain_failure_is_unclassifiable() -> None:
    outside_source_domain = replace(
        yang_audit(flux=20.0),
        broad_component_width=20_001.0,
        fit_valid=False,
        invalid_reason="BROAD_COMPONENT_WIDTH_OUTSIDE_YANG_SOURCE_RANGE",
    )
    assert apply_yang2024_hbeta(
        bright=yang_audit(flux=100.0), faint=outside_source_domain
    ).label is Label.UNCLASSIFIABLE


def test_yang_source_width_domain_is_frozen_in_config() -> None:
    config = load_config()
    assert config.fitting.broad_fwhm_min_kms == 1200.0
    assert config.fitting.yang_broad_fwhm_max_kms == 20_000.0


def test_green_pixel_nsigma_arithmetic_and_threshold_edge() -> None:
    wave = np.arange(4749.0, 4942.0, 1.0)
    faint_values = np.zeros(len(wave))
    bright_values = np.where(wave >= 4820.0, 3.0, 0.0)
    measurement = measure_green_pixel_nsigma(
        green_epoch(bright_values), green_epoch(faint_values)
    )
    assert measurement.measurement_available
    assert measurement.nsigma_hbeta == pytest.approx(3.0)
    # Green's Table-2 catalogue note is inclusive at exactly three.
    assert classify_green_pixel_measurement(
        replace(measurement, nsigma_hbeta=3.0)
    ).label is Label.CL


def test_green_uses_bright_minus_faint_not_absolute_difference() -> None:
    wave = np.arange(4749.0, 4942.0, 1.0)
    faint_values = np.where(wave >= 4820.0, 4.0, 0.0)
    bright_values = np.zeros(len(wave))
    measurement = measure_green_pixel_nsigma(
        green_epoch(bright_values), green_epoch(faint_values)
    )
    assert measurement.measurement_available
    assert measurement.nsigma_hbeta == 0.0


def test_green_mask_gap_and_unsupported_variance_fail_closed() -> None:
    wave = np.arange(4749.0, 4942.0, 1.0)
    values = np.where(wave >= 4820.0, 4.0, 0.0)
    masked = measure_green_pixel_nsigma(
        green_epoch(values, remove=slice(40, 43)), green_epoch(np.zeros(len(wave)))
    )
    unsupported = measure_green_pixel_nsigma(
        green_epoch(values, unsupported_index=80), green_epoch(np.zeros(len(wave)))
    )
    assert not masked.measurement_available
    assert not unsupported.measurement_available
    assert classify_green_pixel_measurement(masked).label is Label.UNCLASSIFIABLE
    assert classify_green_pixel_measurement(unsupported).label is Label.UNCLASSIFIABLE


def test_green_ignores_mask_gaps_outside_required_hbeta_interval() -> None:
    wave = np.arange(4749.0, 4942.0, 1.0)
    values = np.where(wave >= 4820.0, 4.0, 0.0)
    bright = green_epoch(values)
    faint = green_epoch(np.zeros(len(wave)))
    bright_with_remote_gap = GreenEpochSpectrum(
        wavelength_rest=np.concatenate(([4600.0, 4700.0], bright.wavelength_rest)),
        line_flux=np.concatenate(([0.0, 0.0], bright.line_flux)),
        variance=np.concatenate(([1.0, 1.0], bright.variance)),
        variance_provenance=np.concatenate((
            np.array([VarianceProvenance.NATIVE_SDSS_SUPPORTED.value] * 2),
            bright.variance_provenance,
        )),
    )
    assert measure_green_pixel_nsigma(
        bright_with_remote_gap, faint
    ).measurement_available


@pytest.mark.parametrize(
    "provenance",
    [
        VarianceProvenance.NATIVE_LAMOST_SUPPORTED,
        VarianceProvenance.NATIVE_SDSSV_SUPPORTED,
        VarianceProvenance.NATIVE_DESI_EDR_SUPPORTED,
    ],
)
def test_green_accepts_d093_validated_native_variance_domains(
    provenance: VarianceProvenance,
) -> None:
    wave = np.arange(4749.0, 4942.0, 1.0)
    faint_values = np.zeros(len(wave))
    bright_values = np.where(wave >= 4820.0, 3.0, 0.0)
    bright = green_epoch(bright_values)
    faint = green_epoch(faint_values)
    bright = GreenEpochSpectrum(
        bright.wavelength_rest,
        bright.line_flux,
        bright.variance,
        np.full(len(bright.wavelength_rest), provenance.value, dtype=object),
    )
    faint = GreenEpochSpectrum(
        faint.wavelength_rest,
        faint.line_flux,
        faint.variance,
        np.full(len(faint.wavelength_rest), provenance.value, dtype=object),
    )
    result = measure_green_pixel_nsigma(bright, faint)
    assert result.measurement_available
    assert result.all_variance_supported
