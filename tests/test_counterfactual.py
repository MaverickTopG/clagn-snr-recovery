"""The counterfactual engine: S/N degradation and host injection.

These are the tests from the project specification §75. They check the engine
against *analytically known* truth, independently of the spectral fitter, so a
failure here is an engine bug rather than a fitting artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from p3sf.access.spectra import Spectrum
from p3sf.counterfactual.host import (
    Decomposition,
    PairFluxBasis,
    added_star_scale_for_target,
    fixed_continuum_snr_intervention,
    host_scale_for_target,
    inject_host_fraction,
    measure_host_fraction_directly,
    realized_added_star_fraction,
    reference_anchored_pair_amplitude,
)
from p3sf.counterfactual.snr import (
    degradation_ladder,
    degrade_to_snr,
    measure_snr,
)

# ---------------------------------------------------------------------------
# S/N degradation
# ---------------------------------------------------------------------------


def test_measured_snr_matches_construction(clean_spectrum, continuum_window) -> None:
    assert measure_snr(clean_spectrum, continuum_window) == pytest.approx(50.0, rel=1e-6)


@pytest.mark.parametrize("target", [5.0, 10.0, 20.0, 30.0, 40.0])
def test_degradation_achieves_the_requested_snr(clean_spectrum, continuum_window, target) -> None:
    """The error array must actually report the target S/N after degradation."""
    result = degrade_to_snr(clean_spectrum, target, window_rest=continuum_window, seed=1)
    # measure_snr uses flux/error, and the degraded flux carries a noise draw,
    # so compare the intended sigma scaling rather than one noisy realization.
    ratio = result.spectrum.error / clean_spectrum.error
    assert np.allclose(ratio[clean_spectrum.good], 50.0 / target, rtol=1e-9)
    assert result.scale_factor == pytest.approx(50.0 / target, rel=1e-9)


def test_degradation_reduces_snr_on_average_across_realizations(
    clean_spectrum, continuum_window
) -> None:
    achieved = [
        degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=s).achieved_snr
        for s in range(40)
    ]
    assert np.mean(achieved) == pytest.approx(10.0, rel=0.10)


def test_upgrading_is_refused(clean_spectrum, continuum_window) -> None:
    """Degradation is one-directional. This is a preregistered commitment."""
    with pytest.raises(ValueError, match="one-directional"):
        degrade_to_snr(clean_spectrum, 80.0, window_rest=continuum_window, seed=1)


def test_error_message_names_the_exclusion_code(clean_spectrum, continuum_window) -> None:
    with pytest.raises(ValueError, match="REFERENCE_SNR_TOO_LOW_TO_DEGRADE"):
        degrade_to_snr(clean_spectrum, 99.0, window_rest=continuum_window, seed=1)


def test_zero_perturbation_is_an_exact_round_trip(clean_spectrum, continuum_window) -> None:
    """Degrading to the spectrum's own S/N must return it unchanged."""
    result = degrade_to_snr(clean_spectrum, 50.0, window_rest=continuum_window, seed=7)
    assert np.array_equal(result.spectrum.flux, clean_spectrum.flux)
    assert np.allclose(result.spectrum.error, clean_spectrum.error, rtol=0, atol=0)
    assert result.scale_factor == pytest.approx(1.0)


def test_same_seed_reproduces_the_same_spectrum_bit_for_bit(
    clean_spectrum, continuum_window
) -> None:
    a = degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=42)
    b = degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=42)
    assert np.array_equal(a.spectrum.flux, b.spectrum.flux)


def test_different_seeds_give_different_realizations(clean_spectrum, continuum_window) -> None:
    a = degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=1)
    b = degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=2)
    assert not np.array_equal(a.spectrum.flux, b.spectrum.flux)


def test_added_noise_preserves_the_wavelength_structure_of_sigma(
    clean_spectrum, continuum_window
) -> None:
    """Flat white noise would make simulated low-S/N spectra easier than real ones."""
    result = degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=3)
    good = clean_spectrum.good
    ratio = result.spectrum.error[good] / clean_spectrum.error[good]
    assert np.std(ratio) < 1e-12, "sigma scaling must be a single factor, preserving its shape"
    assert np.std(result.spectrum.error[good]) > 0, "sigma must remain wavelength-dependent"


def test_degradation_does_not_bias_the_flux(clean_spectrum, continuum_window) -> None:
    """Added noise is zero-mean, so the ensemble mean flux is unchanged."""
    stack = np.array(
        [
            degrade_to_snr(clean_spectrum, 10.0, window_rest=continuum_window, seed=s).spectrum.flux
            for s in range(200)
        ]
    )
    residual = stack.mean(axis=0) - clean_spectrum.flux
    tolerance = 5.0 * (clean_spectrum.error * 5.0) / np.sqrt(200)
    assert np.all(np.abs(residual) < tolerance)


def test_metadata_records_what_was_done(clean_spectrum, continuum_window) -> None:
    result = degrade_to_snr(clean_spectrum, 15.0, window_rest=continuum_window, seed=9)
    meta = result.spectrum.meta
    assert meta["counterfactual_snr_target"] == 15.0
    assert meta["counterfactual_snr_seed"] == 9
    assert meta["counterfactual_snr_original"] == pytest.approx(50.0, rel=1e-6)


def test_ladder_skips_unreachable_rungs(clean_spectrum, continuum_window) -> None:
    ladder = degradation_ladder(
        clean_spectrum, [5, 10, 20, 30, 40, 80], window_rest=continuum_window, base_seed=314159
    )
    assert set(ladder) == {5, 10, 20, 30, 40}, "80 exceeds the spectrum's own S/N of 50"


def test_ladder_rungs_are_independent_and_order_free(clean_spectrum, continuum_window) -> None:
    forward = degradation_ladder(
        clean_spectrum, [5, 10, 20], window_rest=continuum_window, base_seed=1
    )
    reverse = degradation_ladder(
        clean_spectrum, [20, 10, 5], window_rest=continuum_window, base_seed=1
    )
    for snr in (5, 10, 20):
        assert np.array_equal(forward[snr].spectrum.flux, reverse[snr].spectrum.flux)
    assert not np.array_equal(forward[5].spectrum.flux, forward[10].spectrum.flux)


def test_ladder_seeds_are_stable_across_processes(clean_spectrum, continuum_window) -> None:
    """Uses SHA-256, not hash(), which is salted per process."""
    from p3sf.counterfactual.snr import _derive_seed

    assert _derive_seed(314159, 0, 10.0) == _derive_seed(314159, 0, 10.0)
    assert _derive_seed(314159, 0, 10.0) != _derive_seed(314159, 1, 10.0)
    assert _derive_seed(314159, 0, 10.0) != _derive_seed(314159, 0, 20.0)


# ---------------------------------------------------------------------------
# host injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.0, 0.1, 0.5, 0.85])
def test_additive_external_host_realizes_requested_fraction_for_nonflat_shapes(
    target,
) -> None:
    """D-082 design identity: amplitude and host shape are separate factors."""
    redshift = 0.25
    rest = np.linspace(4700.0, 5300.0, 1201)
    wavelength = rest * (1.0 + redshift)
    baseline = 7.0 + 0.004 * (rest - 5000.0)
    # Deliberately non-flat and with Hbeta absorption: a coefficient shared by
    # different templates cannot stand in for the realized continuum fraction.
    host = (
        1.3 + 0.0015 * (rest - 5000.0)
        - 0.25 * np.exp(-0.5 * ((rest - 4861.33) / 7.0) ** 2)
    )
    window = (5080.0, 5130.0)
    scale = added_star_scale_for_target(
        wavelength, baseline, host, redshift=redshift,
        requested_f_added_star=target, window_rest=window,
    )
    achieved = realized_added_star_fraction(
        wavelength, baseline, scale * host, redshift=redshift, window_rest=window,
    )
    assert achieved == pytest.approx(target, abs=1e-12)


def test_additive_external_host_does_not_require_native_host_separation() -> None:
    rest = np.linspace(5000.0, 5200.0, 401)
    wavelength = rest * 1.2
    # This baseline intentionally mixes AGN and an unknown native host.  The
    # additive design never asks for either component separately.
    observed_baseline = np.full(rest.shape, 12.0)
    external_host = np.linspace(1.0, 2.0, rest.size)
    scale = added_star_scale_for_target(
        wavelength, observed_baseline, external_host, redshift=0.2,
        requested_f_added_star=0.5, window_rest=(5080.0, 5130.0),
    )
    assert realized_added_star_fraction(
        wavelength, observed_baseline, scale * external_host, redshift=0.2,
        window_rest=(5080.0, 5130.0),
    ) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("requested", [0.0, 0.1, 0.5, 0.85])
def test_pair_amplitude_is_solved_on_faint_and_shared_with_bright(requested) -> None:
    rest = np.linspace(5000.0, 5200.0, 401)
    host = np.full(rest.shape, 2.0)
    result = reference_anchored_pair_amplitude(
        rest * 1.2, np.full(rest.shape, 5.0), host,
        rest * 1.2, np.full(rest.shape, 10.0), host,
        reference_redshift=0.2, other_redshift=0.2,
        requested_f_added_star_reference=requested,
        window_rest=(5080.0, 5130.0),
        flux_basis=PairFluxBasis.SAME_SURVEY_SAME_APERTURE,
    )
    assert result.realized_f_added_star_reference == pytest.approx(requested, abs=1e-12)
    assert result.realized_f_added_star_other == pytest.approx(
        requested / (2.0 - requested), abs=1e-12
    )


def test_pair_amplitude_refuses_noncomparable_flux_basis() -> None:
    rest = np.linspace(5000.0, 5200.0, 401)
    with pytest.raises(ValueError, match="comparable flux/aperture basis"):
        reference_anchored_pair_amplitude(
            rest, np.ones_like(rest), np.ones_like(rest),
            rest, np.ones_like(rest), np.ones_like(rest),
            reference_redshift=0.0, other_redshift=0.0,
            requested_f_added_star_reference=0.1,
            window_rest=(5080.0, 5130.0),
            flux_basis=PairFluxBasis.NOT_COMPARABLE,
        )


def test_fixed_continuum_snr_intervention_is_exact_and_deterministic(
    clean_spectrum, continuum_window
) -> None:
    added = 2.0 + 0.1 * np.sin(clean_spectrum.wavelength / 20.0)
    first = fixed_continuum_snr_intervention(
        clean_spectrum, added, window_rest=continuum_window
    )
    second = fixed_continuum_snr_intervention(
        clean_spectrum, added, window_rest=continuum_window
    )
    assert first.achieved_snr == pytest.approx(first.baseline_snr, rel=1e-14)
    assert np.array_equal(first.spectrum.flux, second.spectrum.flux)
    assert np.array_equal(first.spectrum.error, second.spectrum.error)
    assert first.spectrum.meta["counterfactual_random_noise_generated"] is False


def test_fixed_continuum_snr_zero_addition_is_exact_baseline_noop(
    clean_spectrum, continuum_window
) -> None:
    result = fixed_continuum_snr_intervention(
        clean_spectrum, np.zeros_like(clean_spectrum.flux),
        window_rest=continuum_window,
    )
    assert result.spectrum is clean_spectrum
    assert result.uncertainty_scale == 1.0
    assert np.array_equal(result.spectrum.wavelength, clean_spectrum.wavelength)
    assert np.array_equal(result.spectrum.flux, clean_spectrum.flux)
    assert np.array_equal(result.spectrum.error, clean_spectrum.error)
    assert result.spectrum.mask is clean_spectrum.mask


def test_fixture_host_fraction_is_one_half(decomposition) -> None:
    assert decomposition.host_fraction() == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("target", [0.1, 0.3, 0.5, 0.7, 0.85])
def test_injected_host_fraction_is_recovered_without_refitting(
    clean_spectrum, decomposition, target
) -> None:
    """Spec §75: the constructed spectrum must have the requested host fraction.

    Measured directly from the components, so this is an engine check that does
    not depend on the fitter.
    """
    result = inject_host_fraction(clean_spectrum, decomposition, target)
    assert result.achieved_fraction == pytest.approx(target, abs=1e-9)

    rebuilt = Decomposition(
        wavelength=decomposition.wavelength,
        agn_continuum=decomposition.agn_continuum,
        host=result.host_scale * decomposition.host,
        lines=decomposition.lines,
        redshift=decomposition.redshift,
    )
    assert measure_host_fraction_directly(result.spectrum, rebuilt) == pytest.approx(
        target, abs=1e-9
    )


def test_host_scale_solves_the_defining_equation(decomposition) -> None:
    scale = host_scale_for_target(decomposition, 0.8)
    # host and AGN are equal at 5100 A, so a*H/(a*H + A) = 0.8 -> a = 4
    assert scale == pytest.approx(4.0, rel=1e-9)


def test_zero_host_fraction_removes_the_host(clean_spectrum, decomposition) -> None:
    result = inject_host_fraction(clean_spectrum, decomposition, 0.0)
    assert result.host_scale == 0.0
    assert result.achieved_fraction == pytest.approx(0.0, abs=1e-12)


def test_host_fraction_of_one_is_rejected(decomposition) -> None:
    """f_host = 1 means no AGN at all, which is not an observation of this object."""
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        host_scale_for_target(decomposition, 1.0)


def test_missing_host_template_is_refused_not_substituted(decomposition) -> None:
    """Preregistration forbids adding a foreign galaxy spectrum."""
    hostless = Decomposition(
        wavelength=decomposition.wavelength,
        agn_continuum=decomposition.agn_continuum,
        host=np.zeros_like(decomposition.host),
        lines=decomposition.lines,
        redshift=decomposition.redshift,
    )
    with pytest.raises(ValueError, match="NO_HOST_TEMPLATE"):
        host_scale_for_target(hostless, 0.5)


def test_line_flux_rides_with_the_agn_not_the_host(clean_spectrum, decomposition) -> None:
    """Scaling the host must not dilute the emission line we are trying to detect."""
    low = inject_host_fraction(clean_spectrum, decomposition, 0.1, rescale_noise=False)
    high = inject_host_fraction(clean_spectrum, decomposition, 0.85, rescale_noise=False)

    rest = decomposition.rest_wavelength()
    line_region = np.abs(rest - 4861.33) < 40.0

    # Continuum-subtracted line flux, using the known components.
    def line_flux(spectrum: Spectrum, scale: float) -> float:
        continuum = decomposition.agn_continuum + scale * decomposition.host
        return float(np.trapezoid((spectrum.flux - continuum)[line_region], rest[line_region]))

    assert line_flux(low.spectrum, low.host_scale) == pytest.approx(
        line_flux(high.spectrum, high.host_scale), rel=1e-6
    )


def test_noise_is_rescaled_with_the_continuum_level(clean_spectrum, decomposition) -> None:
    """Adding host light must not artificially improve per-pixel S/N."""
    result = inject_host_fraction(clean_spectrum, decomposition, 0.85, rescale_noise=True)
    assert np.all(result.spectrum.error >= clean_spectrum.error)

    unscaled = inject_host_fraction(clean_spectrum, decomposition, 0.85, rescale_noise=False)
    assert np.allclose(unscaled.spectrum.error, clean_spectrum.error)


def test_observed_residual_is_carried_into_the_counterfactual(
    noisy_spectrum, decomposition
) -> None:
    """Real fitting imperfections must survive, or simulations look too clean."""
    result = inject_host_fraction(noisy_spectrum, decomposition, 0.5, rescale_noise=False)
    model = decomposition.agn_continuum + decomposition.host + decomposition.lines
    original_residual = noisy_spectrum.flux - model
    new_model = (
        decomposition.agn_continuum
        + decomposition.lines
        + result.host_scale * decomposition.host
    )
    assert np.allclose(result.spectrum.flux - new_model, original_residual, rtol=0, atol=1e-9)


def test_injection_is_a_no_op_at_the_original_fraction(clean_spectrum, decomposition) -> None:
    result = inject_host_fraction(clean_spectrum, decomposition, 0.5, rescale_noise=False)
    assert np.allclose(result.spectrum.flux, clean_spectrum.flux, rtol=0, atol=1e-9)


def test_mismatched_grids_are_rejected(clean_spectrum, decomposition) -> None:
    shifted = Decomposition(
        wavelength=decomposition.wavelength + 1.0,
        agn_continuum=decomposition.agn_continuum,
        host=decomposition.host,
        lines=decomposition.lines,
        redshift=decomposition.redshift,
    )
    with pytest.raises(ValueError, match="wavelength grids differ"):
        inject_host_fraction(clean_spectrum, shifted, 0.5)


def test_component_shape_mismatch_is_caught_at_construction(decomposition) -> None:
    with pytest.raises(ValueError, match="do not match wavelength"):
        Decomposition(
            wavelength=decomposition.wavelength,
            agn_continuum=decomposition.agn_continuum[:-5],
            host=decomposition.host,
            lines=decomposition.lines,
            redshift=decomposition.redshift,
        )


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_host_then_snr_compose_to_the_requested_condition(
    clean_spectrum, decomposition, continuum_window
) -> None:
    """The factorial grid applies both perturbations; the combination must hold."""
    hosted = inject_host_fraction(clean_spectrum, decomposition, 0.7)
    degraded = degrade_to_snr(
        hosted.spectrum, 10.0, window_rest=continuum_window, seed=11
    )
    assert hosted.achieved_fraction == pytest.approx(0.7, abs=1e-9)
    ratio = degraded.spectrum.error[hosted.spectrum.good] / hosted.spectrum.error[hosted.spectrum.good]
    expected = measure_snr(hosted.spectrum, continuum_window) / 10.0
    assert np.allclose(ratio, expected, rtol=1e-9)
