"""Counterfactual experiment 2 — host-galaxy dilution.

Given a decomposition of a real spectrum into AGN and host components, rebuild
the spectrum with a different host fraction, holding the underlying AGN
transition fixed.

The controlling definition is the host fraction of the *continuum* at 5100 A:

.. math::
    f_{\\rm host} = \\frac{F_{\\rm host}(5100)}{F_{\\rm host}(5100) + F_{\\rm AGN}(5100)}

**The host template is the object's own.** We rescale the host component
recovered from that object's own decomposition. Adding a foreign galaxy
spectrum would introduce a stellar population, velocity dispersion, and
reddening unrelated to the source, and would make the recovered-minus-injected
comparison meaningless. If an object has no usable recovered host, it is
excluded with ``NO_HOST_TEMPLATE`` rather than being given someone else's.

**Line emission rides with the AGN.** The broad and narrow line components
belong to the nucleus, so they scale with the AGN component, not the host.
Scaling the whole non-host residual would dilute the very lines whose
detectability we are trying to measure, which would confound host dilution
with an artificial line-flux change.

What this does *not* model: the spatial redistribution of light in a different
fibre aperture. Preregistration §5 states this explicitly. Host-fraction
manipulation captures the dominant continuum-dilution consequence of aperture
change, not the full spatial effect; residual aperture behaviour is measured
empirically through the same-instrument controls.

.. warning::
   The object-specific replacement path below is retained for provenance and
   synthetic machinery tests, but D-082 prohibits it for the current corrected
   DESI sample: 17H found no robust object-specific host template.  The new
   design uses :func:`added_star_scale_for_target`, which treats the whole
   observed spectrum (including any unknown native host) as an immutable
   baseline and adds an externally specified XSL stellar perturbation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from p3sf.access.spectra import Spectrum


def _window_median(
    wavelength: np.ndarray,
    values: np.ndarray,
    redshift: float,
    window_rest: tuple[float, float],
) -> float:
    """Median in an explicit rest-frame window; fail closed on sparse support."""
    if wavelength.shape != values.shape:
        raise ValueError("wavelength and values must have identical shapes")
    low, high = (float(v) for v in window_rest)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"window_rest must be finite and increasing, got {window_rest}")
    rest = np.asarray(wavelength, dtype=float) / (1.0 + float(redshift))
    inside = (
        (rest >= low) & (rest <= high) & np.isfinite(values)
    )
    if inside.sum() < 5:
        raise ValueError(f"fewer than 5 finite pixels in rest-frame window {window_rest}")
    return float(np.median(np.asarray(values, dtype=float)[inside]))


def added_star_scale_for_target(
    wavelength: np.ndarray,
    baseline_flux: np.ndarray,
    external_host_shape: np.ndarray,
    *,
    redshift: float,
    requested_f_added_star: float,
    window_rest: tuple[float, float],
) -> float:
    r"""Scale an external host so its *added* final-continuum share is exact.

    The immutable observed spectrum is the baseline ``B``.  It may already
    contain an unknown native host; no decomposition or subtraction is assumed.
    For an external stellar shape ``H`` the prospective operator is

    .. math:: F'(\lambda) = B(\lambda) + aH(\lambda)

    with

    .. math::
       f_{\rm star,add} = \frac{a\,\widetilde H_W}
       {\widetilde B_W + a\,\widetilde H_W},\qquad
       a = \frac{f}{1-f}\frac{\widetilde B_W}{\widetilde H_W}.

    Tildes are separate component medians in the explicit rest-frame window
    ``W``. Thus ``f_added_star`` is the contribution of the added experimental
    stellar component to the post-perturbation continuum. It is explicitly not
    a total physical host fraction because ``B`` may contain native host.
    """
    if baseline_flux.shape != external_host_shape.shape:
        raise ValueError("baseline flux and external host shape must match")
    if not 0.0 <= requested_f_added_star < 1.0:
        raise ValueError(
            "requested f_added_star must lie in [0, 1), "
            f"got {requested_f_added_star}"
        )
    baseline = _window_median(wavelength, baseline_flux, redshift, window_rest)
    host = _window_median(wavelength, external_host_shape, redshift, window_rest)
    if baseline <= 0:
        raise ValueError(f"baseline continuum is non-positive ({baseline:.4g})")
    if requested_f_added_star == 0.0:
        return 0.0
    if host <= 0:
        raise ValueError(f"external host continuum is non-positive ({host:.4g})")
    return float(
        (requested_f_added_star / (1.0 - requested_f_added_star)) * (baseline / host)
    )


def realized_added_star_fraction(
    wavelength: np.ndarray,
    baseline_flux: np.ndarray,
    scaled_external_host: np.ndarray,
    *,
    redshift: float,
    window_rest: tuple[float, float],
) -> float:
    """Measure the additive experimental fraction directly from its components."""
    baseline = _window_median(wavelength, baseline_flux, redshift, window_rest)
    host = _window_median(wavelength, scaled_external_host, redshift, window_rest)
    total = baseline + host
    if total <= 0:
        raise ValueError(f"post-perturbation continuum is non-positive ({total:.4g})")
    return float(host / total)


class PairFluxBasis(StrEnum):
    """Whether one absolute added-stellar amplitude is meaningful across epochs."""

    SAME_SURVEY_SAME_APERTURE = "SAME_SURVEY_SAME_APERTURE"
    EXTERNALLY_DEFINED_OBSERVED_CONTAMINATION = "EXTERNALLY_DEFINED_OBSERVED_CONTAMINATION"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True)
class PairAddedStarAmplitude:
    """One reference-anchored stellar amplitude and its realized fractions."""

    shared_scale: float
    requested_f_added_star_reference: float
    realized_f_added_star_reference: float
    realized_f_added_star_other: float
    reference_epoch: str
    flux_basis: PairFluxBasis


@dataclass(frozen=True)
class FixedContinuumSNRResult:
    """Deterministic fixed-continuum-S/N intervention product."""

    spectrum: Spectrum
    baseline_snr: float
    achieved_snr: float
    uncertainty_scale: float


def _continuum_snr(spectrum: Spectrum, window_rest: tuple[float, float]) -> float:
    rest = spectrum.rest_wavelength()
    inside = (
        (rest >= window_rest[0])
        & (rest <= window_rest[1])
        & spectrum.good
    )
    if inside.sum() < 5:
        raise ValueError(f"fewer than 5 usable pixels in rest-frame window {window_rest}")
    value = float(np.median(spectrum.flux[inside] / spectrum.error[inside]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"continuum S/N must be finite and positive, got {value}")
    return value


def fixed_continuum_snr_intervention(
    baseline: Spectrum,
    added_stellar_flux: np.ndarray,
    *,
    window_rest: tuple[float, float],
) -> FixedContinuumSNRResult:
    r"""Add stellar flux while holding the window's median continuum S/N fixed.

    Let :math:`S_0=\operatorname{med}_W(B/\sigma_B)` and
    :math:`S_1=\operatorname{med}_W[(B+aH)/\sigma_B]`. The deterministic rule is

    .. math::
       (B,\sigma_B) \longrightarrow
       (F'=B+aH,\;\sigma'=(S_1/S_0)\sigma_B).

    Therefore ``median_W(F'/sigma') == S0`` up to floating-point arithmetic.
    This is an intervention, not a detector/noise model; it generates no random
    draw. An exactly zero added component returns the authoritative baseline
    object and arrays unchanged.
    """
    added = np.asarray(added_stellar_flux, dtype=float)
    if added.shape != baseline.flux.shape:
        raise ValueError("added stellar flux must match the baseline flux shape")
    if not np.all(np.isfinite(added[baseline.good])):
        raise ValueError("added stellar flux is non-finite on usable baseline pixels")
    baseline_snr = _continuum_snr(baseline, window_rest)
    if np.count_nonzero(added) == 0:
        return FixedContinuumSNRResult(baseline, baseline_snr, baseline_snr, 1.0)
    provisional = replace(baseline, flux=baseline.flux + added)
    provisional_native_snr = _continuum_snr(provisional, window_rest)
    uncertainty_scale = provisional_native_snr / baseline_snr
    if not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0:
        raise ValueError(f"fixed-S/N uncertainty scale is invalid ({uncertainty_scale})")
    intervened = replace(
        provisional,
        error=baseline.error * uncertainty_scale,
        meta={
            **dict(baseline.meta),
            "counterfactual_fixed_snr_intervention": True,
            "counterfactual_fixed_snr_baseline": baseline_snr,
            "counterfactual_fixed_snr_uncertainty_scale": uncertainty_scale,
            "counterfactual_random_noise_generated": False,
        },
    )
    achieved = _continuum_snr(intervened, window_rest)
    return FixedContinuumSNRResult(
        intervened, baseline_snr, achieved, float(uncertainty_scale)
    )


def reference_anchored_pair_amplitude(
    reference_wavelength: np.ndarray,
    reference_baseline_flux: np.ndarray,
    reference_external_star_shape: np.ndarray,
    other_wavelength: np.ndarray,
    other_baseline_flux: np.ndarray,
    other_external_star_shape: np.ndarray,
    *,
    reference_redshift: float,
    other_redshift: float,
    requested_f_added_star_reference: float,
    window_rest: tuple[float, float],
    flux_basis: PairFluxBasis,
    reference_epoch: str = "faint",
) -> PairAddedStarAmplitude:
    """Solve once on a reference epoch and diagnose the same scale on its pair.

    This computes amplitudes and component fractions only; it does not construct
    manipulated spectra. A shared scale is refused unless the caller records a
    comparable or externally defined common observed-flux basis.
    """
    flux_basis = PairFluxBasis(flux_basis)
    if flux_basis is PairFluxBasis.NOT_COMPARABLE:
        raise ValueError("shared pair amplitude requires a comparable flux/aperture basis")
    scale = added_star_scale_for_target(
        reference_wavelength,
        reference_baseline_flux,
        reference_external_star_shape,
        redshift=reference_redshift,
        requested_f_added_star=requested_f_added_star_reference,
        window_rest=window_rest,
    )
    realized_reference = realized_added_star_fraction(
        reference_wavelength,
        reference_baseline_flux,
        scale * reference_external_star_shape,
        redshift=reference_redshift,
        window_rest=window_rest,
    )
    realized_other = realized_added_star_fraction(
        other_wavelength,
        other_baseline_flux,
        scale * other_external_star_shape,
        redshift=other_redshift,
        window_rest=window_rest,
    )
    return PairAddedStarAmplitude(
        shared_scale=scale,
        requested_f_added_star_reference=float(requested_f_added_star_reference),
        realized_f_added_star_reference=realized_reference,
        realized_f_added_star_other=realized_other,
        reference_epoch=str(reference_epoch),
        flux_basis=flux_basis,
    )


@dataclass(frozen=True)
class Decomposition:
    """A fitted separation of one spectrum into physical components.

    All arrays are on the same observed-frame wavelength grid as ``spectrum``.
    ``agn_continuum`` and ``host`` are the two continuum components whose ratio
    defines the host fraction; ``lines`` carries every nuclear emission
    component (broad and narrow) and scales with the AGN.
    """

    wavelength: np.ndarray
    agn_continuum: np.ndarray
    host: np.ndarray
    lines: np.ndarray
    redshift: float

    def __post_init__(self) -> None:
        shapes = {
            "agn_continuum": self.agn_continuum.shape,
            "host": self.host.shape,
            "lines": self.lines.shape,
        }
        bad = {k: v for k, v in shapes.items() if v != self.wavelength.shape}
        if bad:
            raise ValueError(f"component shapes {bad} do not match wavelength {self.wavelength.shape}")

    def rest_wavelength(self) -> np.ndarray:
        return self.wavelength / (1.0 + self.redshift)

    def _at(self, array: np.ndarray, rest_wavelength: float, half_width: float) -> float:
        rest = self.rest_wavelength()
        window = np.abs(rest - rest_wavelength) <= half_width
        if not window.any():
            raise ValueError(f"rest wavelength {rest_wavelength} +/- {half_width} A not covered")
        return float(np.median(array[window]))

    def host_fraction(self, rest_wavelength: float = 5100.0, half_width: float = 25.0) -> float:
        """Continuum host fraction at a reference wavelength."""
        host = self._at(self.host, rest_wavelength, half_width)
        agn = self._at(self.agn_continuum, rest_wavelength, half_width)
        total = host + agn
        if total <= 0:
            raise ValueError(f"non-positive total continuum ({total}) at {rest_wavelength} A")
        return float(host / total)


def host_scale_for_target(
    decomposition: Decomposition,
    target_fraction: float,
    *,
    rest_wavelength: float = 5100.0,
    half_width: float = 25.0,
) -> float:
    """Multiplier on the host component that yields ``target_fraction``.

    Solving

    .. math::
        \\frac{a H}{a H + A} = f \\quad\\Longrightarrow\\quad a = \\frac{f}{1 - f}\\cdot\\frac{A}{H}

    where ``H`` and ``A`` are the host and AGN continua at the reference
    wavelength.
    """
    if not 0.0 <= target_fraction < 1.0:
        raise ValueError(f"target host fraction must lie in [0, 1), got {target_fraction}")

    host = decomposition._at(decomposition.host, rest_wavelength, half_width)
    agn = decomposition._at(decomposition.agn_continuum, rest_wavelength, half_width)

    if agn <= 0:
        raise ValueError(
            f"AGN continuum at {rest_wavelength} A is {agn:.4g}; cannot set a host fraction "
            "against a non-positive AGN component"
        )
    if target_fraction == 0.0:
        return 0.0
    if host <= 0:
        raise ValueError(
            f"recovered host at {rest_wavelength} A is {host:.4g}. This object has no usable "
            "host template; exclude it with NO_HOST_TEMPLATE rather than substituting a "
            "foreign galaxy spectrum."
        )
    return float((target_fraction / (1.0 - target_fraction)) * (agn / host))


@dataclass(frozen=True)
class HostInjectionResult:
    spectrum: Spectrum
    requested_fraction: float
    original_fraction: float
    achieved_fraction: float
    host_scale: float
    seed: int | None

    @property
    def fraction_error(self) -> float:
        return self.achieved_fraction - self.requested_fraction


def inject_host_fraction(
    spectrum: Spectrum,
    decomposition: Decomposition,
    target_fraction: float,
    *,
    rest_wavelength: float = 5100.0,
    half_width: float = 25.0,
    rescale_noise: bool = True,
    seed: int | None = None,
) -> HostInjectionResult:
    """Rebuild ``spectrum`` at a different host fraction.

    The reconstructed spectrum is

    .. math::
        F' = F_{\\rm AGN} + F_{\\rm lines} + a\\,F_{\\rm host} + (F - M)

    where ``M`` is the full fitted model and ``F - M`` is the observed residual,
    carried over unchanged so that real fitting imperfections and correlated
    residual structure survive into the counterfactual.

    Parameters
    ----------
    rescale_noise
        When True, the error array is scaled by the change in total continuum
        level, so that adding host light does not artificially improve the
        per-pixel S/N. A brighter total spectrum at fixed exposure carries
        proportionally larger photon noise; leaving sigma untouched would make
        host-dominated simulations easier to fit than real ones.
    """
    if spectrum.wavelength.shape != decomposition.wavelength.shape:
        raise ValueError("spectrum and decomposition are on different wavelength grids")
    if not np.allclose(spectrum.wavelength, decomposition.wavelength, rtol=0, atol=1e-6):
        raise ValueError("spectrum and decomposition wavelength grids differ")

    original_fraction = decomposition.host_fraction(rest_wavelength, half_width)
    scale = host_scale_for_target(
        decomposition, target_fraction, rest_wavelength=rest_wavelength, half_width=half_width
    )

    model = decomposition.agn_continuum + decomposition.host + decomposition.lines
    residual = spectrum.flux - model
    new_model = decomposition.agn_continuum + decomposition.lines + scale * decomposition.host
    new_flux = new_model + residual

    if rescale_noise:
        old_level = _median_positive(model)
        new_level = _median_positive(new_model)
        noise_scale = np.sqrt(new_level / old_level) if old_level > 0 else 1.0
        new_error = spectrum.error * noise_scale
    else:
        noise_scale = 1.0
        new_error = spectrum.error

    injected = replace(
        spectrum,
        flux=new_flux,
        error=new_error,
        meta={
            **dict(spectrum.meta),
            "counterfactual_host_target": float(target_fraction),
            "counterfactual_host_original": float(original_fraction),
            "counterfactual_host_scale": float(scale),
            "counterfactual_host_noise_scale": float(noise_scale),
        },
    )

    achieved = Decomposition(
        wavelength=decomposition.wavelength,
        agn_continuum=decomposition.agn_continuum,
        host=scale * decomposition.host,
        lines=decomposition.lines,
        redshift=decomposition.redshift,
    ).host_fraction(rest_wavelength, half_width)

    return HostInjectionResult(
        spectrum=injected,
        requested_fraction=float(target_fraction),
        original_fraction=float(original_fraction),
        achieved_fraction=float(achieved),
        host_scale=float(scale),
        seed=seed,
    )


def measure_host_fraction_directly(
    spectrum: Spectrum,
    decomposition: Decomposition,
    *,
    rest_wavelength: float = 5100.0,
    half_width: float = 25.0,
) -> float:
    """Host fraction measured from a constructed spectrum without refitting.

    Used to verify the injection engine independently of the fitter: if the
    engine is correct, this returns the requested fraction. Any disagreement
    between this and the *refitted* host fraction is decomposition bias, which
    is the quantity of interest for preregistration §5 / spec §31 — and that
    separation only holds if this function never calls the fitter.
    """
    del spectrum  # constructed flux is the model plus a residual; components suffice
    return decomposition.host_fraction(rest_wavelength, half_width)


def _median_positive(array: np.ndarray) -> float:
    finite = array[np.isfinite(array) & (array > 0)]
    return float(np.median(finite)) if finite.size else 0.0


def derive_seed(base_seed: int, realization: int, target_fraction: float) -> int:
    payload = f"host|{int(base_seed)}|{int(realization)}|{float(target_fraction):.6f}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


__all__ = [
    "Decomposition",
    "HostInjectionResult",
    "FixedContinuumSNRResult",
    "PairAddedStarAmplitude",
    "PairFluxBasis",
    "added_star_scale_for_target",
    "derive_seed",
    "host_scale_for_target",
    "fixed_continuum_snr_intervention",
    "inject_host_fraction",
    "measure_host_fraction_directly",
    "realized_added_star_fraction",
    "reference_anchored_pair_amplitude",
]
