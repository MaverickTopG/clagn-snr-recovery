"""Component bookkeeping and the Paper-3 canonical host fractions.

Two jobs, in strict order.

**1. Reconstruction identity — the hard gate.** Before any host fraction is
quoted, the separately extracted components must add back up to the fitter's own
best fit:

    F_bestfit  ==  sum_k F_k

If they do not, the host fraction has a denominator nobody verified, and a
missing nuisance family would silently inflate or deflate it. This is checked
numerically and no fraction is computed when it fails.

**2. Canonical host fractions.** Two definitions, computed identically for every
pipeline from *reconstructed component spectra* — never from coefficient sums,
because template families are conditioned differently and weight sums are not
flux ratios:

    f_host_cont   = F_star / (F_star + F_AGN,smooth)
    f_host_pseudo = F_star / (F_star + F_AGN,smooth + F_FeII + F_BalmerCont)

Broad and narrow emission lines are excluded from both denominators. They are
nuclear but not continuum, and including them would make a *continuum* dilution
measure depend on line strength — the very quantity Q2 varies.

Each method's native host fraction is preserved separately and never
overwritten, so "the algorithms disagree" stays distinguishable from "we
compared two different definitions".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p3sf.spectral_domain import host_fraction_in_window

#: Families a decomposition may emit. `stellar` is the host; `agn_smooth` is the
#: featureless nuclear continuum; `feii` and `balmer_cont` are nuclear
#: pseudocontinuum; `lines` are excluded from every host-fraction denominator.
FAMILIES = ("stellar", "agn_smooth", "feii", "balmer_cont", "lines")

#: Relative tolerance for the reconstruction identity, against the median of
#: |bestfit|. Generous enough for float accumulation over thousands of columns,
#: tight enough that a dropped family cannot pass.
RECONSTRUCTION_RTOL = 1e-6


@dataclass(frozen=True)
class ReconstructionCheck:
    """Whether extracted components reproduce the fitter's own best fit."""

    passed: bool
    max_abs_difference: float
    relative_difference: float
    reference_scale: float
    n_pixels: int
    detail: str = ""

    def require(self) -> None:
        """Raise unless the identity holds. Call before quoting any fraction."""
        if not self.passed:
            raise ValueError(
                f"reconstruction identity failed: max|diff| {self.max_abs_difference:.3e}, "
                f"relative {self.relative_difference:.3e} > {RECONSTRUCTION_RTOL:.1e}. "
                f"{self.detail} No host fraction may be derived from these components."
            )


@dataclass(frozen=True)
class ComponentModel:
    """A decomposition split into named families on one wavelength grid."""

    wavelength: np.ndarray                       # rest frame
    components: dict[str, np.ndarray]
    bestfit: np.ndarray | None = None
    method: str = ""
    native_f_host: float | None = None           # the pipeline's own number, untouched
    meta: dict[str, object] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray:
        """A family's spectrum, or zeros when the model does not emit it."""
        value = self.components.get(name)
        return np.zeros_like(self.wavelength) if value is None else value

    @property
    def total(self) -> np.ndarray:
        return np.sum([self.get(name) for name in FAMILIES], axis=0)

    def check_reconstruction(self) -> ReconstructionCheck:
        """Compare the family sum against the fitter's own bestfit."""
        if self.bestfit is None:
            return ReconstructionCheck(
                False, float("nan"), float("nan"), float("nan"), 0,
                "no bestfit supplied, so the identity cannot be tested",
            )
        total = self.total
        if total.shape != self.bestfit.shape:
            return ReconstructionCheck(
                False, float("nan"), float("nan"), float("nan"), 0,
                f"shape mismatch: components {total.shape} vs bestfit {self.bestfit.shape}",
            )

        difference = np.abs(total - self.bestfit)
        scale = float(np.median(np.abs(self.bestfit)))
        max_difference = float(np.max(difference))
        relative = max_difference / scale if scale > 0 else float("inf")
        return ReconstructionCheck(
            passed=bool(relative <= RECONSTRUCTION_RTOL),
            max_abs_difference=max_difference,
            relative_difference=relative,
            reference_scale=scale,
            n_pixels=int(total.size),
            detail="" if relative <= RECONSTRUCTION_RTOL else
                   "a family is missing from the extraction or double counted",
        )

    # -- canonical fractions -------------------------------------------------

    def f_host_cont(self, window: tuple[float, float]):
        """Host over host plus *smooth* nuclear continuum only."""
        return host_fraction_in_window(
            self.wavelength, self.get("stellar"), self.get("agn_smooth"), window
        )

    def f_host_pseudo(self, window: tuple[float, float]):
        """Host over host plus all nuclear pseudocontinuum, lines still excluded."""
        nuclear = self.get("agn_smooth") + self.get("feii") + self.get("balmer_cont")
        return host_fraction_in_window(self.wavelength, self.get("stellar"), nuclear, window)

    def canonical_fractions(self, window: tuple[float, float]) -> dict[str, object]:
        """Both definitions, gated on the reconstruction identity."""
        check = self.check_reconstruction()
        check.require()
        cont, pseudo = self.f_host_cont(window), self.f_host_pseudo(window)
        return {
            "method": self.method,
            "f_host_cont": cont.value,
            "f_host_cont_pixels": cont.n_pixels,
            "f_host_pseudo": pseudo.value,
            "f_host_pseudo_pixels": pseudo.n_pixels,
            "window_coverage": cont.window_coverage,
            # Never overwritten by the common definition.
            "native_f_host": self.native_f_host,
            "reconstruction_relative_difference": check.relative_difference,
        }


def from_ppxf(
    fit,
    wavelength: np.ndarray,
    n_stellar: int,
    nonstellar_names: list[str],
    *,
    method: str = "ppxf",
    native_f_host: float | None = None,
) -> ComponentModel:
    """Split a pPXF solution into Paper-3 families.

    Uses ``fit.matrix`` — the design matrix pPXF actually fitted, carrying the
    templates after convolution and resampling — so the extracted families are
    the same objects the solver combined. Reconstructing from the pre-fit
    templates instead would not reproduce ``bestfit``.
    """
    matrix = np.asarray(fit.matrix)
    weights = np.asarray(fit.weights)
    # Any leading polynomial columns sit before the templates.
    offset = matrix.shape[1] - (n_stellar + len(nonstellar_names))
    if offset < 0:
        raise ValueError(
            f"design matrix has {matrix.shape[1]} columns, fewer than the "
            f"{n_stellar + len(nonstellar_names)} templates supplied"
        )

    template_block = matrix[:, offset:]
    template_weights = weights[offset:] if weights.size == matrix.shape[1] else weights

    stellar = template_block[:, :n_stellar] @ template_weights[:n_stellar]
    nonstellar_block = template_block[:, n_stellar:]
    nonstellar_weights = template_weights[n_stellar:]

    def family(predicate) -> np.ndarray:
        mask = np.array([predicate(name) for name in nonstellar_names], dtype=bool)
        if not mask.any():
            return np.zeros_like(stellar)
        return nonstellar_block[:, mask] @ nonstellar_weights[mask]

    components = {
        "stellar": stellar,
        "agn_smooth": family(lambda n: n.startswith("powerlaw")),
        "feii": family(lambda n: n.startswith("feii")),
        "balmer_cont": family(lambda n: n.startswith("balmer")),
        "lines": family(
            lambda n: not (
                n.startswith("powerlaw") or n.startswith("feii") or n.startswith("balmer")
            )
        ),
    }

    # Polynomial columns, if any were enabled, are counted so the identity test
    # sees the complete model rather than silently failing.
    if offset > 0:
        components["lines"] = components["lines"] + matrix[:, :offset] @ weights[:offset]

    return ComponentModel(
        wavelength=np.asarray(wavelength),
        components=components,
        bestfit=np.asarray(fit.bestfit),
        method=method,
        native_f_host=native_f_host,
        meta={"n_stellar": n_stellar, "n_nonstellar": len(nonstellar_names),
              "n_polynomial": int(offset)},
    )


def from_pyqsofit(fit, redshift: float, *, native_f_host: float | None = None) -> ComponentModel:
    """Split a PyQSOFit solution into the same families.

    PyQSOFit's ``qso`` is a PCA reconstruction of the *whole* nuclear spectrum,
    not a power law, so it cannot be separated into smooth continuum, Fe II and
    lines the way the pPXF model can. It is therefore assigned to
    ``agn_smooth`` and both canonical fractions collapse to the same value for
    this method. That limitation is recorded in ``meta`` rather than hidden: the
    comparison to pPXF's ``f_host_cont`` is meaningful, while a
    pseudocontinuum-specific contrast is not available from this pipeline.
    """
    host = np.asarray(getattr(fit, "host", []), dtype=float)
    qso = np.asarray(getattr(fit, "qso", []), dtype=float)
    wave = np.asarray(getattr(fit, "wave", []), dtype=float)
    if host.size == 0 or host.size != wave.size or qso.size != wave.size:
        raise ValueError("PyQSOFit host/qso/wave arrays missing or mismatched")

    return ComponentModel(
        wavelength=wave / (1.0 + redshift),
        components={"stellar": host, "agn_smooth": qso},
        bestfit=host + qso,
        method="pyqsofit",
        native_f_host=native_f_host,
        meta={
            "qso_is_pca_reconstruction": True,
            "pseudo_equals_cont": True,
            "note": "qso PCA spans continuum, Fe II and lines together; "
                    "f_host_pseudo is not separately defined for this method",
        },
    )


__all__ = [
    "FAMILIES",
    "RECONSTRUCTION_RTOL",
    "ComponentModel",
    "ReconstructionCheck",
    "from_ppxf",
    "from_pyqsofit",
]
