"""Source-verified CLAGN rules and search-protocol gates.

These objects deliberately separate a final classification rule from a
candidate-search stage.  A passing search stage is not a CL label and never
belongs in a final-classifier denominator.

The numerical inputs are *published-protocol measurements*.  This decision
layer does not silently substitute fitted integrated-line fluxes for the
pixel-level MacLeod/Green statistic, nor invent unspecified smoothing or
integration details for Guo or Potts--Villforth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TYPE_CHECKING, TypeGuard

from p3sf.criteria.base import (
    CriterionResult,
    Direction,
    Label,
    require_verified,
    unclassifiable,
)

if TYPE_CHECKING:
    from p3sf.fitting.pyqsofit_driver import YangHbetaFitAudit

MACLEOD_GREEN_HBETA_RANGE_ANGSTROM = (4750.0, 4940.0)
MACLEOD_GREEN_REBIN_ANGSTROM_PER_PIXEL = 2.0
MACLEOD_GREEN_MEDIAN_WIDTH_ANGSTROM = 32.0
MACLEOD_GREEN_THRESHOLD = 3.0
YANG_HBETA_FAINT_TO_BRIGHT_THRESHOLD = 0.3

GUO_NSIGMA_THRESHOLD = 3.0
GUO_FRACTIONAL_FLUX_CHANGE_THRESHOLD = 1.5
GUO_OIII_RELATIVE_DIFFERENCE_LIMIT = 0.20
GUO_PSEUDOPHOTOMETRY_MAG_DIFFERENCE_LIMIT = 0.5
GUO_LINES = ("MgII", "Hbeta", "Halpha")

POTTS_PRESELECTION_FLUX_DENSITY = 3.0e-18
POTTS_PRESELECTION_RANGE_ANGSTROM = (4100.0, 5500.0)
POTTS_CONTINUUM_SLOPE_RANGE_ANGSTROM = (3500.0, 3700.0)
POTTS_LINE_PEAK_RANGES_ANGSTROM = {
    "Hbeta": (4856.0, 4866.0),
    "Halpha": (6558.0, 6568.0),
}
POTTS_STRENGTH_THRESHOLDS = {
    "strong": {"continuum": -2.0e-2, "Halpha": 3.5, "Hbeta": 5.25},
    "intermediate": {"continuum": -1.8e-3, "Halpha": 1.8, "Hbeta": 2.7},
    "weak": {"continuum": -6.0e-4, "Halpha": 0.8, "Hbeta": 1.2},
}


class AnalysisRole(StrEnum):
    """Whether an outcome is allowed to contribute a final CL label."""

    FINAL_CLASSIFIER = "FINAL_CLASSIFIER"
    SEARCH_STAGE = "SEARCH_STAGE"
    CONFIRMATION_PROTOCOL = "CONFIRMATION_PROTOCOL"


class ProtocolDisposition(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNCLASSIFIABLE = "unclassifiable"


@dataclass(frozen=True)
class ProtocolResult:
    """Outcome of a non-classifier protocol stage."""

    protocol: str
    stage: str
    role: AnalysisRole
    disposition: ProtocolDisposition
    reason: str = ""

    @property
    def counts_toward_classifier_denominator(self) -> bool:
        return False


def _finite(value: float | None) -> TypeGuard[float]:
    return value is not None and isfinite(value)


def apply_yang2024_hbeta(
    *,
    bright: YangHbetaFitAudit,
    faint: YangHbetaFitAudit,
    direction: Direction = "indeterminate",
) -> CriterionResult:
    """Apply Yang's strict faint/bright broad-H-beta flux ratio.

    The source explicitly permits ``f_faint=0`` when the broad line is validly
    undetected. That criterion-local convention is never applied to a failed
    fit, an unavailable component, or an arbitrary numerical zero.
    """
    name = "yang2024_hbeta"
    require_verified("B_FLUX_RATIO", name)
    if not bright.fit_valid or not faint.fit_valid:
        reasons = [reason for reason in (bright.invalid_reason, faint.invalid_reason) if reason]
        return unclassifiable(name, "invalid Yang Hbeta fit: " + "|".join(reasons), direction)
    if (
        not bright.broad_component_present
        or not _finite(bright.broad_flux)
        or float(bright.broad_flux) <= 0
    ):
        return unclassifiable(name, "positive valid bright-state broad Hbeta unavailable", direction)
    if faint.broad_component_present:
        if not _finite(faint.broad_flux) or float(faint.broad_flux) < 0:
            return unclassifiable(name, "finite faint-state broad Hbeta unavailable", direction)
        faint_flux = float(faint.broad_flux)
        reason = "finite fitted faint-state broad Hbeta"
    elif faint.valid_nondetection:
        faint_flux = 0.0
        reason = "Yang source convention: valid fitted faint nondetection has flux zero"
    else:
        return unclassifiable(name, "faint component unavailable but not a valid nondetection", direction)
    statistic = faint_flux / float(bright.broad_flux)
    return CriterionResult(
        criterion=name,
        label=(
            Label.CL
            if statistic < YANG_HBETA_FAINT_TO_BRIGHT_THRESHOLD
            else Label.NON_CL
        ),
        statistic=statistic,
        threshold=YANG_HBETA_FAINT_TO_BRIGHT_THRESHOLD,
        direction=direction,
        reason=reason,
    )


def apply_macleod2019(
    *,
    nsigma_hbeta: float | None,
    visual_broad_hbeta_appearance_or_disappearance: bool | None,
    direction: Direction = "indeterminate",
    measurement_available: bool = True,
) -> CriterionResult:
    """Apply MacLeod et al. (2019)'s final H-beta definition.

    ``nsigma_hbeta`` must be measured after the paper's [O III]/host rescaling,
    continuum subtraction, rest-frame 2 A/pixel rebinning and 32 A running
    median.  It is the maximum 4750--4940 A pixel deviation relative to the
    smoothed 4750 A value.  It is not an integrated line-flux S/N.
    """
    name = "macleod2019_hbeta"
    require_verified("A_PIXEL", name)
    if not measurement_available or not _finite(nsigma_hbeta):
        return unclassifiable(name, "source-protocol Hbeta N_sigma is unavailable", direction)
    if visual_broad_hbeta_appearance_or_disappearance is None:
        return unclassifiable(name, "required visual Hbeta transition adjudication is unavailable", direction)
    passed = bool(visual_broad_hbeta_appearance_or_disappearance) and (
        float(nsigma_hbeta) > MACLEOD_GREEN_THRESHOLD
    )
    return CriterionResult(
        criterion=name,
        label=Label.CL if passed else Label.NON_CL,
        statistic=float(nsigma_hbeta),
        threshold=MACLEOD_GREEN_THRESHOLD,
        direction=direction,
        reason="visual Hbeta transition required",
    )


def apply_green2022(
    *,
    nsigma_hbeta: float | None,
    direction: Direction = "indeterminate",
    measurement_available: bool = True,
) -> CriterionResult:
    """Apply Green et al. (2022)'s bona-fide H-beta rule.

    Green et al. first assembled candidates by visual inspection; that parent
    search is not added as a final scalar cut here.  Within the candidate
    sample, bona-fide status is the source-protocol H-beta N_sigma threshold.
    The paper's table uses ``>= 3`` (while nearby prose sometimes says ``>``),
    so equality follows the explicit catalogue note.
    """
    name = "green2022_hbeta"
    require_verified("A_PIXEL", name)
    if not measurement_available or not _finite(nsigma_hbeta):
        return unclassifiable(name, "source-protocol Hbeta N_sigma is unavailable", direction)
    statistic = float(nsigma_hbeta)
    return CriterionResult(
        criterion=name,
        label=Label.CL if statistic >= MACLEOD_GREEN_THRESHOLD else Label.NON_CL,
        statistic=statistic,
        threshold=MACLEOD_GREEN_THRESHOLD,
        direction=direction,
        reason="Green catalogue note uses N_sigma(Hbeta) >= 3",
    )


def guo_fractional_flux_change(bright_flux: float, dim_flux: float) -> float | None:
    """Guo R = (F_bright - F_dim) / F_dim; not F_bright/F_dim.

    The paper supplies no nondetection/zero-denominator convention.  Such a
    case is therefore undefined rather than being coerced to zero or infinity.
    """
    if not isfinite(bright_flux) or not isfinite(dim_flux) or dim_flux <= 0:
        return None
    value = (bright_flux - dim_flux) / dim_flux
    return value if isfinite(value) else None


def apply_guo_quick_screen(
    line_measurements: Mapping[str, tuple[float | None, float | None]],
) -> ProtocolResult:
    """Apply Guo step 1, explicitly as a search stage rather than a classifier.

    Values are ``(Max(N_sigma), R)`` for each available BEL.  Both strict cuts
    must pass for the *same* one of Mg II, H-beta, or H-alpha.  Missing values
    never pass and never become a final non-CL label.
    """
    require_verified("GUO_PROTOCOL", "guo_quick_screen")
    available = 0
    for line in GUO_LINES:
        nsigma, fractional_change = line_measurements.get(line, (None, None))
        if _finite(nsigma) and _finite(fractional_change):
            available += 1
            if (
                float(nsigma) > GUO_NSIGMA_THRESHOLD
                and float(fractional_change) > GUO_FRACTIONAL_FLUX_CHANGE_THRESHOLD
            ):
                return ProtocolResult(
                    protocol="guo2024_desi",
                    stage="STEP1_SPECTRAL_INTEGRATION_QUICK_SCREEN",
                    role=AnalysisRole.SEARCH_STAGE,
                    disposition=ProtocolDisposition.PASS,
                    reason=f"{line} passes both strict cuts",
                )
    if available == 0:
        disposition = ProtocolDisposition.UNCLASSIFIABLE
        reason = "no BEL has finite Max(N_sigma) and R; nondetections have no published substitution"
    else:
        disposition = ProtocolDisposition.FAIL
        reason = "no single BEL passes both strict quick-screen cuts"
    return ProtocolResult(
        protocol="guo2024_desi",
        stage="STEP1_SPECTRAL_INTEGRATION_QUICK_SCREEN",
        role=AnalysisRole.SEARCH_STAGE,
        disposition=disposition,
        reason=reason,
    )


def guo_final_protocol_status(*, all_published_stages_available: bool) -> ProtocolResult:
    """Prevent the Guo quick screen from masquerading as the final catalogue.

    The final protocol additionally requires fitted-line R > 1.5, [O III]
    assessment (normally <20%), pseudo-photometry (normally <0.5 mag), and
    visual exception handling.  Those branches cannot be reconstructed from a
    scalar quick-screen result.
    """
    require_verified("GUO_PROTOCOL", "guo_final_protocol")
    return ProtocolResult(
        protocol="guo2024_desi",
        stage="FINAL_STAGED_CATALOGUE_PROTOCOL",
        role=AnalysisRole.CONFIRMATION_PROTOCOL,
        disposition=(
            ProtocolDisposition.PASS
            if all_published_stages_available
            else ProtocolDisposition.UNCLASSIFIABLE
        ),
        reason=(
            "all independently recorded protocol stages available"
            if all_published_stages_available
            else "quick screen alone is insufficient; calibration, pseudo-photometry, and VI branches required"
        ),
    )


def potts_villforth_protocol_status(*, measurements_available: bool) -> ProtocolResult:
    """Record why Potts--Villforth cannot be a final scalar classifier.

    Table 1 thresholds and wavelength windows are frozen as constants above.
    The paper does not report the Gaussian smoothing width and its prose and
    Table-1 note give conflicting feature-combination wording.  Its 941
    algorithmic candidates were then visually reduced to six CLQs.  Resolving
    those gaps numerically would invent a classifier.
    """
    require_verified("POTTS_PROTOCOL", "potts_villforth_search")
    reason = (
        "published indicators available, but candidate-combination wording and smoothing width are not uniquely specified"
        if measurements_available
        else "published-protocol indicators unavailable"
    )
    return ProtocolResult(
        protocol="potts_villforth2021",
        stage="DIFFERENCE_SPECTRUM_CANDIDATE_SEARCH",
        role=AnalysisRole.SEARCH_STAGE,
        disposition=ProtocolDisposition.UNCLASSIFIABLE,
        reason=reason,
    )


__all__ = [name for name in globals() if name.startswith(("MACLEOD_", "GUO_", "POTTS_"))] + [
    "AnalysisRole",
    "ProtocolDisposition",
    "ProtocolResult",
    "apply_green2022",
    "apply_guo_quick_screen",
    "apply_macleod2019",
    "guo_final_protocol_status",
    "guo_fractional_flux_change",
    "potts_villforth_protocol_status",
]
