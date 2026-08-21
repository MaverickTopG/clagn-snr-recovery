"""Region-specific quality control: global QC is not Hβ QC.

The experiment is about Hβ. A warning raised somewhere else in the spectrum —
a poorly modelled C IV, a negative Mg II line area — says nothing about whether
Hβ is measurable. Conversely, a spectrum with ``ZWARNING == 0`` and a beautiful
global fit is useless for Q1 if the pixels directly under broad Hβ are masked or
corrupted.

So two verdicts travel together and are never conflated:

``qc_global``
    Is this a usable spectrum at all — identity secure, data present, redshift
    trustworthy enough to place rest-frame windows?

``qc_hbeta``
    Is the Hβ region specifically usable — pixels present, uncertainties sane,
    no local pathology, decomposition stable?

An epoch enters the primary Hβ experiment only when both pass. Recording them
separately means the Methods section can say exactly why each epoch was kept or
dropped, instead of translating an object-level survey warning into an
object-level scientific rejection.

``NEGATIVE_EMISSION`` is the motivating case. SDSS raises bit 6 when *any* of
C IV, C III], Mg II, Hβ or Hα satisfies ``LINEAREA + 3*LINEAREA_ERR < 0``. The
flag alone therefore does not identify Hβ as the culprit, and the adjudication
must find the triggering line before deciding anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class RegionVerdict(StrEnum):
    """Disposition of one spectral region for one epoch."""

    PASS = "PASS"
    #: Usable, but a survey-level warning is attached. Kept out of Gold until
    #: the independent reference adjudication runs.
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    #: Reviewed by a human and judged usable despite the warning.
    REVIEW_PASS = "REVIEW_PASS"
    #: The region is not usable for this experiment.
    FAIL = "FAIL"
    #: Not yet adjudicated. Blocks use; never silently treated as either outcome.
    PENDING = "PENDING"

    @property
    def usable(self) -> bool:
        return self in {
            RegionVerdict.PASS,
            RegionVerdict.PASS_WITH_WARNING,
            RegionVerdict.REVIEW_PASS,
        }


@dataclass(frozen=True)
class HbetaAdjudication:
    """A reviewable record for one epoch flagged by a survey-level warning.

    Every field is evidence a reviewer actually checked. ``None`` means
    "not established", never "fine" — an unchecked box must not read as a pass.
    """

    object_id: str
    mjd: float
    zwarning: int

    # Which line actually raised the flag. For NEGATIVE_EMISSION this is the
    # whole question: a C IV trigger says nothing about Hbeta.
    trigger_line: str | None = None
    trigger_linearea: float | None = None
    trigger_linearea_err: float | None = None

    # Hbeta-region evidence
    hbeta_pixel_mask_ok: bool | None = None
    hbeta_ivar_ok: bool | None = None
    hbeta_local_residual_ok: bool | None = None
    hbeta_pyqsofit_valid: bool | None = None

    # Corroboration
    halpha_available: bool | None = None
    halpha_quality: str | None = None
    global_flux_calibration_ok: bool | None = None
    redshift_secure: bool | None = None

    final_hbeta_disposition: RegionVerdict = RegionVerdict.PENDING
    reviewer_note: str = ""
    reviewer: str = ""

    @property
    def trigger_is_hbeta(self) -> bool | None:
        """None while the triggering line is unknown — not False."""
        if self.trigger_line is None:
            return None
        return self.trigger_line.lower().replace(" ", "") in {"hbeta", "hb", "h_beta", "hβ"}

    @property
    def hbeta_evidence_complete(self) -> bool:
        """Whether enough was checked to justify anything other than PENDING."""
        return all(
            value is not None
            for value in (
                self.hbeta_pixel_mask_ok,
                self.hbeta_ivar_ok,
                self.hbeta_local_residual_ok,
                self.hbeta_pyqsofit_valid,
            )
        )

    @property
    def hbeta_region_clean(self) -> bool:
        """All four Hβ checks explicitly passed."""
        return (
            self.hbeta_pixel_mask_ok is True
            and self.hbeta_ivar_ok is True
            and self.hbeta_local_residual_ok is True
            and self.hbeta_pyqsofit_valid is True
        )

    def recommended_disposition(self) -> RegionVerdict:
        """Suggested verdict from the recorded evidence.

        Advisory only: ``final_hbeta_disposition`` is set by a person. The rule
        follows the review policy —

        * evidence incomplete -> PENDING, never a default pass;
        * flag came from another line and Hβ is clean -> PASS_WITH_WARNING;
        * flag came from Hβ itself but Hβ survives independent checks
          -> REVIEW_PASS, and the object stays out of Gold pending adjudication;
        * Hβ region compromised -> FAIL.
        """
        if not self.hbeta_evidence_complete:
            return RegionVerdict.PENDING
        if not self.hbeta_region_clean:
            return RegionVerdict.FAIL
        if self.trigger_is_hbeta is None:
            return RegionVerdict.PENDING
        return RegionVerdict.REVIEW_PASS if self.trigger_is_hbeta else RegionVerdict.PASS_WITH_WARNING

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row.update(
            trigger_is_hbeta=self.trigger_is_hbeta,
            hbeta_evidence_complete=self.hbeta_evidence_complete,
            hbeta_region_clean=self.hbeta_region_clean,
            recommended_disposition=str(self.recommended_disposition()),
            final_hbeta_disposition=str(self.final_hbeta_disposition),
            # Always retained, so the strict robustness re-run stays available.
            strict_zwarning_pass=self.zwarning == 0,
        )
        return row


ADJUDICATION_COLUMNS = tuple(
    HbetaAdjudication(object_id="", mjd=0.0, zwarning=0).to_row().keys()
)


__all__ = [
    "ADJUDICATION_COLUMNS",
    "HbetaAdjudication",
    "RegionVerdict",
]
