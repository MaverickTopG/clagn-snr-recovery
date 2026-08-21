"""Classification outcomes, epoch pairs, and the literature-verification gate.

Three commitments from the preregistration are enforced here rather than left
to each criterion:

1. **The outcome is three-class.** ``CL`` / ``NON_CL`` / ``UNCLASSIFIABLE``. A
   criterion that cannot be evaluated returns ``UNCLASSIFIABLE``. It must never
   quietly return ``NON_CL``, because "we could not measure Hbeta" and "Hbeta
   did not change" are different facts and pooling them manufactures a
   completeness deficit out of a measurement failure.

2. **Non-detections are upper limits, not zeros.** An epoch with
   ``detected=False`` carries a finite ``upper_limit``. Criteria must consume
   it as a bound.

3. **Thresholds come from the literature, verified.** A criterion whose
   threshold has not been read out of its source paper refuses to run. See
   ``01_literature/selection_definitions.md``.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from p3sf.config import project_root
from p3sf.fitting.results import BroadLineMeasurement


class Label(StrEnum):
    """The three-class classification outcome."""

    CL = "CL"
    NON_CL = "non-CL"
    UNCLASSIFIABLE = "unclassifiable"


Direction = Literal["turn_on", "turn_off", "indeterminate"]


class UnverifiedCriterionError(RuntimeError):
    """Raised when a criterion is invoked before its threshold has been verified.

    Implementing a plausible-looking guess would turn a published criterion into
    a strawman, which is precisely the failure mode this paper exists to expose
    in others.
    """


@dataclass(frozen=True)
class EpochPair:
    """Two epochs of one object, ordered in time."""

    object_id: str
    early: BroadLineMeasurement
    late: BroadLineMeasurement
    early_mjd: float
    late_mjd: float
    line: str = "Hbeta"

    def __post_init__(self) -> None:
        if self.late_mjd <= self.early_mjd:
            raise ValueError(
                f"{self.object_id}: late epoch (MJD {self.late_mjd}) does not follow "
                f"early epoch (MJD {self.early_mjd})"
            )
        if self.early.line != self.late.line:
            raise ValueError(
                f"{self.object_id}: epochs measure different lines "
                f"({self.early.line} vs {self.late.line})"
            )

    @property
    def usable(self) -> bool:
        """Both epochs carry either a detection or a finite upper limit."""
        return self.early.usable and self.late.usable

    @property
    def baseline_days(self) -> float:
        return self.late_mjd - self.early_mjd

    def _representative_flux(self, measurement: BroadLineMeasurement) -> float | None:
        """Flux if detected, else the upper limit as a bound on the true flux."""
        if measurement.detected and measurement.flux is not None:
            return measurement.flux
        return measurement.upper_limit

    @property
    def direction(self) -> Direction:
        """Turn-on if the broad line strengthened, turn-off if it weakened.

        Preregistration §11 requires these to be analysed separately: a line
        appearing above a detection threshold is not statistically equivalent
        to one disappearing below it.
        """
        early = self._representative_flux(self.early)
        late = self._representative_flux(self.late)
        if early is None or late is None:
            return "indeterminate"
        if self.early.detected == self.late.detected:
            return "turn_on" if late > early else "turn_off"
        # Exactly one epoch is a non-detection: the detected one is the bright state.
        return "turn_on" if self.late.detected else "turn_off"

    @property
    def bright(self) -> BroadLineMeasurement:
        return self.late if self.direction == "turn_on" else self.early

    @property
    def faint(self) -> BroadLineMeasurement:
        return self.early if self.direction == "turn_on" else self.late


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's verdict on one epoch pair under one condition."""

    criterion: str
    label: Label
    statistic: float | None
    threshold: float | None
    direction: Direction
    reason: str = ""

    @property
    def is_cl(self) -> bool:
        return self.label is Label.CL

    @property
    def is_unclassifiable(self) -> bool:
        return self.label is Label.UNCLASSIFIABLE


def unclassifiable(criterion: str, reason: str, direction: Direction = "indeterminate") -> CriterionResult:
    """Build an explicit non-answer. Never substitute NON_CL for this."""
    return CriterionResult(
        criterion=criterion,
        label=Label.UNCLASSIFIABLE,
        statistic=None,
        threshold=None,
        direction=direction,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# literature verification gate
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def verified_criterion_families(path: str | None = None) -> frozenset[str]:
    """Criterion families with at least one PDF-verified literature row.

    Reads ``01_literature/literature_matrix.csv``. A family is verified when a
    row naming it has ``verified_against_pdf == 'yes'``.
    """
    target = (
        project_root() / "01_literature" / "literature_matrix.csv"
        if path is None
        else project_root() / path
    )
    if not target.exists():
        return frozenset()
    families: set[str] = set()
    with target.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verified_against_pdf", "").strip().lower() == "yes":
                family = row.get("criterion_family", "").strip()
                if family and family != "TO_VERIFY":
                    families.add(family)
    return frozenset(families)


def require_verified(family: str, criterion_name: str) -> None:
    """Refuse to run a criterion whose literature row is unverified."""
    if family not in verified_criterion_families():
        raise UnverifiedCriterionError(
            f"criterion '{criterion_name}' (family {family}) cannot run: no row in "
            "01_literature/literature_matrix.csv with criterion_family="
            f"'{family}' has verified_against_pdf='yes'. Read the source paper, "
            "transcribe the exact threshold into 01_literature/selection_definitions.md, "
            "then flip the flag. Preregistration §11 requires published thresholds "
            "implemented verbatim."
        )


__all__ = [
    "CriterionResult",
    "Direction",
    "EpochPair",
    "Label",
    "UnverifiedCriterionError",
    "require_verified",
    "unclassifiable",
    "verified_criterion_families",
]
