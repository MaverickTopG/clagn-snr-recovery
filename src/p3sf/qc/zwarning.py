"""SDSS ZWARNING bit classification (D-016).

Not a decoder — a *classification*. Each bit carries its survey meaning, why it
matters for this particular experiment, which spectral property it threatens,
and a disposition: ``PASS``, ``REVIEW`` or ``FAIL``.

The motivating problem: a blind ``ZWARNING == 0`` cut would preferentially
remove early/dim epochs, which are precisely the transition states this paper
needs. ``MANY_OUTLIERS`` in particular is documented to fire on broad-line
galaxies and high-S/N spectra without making the spectrum unusable — so the
naive cut discards good broad-line AGN for being broad-line AGN.

The governing question for every bit is narrow: **does this warning compromise
Hβ, the continuum, the wavelength solution, or the flux calibration?** A warning
about, say, a poorly fit sky region far from Hβ is irrelevant to this
experiment. A warning about the redshift itself is not.

The original bitmask is always preserved, and ``strict_zwarning_pass``
(``ZWARNING == 0``) is retained alongside the bit-aware verdict so every major
result can be re-run on strict-zero spectra alone and reported as a robustness
check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

Threatens = Literal["redshift", "identity", "hbeta", "continuum", "wavelength", "calibration", "none"]


class Disposition(StrEnum):
    """What the pipeline does with a spectrum carrying this bit."""

    PASS = "PASS"        # recorded, no action
    REVIEW = "REVIEW"    # held for manual adjudication before use
    FAIL = "FAIL"        # excluded automatically


@dataclass(frozen=True)
class ZWarningBit:
    bit: int
    name: str
    survey_definition: str
    why_it_matters: str
    threatens: tuple[Threatens, ...]
    disposition: Disposition
    automatic: bool  # False means the disposition requires a human decision

    @property
    def value(self) -> int:
        return 1 << self.bit


# Bit definitions follow the SDSS ZWARNING mask. `threatens` and `disposition`
# are this project's judgements, not SDSS's, and are reviewable.
ZWARNING_BITS: tuple[ZWarningBit, ...] = (
    ZWarningBit(
        bit=0,
        name="SKY",
        survey_definition="Sky fiber: allocated to blank sky rather than to a science target.",
        why_it_matters="Not a science target at all; there is no AGN to classify.",
        threatens=("identity",),
        disposition=Disposition.FAIL,
        automatic=True,
    ),
    ZWarningBit(
        bit=1,
        name="LITTLE_COVERAGE",
        survey_definition="Too few good pixels in the fit range.",
        why_it_matters=(
            "Directly threatens whether Hbeta is measurable. Overlaps with our own "
            "coverage and masked-fraction checks, which are the binding ones."
        ),
        threatens=("hbeta", "continuum"),
        disposition=Disposition.REVIEW,
        automatic=False,
    ),
    ZWarningBit(
        bit=2,
        name="SMALL_DELTA_CHI2",
        survey_definition="Best fit is not significantly better than the next best.",
        why_it_matters=(
            "The redshift may be wrong, which would put the Hbeta window in the wrong "
            "place. Must be reconciled against the independent catalogue redshift."
        ),
        threatens=("redshift", "hbeta"),
        disposition=Disposition.REVIEW,
        automatic=False,
    ),
    ZWarningBit(
        bit=3,
        name="NEGATIVE_MODEL",
        survey_definition="Synthetic spectrum is negative in some region.",
        why_it_matters=(
            "Points at a pathological continuum fit, which is the quantity our host/AGN "
            "decomposition and S/N metric both rest on."
        ),
        threatens=("continuum",),
        disposition=Disposition.REVIEW,
        automatic=False,
    ),
    ZWarningBit(
        bit=4,
        name="MANY_OUTLIERS",
        survey_definition="Fraction of points more than 5 sigma from the best model is too large.",
        why_it_matters=(
            "Fires routinely on broad-line galaxies and high-S/N spectra, because the "
            "redshift template does not model a broad line well. Excluding on this bit "
            "would discard broad-line AGN for being broad-line AGN, and would bite "
            "hardest exactly where the broad line is strong."
        ),
        threatens=("none",),
        disposition=Disposition.PASS,
        automatic=True,
    ),
    ZWarningBit(
        bit=5,
        name="Z_FITLIMIT",
        survey_definition="Chi-squared minimum is at an edge of the redshift fitting range.",
        why_it_matters="Redshift is unreliable, so the rest-frame windows may be misplaced.",
        threatens=("redshift", "hbeta"),
        disposition=Disposition.REVIEW,
        automatic=False,
    ),
    ZWarningBit(
        bit=6,
        name="NEGATIVE_EMISSION",
        survey_definition="A QSO line exhibits negative emission, triggered only in QSO spectra.",
        why_it_matters=(
            "A negative line amplitude in the redshift fit can indicate a calibration or "
            "continuum problem in the very lines we measure."
        ),
        threatens=("hbeta", "calibration"),
        disposition=Disposition.REVIEW,
        automatic=False,
    ),
    ZWarningBit(
        bit=7,
        name="UNPLUGGED",
        survey_definition="The fiber was unplugged or damaged; no data.",
        why_it_matters=(
            "There is no spectrum to measure, so the epoch cannot enter a pair at all. "
            "Excluded as an acquisition failure, not as a measurement outcome."
        ),
        threatens=("identity",),
        disposition=Disposition.FAIL,
        automatic=True,
    ),
    ZWarningBit(
        bit=8,
        name="BAD_TARGET",
        survey_definition="Catastrophically bad targeting data.",
        why_it_matters=(
            "Target identity is not trustworthy, so the epoch may not belong to the "
            "labelled object at all. Pairing it would compare two different sources and "
            "manufacture a spurious transition."
        ),
        threatens=("identity",),
        disposition=Disposition.FAIL,
        automatic=True,
    ),
    ZWarningBit(
        bit=9,
        name="NODATA",
        survey_definition="No data for this fiber.",
        why_it_matters=(
            "No flux was recorded, so nothing can be fitted or classified. DESI documents "
            "that selecting on fiber status alone can retain no-data spectra, which is why "
            "this is checked explicitly rather than inferred from a status word."
        ),
        threatens=("identity",),
        disposition=Disposition.FAIL,
        automatic=True,
    ),
)

BY_BIT = {b.bit: b for b in ZWARNING_BITS}
BY_NAME = {b.name: b for b in ZWARNING_BITS}


@dataclass(frozen=True)
class ZWarningVerdict:
    """Bit-aware assessment of one ZWARNING value."""

    zwarning: int                      # the original mask, always preserved
    bits: tuple[ZWarningBit, ...]
    unknown_bits: tuple[int, ...]
    disposition: Disposition
    requires_manual_review: bool

    @property
    def strict_pass(self) -> bool:
        """``ZWARNING == 0``, kept for the strict-only robustness re-run."""
        return self.zwarning == 0

    @property
    def bit_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bits) + tuple(f"UNKNOWN_{b}" for b in self.unknown_bits)

    @property
    def threatens(self) -> tuple[str, ...]:
        seen: list[str] = []
        for bit in self.bits:
            for threat in bit.threatens:
                if threat != "none" and threat not in seen:
                    seen.append(threat)
        return tuple(seen)

    def threatens_hbeta_measurement(self) -> bool:
        """The question that actually matters for this experiment."""
        relevant = {"redshift", "hbeta", "continuum", "wavelength", "calibration", "identity"}
        return bool(set(self.threatens) & relevant)

    def summary(self) -> str:
        if self.zwarning == 0:
            return "ZWARNING=0"
        names = "|".join(self.bit_names)
        threats = ",".join(self.threatens) or "none"
        return f"ZWARNING={self.zwarning} [{names}] threatens={threats} -> {self.disposition}"


def classify_zwarning(zwarning: int) -> ZWarningVerdict:
    """Classify a ZWARNING mask into PASS / REVIEW / FAIL.

    An unrecognised bit is never silently ignored: it forces REVIEW, because an
    unknown warning is exactly the case where a blind pass is dangerous.
    """
    value = int(zwarning)
    if value < 0:
        raise ValueError(f"ZWARNING must be non-negative, got {value}")

    set_bits = [i for i in range(value.bit_length()) if value & (1 << i)]
    known = tuple(BY_BIT[i] for i in set_bits if i in BY_BIT)
    unknown = tuple(i for i in set_bits if i not in BY_BIT)

    if any(b.disposition is Disposition.FAIL for b in known):
        disposition = Disposition.FAIL
    elif unknown or any(b.disposition is Disposition.REVIEW for b in known):
        disposition = Disposition.REVIEW
    else:
        disposition = Disposition.PASS

    requires_review = disposition is Disposition.REVIEW or any(
        not b.automatic for b in known if b.disposition is not Disposition.PASS
    )

    return ZWarningVerdict(
        zwarning=value,
        bits=known,
        unknown_bits=unknown,
        disposition=disposition,
        requires_manual_review=requires_review,
    )


def describe_bits() -> str:
    """The bit table, for the appendix and for review."""
    lines = [
        f"{'bit':>4} {'name':<18} {'disp':<7} {'auto':<5} {'threatens':<34} definition",
        "-" * 118,
    ]
    for bit in ZWARNING_BITS:
        threats = ",".join(bit.threatens)
        lines.append(
            f"{bit.bit:>4} {bit.name:<18} {bit.disposition:<7} "
            f"{str(bit.automatic):<5} {threats:<34} {bit.survey_definition}"
        )
    return "\n".join(lines)


__all__ = [
    "BY_BIT",
    "BY_NAME",
    "ZWARNING_BITS",
    "Disposition",
    "ZWarningBit",
    "ZWarningVerdict",
    "classify_zwarning",
    "describe_bits",
]
