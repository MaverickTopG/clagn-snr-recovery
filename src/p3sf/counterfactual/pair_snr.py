"""Pair-level S/N counterfactuals — the three Q1 arms (D-010).

A spectrum *pair* does not have one signal-to-noise ratio. It has two,
``SNR_bright`` and ``SNR_faint``, and the faint-state S/N is partly coupled to
the physical transition itself: in a turn-off the continuum falls along with the
broad line, so the largest-amplitude events have the poorest faint-state data.
Collapsing the pair to a single number hides exactly the asymmetry Q1 exists to
measure.

Three arms, in the order they matter:

**Primary — faint only.** Degrade the faint/line-poor epoch; hold the bright
reference epoch at native quality. This asks the question the paper is actually
about: when does a real transition stop being classifiable as the crucial faint
spectrum worsens? It also avoids throwing away a good bright spectrum merely
because its partner is poor, which is why it buys sample support at exactly the
rungs where support is scarce.

**Secondary — matched.** Both epochs degraded to the same target, matching the
convention implicit in the literature, for comparison with the primary arm.

**Exploratory — joint.** The 2-D ``SNR_bright × SNR_faint`` surface, run only
where sample support permits.

Degradation remains one-directional throughout: a target above an epoch's native
S/N is refused, never synthesised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from p3sf.access.spectra import Spectrum
from p3sf.counterfactual.snr import DegradationResult, degrade_to_snr, measure_snr

Arm = Literal["faint_only", "matched", "joint"]

#: Sentinel meaning "leave this epoch at its observed quality".
NATIVE = None


@dataclass(frozen=True)
class PairCondition:
    """One point in the Q1 design: what each epoch's S/N should become."""

    arm: Arm
    snr_faint: float
    snr_bright: float | None = NATIVE  # None = native quality

    def __post_init__(self) -> None:
        if self.snr_faint <= 0:
            raise ValueError(f"snr_faint must be positive, got {self.snr_faint}")
        if self.snr_bright is not None and self.snr_bright <= 0:
            raise ValueError(f"snr_bright must be positive or None, got {self.snr_bright}")
        if self.arm == "faint_only" and self.snr_bright is not None:
            raise ValueError(
                "the faint_only arm holds the bright epoch at native quality; "
                "snr_bright must be None"
            )
        if self.arm == "matched" and self.snr_bright != self.snr_faint:
            raise ValueError(
                f"the matched arm degrades both epochs to the same target; "
                f"got bright={self.snr_bright}, faint={self.snr_faint}"
            )

    @property
    def label(self) -> str:
        bright = "native" if self.snr_bright is None else f"{self.snr_bright:g}"
        return f"{self.arm}:b={bright},f={self.snr_faint:g}"


@dataclass(frozen=True)
class PairDegradationResult:
    """A degraded pair, plus what was actually done to each epoch."""

    condition: PairCondition
    faint: DegradationResult
    bright: DegradationResult | None  # None when held at native quality
    bright_spectrum: Spectrum
    faint_spectrum: Spectrum

    @property
    def bright_was_degraded(self) -> bool:
        return self.bright is not None


def _seed(*parts: object) -> int:
    """Deterministic child seed from a fully-qualified key.

    SHA-256 rather than :func:`hash`, which is salted per process for strings
    and would make runs irreproducible across invocations.
    """
    payload = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _faint_seed(
    base_seed: int, pair_id: str, spectrum_id: str, realization: int, snr_faint: float
) -> int:
    """Common random numbers for the faint epoch, shared across arms.

    The key includes the pair and spectrum identity but **excludes the arm and
    the bright target**. That combination gives both properties we need:

    * same object, rung and realization -> bit-for-bit identical faint spectrum
      in the faint_only, matched and joint arms, making the arm comparison a
      *paired* experiment rather than one contaminated by Monte Carlo noise;
    * different objects -> independent noise, because objects are independent
      experimental units and must not share a random stream.

    Omitting the identity would give every AGN at a given rung and realization
    the same underlying Gaussian draws, correlating supposedly independent
    units across the whole sample.
    """
    return _seed("faint", base_seed, pair_id, spectrum_id, realization, f"{snr_faint:.6f}")


def _bright_seed(
    base_seed: int, pair_id: str, spectrum_id: str, realization: int, snr_bright: float
) -> int:
    """Seed for the bright epoch, independent of the faint one.

    The ``bright`` tag and the distinct spectrum identity both guarantee a
    different stream from the faint epoch, even in the matched arm where the two
    are degraded to the same target. Sharing a stream there would correlate the
    epochs and understate the scatter in every derived statistic.
    """
    return _seed("bright", base_seed, pair_id, spectrum_id, realization, f"{snr_bright:.6f}")


def _spectrum_id(spectrum: Spectrum, fallback: str) -> str:
    """Identity of one epoch, from its metadata when available."""
    value = spectrum.meta.get("spectrum_id") if spectrum.meta else None
    return str(value) if value is not None else fallback


def degrade_pair(
    bright: Spectrum,
    faint: Spectrum,
    condition: PairCondition,
    *,
    window_rest: tuple[float, float],
    base_seed: int,
    pair_id: str,
    realization: int = 0,
    bright_id: str | None = None,
    faint_id: str | None = None,
    seed_namespace: str | None = None,
) -> PairDegradationResult:
    """Apply one pair condition. Raises if either target exceeds native S/N.

    ``pair_id`` is required, not optional: it is what keeps different objects on
    independent random streams. Defaulting it would silently reintroduce
    cross-object correlation.
    """
    if not pair_id:
        raise ValueError(
            "pair_id is required; without it every object would share a random "
            "stream at a given rung and realization"
        )
    qualified_pair_id = (
        f"{seed_namespace}|{pair_id}" if seed_namespace is not None else pair_id
    )

    faint_result = degrade_to_snr(
        faint,
        condition.snr_faint,
        window_rest=window_rest,
        seed=_faint_seed(
            base_seed,
            qualified_pair_id,
            faint_id or _spectrum_id(faint, "faint"),
            realization,
            condition.snr_faint,
        ),
    )

    bright_result: DegradationResult | None = None
    bright_spectrum = bright
    if condition.snr_bright is not None:
        bright_result = degrade_to_snr(
            bright,
            condition.snr_bright,
            window_rest=window_rest,
            seed=_bright_seed(
                base_seed,
                qualified_pair_id,
                bright_id or _spectrum_id(bright, "bright"),
                realization,
                condition.snr_bright,
            ),
        )
        bright_spectrum = bright_result.spectrum

    return PairDegradationResult(
        condition=condition,
        faint=faint_result,
        bright=bright_result,
        bright_spectrum=bright_spectrum,
        faint_spectrum=faint_result.spectrum,
    )


def reachable_conditions(
    bright: Spectrum,
    faint: Spectrum,
    *,
    window_rest: tuple[float, float],
    faint_rungs: list[float],
    matched_rungs: list[float] | None = None,
    bright_rungs: list[float] | None = None,
    arms: tuple[Arm, ...] = ("faint_only", "matched"),
) -> list[PairCondition]:
    """Enumerate the conditions this pair can actually support.

    Unreachable rungs are skipped rather than raising: a mixed-quality reference
    sample will legitimately have objects that cannot climb the whole ladder,
    and the per-rung object count is a reportable quantity, not an error.

    The asymmetry that motivates the primary arm shows up here directly — a pair
    whose bright epoch is strong and faint epoch weak supports more faint_only
    conditions than matched ones.
    """
    snr_bright_native = measure_snr(bright, window_rest)
    snr_faint_native = measure_snr(faint, window_rest)

    conditions: list[PairCondition] = []

    if "faint_only" in arms:
        for rung in faint_rungs:
            if rung <= snr_faint_native:
                conditions.append(PairCondition("faint_only", snr_faint=rung))

    if "matched" in arms:
        for rung in matched_rungs if matched_rungs is not None else faint_rungs:
            if rung <= min(snr_bright_native, snr_faint_native):
                conditions.append(PairCondition("matched", snr_faint=rung, snr_bright=rung))

    if "joint" in arms and bright_rungs is not None:
        for bright_rung in bright_rungs:
            if bright_rung > snr_bright_native:
                continue
            for faint_rung in faint_rungs:
                if faint_rung > snr_faint_native:
                    continue
                conditions.append(
                    PairCondition("joint", snr_faint=faint_rung, snr_bright=bright_rung)
                )

    return conditions


def arm_support(
    native_bright: list[float],
    native_faint: list[float],
    rungs: list[float],
) -> dict[str, dict[float, int]]:
    """How many pairs each arm can place at each rung.

    Used to justify the frozen rung list: a rung supported by one or two objects
    is reported as underpowered rather than carried as a design point.
    """
    faint_only = {r: sum(1 for f in native_faint if f >= r) for r in rungs}
    matched = {
        r: sum(1 for b, f in zip(native_bright, native_faint, strict=True) if min(b, f) >= r)
        for r in rungs
    }
    return {"faint_only": faint_only, "matched": matched}


__all__ = [
    "NATIVE",
    "Arm",
    "PairCondition",
    "PairDegradationResult",
    "arm_support",
    "degrade_pair",
    "reachable_conditions",
]
