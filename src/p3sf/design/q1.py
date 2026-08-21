"""Frozen Q1 measurement-validity state machine (D-088)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q1Validity:
    """The mandatory monotone chain from eligibility to classification."""

    eligible: bool
    degradation_succeeded: bool
    spectral_qc_passed: bool
    fit_completed: bool
    fit_valid: bool
    required_measurement_available: bool
    variance_calibration_supported: bool
    classifiable: bool

    def __post_init__(self) -> None:
        chain = (
            self.eligible,
            self.degradation_succeeded,
            self.spectral_qc_passed,
            self.fit_completed,
            self.fit_valid,
            self.required_measurement_available,
            self.variance_calibration_supported,
            self.classifiable,
        )
        if any(
            later and not earlier
            for earlier, later in zip(chain, chain[1:], strict=False)
        ):
            raise ValueError("Q1 validity stages must be monotone; a failed stage cannot recover")

    @property
    def terminal_status(self) -> str:
        """First failed stage, or ``CLASSIFIABLE`` when the whole chain passes."""
        names = (
            "INELIGIBLE",
            "DEGRADATION_FAILED",
            "SPECTRAL_QC_FAILED",
            "FIT_NOT_COMPLETED",
            "FIT_INVALID",
            "REQUIRED_MEASUREMENT_UNAVAILABLE",
            "VARIANCE_CALIBRATION_UNSUPPORTED",
            "UNCLASSIFIABLE",
        )
        values = (
            self.eligible,
            self.degradation_succeeded,
            self.spectral_qc_passed,
            self.fit_completed,
            self.fit_valid,
            self.required_measurement_available,
            self.variance_calibration_supported,
            self.classifiable,
        )
        for name, value in zip(names, values, strict=True):
            if not value:
                return name
        return "CLASSIFIABLE"


__all__ = ["Q1Validity"]
