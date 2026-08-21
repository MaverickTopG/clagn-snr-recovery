"""Measurement-level uncertainty-calibration support semantics.

Calibration support belongs to a *quantity*, not to an object.  In particular,
an unsupported H-beta uncertainty must not erase a usable continuum estimate or
silently turn an undecidable classification into ``non-CL``.
"""

from __future__ import annotations

from enum import StrEnum


class CalibrationSupport(StrEnum):
    """Support state for one measurement or criterion dependency."""

    CALIBRATED = "CALIBRATED"
    POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED = (
        "POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED"
    )
    UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION = (
        "UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION"
    )
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    DEPENDENCY_NOT_YET_DEFINABLE = "DEPENDENCY_NOT_YET_DEFINABLE"

    @property
    def point_estimate_available(self) -> bool:
        return self in {
            CalibrationSupport.CALIBRATED,
            CalibrationSupport.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED,
        }

    @property
    def classifiable(self) -> bool:
        """Whether calibration support alone permits a criterion decision."""
        return self is CalibrationSupport.CALIBRATED


__all__ = ["CalibrationSupport"]
