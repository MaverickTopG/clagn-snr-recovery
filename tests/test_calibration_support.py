from p3sf.calibration.support import CalibrationSupport


def test_unsupported_uncertainty_preserves_point_estimate_without_classifying() -> None:
    state = CalibrationSupport.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED
    assert state.point_estimate_available
    assert not state.classifiable


def test_uncalibrated_significance_is_not_classifiable() -> None:
    state = CalibrationSupport.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION
    assert not state.point_estimate_available
    assert not state.classifiable


def test_only_calibrated_state_is_classifiable() -> None:
    assert CalibrationSupport.CALIBRATED.classifiable
    assert CalibrationSupport.CALIBRATED.point_estimate_available
    for state in CalibrationSupport:
        if state is not CalibrationSupport.CALIBRATED:
            assert not state.classifiable
