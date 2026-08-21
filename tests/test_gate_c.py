from __future__ import annotations

import numpy as np
import pytest

from p3sf.calibration.gate_c import normalized_difference, robust_width, sn_bin


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "00-10"), (9.99, "00-10"), (10.0, "10-20"), (20.0, "20-30"), (30.0, "30-50"), (50.0, "50+")],
)
def test_frozen_local_snr_bins(value: float, expected: str) -> None:
    assert sn_bin(value) == expected


def test_cross_bin_correction_uses_each_exposures_factor() -> None:
    f1 = np.array([4.0])
    f2 = np.array([1.0])
    s1 = np.array([2.0])
    s2 = np.array([3.0])
    result = normalized_difference(f1, f2, s1, s2, k_a=1.5, k_b=2.0)
    assert result[0] == pytest.approx(3.0 / np.sqrt((1.5 * 2.0) ** 2 + (2.0 * 3.0) ** 2))


def test_robust_width_is_frozen_percentile_span() -> None:
    values = np.arange(100.0)
    expected = (np.percentile(values, 84) - np.percentile(values, 16)) / 2
    assert robust_width(values) == pytest.approx(expected)
    assert robust_width(values[:20]) is None
