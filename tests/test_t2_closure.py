from __future__ import annotations

import numpy as np
import pytest

from p3sf.calibration.t2_closure import closure_sn_bin, local_snr


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (49.999, "30-50"),
        (50.0, "50-55"),
        (52.375850, "50-55"),
        (54.999, "50-55"),
        (55.0, "55+_OUTSIDE_CLOSURE"),
        (100.0, "55+_OUTSIDE_CLOSURE"),
    ],
)
def test_bounded_closure_bin(value: float, expected: str) -> None:
    assert closure_sn_bin(value) == expected


def test_local_snr_matches_frozen_definition() -> None:
    flux = np.array([-2.0, 4.0, 3.0])
    ivar = np.array([4.0, 1.0, 4.0])
    assert local_snr(flux, ivar) == pytest.approx(4.0)
