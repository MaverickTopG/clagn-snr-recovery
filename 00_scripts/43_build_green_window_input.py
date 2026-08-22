#!/usr/bin/env python
"""Stage 43 -- compact redistributable input for the Stage 42 sensitivity.

Stage 42 recomputes the Green statistic under an inflated denominator. It reads
`spectrum_fit_results_d094.parquet`, which is 675 MB: too large for the code
repository and not part of the published dataset. Yet the Green statistic only
ever touches pixels near the rest-frame 4750-4940 A window, so the overwhelming
majority of each stored array is never read.

This script extracts exactly the pixels the statistic can reach, plus a margin,
for exactly the epochs Stage 42 uses, and writes them to a compact parquet with
a recorded SHA-256. Stage 42 falls back to that file when the full product is
absent, so the sensitivity is reproducible from a public checkout.

The extraction is verified, not assumed: every GOLD epoch is rebinned from both
the full array and the window slice, and the run aborts unless the resulting
2 A bins are bitwise identical. Nothing frozen is modified.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.criteria.green_pixel import (  # noqa: E402
    GREEN_HBETA_INTERVAL_ANGSTROM,
    GreenEpochSpectrum,
    _rebin_two_angstrom,
)

ROOT = project_root()
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
FULL = RAW / "spectrum_fit_results_d094.parquet"
WINDOW = RAW / "green_pixel_window_d094.parquet"

# The statistic is maximized over [4750, 4940) after 2 A rebinning, and the gap
# check reaches one bin either side. A 20 A margin puts the truncated array's
# own end pixels -- the only ones whose edges differ under truncation -- well
# outside anything the binning can read.
MARGIN_ANGSTROM = 20.0
LOW = GREEN_HBETA_INTERVAL_ANGSTROM[0] - MARGIN_ANGSTROM
HIGH = GREEN_HBETA_INTERVAL_ANGSTROM[1] + 2.0 + MARGIN_ANGSTROM

ARRAYS = ["green_wavelength_rest", "green_line_flux", "green_variance",
          "green_variance_provenance"]
SCALARS = ["green_preprocessing_valid", "green_invalid_reason",
           "green_scale_factor", "green_scale_provenance"]


def _epoch(row: pd.Series) -> GreenEpochSpectrum:
    return GreenEpochSpectrum(
        wavelength_rest=np.asarray(row.green_wavelength_rest, dtype=float),
        line_flux=np.asarray(row.green_line_flux, dtype=float),
        variance=np.asarray(row.green_variance, dtype=float),
        variance_provenance=np.asarray(row.green_variance_provenance, dtype=str),
        preprocessing_valid=bool(row.green_preprocessing_valid),
        invalid_reason=str(row.green_invalid_reason or ""),
        scale_factor=float(row.green_scale_factor),
        scale_provenance=str(row.green_scale_provenance),
    )


def _same_rebin(full, cut) -> bool:
    """Whether truncation left the 2 A bins bitwise unchanged."""
    if (full is None) != (cut is None):
        return False
    if full is None:
        return True
    return all(np.array_equal(a, b) for a, b in zip(full, cut, strict=True))


def main() -> int:
    if not FULL.exists():
        print(f"ABORT: {FULL.name} is required to build the window input and is absent.")
        return 1

    outcomes = pd.read_csv(RAW / "classifier_outcomes_d094.csv")
    green = outcomes[
        (outcomes.criterion_id == "GREEN2022_FINAL")
        & outcomes.applicable
        & (outcomes.reference_tier == "GOLD")
    ]
    needed = sorted(set(green.bright_task_id) | set(green.faint_task_id))
    print(f"epochs required by Stage 42: {len(needed)}")

    fits = pd.read_parquet(FULL, columns=["task_id", *ARRAYS, *SCALARS]).set_index("task_id")
    fits = fits.loc[needed]

    records, mismatches = [], 0
    for task_id, row in fits.iterrows():
        wave = np.asarray(row.green_wavelength_rest, dtype=float)
        keep = (wave >= LOW) & (wave <= HIGH)
        record = {"task_id": task_id}
        for name in ARRAYS:
            record[name] = np.asarray(row[name])[keep]
        for name in SCALARS:
            record[name] = row[name]
        records.append(record)

        if not _same_rebin(_rebin_two_angstrom(_epoch(row)),
                           _rebin_two_angstrom(_epoch(pd.Series(record)))):
            mismatches += 1

    if mismatches:
        print(f"ABORT: {mismatches} epochs rebin differently after truncation.")
        return 1
    print(f"verified: all {len(records)} epochs rebin bitwise-identically after truncation")

    frame = pd.DataFrame(records)
    frame.to_parquet(WINDOW, index=False, compression="zstd")
    digest = hashlib.sha256(WINDOW.read_bytes()).hexdigest()
    kept = len(records[0]["green_wavelength_rest"])
    print(f"pixels kept per epoch: {kept} (window {LOW:.0f}-{HIGH:.0f} A)")
    print(f"wrote {WINDOW.relative_to(ROOT)}  "
          f"{WINDOW.stat().st_size / 1e6:.1f} MB  (full product {FULL.stat().st_size / 1e6:.0f} MB)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
