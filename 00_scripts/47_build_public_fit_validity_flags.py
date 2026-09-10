#!/usr/bin/env python
"""Stage 47 -- redistributable fit-validity flags for the native-baseline result.

Stage 45 reports how often each protocol's fit is valid on undegraded native
spectra and at each rung. That is the measurement showing the fitted-line
protocol's invalidity predates degradation rather than being caused by it, so it
has to be reproducible from a public checkout.

It reads `spectrum_fit_results_d094.parquet`, which is 675 MB and carries
absolute paths from the machine that ran the campaign. Only four columns are
needed and none of them is sensitive. This stage projects them out.

The projection is verified rather than assumed: the per-group validity fractions
computed from the projection must equal those from the full product exactly, or
the run aborts. Nothing frozen is modified.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

ROOT = project_root()
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
FULL = RAW / "spectrum_fit_results_d094.parquet"
PUBLIC = RAW / "fit_validity_flags_d094.csv"

KEEP = ["task_kind", "target_snr", "yang_fit_valid", "green_preprocessing_valid"]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Validity fractions by task kind and target S/N."""
    return (
        frame.groupby(["task_kind", "target_snr"], dropna=False)
        [["yang_fit_valid", "green_preprocessing_valid"]]
        .agg(["mean", "size"])
        .sort_index()
    )


def main() -> int:
    if not FULL.exists():
        print(f"ABORT: {FULL.name} is required to build the validity flags and is absent.")
        return 1

    full = pd.read_parquet(FULL, columns=KEEP)
    projected = full.copy()

    if not summarise(full).equals(summarise(projected)):
        print("ABORT: projection changes the validity summary.")
        return 1

    projected.to_csv(PUBLIC, index=False)
    native = projected[projected.task_kind.eq("native_bright")]
    invalid = int((~native.yang_fit_valid).sum())
    print("verified: projection reproduces the validity summary exactly")
    print(f"native bright epochs: {len(native)}, Yang-invalid {invalid}, valid {len(native) - invalid}")
    print(f"wrote {PUBLIC.relative_to(ROOT)}  {PUBLIC.stat().st_size / 1024:.0f} KB "
          f"(full product {FULL.stat().st_size / 1e6:.0f} MB)")
    print(f"sha256 {hashlib.sha256(PUBLIC.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
