#!/usr/bin/env python
"""Stage 44 -- minimal public seed manifest for the stage 40 rung-independence check.

Stage 40 reports how many transition-realization pairs reuse a faint-epoch seed
between the two S/N rungs. The designed answer is zero, and reporting the
denominator alongside it keeps the check honest.

That check reads `fit_task_manifest_d094.csv`, which is 3.7 MB and carries
columns that must not be redistributed: `local_path` records absolute paths on
the machine that ran the production campaign, and the manifest also holds
per-spectrum identifiers and source checksums that belong with the survey data
rather than the code.

This script projects out only the five columns the check actually reads, for
only the `faint_shared` rows it looks at, and writes them to a small CSV that is
safe to ship. The projection is verified, not assumed: the overlap count and
denominator computed from the projection must equal those computed from the full
manifest, or the run aborts.

Nothing frozen is modified.
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
FULL = RAW / "fit_task_manifest_d094.csv"
PUBLIC = RAW / "fit_task_seed_manifest_d094.csv"

KEEP = ["task_kind", "transition_id", "realization", "target_snr", "seed"]


def rung_seed_overlap(manifest: pd.DataFrame) -> tuple[int, int]:
    """Shared faint seeds between rungs, and the number of pairs compared."""
    faint = manifest[manifest.task_kind.eq("faint_shared")]
    wide = faint.pivot_table(
        index=["transition_id", "realization"], columns="target_snr",
        values="seed", aggfunc="first",
    ).dropna()
    return int((wide[5.0] == wide[10.0]).sum()), int(len(wide))


def main() -> int:
    if not FULL.exists():
        print(f"ABORT: {FULL.name} is required to build the public seed manifest and is absent.")
        return 1

    full = pd.read_csv(FULL)
    projected = full.loc[full.task_kind.eq("faint_shared"), KEEP].reset_index(drop=True)

    if rung_seed_overlap(full) != rung_seed_overlap(projected):
        print("ABORT: projection changes the rung-independence result.")
        return 1

    dropped = sorted(set(full.columns) - set(KEEP))
    projected.to_csv(PUBLIC, index=False)
    overlap, pairs = rung_seed_overlap(projected)
    print(f"verified: projection reproduces the rung check exactly ({overlap} of {pairs} pairs)")
    print(f"columns dropped: {', '.join(dropped)}")
    print(f"wrote {PUBLIC.relative_to(ROOT)}  {PUBLIC.stat().st_size / 1024:.0f} KB "
          f"(full manifest {FULL.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {hashlib.sha256(PUBLIC.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
