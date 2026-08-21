#!/usr/bin/env python
"""Stage 11 (slice subset) — assemble reference-event candidates.

For the feasibility slice this selects a handful of literature-labelled
changing-look AGN that are eligible for the counterfactual experiment, and
records why every rejected object was rejected.

Selection is deliberately *mechanical*: every filter is a frozen-config value
or a structural requirement, and nothing here inspects an outcome. Gold status
is not conferred by this script — blind adjudication does that later
(preregistration §6). These are candidates.

Usage
-----
    uv run python 00_scripts/07_construct_reference_candidates.py --slice
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.infra.exclusions import record_exclusions  # noqa: E402
from p3sf.infra.splits import assign_splits  # noqa: E402

SCRIPT = "07_construct_reference_candidates.py"

# One labelled input among several. Its own split column is discarded (D-005).
BENCHMARK = Path(
    os.environ.get("P3SF_BENCHMARK_CSV", "external/benchmark_master.csv")
)

SDSS_NAME = re.compile(r"^SDSSJ+\d{6}\.\d+[+-]\d{6}", re.IGNORECASE)


def _is_mjd(value: object) -> bool:
    """True when an epoch is a numeric MJD rather than a calendar string.

    Objects labelled from mid-century photographic or IUE campaigns carry
    date strings; they are real CLAGN but have no SDSS/DESI epoch pair and
    cannot enter a counterfactual built on those two surveys.
    """
    try:
        mjd = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return 40000.0 < mjd < 70000.0


def load_candidates(path: Path = BENCHMARK) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"labelled input not found: {path}. See 00_admin/decisions_log.md D-005."
        )
    frame = pd.read_csv(path)
    return frame.drop(columns=[c for c in ("split", "y_true", "mag") if c in frame.columns])


def select(frame: pd.DataFrame, *, development_only: bool) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Apply the mechanical filters. Returns (kept, exclusion rows)."""
    cfg = load_config()
    excluded: list[tuple[str, str, str]] = []

    working = frame[frame["class"] == "clagn"].copy()
    working["object_id"] = working["canonical_id"].astype(str)
    working = working.drop_duplicates(subset="object_id")

    def drop(mask: pd.Series, code: str, detail: str) -> None:
        nonlocal working
        for oid in working.loc[mask, "object_id"]:
            excluded.append((str(oid), code, detail))
        working = working.loc[~mask].copy()

    drop(
        working.redshift.isna() | (working.redshift <= cfg.redshift.primary_min),
        "BAD_REDSHIFT",
        "missing or non-positive redshift",
    )
    drop(
        working.redshift > cfg.redshift.primary_max,
        "OUT_OF_REDSHIFT_RANGE",
        f"z > {cfg.redshift.primary_max} (primary cohort)",
    )
    drop(
        ~working.canonical_id.astype(str).str.match(SDSS_NAME),
        "NO_SDSS_DESI_MATCH",
        "identifier is not an SDSS designation; no SDSS/DESI epoch pair expected",
    )
    drop(
        ~(working.epoch_1.map(_is_mjd) & working.epoch_2.map(_is_mjd)),
        "EPOCHS_NOT_ORDERED",
        "epoch recorded as a calendar date rather than an MJD",
    )

    working["epoch_1"] = working.epoch_1.astype(float)
    working["epoch_2"] = working.epoch_2.astype(float)
    drop(
        working.epoch_2 <= working.epoch_1,
        "EPOCHS_NOT_ORDERED",
        "second epoch does not follow the first",
    )

    splits = assign_splits(working.object_id.tolist())
    working = working.merge(splits, on="object_id", how="left")

    if development_only:
        not_dev = working.stratum != "development"
        for oid in working.loc[not_dev, "object_id"]:
            excluded.append(
                (str(oid), "HELD_OUT_STRATUM", "reserved for validation/replication")
            )
        working = working.loc[~not_dev].copy()

    working = working.sort_values("object_id").reset_index(drop=True)
    return working, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice", action="store_true", help="feasibility slice: development stratum, N from config"
    )
    parser.add_argument("--limit", type=int, default=None, help="override the object count")
    args = parser.parse_args()

    cfg = load_config()
    root = project_root()

    frame = load_candidates()
    kept, excluded = select(frame, development_only=args.slice)

    if args.slice:
        limit = args.limit or cfg.slice_grid.n_objects
        chosen = kept.head(limit).copy()
        for oid in kept.object_id.iloc[limit:]:
            excluded.append(
                (str(oid), "SLICE_BUDGET", "beyond the feasibility-slice object budget")
            )
        destination = root / "04_reference_sample" / "candidates" / "slice_candidates.csv"
    else:
        chosen = kept
        destination = root / "04_reference_sample" / "candidates" / "reference_candidates.csv"

    columns = [
        "object_id", "ra", "dec", "redshift", "epoch_1", "epoch_2",
        "spectral_state_1", "spectral_state_2", "label_source", "label_method",
        "confidence", "bucket", "stratum",
    ]
    chosen[[c for c in columns if c in chosen.columns]].to_csv(destination, index=False)

    n_excluded = record_exclusions(excluded, script=SCRIPT)

    print(f"input                {len(frame):>5}")
    print(f"clagn, filters passed{len(kept):>5}")
    print(f"selected             {len(chosen):>5}  -> {destination.relative_to(root)}")
    print(f"exclusions recorded  {n_excluded:>5}  -> 00_admin/exclusions_log.csv")
    print()
    if not chosen.empty:
        print(chosen[["object_id", "redshift", "epoch_1", "epoch_2", "stratum"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
