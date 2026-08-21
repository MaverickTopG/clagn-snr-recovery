#!/usr/bin/env python
"""Stage 4 — resolve DESI DR1 counterparts for the reference candidates.

Cone-searches ``desi_dr1.zpix`` around each candidate through the NOIRLab Data
Lab query client. The TAP ``/sync`` endpoint is not usable here: it rejects both
``q3c_*`` functions and standard ADQL ``CONTAINS``/``POINT`` geometry (D-009).

**TARGETID must cross the wire as text.** It is an int64 with 17 significant
digits, beyond the ~15--16 digits float64 represents exactly, so a numeric
round trip silently corrupts the low digits — observed as
``39633043169740264`` against the coadd's true ``39633043169740267``, which
then looks like "target absent from coadd" rather than like a precision bug.
Every 64-bit identifier is therefore cast to VARCHAR in SQL and kept as a
Python string until it is used.

Quality flags are recorded, never applied. Redshift disagreement is likewise
recorded here and adjudicated downstream.

Usage
-----
    uv run python 00_scripts/03_crossmatch_catalogs.py --slice
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

SCRIPT = "03_crossmatch_catalogs.py"
MATCH_RADIUS_ARCSEC = 1.5

# CAST every 64-bit identifier to VARCHAR. See the module docstring.
ZPIX_COLUMNS = """
  CAST(targetid AS VARCHAR) AS targetid,
  CAST(id AS VARCHAR) AS zpix_id,
  healpix, survey, program,
  z, zerr, zwarn, spectype, subtype,
  coadd_fiberstatus, coadd_numexp, coadd_numnight, coadd_exptime,
  zcat_primary, zcat_nspec,
  min_mjd, mean_mjd, max_mjd,
  mean_fiber_ra, mean_fiber_dec
"""


def cone_query(ra: float, dec: float, radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> str:
    return (
        f"SELECT {ZPIX_COLUMNS} FROM desi_dr1.zpix "
        f"WHERE q3c_radial_query(mean_fiber_ra, mean_fiber_dec, "
        f"{ra}, {dec}, {radius_arcsec / 3600.0})"
    )


def resolve(candidates: pd.DataFrame) -> pd.DataFrame:
    from dl import queryClient as qc

    frames: list[pd.DataFrame] = []
    for candidate in candidates.itertuples():
        try:
            text = qc.query(sql=cone_query(candidate.ra, candidate.dec), fmt="csv")
        except Exception as error:  # noqa: BLE001
            print(f"{candidate.object_id:26s} QUERY FAILED {str(error)[:60]}")
            continue

        if not text.strip() or "\n" not in text.strip():
            print(f"{candidate.object_id:26s} desi_n=0")
            continue

        # targetid/zpix_id as str: pandas would otherwise re-infer them as int64
        # here, which is safe, but any later float coercion would not be.
        found = pd.read_csv(io.StringIO(text), dtype={"targetid": str, "zpix_id": str})
        found.insert(0, "object_id", candidate.object_id)
        found["catalogue_redshift"] = candidate.redshift
        found["delta_z"] = found.z - candidate.redshift
        frames.append(found)

        flags = []
        if (found.zwarn != 0).any():
            flags.append("ZWARN")
        if (found.coadd_fiberstatus != 0).any():
            flags.append("FIBERSTATUS")
        if (found.delta_z.abs() > 0.05 * (1 + candidate.redshift)).any():
            flags.append("REDSHIFT_CONFLICT")
        note = ("  <-- " + ",".join(flags)) if flags else ""
        print(f"{candidate.object_id:26s} desi_n={len(found)}{note}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    root = project_root()
    suffix = "_slice" if args.slice else ""
    candidates = pd.read_csv(
        root / "04_reference_sample" / "candidates"
        / ("slice_candidates.csv" if args.slice else "reference_candidates.csv")
    )

    matches = resolve(candidates)
    destination = root / "02_catalogs" / "crossmatches" / f"desi_dr1_matches{suffix}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(destination, index=False)

    print()
    print(f"candidates queried     {len(candidates):>4}")
    if matches.empty:
        print("no DESI matches")
        return 0
    print(f"DESI epochs found      {len(matches):>4}")
    print(f"objects with an epoch  {matches.object_id.nunique():>4}")
    print(f"nonzero ZWARN          {int((matches.zwarn != 0).sum()):>4}")
    print(f"nonzero FIBERSTATUS    {int((matches.coadd_fiberstatus != 0).sum()):>4}")
    conflicts = matches[matches.delta_z.abs() > 0.05 * (1 + matches.catalogue_redshift)]
    print(f"redshift conflicts     {len(conflicts):>4}")
    for row in conflicts.itertuples():
        print(f"    {row.object_id}: DESI z={row.z:.4f} vs catalogue {row.catalogue_redshift:.4f}")
    print(f"written -> {destination.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
