#!/usr/bin/env python
"""Stage 6 — resolve and download SDSS spectra for the reference candidates.

Writes raw FITS to ``03_spectra/raw_sdss/`` together with a manifest carrying
everything specification §74 requires: survey, release, spectrum identifier,
MJD, source URL, sha256, and the processing version. Raw files are gitignored;
the manifest is committed, so any discarded file is refetchable.

Epoch selection is mechanical. For each object we take the two spectra nearest
the epoch MJDs recorded by the labelling paper, and never choose on line
strength — choosing epochs by how much the line changed would build the
outcome into the sample.

Quality flags are *recorded, not applied*. ZWARNING filtering happens in
``06_spectrum_quality_control.py`` against the frozen config, so that the
number of spectra lost to each flag is itself a reportable quantity.

Usage
-----
    uv run python 00_scripts/04_download_sdss_spectra.py --slice
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import astropy.units as u
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.infra.exclusions import record_exclusions  # noqa: E402

SCRIPT = "04_download_sdss_spectra.py"
PROCESSING_VERSION = "p3sf-0.1.0"
SAS_BASE = "https://data.sdss.org/sas/dr17"
MATCH_RADIUS = 3.0 * u.arcsec

SPEC_FIELDS = [
    "plate", "mjd", "fiberID", "run2d", "z", "zWarning", "class", "subclass",
    "instrument", "survey", "programname", "snMedian", "ra", "dec",
]


def sas_url(plate: int, mjd: int, fiberid: int, run2d: str) -> str:
    """Lite spectrum URL. Legacy (run2d='26') and BOSS trees differ."""
    stem = f"spec-{plate:04d}-{mjd:05d}-{fiberid:04d}.fits"
    if str(run2d) == "26":
        return f"{SAS_BASE}/sdss/spectro/redux/26/spectra/lite/{plate:04d}/{stem}"
    return f"{SAS_BASE}/eboss/spectro/redux/{run2d}/spectra/lite/{plate:04d}/{stem}"


def resolve(object_id: str, ra: float, dec: float) -> pd.DataFrame:
    """Every SDSS spectrum within the match radius. Empty frame when none."""
    table = SDSS.query_region(
        SkyCoord(ra, dec, unit="deg"),
        radius=MATCH_RADIUS,
        spectro=True,
        specobj_fields=SPEC_FIELDS,
    )
    if table is None or len(table) == 0:
        return pd.DataFrame()
    frame = table.to_pandas()
    frame.insert(0, "object_id", object_id)
    return frame


def choose_epochs(frame: pd.DataFrame, epoch_1: float, epoch_2: float) -> pd.DataFrame:
    """Pick the spectra nearest each labelled epoch, without duplication.

    Never selects on line strength or S/N: the epoch pair must be fixed by the
    published event, not by the size of the change we are about to measure.
    """
    if frame.empty:
        return frame

    remaining = frame.copy()
    chosen: list[pd.Series] = []
    for label, wanted in (("early", epoch_1), ("late", epoch_2)):
        if remaining.empty:
            break
        index = (remaining.mjd - wanted).abs().idxmin()
        row = remaining.loc[index].copy()
        row["epoch_role"] = label
        row["target_mjd"] = wanted
        row["mjd_offset_days"] = float(row.mjd - wanted)
        chosen.append(row)
        remaining = remaining.drop(index=index)

    return pd.DataFrame(chosen)


def download(url: str, destination: Path, *, timeout: float = 180.0) -> tuple[bool, str, int]:
    """Fetch one spectrum. Returns (ok, sha256_or_error, n_bytes)."""
    if destination.exists():
        payload = destination.read_bytes()
        return True, hashlib.sha256(payload).hexdigest(), len(payload)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        return False, f"{type(error).__name__}: {error}", 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return True, hashlib.sha256(response.content).hexdigest(), len(response.content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true", help="use the feasibility-slice candidates")
    parser.add_argument("--dry-run", action="store_true", help="resolve identifiers, download nothing")
    args = parser.parse_args()

    root = project_root()
    candidates_path = root / "04_reference_sample" / "candidates" / (
        "slice_candidates.csv" if args.slice else "reference_candidates.csv"
    )
    if not candidates_path.exists():
        print(f"missing {candidates_path}; run 07_construct_reference_candidates.py first", file=sys.stderr)
        return 1

    candidates = pd.read_csv(candidates_path)
    raw_dir = root / "03_spectra" / "raw_sdss"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    exclusions: list[tuple[str, str, str]] = []

    for candidate in candidates.itertuples():
        found = resolve(candidate.object_id, candidate.ra, candidate.dec)
        if found.empty:
            exclusions.append(
                (candidate.object_id, "NO_SDSS_DESI_MATCH", "no SDSS spectrum within 3 arcsec")
            )
            print(f"{candidate.object_id:26s} no SDSS spectra")
            continue

        selected = choose_epochs(found, candidate.epoch_1, candidate.epoch_2)
        if len(selected) < 2:
            exclusions.append(
                (
                    candidate.object_id,
                    "INSUFFICIENT_EPOCHS",
                    f"only {len(selected)} SDSS spectrum available",
                )
            )
            print(f"{candidate.object_id:26s} only {len(selected)} epoch(s), skipped")
            continue

        print(f"{candidate.object_id:26s} {len(found)} spectra, using {len(selected)}")
        for record in selected.itertuples():
            url = sas_url(int(record.plate), int(record.mjd), int(record.fiberID), record.run2d)
            filename = f"{candidate.object_id.replace(' ', '')}__{record.epoch_role}__" \
                       f"spec-{int(record.plate):04d}-{int(record.mjd):05d}-{int(record.fiberID):04d}.fits"
            destination = raw_dir / filename

            if args.dry_run:
                ok, digest, size = True, "", 0
            else:
                ok, digest, size = download(url, destination)
                time.sleep(0.3)  # courtesy to the archive

            if not ok:
                exclusions.append(
                    (candidate.object_id, "CALIBRATION_FAILURE", f"download failed: {digest}")
                )
                print(f"    FAIL {record.epoch_role:5s} {url}\n         {digest}")
                continue

            rows.append(
                {
                    "object_id": candidate.object_id,
                    "epoch_role": record.epoch_role,
                    "survey": "SDSS",
                    "release": "DR17",
                    "instrument": record.instrument,
                    "sdss_survey": record.survey,
                    "spectrum_id": f"{int(record.plate)}-{int(record.mjd)}-{int(record.fiberID)}",
                    "plate": int(record.plate),
                    "mjd": int(record.mjd),
                    "fiberid": int(record.fiberID),
                    "run2d": record.run2d,
                    "target_mjd": record.target_mjd,
                    "mjd_offset_days": record.mjd_offset_days,
                    "redshift": record.z,
                    "zwarning": int(record.zWarning),
                    "sdss_class": getattr(record, "_asdict", lambda: {})().get("class", ""),
                    "sn_median": record.snMedian,
                    "ra": record.ra,
                    "dec": record.dec,
                    "url": url,
                    "path": str(destination.relative_to(root)),
                    "n_bytes": size,
                    "sha256": digest,
                    "processing_version": PROCESSING_VERSION,
                }
            )
            print(
                f"    {record.epoch_role:5s} MJD {int(record.mjd)} "
                f"(offset {record.mjd_offset_days:+.0f} d)  zw={int(record.zWarning)}  "
                f"{size / 1e6:.2f} MB"
            )

    manifest_path = raw_dir / ("manifest_slice.csv" if args.slice else "manifest.csv")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)

    n_excluded = record_exclusions(exclusions, script=SCRIPT)

    print()
    print(f"spectra downloaded  {len(manifest):>4}")
    print(f"objects with a pair {manifest.object_id.nunique() if not manifest.empty else 0:>4}")
    print(f"exclusions recorded {n_excluded:>4}")
    print(f"manifest            {manifest_path.relative_to(root)}")
    if not manifest.empty:
        flagged = int((manifest.zwarning != 0).sum())
        print(f"nonzero ZWARNING    {flagged:>4}  (recorded, not applied; see 06_spectrum_quality_control.py)")
        print(f"total size          {manifest.n_bytes.sum() / 1e6:>6.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
