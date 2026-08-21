#!/usr/bin/env python
"""Stage 6 — resolve and extract DESI DR1 spectra for the reference candidates.

A DESI healpix coadd holds every target in its pixel, so the file is opened once
per (survey, program, healpix) via HTTP range reads and every wanted target is
drained from that single handle. What lands on disk is a per-target extraction,
not the whole coadd, so provenance records the *source coadd URL* plus TARGETID
alongside the checksum of the extracted product.

Per D-017 each spectrum carries TARGETID, HEALPIX, survey/program, the spectro
production (``iron`` for DR1), ZWARN, COADD_FIBERSTATUS and raw-product
provenance. Quality flags are recorded, never applied here: DESI warns that
selecting on COADD_FIBERSTATUS alone can retain no-data spectra, so ZWARN must
be considered too, and both decisions belong in QC where the loss is countable.

The B, R and Z cameras are combined with official desispec ``coadd_cameras`` on
exactly one published healpix-coadd row at a time (D-077). MASK and the complete
EXP_FIBERMAP are preserved. There is deliberately no concatenate/sort fallback.

Usage
-----
    uv run python 00_scripts/05_download_desi_spectra.py --slice
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.infra.exclusions import record_exclusions  # noqa: E402
from p3sf.ingest.desi import (  # noqa: E402
    PRODUCT_TYPE,
    ScienceRecord,
    exposure_summary,
    merge_cameras_official,
    parse_catalog_bool,
    relative_product_path,
)

SCRIPT = "05_download_desi_spectra.py"
PROCESSING_VERSION = "p3sf-0.2.0-desi-ingest-d077"
DESI_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux"
SPECPROD = "iron"  # DR1 reduction
BANDS = ("B", "R", "Z")

# Redshift disagreement beyond this fraction means the DESI target is probably
# not the labelled object. Fails closed rather than silently pairing epochs.
REDSHIFT_TOLERANCE = 0.05


def coadd_url(survey: str, program: str, healpix: int) -> str:
    hp = int(healpix)
    return (
        f"{DESI_BASE}/{SPECPROD}/healpix/{survey}/{program}/{hp // 100}/{hp}/"
        f"coadd-{survey}-{program}-{hp}.fits"
    )


def open_remote(url: str, timeout: float = 600.0):
    """Open a remote FITS for range reads, so only the needed rows transfer."""
    import fsspec

    return fsspec.open(url, block_size=1 << 20, cache_type="readahead").open()


def extract_targets(url: str, targetids: list[int]) -> dict[int, dict[str, Any]]:
    """Drain the wanted targets from one coadd on a single open handle.

    A broken stream loses the rest of this file, not what was already read; the
    caller re-requests anything missing. Silently accepting a short mapping
    would trade speed for a hole in the sample.
    """
    out: dict[int, dict[str, Any]] = {}
    handle = open_remote(url)
    try:
        with fits.open(handle, memmap=False) as hdul:
            fibermap = hdul["FIBERMAP"].data
            ids = np.asarray(fibermap["TARGETID"])

            waves = {
                b.lower(): np.asarray(hdul[f"{b}_WAVELENGTH"].data, dtype=float) for b in BANDS
            }
            if "EXP_FIBERMAP" not in hdul:
                raise RuntimeError(f"source coadd lacks EXP_FIBERMAP: {url}")
            exp_fibermap = hdul["EXP_FIBERMAP"].data

            for tid in targetids:
                rows = np.where(ids == int(tid))[0]
                if rows.size == 0:
                    continue
                row = int(rows[0])

                selected_exp = np.asarray(exp_fibermap[exp_fibermap["TARGETID"] == int(tid)])
                source_fibermap = np.asarray(fibermap[row : row + 1])
                flux = {
                    b.lower(): np.asarray(hdul[f"{b}_FLUX"].section[row : row + 1], dtype=float)
                    for b in BANDS
                }
                ivar = {
                    b.lower(): np.asarray(hdul[f"{b}_IVAR"].section[row : row + 1], dtype=float)
                    for b in BANDS
                }
                mask = {
                    b.lower(): np.asarray(
                        hdul[f"{b}_MASK"].section[row : row + 1], dtype=np.uint32
                    )
                    for b in BANDS
                }
                resolution = {
                    b.lower(): np.asarray(
                        hdul[f"{b}_RESOLUTION"].section[row : row + 1], dtype=float
                    )
                    for b in BANDS
                }
                merged = merge_cameras_official(
                    wave=waves,
                    flux=flux,
                    ivar=ivar,
                    mask=mask,
                    resolution_data=resolution,
                    fibermap=source_fibermap,
                    exp_fibermap=selected_exp,
                )
                merged["iron_resolution_by_camera"] = {
                    band: values[0] for band, values in resolution.items()
                }
                out[int(tid)] = merged
    except Exception as error:  # noqa: BLE001 - partial reads are a normal outcome
        if not out:
            raise RuntimeError(f"no targets recovered from {url}: {error}") from error
    finally:
        handle.close()
    return out


def write_spectrum(destination: Path, payload: dict[str, Any], header: dict[str, object]) -> str:
    """Write the per-target extraction and return its sha256."""
    primary = fits.PrimaryHDU()
    for key, value in header.items():
        primary.header[key[:8].upper()] = value

    hdus = [primary]
    hdus.append(fits.BinTableHDU(data=payload["fibermap"], name="FIBERMAP"))
    hdus.append(fits.BinTableHDU(data=payload["exp_fibermap"], name="EXP_FIBERMAP"))
    for name in ("wavelength", "flux", "ivar", "mask", "resolution"):
        hdus.append(fits.ImageHDU(data=payload[name], name=name.upper()))
    hdus.append(fits.ImageHDU(data=payload["iron_resolution"], name="IRON_RESOLUTION"))
    for band, values in payload["iron_resolution_by_camera"].items():
        hdus.append(fits.ImageHDU(data=values, name=f"IRON_RES_{band.upper()}"))

    destination.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(destination, overwrite=True)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    parser.add_argument(
        "--resolution-overrides",
        type=Path,
        help="directory of validated zpix<id>.npz corrected-resolution fixtures",
    )
    args = parser.parse_args()

    root = project_root()
    suffix = "_slice" if args.slice else ""

    matches_path = root / "02_catalogs" / "crossmatches" / f"desi_dr1_matches{suffix}.csv"
    candidates_path = root / "04_reference_sample" / "candidates" / (
        "slice_candidates.csv" if args.slice else "reference_candidates.csv"
    )
    if not matches_path.exists():
        print(f"missing {matches_path}", file=sys.stderr)
        return 1

    matches = pd.read_csv(matches_path, dtype={"targetid": str, "zpix_id": str})
    candidates = pd.read_csv(candidates_path).set_index("object_id")
    raw_dir = root / "03_spectra" / "raw_desi"

    exclusions: list[tuple[str, str, str]] = []
    usable: list[pd.Series] = []

    # Identity check before any download: a large redshift disagreement means the
    # target inside the match radius is probably not the labelled object.
    for record in matches.itertuples():
        expected = float(candidates.loc[record.object_id, "redshift"])
        observed = float(record.z)
        if abs(observed - expected) > REDSHIFT_TOLERANCE * (1.0 + expected):
            exclusions.append(
                (
                    record.object_id,
                    "IDENTITY_AMBIGUOUS",
                    f"DESI z={observed:.4f} vs catalogue z={expected:.4f} "
                    f"(targetid {record.targetid})",
                )
            )
            print(
                f"{record.object_id:26s} REJECT  DESI z={observed:.4f} "
                f"vs catalogue z={expected:.4f}"
            )
            continue
        usable.append(record)

    grouped: dict[tuple[str, str, int], list] = defaultdict(list)
    for record in usable:
        grouped[(record.survey, record.program, int(record.healpix))].append(record)

    rows: list[dict[str, object]] = []
    for (survey, program, healpix), records in sorted(grouped.items()):
        url = coadd_url(survey, program, healpix)
        wanted = [int(r.targetid) for r in records]  # str -> exact int, no float step
        print(f"\ncoadd {survey}/{program}/{healpix}  ({len(wanted)} target(s))")
        try:
            payloads = extract_targets(url, wanted)
        except Exception as error:  # noqa: BLE001
            for record in records:
                exclusions.append(
                    (record.object_id, "CALIBRATION_FAILURE", f"coadd read failed: {error}")
                )
            print(f"  FAIL {type(error).__name__}: {str(error)[:90]}")
            continue

        for record in records:
            tid = int(record.targetid)
            if tid not in payloads:
                exclusions.append(
                    (record.object_id, "NO_SDSS_DESI_MATCH", f"targetid {tid} absent from coadd")
                )
                print(f"  {record.object_id:26s} targetid {tid} not in coadd")
                continue

            payload = payloads[tid]
            science_record = ScienceRecord(
                object_id=record.object_id,
                specprod=SPECPROD,
                survey=survey,
                program=program,
                healpix=healpix,
                targetid=tid,
                zpix_id=str(record.zpix_id) if pd.notna(record.zpix_id) else None,
            )
            summary = exposure_summary(payload["exp_fibermap"])
            payload["iron_resolution"] = np.asarray(payload["resolution"]).copy()
            if int(summary["n_exp"]) > 1:
                if args.resolution_overrides is None:
                    raise RuntimeError(
                        f"{science_record.spectrum_id}: multi-exposure record requires the "
                        "validated resolution-policy override"
                    )
                override_path = (
                    args.resolution_overrides / f"zpix{science_record.zpix_coordinate}.npz"
                )
                if not override_path.exists():
                    raise RuntimeError(f"missing corrected resolution fixture {override_path}")
                with np.load(override_path) as override:
                    if not np.array_equal(override["wavelength"], payload["wavelength"]):
                        raise RuntimeError(f"corrected resolution wavelength mismatch: {override_path}")
                    corrected_resolution = np.asarray(override["resolution"])
                if corrected_resolution.shape != payload["resolution"].shape:
                    raise RuntimeError(f"corrected resolution shape mismatch: {override_path}")
                payload["resolution"] = corrected_resolution
            if int(summary["n_exp"]) != int(record.coadd_numexp):
                raise RuntimeError(
                    f"{science_record.spectrum_id}: EXP_FIBERMAP count "
                    f"{summary['n_exp']} != catalog {record.coadd_numexp}"
                )
            if int(summary["n_night"]) != int(record.coadd_numnight):
                raise RuntimeError(
                    f"{science_record.spectrum_id}: night count "
                    f"{summary['n_night']} != catalog {record.coadd_numnight}"
                )
            filename = science_record.filename
            destination = raw_dir / filename
            header = {
                "OBJID": record.object_id[:60],
                "TARGETID": tid,
                "HEALPIX": healpix,
                "SURVEY": survey,
                "PROGRAM": program,
                "PRODTYPE": PRODUCT_TYPE,
                "SPECPROD": SPECPROD,
                "ZPIXID": science_record.zpix_coordinate,
                "SCIRECID": science_record.spectrum_id,
                "RELEASE": "DR1",
                "Z": float(record.z),
                "ZWARN": int(record.zwarn),
                "FIBSTAT": int(record.coadd_fiberstatus),
                "MJD_MIN": float(summary["mjd_min"]),
                "MJD_EFF": float(summary["mjd_effective"]),
                "MJD_MAX": float(summary["mjd_max"]),
                "NEXP": int(summary["n_exp"]),
                "NNIGHT": int(summary["n_night"]),
                "PROCVER": PROCESSING_VERSION,
            }
            digest = write_spectrum(destination, payload, header)

            good = payload["ivar"] > 0
            rows.append(
                {
                    "object_id": record.object_id,
                    "epoch_role": "desi",
                    "survey": "DESI",
                    "release": "DR1",
                    "specprod": SPECPROD,
                    "product_type": PRODUCT_TYPE,
                    "desi_survey": survey,
                    "program": program,
                    "spectrum_id": science_record.spectrum_id,
                    "targetid": tid,
                    "zpix_id": science_record.zpix_coordinate,
                    "healpix": healpix,
                    **summary,
                    "redshift": float(record.z),
                    "zerr": float(record.zerr),
                    "zwarn": int(record.zwarn),
                    "coadd_fiberstatus": int(record.coadd_fiberstatus),
                    "zcat_primary": parse_catalog_bool(record.zcat_primary),
                    "spectype": record.spectype,
                    "coadd_exptime": float(record.coadd_exptime),
                    "ra": float(record.mean_fiber_ra),
                    "dec": float(record.mean_fiber_dec),
                    "n_pixels": int(payload["wavelength"].size),
                    "n_good_pixels": int(good.sum()),
                    "n_masked": int((payload["mask"] != 0).sum()),
                    "url": url,
                    "path": relative_product_path(destination, root),
                    "n_bytes": destination.stat().st_size,
                    "sha256": digest,
                    "processing_version": PROCESSING_VERSION,
                }
            )
            print(
                f"  {record.object_id:26s} tid={tid}  {payload['wavelength'].size} pix  "
                f"masked {int((payload['mask'] != 0).sum())}  "
                f"{destination.stat().st_size / 1e6:.2f} MB"
            )

    manifest = pd.DataFrame(rows)
    if args.slice and len(usable) == 8:
        if len(manifest) != 8:
            raise RuntimeError(f"intended 8 DESI source records, extracted {len(manifest)}")
        if manifest.spectrum_id.nunique() != 8 or manifest.path.nunique() != 8:
            raise RuntimeError("eight intended DESI records did not produce eight unique products")
    manifest_path = raw_dir / f"manifest{suffix}.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    n_excluded = record_exclusions(exclusions, script=SCRIPT)

    print()
    print(f"DESI spectra extracted {len(manifest):>4}")
    print(f"objects                {manifest.object_id.nunique() if not manifest.empty else 0:>4}")
    print(f"exclusions recorded    {n_excluded:>4}")
    print(f"manifest               {manifest_path.relative_to(root)}")
    if not manifest.empty:
        print(f"total size             {manifest.n_bytes.sum() / 1e6:>6.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
