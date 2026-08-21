#!/usr/bin/env python
"""Build the bounded OLD-vs-CORRECTED DESI ingest impact audit (D-077)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.ingest.desi import count_tolerance_duplicates  # noqa: E402

TILES = [(5200.0 + 200 * index, 5400.0 + 200 * index) for index in range(9)]
TILE_NAMES = [f"T{index + 1}" for index in range(9)]
HBETA_REGION = (4700.0, 5100.0)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        return {
            "wavelength": np.asarray(hdul["WAVELENGTH"].data, dtype=float),
            "flux": np.asarray(hdul["FLUX"].data, dtype=float),
            "ivar": np.asarray(hdul["IVAR"].data, dtype=float),
            "mask": np.asarray(hdul["MASK"].data),
            "exp_fibermap": np.asarray(hdul["EXP_FIBERMAP"].data)
            if "EXP_FIBERMAP" in hdul
            else None,
        }


def _raw_hbeta_inputs(arrays: dict[str, Any], redshift: float) -> dict[str, float | int]:
    rest = arrays["wavelength"] / (1.0 + redshift)
    good = (arrays["ivar"] > 0) & (arrays["mask"] == 0) & np.isfinite(arrays["flux"])
    inside = good & (rest >= HBETA_REGION[0]) & (rest <= HBETA_REGION[1])
    if inside.sum() < 2:
        return {"hbeta_n_good": int(inside.sum()), "hbeta_median_flux": np.nan, "hbeta_raw_integral": np.nan}
    return {
        "hbeta_n_good": int(inside.sum()),
        "hbeta_median_flux": float(np.median(arrays["flux"][inside])),
        "hbeta_raw_integral": float(
            np.trapezoid(arrays["flux"][inside], arrays["wavelength"][inside])
        ),
    }


def _local_tile_snr(arrays: dict[str, Any], tile_names: set[str]) -> str:
    good = (
        (arrays["ivar"] > 0)
        & (arrays["mask"] == 0)
        & np.isfinite(arrays["flux"])
        & np.isfinite(arrays["ivar"])
    )
    values = []
    for name, (lower, upper) in zip(TILE_NAMES, TILES, strict=True):
        if name not in tile_names:
            continue
        inside = good & (arrays["wavelength"] >= lower) & (arrays["wavelength"] < upper)
        if inside.sum() < 30:
            continue
        snr = float(
            np.median(np.abs(arrays["flux"][inside]) * np.sqrt(arrays["ivar"][inside]))
        )
        values.append(f"{name}:{snr:.12g}")
    return "|".join(values)


def _provenance_string(exp_fibermap: np.ndarray) -> str:
    required = ("NIGHT", "EXPID", "TILEID", "MJD")
    return "|".join(
        "/".join(str(row[name]) for name in required) for row in exp_fibermap
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    root = project_root()
    suffix = "_slice" if args.slice else ""
    audit_dir = root / "05_analysis" / "desi_ingest"
    old = pd.read_csv(
        audit_dir / f"old_snapshot{suffix}.csv", dtype={"targetid": str, "zpix_id": str}
    )
    new = pd.read_csv(
        root / "03_spectra" / "raw_desi" / f"manifest{suffix}.csv",
        dtype={"targetid": str, "zpix_id": str, "spectrum_id": str},
    )
    qc = pd.read_csv(
        root / "03_spectra" / "qc" / f"qc_verdicts{suffix}.csv", dtype={"spectrum_id": str}
    )
    qc = qc[qc.survey == "DESI"]
    combined = old.merge(new, on=["object_id", "program", "targetid", "zpix_id"], validate="one_to_one")
    combined = combined.merge(qc, on=["object_id", "spectrum_id"], validate="one_to_one", suffixes=("", "_qc"))

    rows: list[dict[str, object]] = []
    for row in combined.itertuples():
        old_path = root / row.old_path
        new_path = root / row.path
        old_arrays = _arrays(old_path)
        new_arrays = _arrays(new_path)
        exp = new_arrays["exp_fibermap"]
        if exp is None:
            raise RuntimeError(f"missing EXP_FIBERMAP: {new_path}")
        if len(np.unique(exp["EXPID"])) != row.n_exp:
            raise RuntimeError(f"exposure provenance count mismatch: {new_path}")
        if len(np.unique(exp["NIGHT"])) != row.n_night:
            raise RuntimeError(f"night provenance count mismatch: {new_path}")
        old_hbeta = _raw_hbeta_inputs(old_arrays, row.redshift)
        new_hbeta = _raw_hbeta_inputs(new_arrays, row.redshift)
        rows.append(
            {
                "object_id": row.object_id,
                "program": row.program,
                "spectrum_id": row.spectrum_id,
                "targetid": row.targetid,
                "zpix_id": row.zpix_id,
                "old_filename": row.old_filename,
                "corrected_filename": Path(row.path).name,
                "old_manifest_sha256": row.old_manifest_sha256,
                "old_physical_sha256": row.old_physical_sha256,
                "old_file_matches_manifest": row.old_file_matches_manifest,
                "old_measurement_valid": bool(row.old_file_matches_manifest),
                "corrected_sha256": row.sha256,
                "corrected_hash_verified": _digest(new_path) == row.sha256,
                "old_pixel_count": row.old_pixel_count,
                "corrected_pixel_count": len(new_arrays["wavelength"]),
                "old_tolerance_duplicate_count": row.old_tolerance_duplicate_count,
                "corrected_tolerance_duplicate_count": count_tolerance_duplicates(
                    new_arrays["wavelength"]
                ),
                "n_exp": row.n_exp,
                "n_night": row.n_night,
                "mjd_min": row.mjd_min,
                "mjd_effective": row.mjd_effective,
                "mjd_max": row.mjd_max,
                "exposure_provenance": _provenance_string(exp),
                "old_local_tile_snr": row.old_local_tile_snr,
                "corrected_local_tile_snr": _local_tile_snr(
                    new_arrays,
                    {value.split(":", 1)[0] for value in row.old_local_tile_snr.split("|")},
                ),
                "old_snr_continuum_5100": row.old_snr_continuum_5100,
                "corrected_snr_continuum_5100": row.snr_continuum_5100,
                "old_snr_hbeta_window": row.old_snr_hbeta_window,
                "corrected_snr_hbeta_window": row.snr_hbeta_window,
                "old_continuum_level_5100": row.old_continuum_level_5100,
                "corrected_continuum_level_5100": row.continuum_level_5100,
                "old_masked_fraction_hbeta": row.old_masked_fraction_hbeta,
                "corrected_masked_fraction_hbeta": row.masked_fraction_hbeta,
                "old_qc_pass": row.old_qc_pass,
                "corrected_qc_pass": row.qc_pass,
                "old_qc_failures": row.old_qc_failures,
                "corrected_qc_failures": row.qc_failures,
                "old_hbeta_n_good": old_hbeta["hbeta_n_good"],
                "corrected_hbeta_n_good": new_hbeta["hbeta_n_good"],
                "old_hbeta_median_flux": old_hbeta["hbeta_median_flux"],
                "corrected_hbeta_median_flux": new_hbeta["hbeta_median_flux"],
                "old_hbeta_raw_integral": old_hbeta["hbeta_raw_integral"],
                "corrected_hbeta_raw_integral": new_hbeta["hbeta_raw_integral"],
            }
        )

    audit = pd.DataFrame(rows)
    if len(audit) != 8:
        raise RuntimeError(f"intended eight-record audit, found {len(audit)}")
    invariants = {
        "unique_science_record_ids": audit.spectrum_id.nunique() == 8,
        "unique_corrected_filenames": audit.corrected_filename.nunique() == 8,
        "unique_corrected_hashes": audit.corrected_sha256.nunique() == 8,
        "all_hashes_verified": bool(audit.corrected_hash_verified.all()),
        "all_7781_fixture_pixels": bool((audit.corrected_pixel_count == 7781).all()),
        "strictly_zero_tolerance_duplicates": bool(
            (audit.corrected_tolerance_duplicate_count == 0).all()
        ),
        "all_qc_pass": bool(audit.corrected_qc_pass.all()),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"corrected eight-record audit failed: {invariants}")
    audit.to_csv(audit_dir / f"old_vs_corrected{suffix}.csv", index=False)
    pd.DataFrame([invariants]).to_csv(audit_dir / f"rebuild_invariants{suffix}.csv", index=False)

    impacts = [
        ("corrected DESI FITS + manifest", True, True, False, "D-077 ingest output"),
        ("DESI QC + S/N census", True, True, False, "recomputed in bounded impact audit"),
        ("local science-tile S/N map", True, True, False, "recomputed values stored in audit"),
        ("PyQSOFit measurements", True, True, True, "corrected science arrays and J161 record identity"),
        ("pPXF decomposition", True, True, True, "corrected spectra and unique epoch identity"),
        ("host stability/leakage", True, True, True, "depends on PyQSOFit/pPXF products"),
        ("per-tile Gate-C cross-bin validation", False, False, True, "pre-existing pending validation; do not run yet"),
        ("per-tile LOO", False, False, True, "pre-existing pending validation; do not run yet"),
        ("k_adopted_primary", True, True, True, "depends on corrected science-domain mapping and Gate C"),
        ("17H", True, True, True, "depends on corrected fits and adopted uncertainty model"),
        ("host injection", True, True, True, "depends on corrected host decomposition"),
        ("all-17", True, True, True, "depends on corrected measurements and 17H"),
        ("Q2", True, True, True, "depends on corrected host products and full cohort"),
    ]
    pd.DataFrame(
        impacts,
        columns=["artifact", "affected_by_overlap", "affected_by_overwrite", "needs_recompute", "dependency"],
    ).to_csv(audit_dir / f"recomputation_manifest{suffix}.csv", index=False)

    j161 = audit[audit.object_id == "SDSSJ161711.42+063833.4"].set_index("program")
    ratio = float(
        j161.loc["bright", "corrected_continuum_level_5100"]
        / j161.loc["dark", "corrected_continuum_level_5100"]
    )
    physical_bright = "bright" if ratio >= 1 else "dark"
    print(f"audit invariants: {invariants}")
    print(
        "J161 continuum: "
        f"PROGRAM bright={j161.loc['bright', 'corrected_continuum_level_5100']:.9g}, "
        f"PROGRAM dark={j161.loc['dark', 'corrected_continuum_level_5100']:.9g}, "
        f"ratio={max(ratio, 1 / ratio):.9g}, physical bright={physical_bright}"
    )
    j082 = audit[audit.object_id == "SDSSJ082942.66+415436.8"].iloc[0]
    print(f"J082 corrected local tile S/N: {j082.corrected_local_tile_snr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
