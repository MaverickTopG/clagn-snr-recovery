#!/usr/bin/env python
"""Targeted Iron multi-exposure resolution experiment required by D-077.

Only the three Paper-3 science records whose EXP_FIBERMAP has more than one
exposure are in scope.  The exposure-preserving ``spectra`` product is coadded
with modern desispec, compared to the published Iron healpix coadd, and a small
validated corrected-resolution fixture is emitted for production extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.ingest.desi import ScienceRecord  # noqa: E402

BANDS = ("b", "r", "z")
DESI_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"


def _open_remote(url: str):
    import fsspec

    return fsspec.open(url, block_size=1 << 20, cache_type="readahead").open()


def _spectra_from_hdul(hdul: fits.HDUList, rows: np.ndarray, exp_fibermap: Any = None):
    from desispec.spectra import Spectra

    fibermap = np.asarray(hdul["FIBERMAP"].data[rows])
    wave = {band: np.asarray(hdul[f"{band.upper()}_WAVELENGTH"].data) for band in BANDS}
    flux = {
        band: np.asarray(hdul[f"{band.upper()}_FLUX"].data[rows], dtype=float) for band in BANDS
    }
    ivar = {
        band: np.asarray(hdul[f"{band.upper()}_IVAR"].data[rows], dtype=float) for band in BANDS
    }
    mask = {
        band: np.asarray(hdul[f"{band.upper()}_MASK"].data[rows], dtype=np.uint32)
        for band in BANDS
    }
    resolution = {
        band: np.asarray(hdul[f"{band.upper()}_RESOLUTION"].data[rows], dtype=float)
        for band in BANDS
    }
    return Spectra(
        bands=list(BANDS),
        wave=wave,
        flux=flux,
        ivar=ivar,
        mask=mask,
        resolution_data=resolution,
        fibermap=fibermap,
        exp_fibermap=exp_fibermap,
    )


def _read_exposures(path: Path, targetid: int):
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        ids = np.asarray(hdul["FIBERMAP"].data["TARGETID"])
        rows = np.flatnonzero(ids == targetid)
        if rows.size < 2:
            raise RuntimeError(f"expected multiple exposure rows for TARGETID {targetid}: {path}")
        return _spectra_from_hdul(hdul, rows)


def _read_published(url: str, targetid: int):
    handle = _open_remote(url)
    try:
        with fits.open(handle, memmap=False, lazy_load_hdus=True) as hdul:
            ids = np.asarray(hdul["FIBERMAP"].data["TARGETID"])
            rows = np.flatnonzero(ids == targetid)
            if rows.size != 1:
                raise RuntimeError(f"expected one published coadd row for TARGETID {targetid}")
            exp = np.asarray(
                hdul["EXP_FIBERMAP"].data[hdul["EXP_FIBERMAP"].data["TARGETID"] == targetid]
            )
            return _spectra_from_hdul(hdul, rows, exp_fibermap=exp)
    finally:
        handle.close()


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def _allclose(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=1e-6, atol=1e-6, equal_nan=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-dir", type=Path, required=True)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    from desispec.coaddition import coadd, coadd_cameras

    root = project_root()
    suffix = "_slice" if args.slice else ""
    matches = pd.read_csv(
        root / "02_catalogs" / "crossmatches" / f"desi_dr1_matches{suffix}.csv",
        dtype={"targetid": str, "zpix_id": str},
    )
    multi = matches[(matches.coadd_numexp > 1) & (matches.object_id != "SDSSJ210200.42+000501.8")]
    observed = {(row.object_id, row.program) for row in multi.itertuples()}
    expected = {
        ("SDSSJ141213.61+021202.1", "bright"),
        ("SDSSJ161711.42+063833.4", "bright"),
        ("SDSSJ082942.66+415436.8", "dark"),
    }
    if observed != expected:
        raise RuntimeError(f"multi-exposure record set changed: expected {expected}, found {observed}")

    output_dir = root / "05_analysis" / "desi_ingest"
    override_dir = output_dir / "corrected_resolution"
    override_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for row in multi.sort_values(["object_id", "program"]).itertuples():
        targetid = int(row.targetid)
        science_record = ScienceRecord(
            object_id=row.object_id,
            specprod="iron",
            survey=row.survey,
            program=row.program,
            healpix=int(row.healpix),
            targetid=targetid,
            zpix_id=str(row.zpix_id),
        )
        basename = f"spectra-{row.survey}-{row.program}-{int(row.healpix)}.fits"
        source_path = args.spectra_dir / basename
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        url = (
            f"{DESI_BASE}/{row.survey}/{row.program}/{int(row.healpix) // 100}/"
            f"{int(row.healpix)}/coadd-{row.survey}-{row.program}-{int(row.healpix)}.fits"
        )

        regenerated = _read_exposures(source_path, targetid)
        input_rows = regenerated.num_spectra()
        coadd(regenerated)
        if regenerated.num_spectra() != 1:
            raise RuntimeError(f"exposure coadd changed {input_rows} rows to {regenerated.num_spectra()}")
        published = _read_published(url, targetid)

        science_material = False
        metrics: dict[str, object] = {}
        for band in BANDS:
            wave_equal = np.array_equal(regenerated.wave[band], published.wave[band])
            flux_close = _allclose(regenerated.flux[band], published.flux[band])
            ivar_close = _allclose(regenerated.ivar[band], published.ivar[band])
            mask_equal = np.array_equal(regenerated.mask[band], published.mask[band])
            science_material |= not (wave_equal and flux_close and ivar_close and mask_equal)
            metrics.update(
                {
                    f"{band}_wave_bitwise": wave_equal,
                    f"{band}_flux_bitwise": np.array_equal(
                        regenerated.flux[band], published.flux[band]
                    ),
                    f"{band}_flux_bitwise_after_f32": np.array_equal(
                        regenerated.flux[band].astype(np.float32),
                        published.flux[band].astype(np.float32),
                    ),
                    f"{band}_flux_max_abs": _max_abs(
                        regenerated.flux[band], published.flux[band]
                    ),
                    f"{band}_ivar_bitwise": np.array_equal(
                        regenerated.ivar[band], published.ivar[band]
                    ),
                    f"{band}_ivar_bitwise_after_f32": np.array_equal(
                        regenerated.ivar[band].astype(np.float32),
                        published.ivar[band].astype(np.float32),
                    ),
                    f"{band}_ivar_max_abs": _max_abs(
                        regenerated.ivar[band], published.ivar[band]
                    ),
                    f"{band}_mask_bitwise": mask_equal,
                    f"{band}_resolution_max_abs": _max_abs(
                        regenerated.resolution_data[band], published.resolution_data[band]
                    ),
                }
            )

        regenerated_merged = coadd_cameras(regenerated)
        published_merged = coadd_cameras(published)
        regenerated_band = regenerated_merged.bands[0]
        published_band = published_merged.bands[0]
        if not np.array_equal(
            regenerated_merged.wave[regenerated_band], published_merged.wave[published_band]
        ):
            science_material = True
        if science_material:
            raise RuntimeError(
                f"STOP: wavelength/FLUX/IVAR/MASK materially changed for "
                f"{science_record.spectrum_id}"
            )

        override_path = override_dir / f"zpix{science_record.zpix_coordinate}.npz"
        np.savez_compressed(
            override_path,
            wavelength=regenerated_merged.wave[regenerated_band],
            resolution=regenerated_merged.resolution_data[regenerated_band][0],
        )
        results.append(
            {
                "object_id": row.object_id,
                "program": row.program,
                "spectrum_id": science_record.spectrum_id,
                "targetid": targetid,
                "zpix_id": science_record.zpix_coordinate,
                "n_input_exposure_rows": input_rows,
                "n_published_exp_fibermap": len(published.exp_fibermap),
                "science_arrays_materially_changed": science_material,
                "merged_resolution_max_abs": _max_abs(
                    regenerated_merged.resolution_data[regenerated_band],
                    published_merged.resolution_data[published_band],
                ),
                "override_path": str(override_path.relative_to(root)),
                **metrics,
            }
        )
        print(f"PASS {science_record.spectrum_id}: {input_rows} exposure rows -> 1 coadd row")

    result = pd.DataFrame(results)
    result.to_csv(output_dir / f"resolution_policy{suffix}.csv", index=False)
    print(f"wrote {len(result)} records to {output_dir.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
