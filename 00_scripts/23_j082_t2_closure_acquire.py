#!/usr/bin/env python
"""Prospective staged acquisition and pre-calibration census for J082 T2-B.

This script cannot estimate or inspect k.  It freezes the five-product order,
downloads only the requested stage, preserves the lifecycle, and freezes a
T2-B/50-55 census.  Calibration is deliberately a separate later executable.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.calibration.t2_closure import (  # noqa: E402
    T2_INTERVAL,
    closure_sn_bin,
    local_snr,
)
from p3sf.config import project_root  # noqa: E402

BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/tiles/cumulative"
KNOWN_INVALID = {(20211212, 9)}
MIN_VALID_PIXELS = 30
MIN_COVERAGE_FRACTION = 0.5

# Orders 1-3 are exactly the products named in the accepted D-079 plan.
# Orders 4-5 were prospectively frozen before acquisition from metadata only.
CANDIDATES = [
    (1, 1, 560, 20210509, 8, "D079_NAMED_INITIAL"),
    (2, 1, 21347, 20220109, 2, "D079_NAMED_INITIAL"),
    (3, 1, 23981, 20211117, 3, "D079_NAMED_INITIAL"),
    (4, 2, 26147, 20211219, 4, "PROSPECTIVE_METADATA_ONLY_CONTINGENCY"),
    (5, 2, 20389, 20210619, 8, "PROSPECTIVE_METADATA_ONLY_CONTINGENCY"),
]


def url_for(tileid: int, night: int, petal: int) -> str:
    return f"{BASE}/{tileid}/{night}/spectra-{petal}-{tileid}-thru{night}.fits.gz"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def append_lifecycle(row: pd.Series, state: str) -> None:
    history = str(row.get("lifecycle_history", ""))
    if history in {"", "nan"}:
        history = state
    elif history.split(">")[-1] != state:
        history = f"{history}>{state}"
    row["lifecycle"] = state
    row["lifecycle_history"] = history


def freeze_manifest(root: Path) -> pd.DataFrame:
    final = root / "02_catalogs" / "final"
    path = final / "j082_t2_closure_stage_manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    metadata = pd.read_csv(final / "c0_cluster_frame_desi.csv")
    records: list[dict[str, object]] = []
    for order, stage, tileid, night, petal, basis in CANDIDATES:
        if (night, petal) in KNOWN_INVALID:
            raise RuntimeError(f"prospective candidate intersects known invalid {(night, petal)}")
        match = metadata[
            (metadata.tileid == tileid)
            & (metadata.night == night)
            & (metadata.petal == petal)
        ]
        if len(match) != 1:
            raise RuntimeError(f"candidate not unique in frozen metadata: {tileid}-{night}-{petal}")
        source = match.iloc[0]
        cluster_id = f"{tileid}-{night}-{petal}"
        local_path = (
            Path("03_spectra/raw_desi/j082_t2_closure")
            / f"spectra-{petal}-{tileid}-{night}.fits.gz"
        )
        records.append(
            {
                "candidate_order": order,
                "planned_stage": stage,
                "cluster_id": cluster_id,
                "tileid": tileid,
                "night": night,
                "petal": petal,
                "url": url_for(tileid, night, petal),
                "local_path": str(local_path),
                "n_targets_expected": int(source.n_targets),
                "n_low_proxy": int(source.n_low),
                "n_mid_proxy": int(source.n_mid),
                "n_high_proxy": int(source.n_high),
                "selection_basis": basis,
                "lifecycle": "DISCOVERED",
                "lifecycle_history": "DISCOVERED",
                "failure_reason": "",
                "n_bytes": np.nan,
                "sha256": "",
                "n_file_rows": np.nan,
                "n_unique_file_targets": np.nan,
                "n_eligible_targets": np.nan,
                "n_exposure_rows": np.nan,
                "t2_coverage_failures": np.nan,
                "qc_failures": np.nan,
                "n_closure_eligible_pairs": np.nan,
            }
        )
    frame = pd.DataFrame(records)
    atomic_csv(frame, path)
    return frame


def download(url: str, destination: Path) -> tuple[int, str]:
    import requests

    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination.stat().st_size, ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with requests.get(url, stream=True, timeout=3600) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1 << 22):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        temporary.replace(destination)
        return size, digest.hexdigest()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_population(path: Path, candidate: pd.Series) -> pd.DataFrame:
    """Resolve the frozen DR1 BRIGHT/STAR/repeat/zcat-primary population."""
    import io

    from astropy.io import fits
    from dl import queryClient as query_client

    with fits.open(path, memmap=False) as hdul:
        targetids = np.unique(
            np.asarray(hdul["FIBERMAP"].data["TARGETID"]).astype(str)
        )
    frames: list[pd.DataFrame] = []
    for chunk in np.array_split(targetids, max(1, (len(targetids) + 199) // 200)):
        identifiers = ",".join(chunk)
        sql = f"""
        SELECT CAST(targetid AS VARCHAR) AS targetid,
               survey, program, spectype, tsnr2_bgs_z,
               zwarn, coadd_fiberstatus, coadd_numexp, coadd_numnight,
               zcat_primary
        FROM desi_dr1.zpix
        WHERE targetid IN ({identifiers})
          AND program = 'bright'
          AND spectype = 'STAR'
          AND zwarn = 0
          AND coadd_fiberstatus = 0
          AND coadd_numexp > 1
          AND coadd_numnight = 1
          AND zcat_primary = 't'
        """
        text = query_client.query(sql=sql, fmt="csv", timeout=180)
        frames.append(pd.read_csv(io.StringIO(text), dtype={"targetid": str}))
    population = pd.concat(frames, ignore_index=True).drop_duplicates("targetid")
    population.insert(0, "cluster_id", candidate.cluster_id)
    population.insert(1, "tileid", int(candidate.tileid))
    population.insert(2, "night", int(candidate.night))
    population.insert(3, "petal", int(candidate.petal))
    population["tsnr2_discovery_field"] = "TSNR2_BGS_Z"
    population["population_status"] = "FROZEN_BRIGHT_STAR_REPEAT_ZCAT_PRIMARY"
    if len(population) != int(candidate.n_targets_expected):
        raise RuntimeError(
            f"population count {len(population)} != frozen metadata "
            f"{int(candidate.n_targets_expected)}"
        )
    return population


def inspect_product(
    path: Path, candidate: pd.Series, eligible: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        required = ["FIBERMAP", "B_WAVELENGTH", "B_FLUX", "B_IVAR", "B_MASK"]
        missing = [name for name in required if name not in hdul]
        if missing:
            raise RuntimeError(f"missing required HDUs: {missing}")
        fibermap = hdul["FIBERMAP"].data
        for column in ("TARGETID", "NIGHT", "EXPID"):
            if column not in fibermap.columns.names:
                raise RuntimeError(f"missing FIBERMAP column {column}")
        wave = np.asarray(hdul["B_WAVELENGTH"].data, dtype=float)
        flux = np.asarray(hdul["B_FLUX"].data, dtype=float)
        ivar = np.asarray(hdul["B_IVAR"].data, dtype=float)
        mask = np.asarray(hdul["B_MASK"].data)
        if not (len(fibermap) == flux.shape[0] == ivar.shape[0] == mask.shape[0]):
            raise RuntimeError("FIBERMAP/spectral row mismatch")
        targetids = np.asarray(fibermap["TARGETID"]).astype(str)
        nights = np.asarray(fibermap["NIGHT"]).astype(int)
        expids = np.asarray(fibermap["EXPID"]).astype(int)
        mjd = (
            np.asarray(fibermap["MJD"], dtype=float)
            if "MJD" in fibermap.columns.names
            else np.full(len(fibermap), np.nan)
        )
        in_t2 = (wave >= T2_INTERVAL[0]) & (wave < T2_INTERVAL[1])
        if not in_t2.any():
            raise RuntimeError("no B-arm T2 pixels")

        exposure_rows: list[dict[str, object]] = []
        pair_rows: list[dict[str, object]] = []
        order = np.argsort(targetids, kind="stable")
        for targetid in np.unique(targetids):
            if targetid not in eligible:
                continue
            lo = np.searchsorted(targetids[order], targetid, side="left")
            hi = np.searchsorted(targetids[order], targetid, side="right")
            indices = order[lo:hi]
            indices = indices[nights[indices] == int(candidate.night)]
            if indices.size < 2:
                continue
            target_exposures: list[dict[str, object]] = []
            for index in indices:
                good = (
                    in_t2
                    & (ivar[index] > 0)
                    & (mask[index] == 0)
                    & np.isfinite(flux[index])
                )
                n_valid = int(good.sum())
                span = float(wave[good].max() - wave[good].min()) if n_valid else np.nan
                coverage = span / (T2_INTERVAL[1] - T2_INTERVAL[0]) if n_valid else 0.0
                coverage_pass = n_valid >= MIN_VALID_PIXELS and coverage >= MIN_COVERAGE_FRACTION
                snr = local_snr(flux[index][good], ivar[index][good]) if coverage_pass else np.nan
                record = {
                    "cluster_id": candidate.cluster_id,
                    "tileid": int(candidate.tileid),
                    "night": int(candidate.night),
                    "petal": int(candidate.petal),
                    "targetid": targetid,
                    "expid": int(expids[index]),
                    "mjd": float(mjd[index]),
                    "n_t2_pixels": int(in_t2.sum()),
                    "n_valid_t2_pixels": n_valid,
                    "t2_span": span,
                    "coverage_status": "T2_COVERAGE_PASS" if coverage_pass else "T2_COVERAGE_FAILED",
                    "spectral_qc_status": "SPECTRAL_QC_PASS" if coverage_pass else "NOT_APPLICABLE",
                    "snr_pix_t2": snr,
                    "sn_bin_t2": closure_sn_bin(snr),
                }
                exposure_rows.append(record)
                target_exposures.append(record)
            for first, second in combinations(target_exposures, 2):
                pair_eligible = (
                    first["spectral_qc_status"] == "SPECTRAL_QC_PASS"
                    and second["spectral_qc_status"] == "SPECTRAL_QC_PASS"
                )
                pair_rows.append(
                    {
                        "cluster_id": candidate.cluster_id,
                        "tileid": int(candidate.tileid),
                        "night": int(candidate.night),
                        "petal": int(candidate.petal),
                        "targetid": targetid,
                        "expid_1": first["expid"],
                        "expid_2": second["expid"],
                        "snr_1": first["snr_pix_t2"],
                        "snr_2": second["snr_pix_t2"],
                        "bin_1": first["sn_bin_t2"],
                        "bin_2": second["sn_bin_t2"],
                        "same_bin_50_55": (
                            first["sn_bin_t2"] == second["sn_bin_t2"] == "50-55"
                        ),
                        "cross_30_50_to_50_55": {
                            first["sn_bin_t2"],
                            second["sn_bin_t2"],
                        }
                        == {"30-50", "50-55"},
                        "pair_status": (
                            "CLOSURE_ELIGIBLE"
                            if pair_eligible
                            else "SPECTRAL_QC_FAILED"
                        ),
                    }
                )
    exposure_frame = pd.DataFrame(exposure_rows)
    pair_frame = pd.DataFrame(pair_rows)
    stats = {
        "n_file_rows": len(fibermap),
        "n_unique_file_targets": len(np.unique(targetids)),
        "n_eligible_targets": pair_frame.targetid.nunique() if not pair_frame.empty else 0,
        "n_exposure_rows": len(exposure_frame),
        "t2_coverage_failures": int((exposure_frame.coverage_status != "T2_COVERAGE_PASS").sum()),
        "qc_failures": int((pair_frame.pair_status != "CLOSURE_ELIGIBLE").sum()),
        "n_closure_eligible_pairs": int((pair_frame.pair_status == "CLOSURE_ELIGIBLE").sum()),
    }
    return exposure_frame, pair_frame, stats


def acquire_stage(root: Path, stage: int) -> None:
    final = root / "02_catalogs" / "final"
    manifest_path = final / "j082_t2_closure_stage_manifest.csv"
    exposure_path = final / "j082_t2_closure_exposures.csv"
    pair_path = final / "j082_t2_closure_pairs.csv"
    population_path = final / "j082_t2_closure_population.csv"
    manifest = freeze_manifest(root)
    for column in ("sha256", "failure_reason", "lifecycle", "lifecycle_history"):
        manifest[column] = manifest[column].fillna("").astype(object)
    exposures = pd.read_csv(exposure_path, dtype={"targetid": str}) if exposure_path.exists() else pd.DataFrame()
    pairs = pd.read_csv(pair_path, dtype={"targetid": str}) if pair_path.exists() else pd.DataFrame()
    populations = (
        pd.read_csv(population_path, dtype={"targetid": str})
        if population_path.exists()
        else pd.DataFrame()
    )

    for index in manifest.index[manifest.planned_stage == stage]:
        row = manifest.loc[index].copy()
        if row.lifecycle == "CLOSURE_ELIGIBLE":
            if str(row.get("failure_reason", "")) not in {"", "nan"}:
                row["failure_reason"] = ""
                manifest.loc[index] = row
                atomic_csv(manifest, manifest_path)
            print(f"{row.cluster_id}: already CLOSURE_ELIGIBLE", flush=True)
            continue
        append_lifecycle(row, "EXPOSURES_RESOLVED")
        manifest.loc[index] = row
        atomic_csv(manifest, manifest_path)
        destination = root / str(row.local_path)
        try:
            row["failure_reason"] = ""
            size, digest = download(str(row.url), destination)
            row["n_bytes"] = size
            row["sha256"] = digest or file_sha256(destination)
            append_lifecycle(row, "DOWNLOAD_COMPLETE")
            manifest.loc[index] = row
            atomic_csv(manifest, manifest_path)
            existing_population = (
                populations[populations.cluster_id == row.cluster_id]
                if not populations.empty
                else pd.DataFrame()
            )
            if existing_population.empty:
                existing_population = resolve_population(destination, row)
                populations = pd.concat(
                    [populations, existing_population], ignore_index=True
                )
                atomic_csv(populations, population_path)
            eligible = set(existing_population.targetid)
            exposure_frame, pair_frame, stats = inspect_product(
                destination, row, eligible
            )
            append_lifecycle(row, "INTEGRITY_PASS")
            if stats["t2_coverage_failures"] < stats["n_exposure_rows"]:
                append_lifecycle(row, "T2_COVERAGE_PASS")
            if stats["n_closure_eligible_pairs"]:
                append_lifecycle(row, "SPECTRAL_QC_PASS")
                append_lifecycle(row, "CLOSURE_ELIGIBLE")
            else:
                row["failure_reason"] = "no closure-eligible pairs"
            for key, value in stats.items():
                row[key] = value
            exposures = exposures[exposures.cluster_id != row.cluster_id] if not exposures.empty else exposures
            pairs = pairs[pairs.cluster_id != row.cluster_id] if not pairs.empty else pairs
            exposures = pd.concat([exposures, exposure_frame], ignore_index=True)
            pairs = pd.concat([pairs, pair_frame], ignore_index=True)
            atomic_csv(exposures, exposure_path)
            atomic_csv(pairs, pair_path)
        except Exception as error:  # noqa: BLE001
            row["failure_reason"] = f"{type(error).__name__}: {error}"
        manifest.loc[index] = row
        atomic_csv(manifest, manifest_path)
        print(
            f"{row.cluster_id}: {row.lifecycle}; {float(row.get('n_bytes', 0) or 0)/1e9:.3f} GB; "
            f"pairs={row.get('n_closure_eligible_pairs', 'NA')}",
            flush=True,
        )


def freeze_bounded_unsupported_cell(final: Path) -> None:
    """Replace only the old T2-B open-ended unsupported placeholder."""
    applicability_path = final / "c6c_science_applicability.csv"
    applicability = pd.read_csv(applicability_path, dtype={"science_record_id": str})
    science_row = (
        applicability.object_id.eq("SDSSJ082942.66+415436.8")
        & applicability.measurement_region.eq("hbeta_primary")
        & applicability.tile.eq("T2")
        & applicability.arm.eq("B")
    )
    if int(science_row.sum()) != 1 or applicability.loc[science_row, "k_empirical"].notna().any():
        raise RuntimeError("unexpected J082 T2 applicability state")
    applicability.loc[science_row, "sn_bin"] = "50-55"
    applicability.loc[science_row, "calibration_cell"] = "T2-B/50-55"
    applicability.loc[science_row, "empirical_support_status"] = (
        "T2-B_50-55_EMPIRICAL_SUPPORT_INSUFFICIENT"
    )
    atomic_csv(applicability, applicability_path)

    for filename in (
        "c6c_k_empirical_validated.csv",
        "c6c_k_adopted_primary.csv",
        "c6c_k_adopted_faithful.csv",
    ):
        path = final / filename
        frame = pd.read_csv(path)
        placeholder = (
            frame.tile.eq("T2")
            & frame.arm.eq("B")
            & frame.sn_bin.isin(["50+", "50-55"])
        )
        if int(placeholder.sum()) != 1 or frame.loc[placeholder, "k_empirical"].notna().any():
            raise RuntimeError(f"unexpected T2-B placeholder in {filename}")
        frame.loc[placeholder, "sn_bin"] = "50-55"
        frame.loc[placeholder, "empirical_support_status"] = (
            "T2-B_50-55_EMPIRICAL_SUPPORT_INSUFFICIENT"
        )
        frame.loc[placeholder, "primary_science_relevant"] = True
        frame.loc[placeholder, "cross_bin_validation_status"] = (
            "NOT_RUN_SUPPORT_GATE_FAILED"
        )
        frame.loc[placeholder, "loo_validation_status"] = (
            "NOT_RUN_SUPPORT_GATE_FAILED"
        )
        if "adoption_status" in frame:
            frame.loc[placeholder, "adoption_status"] = (
                "T2-B_50-55_EMPIRICAL_SUPPORT_INSUFFICIENT"
            )
        atomic_csv(frame, path)


def freeze_census(root: Path, stage: int) -> pd.DataFrame:
    final = root / "02_catalogs" / "final"
    manifest = pd.read_csv(final / "j082_t2_closure_stage_manifest.csv")
    included = manifest[manifest.planned_stage <= stage]
    acquired = included[included.lifecycle == "CLOSURE_ELIGIBLE"]
    pairs = pd.read_csv(final / "j082_t2_closure_pairs.csv", dtype={"targetid": str})
    exposures = pd.read_csv(final / "j082_t2_closure_exposures.csv", dtype={"targetid": str})
    cluster_ids = set(acquired.cluster_id)
    pairs = pairs[pairs.cluster_id.isin(cluster_ids)]
    exposures = exposures[exposures.cluster_id.isin(cluster_ids)]
    valid_pairs = pairs[pairs.pair_status == "CLOSURE_ELIGIBLE"]
    valid_exposures = exposures[exposures.spectral_qc_status == "SPECTRAL_QC_PASS"]
    bounded = valid_pairs[valid_pairs.same_bin_50_55]
    cross = valid_pairs[valid_pairs.cross_30_50_to_50_55]
    all_cross = valid_pairs[valid_pairs.bin_1 != valid_pairs.bin_2]
    snr = valid_exposures.snr_pix_t2.dropna()
    bounded_snr = snr[(snr >= 50) & (snr < 55)]
    record = {
        "stage": stage,
        "n_products_planned": len(included),
        "n_products_downloaded": int(included.n_bytes.notna().sum()),
        "total_added_bytes": int(included.n_bytes.fillna(0).sum()),
        "population_targets": int(included.n_eligible_targets.fillna(0).sum()),
        "unique_targets": valid_pairs.targetid.nunique(),
        "resolved_exposure_rows": len(exposures),
        "exposure_rows": len(valid_exposures),
        "exposures_50_55": len(bounded_snr),
        "same_bin_50_55_pairs": len(bounded),
        "same_bin_50_55_nights": bounded.night.nunique(),
        "all_cross_bin_pairs": len(all_cross),
        "all_cross_bin_nights": all_cross.night.nunique(),
        "cross_30_50_to_50_55_pairs": len(cross),
        "cross_30_50_to_50_55_nights": cross.night.nunique(),
        "independent_nights": valid_pairs.night.nunique(),
        "clusters": valid_pairs.cluster_id.nunique(),
        "t2_coverage_failures": int(included.t2_coverage_failures.fillna(0).sum()),
        "qc_failures": int(included.qc_failures.fillna(0).sum()),
        "snr_min": float(snr.min()),
        "snr_median": float(snr.median()),
        "snr_max": float(snr.max()),
        "snr_50_55_min": float(bounded_snr.min()) if len(bounded_snr) else np.nan,
        "snr_50_55_median": (
            float(bounded_snr.median()) if len(bounded_snr) else np.nan
        ),
        "snr_50_55_max": float(bounded_snr.max()) if len(bounded_snr) else np.nan,
        "support_gate": (
            "ADEQUATE"
            if len(bounded) >= 20 and bounded.night.nunique() >= 3
            else "INSUFFICIENT"
        ),
        "calibration_inspected": False,
    }
    output = pd.DataFrame([record])
    atomic_csv(output, final / f"j082_t2_closure_stage{stage}_census.csv")
    if stage == 2:
        adequate = record["support_gate"] == "ADEQUATE"
        result = pd.DataFrame(
            [
                {
                    "calibration_cell": "T2-B / 50-55",
                    "support_status": (
                        "SUPPORT_GATE_ADEQUATE_PENDING_CALIBRATION"
                        if adequate
                        else "T2-B_50-55_EMPIRICAL_SUPPORT_INSUFFICIENT"
                    ),
                    "same_bin_pairs": len(bounded),
                    "independent_nights": bounded.night.nunique(),
                    "minimum_pairs": 20,
                    "minimum_nights": 3,
                    "k_empirical": np.nan,
                    "same_bin_native_width": np.nan,
                    "loo_status": "NOT_RUN_SUPPORT_GATE_FAILED",
                    "cross_bin_status": "NOT_RUN_SUPPORT_GATE_FAILED",
                    "falsification_status": "NOT_RUN_SUPPORT_GATE_FAILED",
                    "k_adopted_primary": np.nan,
                    "k_adopted_faithful": np.nan,
                    "gate_c_verdict": (
                        "PENDING_CALIBRATION" if adequate else
                        "GATE_C_PRIMARY_SCIENCE_VALIDATED_PARTIAL"
                    ),
                }
            ]
        )
        atomic_csv(result, final / "j082_t2_closure_result.csv")
        if not adequate:
            freeze_bounded_unsupported_cell(final)
    print(output.to_string(index=False))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--acquire-stage", type=int, choices=(1, 2))
    parser.add_argument("--census-stage", type=int, choices=(1, 2))
    args = parser.parse_args()
    root = project_root()
    manifest = freeze_manifest(root)
    if args.freeze:
        print(manifest.to_string(index=False))
    if args.acquire_stage:
        acquire_stage(root, args.acquire_stage)
    if args.census_stage:
        freeze_census(root, args.census_stage)
    if not (args.freeze or args.acquire_stage or args.census_stage):
        parser.error("choose --freeze, --acquire-stage, or --census-stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
