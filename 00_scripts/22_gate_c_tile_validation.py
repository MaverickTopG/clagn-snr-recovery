#!/usr/bin/env python
"""Resume Gate C after D-077: corrected applicability, cross-bin, and LOO.

The C6-B empirical factors are loaded unchanged.  Existing local repeat-star
products provide the validation sample; no calibration product is downloaded
and native IVAR is never modified.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.calibration.gate_c import (  # noqa: E402
    SN_LABELS,
    normalized_difference,
    robust_width,
    sn_bin,
)
from p3sf.config import load_config, project_root  # noqa: E402

TILES = {f"T{index + 1}": (5200.0 + 200 * index, 5400.0 + 200 * index) for index in range(9)}
MIN_SUPPORT_PAIRS = 20
MIN_SUPPORT_NIGHTS = 3
LOW_SN_LABELS = set(SN_LABELS[:4])


def _source_camera_arrays(url: str, targetid: int) -> dict[str, dict[str, np.ndarray]]:
    """Read the authoritative source row through HTTP ranges; save no product."""
    import fsspec

    handle = fsspec.open(url, block_size=1 << 20, cache_type="readahead").open()
    try:
        with fits.open(handle, memmap=False, lazy_load_hdus=True) as hdul:
            ids = np.asarray(hdul["FIBERMAP"].data["TARGETID"])
            rows = np.flatnonzero(ids == int(targetid))
            if rows.size != 1:
                raise RuntimeError(f"expected one source coadd row for TARGETID {targetid}")
            row = int(rows[0])
            return {
                band: {
                    "wave": np.asarray(hdul[f"{band}_WAVELENGTH"].data, dtype=float),
                    "flux": np.asarray(hdul[f"{band}_FLUX"].section[row], dtype=float),
                    "ivar": np.asarray(hdul[f"{band}_IVAR"].section[row], dtype=float),
                    "mask": np.asarray(hdul[f"{band}_MASK"].section[row]),
                }
                for band in ("B", "R")
            }
    finally:
        handle.close()


def _tile_snr(camera: dict[str, np.ndarray], tile: str) -> float:
    lower, upper = TILES[tile]
    good = (
        (camera["wave"] >= lower)
        & (camera["wave"] < upper)
        & (camera["ivar"] > 0)
        & (camera["mask"] == 0)
        & np.isfinite(camera["flux"])
    )
    if good.sum() < 30:
        return float("nan")
    return float(np.median(np.abs(camera["flux"][good]) * np.sqrt(camera["ivar"][good])))


def _physical_states(manifest: pd.DataFrame, qc: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    usable = qc[qc.qc_pass & qc.continuum_level_5100.notna()]
    for object_id, records in manifest.groupby("object_id"):
        object_qc = usable[usable.object_id == object_id].sort_values(
            "continuum_level_5100", ascending=False
        )
        desi_ids = set(records.spectrum_id)
        if len(object_qc) == 1:
            for spectrum_id in desi_ids:
                result[spectrum_id] = "UNASSIGNED_SINGLE_RECORD"
            continue
        for rank, row in enumerate(object_qc.itertuples()):
            if row.spectrum_id not in desi_ids:
                continue
            if rank == 0:
                result[row.spectrum_id] = "bright"
            elif rank == len(object_qc) - 1:
                result[row.spectrum_id] = "faint"
            else:
                result[row.spectrum_id] = "intermediate"
    return result


def _intersection(lower: float, upper: float, other_lower: float, other_upper: float) -> tuple[float, float] | None:
    lo = max(lower, other_lower)
    hi = min(upper, other_upper)
    return (lo, hi) if hi > lo else None


def build_applicability(root: Path, k_table: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(
        root / "03_spectra/raw_desi/manifest_slice.csv",
        dtype={"targetid": str, "zpix_id": str, "spectrum_id": str},
    )
    qc = pd.read_csv(root / "03_spectra/qc/qc_verdicts_slice.csv", dtype={"spectrum_id": str})
    states = _physical_states(manifest, qc)
    cfg = load_config()
    regions = {
        "hbeta_primary": tuple(cfg.windows.hbeta_region),
        "continuum_5100": tuple(cfg.windows.continuum_5100),
    }
    k_lookup = {
        (row.tile, row.arm, row.sn_bin): row.k_empirical for row in k_table.itertuples()
    }
    rows: list[dict[str, object]] = []

    for record in manifest.itertuples():
        cameras = _source_camera_arrays(record.url, int(record.targetid))
        tile_snrs = {
            (tile, arm): _tile_snr(cameras[arm], tile)
            for tile in TILES
            for arm in ("B", "R")
        }
        for region_name, (rest_lower, rest_upper) in regions.items():
            observed_lower = rest_lower * (1.0 + record.redshift)
            observed_upper = rest_upper * (1.0 + record.redshift)
            for tile, (tile_lower, tile_upper) in TILES.items():
                tile_overlap = _intersection(
                    observed_lower, observed_upper, tile_lower, tile_upper
                )
                if tile_overlap is None:
                    continue
                segments: list[tuple[str, str, float, float]]
                if tile == "T3":
                    segments = [
                        ("B", "B_ONLY", 5600.0, 5760.0),
                        ("B", "B+R_OFFICIAL_COADD", 5760.0, 5800.0),
                        ("R", "B+R_OFFICIAL_COADD", 5760.0, 5800.0),
                    ]
                elif tile in {"T1", "T2"}:
                    segments = [("B", "B_ONLY", tile_lower, tile_upper)]
                else:
                    segments = [("R", "R_ONLY", tile_lower, tile_upper)]

                for arm, sampling, segment_lower, segment_upper in segments:
                    interval = _intersection(
                        tile_overlap[0], tile_overlap[1], segment_lower, segment_upper
                    )
                    if interval is None:
                        continue
                    local_snr = tile_snrs[(tile, arm)]
                    label = sn_bin(local_snr)
                    cell = (tile, arm, label)
                    if cell in k_lookup:
                        support = "EMPIRICALLY_SUPPORTED"
                        empirical = k_lookup[cell]
                    elif label == "50+":
                        support = "EMPIRICAL_HIGH_SN_UNSUPPORTED"
                        empirical = np.nan
                    else:
                        support = "EMPIRICAL_CELL_UNSUPPORTED"
                        empirical = np.nan
                    rows.append(
                        {
                            "science_record_id": record.spectrum_id,
                            "object_id": record.object_id,
                            "physical_state": states[record.spectrum_id],
                            "desi_program": record.program,
                            "measurement_region": region_name,
                            "tile": tile,
                            "arm": arm,
                            "science_sampling": sampling,
                            "observed_interval_min": interval[0],
                            "observed_interval_max": interval[1],
                            "local_snr_pix": local_snr,
                            "sn_bin": label,
                            "calibration_cell": f"{tile}-{arm}/{label}",
                            "k_empirical": empirical,
                            "empirical_support_status": support,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["object_id", "science_record_id", "measurement_region", "observed_interval_min", "arm"]
    )


def collect_validation_pairs(
    root: Path, k_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct only cross-bin z widths and T2 high-S/N inventory."""
    final = root / "02_catalogs/final"
    eligible = set(
        pd.read_csv(final / "c0_eligible_targets.csv", dtype={"targetid": str}).targetid
    )
    manifest = pd.read_csv(final / "c0_acquisition_manifest.csv")
    k_lookup = {
        (row.tile, row.arm, row.sn_bin): float(row.k_empirical)
        for row in k_table.itertuples()
    }
    cross_rows: list[dict[str, object]] = []
    high_rows: list[dict[str, object]] = []

    for entry in manifest.itertuples():
        path = root / entry.local_path
        with fits.open(path, memmap=False) as hdul:
            fibermap = hdul["FIBERMAP"].data
            targetids = np.asarray(fibermap["TARGETID"]).astype(str)
            order = np.argsort(targetids, kind="stable")
            arms = {
                arm: {
                    "wave": np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float),
                    "flux": np.asarray(hdul[f"{arm}_FLUX"].data, dtype=float),
                    "ivar": np.asarray(hdul[f"{arm}_IVAR"].data, dtype=float),
                    "mask": np.asarray(hdul[f"{arm}_MASK"].data),
                }
                for arm in ("B", "R")
            }

            for targetid in np.unique(targetids):
                if targetid not in eligible:
                    continue
                left = np.searchsorted(targetids[order], targetid, side="left")
                right = np.searchsorted(targetids[order], targetid, side="right")
                indices = order[left:right]
                if indices.size != 2:
                    continue
                first, second = int(indices[0]), int(indices[1])
                for arm, arrays in arms.items():
                    for tile, (lower, upper) in TILES.items():
                        if (arm == "B" and tile not in {"T1", "T2", "T3"}) or (
                            arm == "R" and tile not in {"T3", "T4", "T5", "T6", "T7", "T8", "T9"}
                        ):
                            continue
                        good = (
                            (arrays["wave"] >= lower)
                            & (arrays["wave"] < upper)
                            & (arrays["ivar"][first] > 0)
                            & (arrays["mask"][first] == 0)
                            & (arrays["ivar"][second] > 0)
                            & (arrays["mask"][second] == 0)
                            & np.isfinite(arrays["flux"][first])
                            & np.isfinite(arrays["flux"][second])
                        )
                        if good.sum() < 30:
                            continue
                        flux_a = arrays["flux"][first][good]
                        flux_b = arrays["flux"][second][good]
                        sigma_a = 1.0 / np.sqrt(arrays["ivar"][first][good])
                        sigma_b = 1.0 / np.sqrt(arrays["ivar"][second][good])
                        snr_a = float(np.median(np.abs(flux_a) / sigma_a))
                        snr_b = float(np.median(np.abs(flux_b) / sigma_b))
                        bin_a, bin_b = sn_bin(snr_a), sn_bin(snr_b)

                        if tile == "T2" and arm == "B" and (
                            snr_a >= 50 or snr_b >= 50
                        ):
                            high_rows.append(
                                {
                                    "targetid": targetid,
                                    "night": int(entry.night),
                                    "cluster_id": entry.cluster_id,
                                    "snr_a": snr_a,
                                    "snr_b": snr_b,
                                    "bin_a": bin_a,
                                    "bin_b": bin_b,
                                    "both_in_50_55": 50 <= snr_a <= 55 and 50 <= snr_b <= 55,
                                }
                            )
                        if bin_a == bin_b or bin_a not in LOW_SN_LABELS or bin_b not in LOW_SN_LABELS:
                            continue
                        native = robust_width(
                            normalized_difference(flux_a, flux_b, sigma_a, sigma_b)
                        )
                        k_a = k_lookup.get((tile, arm, bin_a))
                        k_b = k_lookup.get((tile, arm, bin_b))
                        corrected = (
                            robust_width(
                                normalized_difference(
                                    flux_a, flux_b, sigma_a, sigma_b, k_a, k_b
                                )
                            )
                            if k_a is not None and k_b is not None
                            else None
                        )
                        labels = sorted((bin_a, bin_b), key=SN_LABELS.index)
                        cross_rows.append(
                            {
                                "tile": tile,
                                "arm": arm,
                                "bin_a": labels[0],
                                "bin_b": labels[1],
                                "targetid": targetid,
                                "night": int(entry.night),
                                "native_z_width": native,
                                "corrected_z_width": corrected,
                                "both_factors_supported": k_a is not None and k_b is not None,
                            }
                        )
        print(f"validated source cluster {entry.cluster_id}", flush=True)
    return pd.DataFrame(cross_rows), pd.DataFrame(high_rows)


def aggregate_cross_bin(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pair_rows.groupby(["tile", "arm", "bin_a", "bin_b"]):
        tile, arm, bin_a, bin_b = keys
        informative = len(group) >= MIN_SUPPORT_PAIRS and group.night.nunique() >= MIN_SUPPORT_NIGHTS
        supported = bool(group.both_factors_supported.all())
        rows.append(
            {
                "tile": tile,
                "arm": arm,
                "bin_a": bin_a,
                "bin_b": bin_b,
                "n_target_pairs": len(group),
                "n_nights": group.night.nunique(),
                "native_z_width": float(group.native_z_width.median()),
                "corrected_z_width": (
                    float(group.corrected_z_width.median()) if supported else np.nan
                ),
                "validation_status": (
                    "UNVALIDATED_UNSUPPORTED_FACTOR"
                    if not supported
                    else "INFORMATIVE_NO_NUMERICAL_PASS_THRESHOLD"
                    if informative
                    else "DESCRIPTIVE_SPARSE"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["tile", "arm", "bin_a", "bin_b"])


def run_loo(
    residuals: pd.DataFrame, k_table: pd.DataFrame, relevant_cells: set[tuple[str, str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k_lookup = {
        (row.tile, row.arm, row.sn_bin): float(row.k_empirical)
        for row in k_table.itertuples()
    }
    summaries: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for tile, arm, label in sorted(relevant_cells):
        cell = residuals[
            residuals.same_bin
            & (residuals.tile == tile)
            & (residuals.arm == arm)
            & (residuals.b1 == label)
        ].dropna(subset=["width"])
        if cell.empty or (tile, arm, label) not in k_lookup:
            continue
        for night in sorted(cell.night.unique()):
            training = cell[cell.night != night]
            heldout = cell[cell.night == night]
            if training.empty or heldout.empty:
                continue
            training_k = float(training.width.median())
            heldout_summary = float((heldout.width / training_k).median())
            folds.append(
                {
                    "tile": tile,
                    "arm": arm,
                    "sn_bin": label,
                    "heldout_night": int(night),
                    "n_training_pairs": len(training),
                    "n_heldout_pairs": len(heldout),
                    "loo_k": training_k,
                    "heldout_corrected_width": heldout_summary,
                }
            )
        cell_folds = pd.DataFrame(
            [
                row
                for row in folds
                if row["tile"] == tile and row["arm"] == arm and row["sn_bin"] == label
            ]
        )
        if cell_folds.empty:
            continue
        deviation = (cell_folds.heldout_corrected_width - 1.0).abs()
        max_row = cell_folds.loc[deviation.idxmax()]
        summaries.append(
            {
                "tile": tile,
                "arm": arm,
                "sn_bin": label,
                "n_pairs": len(cell),
                "n_nights": cell.night.nunique(),
                "full_sample_k": k_lookup[(tile, arm, label)],
                "loo_k_min": cell_folds.loo_k.min(),
                "loo_k_max": cell_folds.loo_k.max(),
                "heldout_corrected_width_summary": (
                    f"median={cell_folds.heldout_corrected_width.median():.6f};"
                    f"min={cell_folds.heldout_corrected_width.min():.6f};"
                    f"max={cell_folds.heldout_corrected_width.max():.6f}"
                ),
                "anomalous_night_if_any": (
                    "no post-hoc threshold applied; max absolute deviation "
                    f"night={int(max_row.heldout_night)} width={max_row.heldout_corrected_width:.6f}"
                ),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    fold_frame = pd.DataFrame(folds)
    fold_frame["absolute_deviation_from_unity"] = (
        fold_frame.heldout_corrected_width - 1.0
    ).abs()
    max_rows = fold_frame.loc[
        fold_frame.groupby(["tile", "arm", "sn_bin"])[
            "absolute_deviation_from_unity"
        ].idxmax()
    ]
    recurrence = max_rows.heldout_night.value_counts()
    for index, row in summary_frame.iterrows():
        cell_max = max_rows[
            (max_rows.tile == row.tile)
            & (max_rows.arm == row.arm)
            & (max_rows.sn_bin == row.sn_bin)
        ].iloc[0]
        night = int(cell_max.heldout_night)
        summary_frame.loc[index, "anomalous_night_if_any"] += (
            f"; same night is maximum-deviation fold for {int(recurrence[night])} "
            "of the relevant cells"
        )
    night_summary = (
        fold_frame.groupby("heldout_night", as_index=False)
        .agg(
            n_cells=("sn_bin", "size"),
            heldout_width_median=("heldout_corrected_width", "median"),
            heldout_width_min=("heldout_corrected_width", "min"),
            heldout_width_max=("heldout_corrected_width", "max"),
        )
        .sort_values("heldout_night")
    )
    night_summary["n_cells_where_maximum_absolute_deviation"] = (
        night_summary.heldout_night.map(recurrence).fillna(0).astype(int)
    )
    return summary_frame, fold_frame, night_summary


def build_calibration_tables(
    k_table: pd.DataFrame,
    applicability: pd.DataFrame,
    cross: pd.DataFrame,
    loo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = [("T1", "B"), ("T2", "B"), ("T3", "B"), ("T3", "R")]
    arms.extend((f"T{index}", "R") for index in range(4, 10))
    empirical_lookup = {
        (row.tile, row.arm, row.sn_bin): row for row in k_table.itertuples()
    }
    relevant = {
        (row.tile, row.arm, row.sn_bin)
        for row in applicability.itertuples()
        if row.empirical_support_status == "EMPIRICALLY_SUPPORTED"
    }
    cross_cells: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in cross.itertuples():
        for label in (row.bin_a, row.bin_b):
            cross_cells[(row.tile, row.arm, label)].append(row.validation_status)
    loo_cells = {(row.tile, row.arm, row.sn_bin) for row in loo.itertuples()}

    rows = []
    for tile, arm in arms:
        for label in SN_LABELS:
            key = (tile, arm, label)
            source = empirical_lookup.get(key)
            measured = source is not None
            if measured:
                empirical_status = "EMPIRICALLY_SUPPORTED"
                value = float(source.k_empirical)
            elif label == "50+":
                empirical_status = "EMPIRICAL_HIGH_SN_UNSUPPORTED"
                value = np.nan
            else:
                empirical_status = "EMPIRICAL_CELL_UNSUPPORTED"
                value = np.nan
            cross_statuses = cross_cells.get(key, [])
            informative_cross = "INFORMATIVE_NO_NUMERICAL_PASS_THRESHOLD" in cross_statuses
            validated = key in relevant and informative_cross and key in loo_cells
            rows.append(
                {
                    "tile": tile,
                    "arm": arm,
                    "sn_bin": label,
                    "k_empirical": value,
                    "ci_lo": float(source.ci_lo) if measured else np.nan,
                    "ci_hi": float(source.ci_hi) if measured else np.nan,
                    "n_pairs": int(source.n_pairs) if measured else 0,
                    "n_nights": int(source.n_nights) if measured else 0,
                    "empirical_support_status": empirical_status,
                    "primary_science_relevant": key in relevant,
                    "cross_bin_validation_status": (
                        "INFORMATIVE" if informative_cross else "SPARSE_ONLY" if cross_statuses else "NOT_RUN_OR_UNPOPULATED"
                    ),
                    "loo_validation_status": "RUN" if key in loo_cells else "NOT_REQUIRED_OR_UNSUPPORTED",
                    "validated_for_primary_adoption": validated,
                }
            )
    empirical = pd.DataFrame(rows)
    primary = empirical.copy()
    primary["k_adopted_primary"] = np.where(
        primary.validated_for_primary_adoption,
        np.maximum(1.0, primary.k_empirical),
        np.nan,
    )
    primary["adoption_status"] = np.where(
        primary.validated_for_primary_adoption,
        "VALIDATED_CONSERVATIVE_MAX_1_K_EMPIRICAL",
        primary.empirical_support_status,
    )
    faithful = empirical.copy()
    faithful["k_adopted_faithful"] = np.where(
        faithful.empirical_support_status == "EMPIRICALLY_SUPPORTED",
        faithful.k_empirical,
        np.nan,
    )
    faithful["adoption_status"] = np.select(
        [
            faithful.validated_for_primary_adoption,
            faithful.empirical_support_status == "EMPIRICALLY_SUPPORTED",
        ],
        [
            "VALIDATED_FAITHFUL_EQUALS_K_EMPIRICAL",
            "EMPIRICAL_SENSITIVITY_ONLY_NOT_PRIMARY_VALIDATED",
        ],
        default=faithful.empirical_support_status,
    )
    return empirical, primary, faithful


def main() -> int:
    root = project_root()
    final = root / "02_catalogs/final"
    analysis = root / "05_analysis/gate_c"
    analysis.mkdir(parents=True, exist_ok=True)

    original_k = pd.read_csv(final / "c6b_k_empirical.csv")
    applicability = build_applicability(root, original_k)
    applicability.to_csv(final / "c6c_science_applicability.csv", index=False)
    unsupported = applicability[
        applicability.empirical_support_status != "EMPIRICALLY_SUPPORTED"
    ]
    print(f"applicability rows={len(applicability)} unsupported={len(unsupported)}")

    cross_pairs, high_inventory = collect_validation_pairs(root, original_k)
    cross_pairs.to_csv(analysis / "c6c_cross_bin_pair_details.csv", index=False)
    high_inventory.to_csv(analysis / "c6c_t2b_high_snr_inventory.csv", index=False)
    cross = aggregate_cross_bin(cross_pairs)
    cross.to_csv(final / "c6c_cross_bin_validation.csv", index=False)

    residuals = pd.read_csv(final / "c6b_tile_residuals.csv", dtype={"targetid": str})
    relevant_cells = {
        (row.tile, row.arm, row.sn_bin)
        for row in applicability.itertuples()
        if row.empirical_support_status == "EMPIRICALLY_SUPPORTED"
    }
    loo, loo_folds, loo_nights = run_loo(residuals, original_k, relevant_cells)
    loo.to_csv(final / "c6c_loo_validation.csv", index=False)
    loo_folds.to_csv(analysis / "c6c_loo_fold_details.csv", index=False)
    loo_nights.to_csv(analysis / "c6c_loo_night_summary.csv", index=False)

    empirical, primary, faithful = build_calibration_tables(
        original_k, applicability, cross, loo
    )
    empirical.to_csv(final / "c6c_k_empirical_validated.csv", index=False)
    primary.to_csv(final / "c6c_k_adopted_primary.csv", index=False)
    faithful.to_csv(final / "c6c_k_adopted_faithful.csv", index=False)

    candidates = pd.read_csv(final / "c0_cluster_frame_desi.csv")
    selected = pd.read_csv(final / "c0_selected_clusters.csv")
    selected_keys = set(zip(selected.tileid, selected.night, selected.petal, strict=True))
    candidates["already_selected"] = [
        (row.tileid, row.night, row.petal) in selected_keys for row in candidates.itertuples()
    ]
    candidates[~candidates.already_selected].sort_values(
        ["n_high", "n_targets"], ascending=False
    ).to_csv(analysis / "c6c_unused_cluster_metadata.csv", index=False)

    print(f"cross-bin combinations={len(cross)}; LOO cells={len(loo)}")
    print(
        f"T2-B existing >=50 pairs={len(high_inventory)} "
        f"nights={high_inventory.night.nunique() if not high_inventory.empty else 0}; "
        f"both in 50-55={int(high_inventory.both_in_50_55.sum()) if not high_inventory.empty else 0}"
    )
    print("native IVAR unchanged; no downstream spectrum fit run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
