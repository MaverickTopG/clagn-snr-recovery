#!/usr/bin/env python3
"""Execute the frozen D-093 full-Q1 matrix and freeze D-094 raw products.

This runner deliberately separates three phases:

1. prepare and validate the exact reusable spectrum-fit task manifest;
2. execute/checkpoint every spectrum fit and freeze condition-level outcomes;
3. run production QC and record completion only when all invariants pass.

Scientific aggregation is performed by 35_analyze_full_q1.py only after this
script has frozen raw products and emitted a passing completion record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.counterfactual.pair_snr import _bright_seed, _faint_seed  # noqa: E402
from p3sf.counterfactual.snr import degrade_to_snr  # noqa: E402
from p3sf.criteria.green_pixel import (  # noqa: E402
    GreenEpochSpectrum,
    VarianceProvenance,
    classify_green_pixel_measurement,
    fit_green_epoch_line_spectrum_pyqsofit,
    measure_green_pixel_nsigma,
)
from p3sf.criteria.source_verified import apply_yang2024_hbeta  # noqa: E402
from p3sf.fitting.pyqsofit_driver import (  # noqa: E402
    YangHbetaFitAudit,
    fit_yang_hbeta_audit_pyqsofit,
)

DESIGN = ROOT / "05_analysis" / "q1_design" / "d093"
OUT = ROOT / "05_analysis" / "q1_production" / "d094"
RAW = OUT / "raw"
BATCHES = RAW / "fit_batches"
MATRIX_PATH = DESIGN / "final_q1_matrix_d093_unexecuted.csv"
APP_PATH = DESIGN / "final_classifier_applicability_d093.csv"
SENSITIVITY_SAMPLE_PATH = DESIGN / "final_sensitivity_sample_d093.csv"
IDENTITY_PATH = DESIGN / "endpoint_identity_validation_d093.csv"
NEW_MANIFEST_PATH = DESIGN / "endpoint_acquisition_manifest_d093.csv"
NEW_QC_PATH = DESIGN / "native_qc_d093.csv"
OLD_SUPPORT_PATH = ROOT / "05_analysis" / "q1_design" / "expanded_native_snr_support_d091.csv"
OLD_QC_PATH = ROOT / "05_analysis" / "q1_design" / "native_qc_d091.csv"
OLD_IDENTITY_PATH = ROOT / "05_analysis" / "q1_design" / "endpoint_identity_validation_d091.csv"
SLICE_QC_PATH = ROOT / "03_spectra" / "qc" / "qc_verdicts_slice.csv"
EXPANSION_MANIFEST_PATH = ROOT / "03_spectra" / "raw_sdss" / "manifest_d091_expansion.csv"
SLICE_MANIFEST_PATH = ROOT / "03_spectra" / "raw_sdss" / "manifest_slice.csv"

SEED_BASE = 314159
SEED_NAMESPACE = "p3sf:q1:full:v1"
REALIZATIONS = 50
EXPECTED_CONDITIONS = 167
EXPECTED_CONDITION_REALIZATIONS = 8350
EXPECTED_FIT_INPUTS = 8412
EXPECTED_OUTCOMES = 25050
SNR_WINDOW_REST = (4700.0, 5100.0)
VENDOR_PATH = ROOT / "06_fitting" / "pyqsofit" / "vendor" / "PyQSOFit"
CRITERIA = ("YANG2024_FINAL", "GREEN2022_FINAL", "MACLEOD2019_FINAL")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _value(row: pd.Series, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _resolve_local_path(value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _normalize_record_id(value: str) -> str:
    return str(value).removeprefix("sdss:")


def _iau_coordinates(object_id: str) -> tuple[float, float]:
    match = re.search(
        r"J(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)$",
        str(object_id),
    )
    if match is None:
        return np.nan, np.nan
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra = 15.0 * (float(hh) + float(mm) / 60.0 + float(ss) / 3600.0)
    dec = float(dd) + float(dm) / 60.0 + float(ds) / 3600.0
    return ra, dec if sign == "+" else -dec


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _endpoint_catalog() -> pd.DataFrame:
    """Return one row per locally validated historical endpoint."""
    rows: list[dict[str, Any]] = []

    new_manifest = _read_table(NEW_MANIFEST_PATH)
    new_qc = _read_table(NEW_QC_PATH)
    identity = _read_table(IDENTITY_PATH)
    for _, r in new_manifest.iterrows():
        transition_id = str(_value(r, "transition_id"))
        role = str(_value(r, "physical_role", "endpoint_role", "role")).lower()
        q = new_qc.loc[new_qc["transition_id"].astype(str).eq(transition_id)]
        q_role = next(c for c in ("physical_role", "endpoint_role", "role") if c in q.columns)
        q = q.loc[q[q_role].astype(str).str.lower().eq(role)]
        qr = q.iloc[0] if len(q) else pd.Series(dtype=object)
        i = identity.loc[identity["transition_id"].astype(str).eq(transition_id)]
        i_role = next(c for c in ("physical_role", "endpoint_role", "role") if c in i.columns)
        i = i.loc[i[i_role].astype(str).str.lower().eq(role)]
        ir = i.iloc[0] if len(i) else pd.Series(dtype=object)
        rows.append(
            {
                "transition_id": transition_id,
                "role": role,
                "spectrum_id": str(
                    _value(
                        r,
                        "immutable_source_identifier",
                        "immutable_identifier",
                        "spectrum_id",
                        "source_identifier",
                    )
                ),
                "local_path": str(
                    _resolve_local_path(str(_value(r, "local_filename", "local_path")))
                ),
                "instrument_survey": str(
                    _value(r, "instrument_survey", "survey", default="UNKNOWN")
                ),
                "redshift": float(
                    _value(qr, "canonical_redshift", "redshift", default=_value(ir, "redshift"))
                ),
                "ra_deg": float(
                    _value(ir, "archive_ra", "archive_ra_deg", "ra_deg", default=np.nan)
                ),
                "dec_deg": float(
                    _value(ir, "archive_dec", "archive_dec_deg", "dec_deg", default=np.nan)
                ),
                "expected_sha256": str(_value(r, "sha256", "checksum_sha256", default="")),
                "variance_provenance": str(_value(qr, "variance_provenance", default="")),
                "catalog_source": "D093",
            }
        )

    support = _read_table(OLD_SUPPORT_PATH)
    manifests = pd.concat(
        [_read_table(EXPANSION_MANIFEST_PATH), _read_table(SLICE_MANIFEST_PATH)],
        ignore_index=True,
        sort=False,
    )
    spectrum_col = next(
        c for c in ("spectrum_id", "immutable_identifier", "record_id") if c in manifests.columns
    )
    manifest_map = {_normalize_record_id(str(r[spectrum_col])): r for _, r in manifests.iterrows()}
    old_qc = pd.concat(
        [_read_table(OLD_QC_PATH), _read_table(SLICE_QC_PATH)], ignore_index=True, sort=False
    )
    old_identity = _read_table(OLD_IDENTITY_PATH)
    already = {r["transition_id"] for r in rows}
    for _, r in support.iterrows():
        transition_id = str(r["transition_id"])
        if transition_id in already:
            continue
        for role in ("bright", "faint"):
            record = _normalize_record_id(str(r[f"{role}_science_record_id"]))
            if record not in manifest_map:
                raise RuntimeError(
                    f"No local manifest binding for {transition_id} {role}: {record}"
                )
            mr = manifest_map[record]
            q = old_qc.loc[
                old_qc["object_id"].astype(str).eq(transition_id)
                & old_qc["spectrum_id"].astype(str).map(_normalize_record_id).eq(record)
            ]
            if len(q) != 1:
                raise RuntimeError(f"No unique old QC binding for {transition_id} {record}")
            qr = q.iloc[0]
            i = old_identity.loc[
                old_identity["object"].astype(str).eq(transition_id)
                & old_identity["expected_spectrum_id"]
                .astype(str)
                .map(_normalize_record_id)
                .eq(record)
            ]
            ir = i.iloc[0] if len(i) == 1 else pd.Series(dtype=object)
            parsed_ra, parsed_dec = _iau_coordinates(transition_id)
            local_value = _value(
                qr, "path", default=_value(mr, "local_filename", "local_path", "path")
            )
            rows.append(
                {
                    "transition_id": transition_id,
                    "role": role,
                    "spectrum_id": record,
                    "local_path": str(_resolve_local_path(str(local_value))),
                    "instrument_survey": "SDSS_LEGACY",
                    "redshift": float(_value(qr, "redshift")),
                    "ra_deg": float(_value(ir, "archive_ra", default=parsed_ra)),
                    "dec_deg": float(_value(ir, "archive_dec", default=parsed_dec)),
                    "expected_sha256": str(_value(mr, "sha256", "checksum_sha256", default="")),
                    "variance_provenance": "NATIVE_SDSS_SUPPORTED",
                    "catalog_source": "D091_OR_EARLIER",
                }
            )
    out = pd.DataFrame(rows)
    if out.duplicated(["transition_id", "role"]).any():
        dup = out.loc[out.duplicated(["transition_id", "role"], keep=False)]
        raise RuntimeError(f"Duplicate endpoint bindings:\n{dup}")
    return out.sort_values(["transition_id", "role"]).reset_index(drop=True)


def _condition_columns(matrix: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "condition_id": ("condition_id", "q1_condition_id"),
        "transition_id": ("transition_id",),
        "object_id": ("object_id", "object"),
        "reference_tier": ("reference_tier",),
        "source_family": ("source_family",),
        "arm": ("arm",),
        "snr": ("snr", "snr_rung", "target_snr"),
    }
    result: dict[str, str] = {}
    for logical, options in candidates.items():
        match = next((c for c in options if c in matrix.columns), None)
        if match is None and logical in {"condition_id", "object_id"}:
            continue
        if match is None:
            raise RuntimeError(
                f"Missing matrix column for {logical}; columns={list(matrix.columns)}"
            )
        result[logical] = match
    return result


def prepare_task_manifest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = _read_table(MATRIX_PATH)
    cols = _condition_columns(matrix)
    matrix = matrix.rename(columns={v: k for k, v in cols.items() if v != k})
    matrix["transition_id"] = matrix["transition_id"].astype(str)
    if "object_id" not in matrix:
        matrix["object_id"] = matrix["transition_id"]
    matrix["snr"] = matrix["snr"].astype(int)
    if "condition_id" not in matrix:
        matrix["condition_id"] = [
            f"{t}|{a}|snr{s}"
            for t, a, s in zip(matrix.transition_id, matrix.arm, matrix.snr, strict=True)
        ]
    if (
        len(matrix) != EXPECTED_CONDITIONS
        or matrix["condition_id"].nunique() != EXPECTED_CONDITIONS
    ):
        raise RuntimeError(f"Frozen condition count mismatch: {len(matrix)}")
    if set(matrix["snr"]) != {5, 10} or set(matrix["arm"]) != {"faint_only", "matched"}:
        raise RuntimeError("Frozen S/N grid or arms differ from D-093")

    endpoints = _endpoint_catalog()
    frozen_reference = set(_read_table(SENSITIVITY_SAMPLE_PATH)["transition_id"].astype(str))
    endpoints = endpoints.loc[endpoints["transition_id"].isin(frozen_reference)].reset_index(
        drop=True
    )
    bound_transitions = set(endpoints["transition_id"])
    if bound_transitions != frozen_reference:
        raise RuntimeError(
            f"Frozen 62-transition endpoint binding mismatch: missing={sorted(frozen_reference - bound_transitions)}"
        )
    matrix_transitions = set(matrix["transition_id"])
    missing = matrix_transitions - bound_transitions
    if missing:
        raise RuntimeError(f"Missing endpoint bindings: {sorted(missing)}")
    # The frozen 8,412-input workload includes one native-bright audit for all
    # 62 reference transitions, including ten without a supported degradation
    # condition. Degraded tasks exist only for the 52 transitions in the matrix.
    required_transitions = bound_transitions
    endpoint_map = {(r["transition_id"], r["role"]): r for _, r in endpoints.iterrows()}

    tasks: list[dict[str, Any]] = []
    consumers: dict[str, int] = {}
    for transition_id in sorted(required_transitions):
        bright = endpoint_map[(transition_id, "bright")]
        task_id = f"{transition_id}|native_bright"
        consumers[task_id] = int((matrix["transition_id"] == transition_id).sum() * REALIZATIONS)
        tasks.append(_task_row(task_id, transition_id, "native_bright", bright, None, None))

    for _, c in matrix.sort_values("condition_id").iterrows():
        transition_id = c["transition_id"]
        faint = endpoint_map[(transition_id, "faint")]
        bright = endpoint_map[(transition_id, "bright")]
        pair_qualified = f"{SEED_NAMESPACE}|{transition_id}"
        for realization in range(REALIZATIONS):
            faint_id = f"{transition_id}|faint|snr{c['snr']}|r{realization:02d}"
            consumers[faint_id] = consumers.get(faint_id, 0) + 1
            if consumers[faint_id] == 1:
                seed = _faint_seed(
                    SEED_BASE, pair_qualified, str(faint["spectrum_id"]), realization, int(c["snr"])
                )
                tasks.append(
                    _task_row(
                        faint_id,
                        transition_id,
                        "faint_shared",
                        faint,
                        int(c["snr"]),
                        seed,
                        realization,
                    )
                )
            if c["arm"] == "matched":
                bright_id = f"{transition_id}|bright|snr{c['snr']}|r{realization:02d}"
                consumers[bright_id] = consumers.get(bright_id, 0) + 1
                if consumers[bright_id] == 1:
                    seed = _bright_seed(
                        SEED_BASE,
                        pair_qualified,
                        str(bright["spectrum_id"]),
                        realization,
                        int(c["snr"]),
                    )
                    tasks.append(
                        _task_row(
                            bright_id,
                            transition_id,
                            "matched_bright",
                            bright,
                            int(c["snr"]),
                            seed,
                            realization,
                        )
                    )

    task_df = (
        pd.DataFrame(tasks).drop_duplicates("task_id").sort_values("task_id").reset_index(drop=True)
    )
    task_df["consumer_count"] = task_df["task_id"].map(consumers).astype(int)
    task_df.insert(0, "task_sequence", np.arange(len(task_df), dtype=int))
    if len(task_df) != EXPECTED_FIT_INPUTS:
        raise RuntimeError(
            f"Reusable fit input count mismatch: {len(task_df)} != {EXPECTED_FIT_INPUTS}"
        )
    degraded = task_df[task_df["task_kind"] != "native_bright"]
    if degraded["seed"].isna().any() or degraded["seed"].duplicated().any():
        raise RuntimeError("Missing or colliding degraded-spectrum seeds")
    if (task_df["seed_namespace"] != SEED_NAMESPACE).any():
        raise RuntimeError("Seed namespace mismatch")

    for _, endpoint in endpoints.iterrows():
        path = Path(endpoint["local_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = endpoint["expected_sha256"]
        if expected and expected.lower() not in {"nan", "none"} and actual != expected:
            raise RuntimeError(f"Endpoint checksum mismatch: {path}")
        endpoints.loc[endpoints.index == endpoint.name, "actual_sha256"] = actual
    checksum_map = endpoints.set_index(["transition_id", "role"])["actual_sha256"].to_dict()
    task_df["source_file_sha256"] = [
        checksum_map[
            (
                r.transition_id,
                "bright" if r.task_kind in {"native_bright", "matched_bright"} else "faint",
            )
        ]
        for r in task_df.itertuples()
    ]
    return matrix, endpoints, task_df


def _task_row(
    task_id: str,
    transition_id: str,
    task_kind: str,
    endpoint: pd.Series,
    target_snr: int | None,
    seed: int | None,
    realization: int | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "transition_id": transition_id,
        "task_kind": task_kind,
        "endpoint_role": "bright" if task_kind in {"native_bright", "matched_bright"} else "faint",
        "spectrum_id": endpoint["spectrum_id"],
        "local_path": endpoint["local_path"],
        "instrument_survey": endpoint["instrument_survey"],
        "redshift": float(endpoint["redshift"]),
        "ra_deg": float(endpoint["ra_deg"]),
        "dec_deg": float(endpoint["dec_deg"]),
        "target_snr": target_snr,
        "realization": realization,
        "seed": seed,
        "seed_namespace": SEED_NAMESPACE,
        "seed_base": SEED_BASE,
        "variance_provenance": endpoint.get("variance_provenance", ""),
    }


def _load_spectrum(row: dict[str, Any]) -> Spectrum:
    path = Path(row["local_path"])
    survey = str(row["instrument_survey"]).upper()
    with fits.open(path, memmap=False) as hdul:
        if "LAMOST" in survey:
            data = hdul[1].data
            names = {n.upper(): n for n in data.names}
            first = data[0]
            wave = np.asarray(first[names["WAVELENGTH"]], dtype=float)
            flux = np.asarray(first[names["FLUX"]], dtype=float)
            ivar = np.asarray(first[names["IVAR"]], dtype=float)
            mask = ~np.isfinite(ivar) | (ivar <= 0)
        elif "DESI" in survey:
            wave = np.asarray(hdul["WAVELENGTH"].data, dtype=float).ravel()
            flux = np.asarray(hdul["FLUX"].data, dtype=float).ravel()
            ivar = np.asarray(hdul["IVAR"].data, dtype=float).ravel()
            mask = (
                np.asarray(hdul["MASK"].data).ravel() != 0
                if "MASK" in hdul
                else np.zeros_like(wave, dtype=bool)
            )
            mask |= ~np.isfinite(ivar) | (ivar <= 0)
        else:
            data = hdul[1].data
            names = {n.lower(): n for n in data.names}
            wave = np.power(10.0, np.asarray(data[names["loglam"]], dtype=float))
            flux = np.asarray(data[names["flux"]], dtype=float)
            ivar = np.asarray(data[names["ivar"]], dtype=float)
            # Frozen D-089/D-093 measurement contract: SDSS pixel validity is
            # positive IVAR. Survey bitmasks remain provenance/QC metadata and
            # are not an additional fitting mask (32_q1...::_read_sdss).
            mask = ~np.isfinite(ivar) | (ivar <= 0)
    error = np.full_like(ivar, np.inf, dtype=float)
    good = np.isfinite(ivar) & (ivar > 0)
    error[good] = 1.0 / np.sqrt(ivar[good])
    mask |= ~np.isfinite(wave) | ~np.isfinite(flux) | ~np.isfinite(error)
    return Spectrum(
        wavelength=wave,
        flux=flux,
        error=error,
        redshift=float(row["redshift"]),
        mask=mask.astype(int),
        meta={"spectrum_id": str(row["spectrum_id"])},
    )


def _process_task(row: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        native = _load_spectrum(row)
        if row["task_kind"] == "native_bright":
            spec = native
            degradation = None
        else:
            degradation = degrade_to_snr(
                native,
                float(row["target_snr"]),
                window_rest=SNR_WINDOW_REST,
                seed=int(row["seed"]),
            )
            spec = degradation.spectrum
        arrays_sha = hash_arrays(
            spec.wavelength, spec.flux, spec.error, np.asarray(spec.mask).astype(np.uint8)
        )
        _, yang = fit_yang_hbeta_audit_pyqsofit(
            spec.wavelength,
            spec.flux,
            spec.error,
            redshift=float(row["redshift"]),
            path=str(VENDOR_PATH),
        )
        provenance_text = str(row.get("variance_provenance", ""))
        try:
            provenance = VarianceProvenance(provenance_text)
        except ValueError:
            survey = str(row["instrument_survey"]).upper()
            if "LAMOST" in survey:
                provenance = VarianceProvenance.NATIVE_LAMOST_SUPPORTED
            elif "SDSSV" in survey:
                provenance = VarianceProvenance.NATIVE_SDSSV_SUPPORTED
            elif "DESI" in survey:
                provenance = VarianceProvenance.NATIVE_DESI_EDR_SUPPORTED
            else:
                provenance = VarianceProvenance.NATIVE_SDSS_SUPPORTED
        green = fit_green_epoch_line_spectrum_pyqsofit(
            spec.wavelength,
            spec.flux,
            spec.error,
            redshift=float(row["redshift"]),
            path=VENDOR_PATH,
            variance_provenance=provenance,
            ra=float(row["ra_deg"]),
            dec=float(row["dec_deg"]),
            scale_factor=1.0,
            scale_provenance="NO_RESCALE_Q1_OBSERVED_HISTORICAL_ENDPOINT",
        )
        result: dict[str, Any] = {
            **row,
            "job_status": "COMPLETED",
            "failure_reason": "",
            "spectrum_array_sha256": arrays_sha,
            "elapsed_seconds": time.monotonic() - started,
            "requested_snr": float(row["target_snr"])
            if row.get("target_snr") is not None
            else np.nan,
            "original_snr": degradation.original_snr if degradation is not None else np.nan,
            "achieved_snr": degradation.achieved_snr if degradation is not None else np.nan,
            "snr_scale_factor": degradation.scale_factor if degradation is not None else 1.0,
        }
        result.update({f"yang_{k}": v for k, v in asdict(yang).items()})
        result.update(
            {
                "green_preprocessing_valid": green.preprocessing_valid,
                "green_invalid_reason": green.invalid_reason,
                "green_scale_factor": green.scale_factor,
                "green_scale_provenance": green.scale_provenance,
                "green_wavelength_rest": green.wavelength_rest.tolist(),
                "green_line_flux": green.line_flux.tolist(),
                "green_variance": green.variance.tolist(),
                "green_variance_provenance": green.variance_provenance.tolist(),
            }
        )
        return result
    except Exception as exc:  # fail closed and retain exact failure provenance
        return {
            **row,
            "job_status": "FAILED",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "spectrum_array_sha256": "",
            "elapsed_seconds": time.monotonic() - started,
            "green_wavelength_rest": [],
            "green_line_flux": [],
            "green_variance": [],
            "green_variance_provenance": [],
        }


def _write_atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _existing_fit_results() -> pd.DataFrame:
    files = sorted(BATCHES.glob("fit_batch_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    out = pd.concat(frames, ignore_index=True)
    if out["task_id"].duplicated().any():
        raise RuntimeError("Duplicate task IDs across fit checkpoints")
    return out


def invalidate_wrong_sdss_mask_checkpoints() -> dict[str, Any]:
    """Remove only provisional fits made with the rejected SDSS bitmask reader."""
    BATCHES.mkdir(parents=True, exist_ok=True)
    affected = 0
    retained = 0
    files = sorted(BATCHES.glob("fit_batch_*.parquet"))
    for path in files:
        frame = pd.read_parquet(path)
        bad = frame["instrument_survey"].astype(str).str.upper().str.contains("SDSS")
        affected += int(bad.sum())
        keep = frame.loc[~bad].copy()
        retained += len(keep)
        if len(keep):
            _write_atomic_parquet(keep, path)
        else:
            path.unlink()
    audit = {
        "repair": "INVALIDATE_REJECTED_SDSS_BITMASK_READER_OUTPUTS",
        "reason": "Frozen D-089/D-093 contract masks SDSS by nonpositive IVAR only",
        "affected_tasks_to_rerun": affected,
        "unaffected_tasks_retained": retained,
        "seeds_changed": False,
        "matrix_changed": False,
    }
    (OUT / "engineering_repair_audit_d094.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def execute_fits(task_df: pd.DataFrame, workers: int, batch_size: int) -> pd.DataFrame:
    BATCHES.mkdir(parents=True, exist_ok=True)
    existing = _existing_fit_results()
    complete = set(existing["task_id"]) if len(existing) else set()
    pending = task_df.loc[~task_df["task_id"].isin(complete)].to_dict("records")
    existing_batch_indices = [
        int(path.stem.rsplit("_", 1)[-1]) for path in BATCHES.glob("fit_batch_*.parquet")
    ]
    batch_index = max(existing_batch_indices, default=-1) + 1
    print(
        f"fit tasks: total={len(task_df)} checkpointed={len(complete)} pending={len(pending)}",
        flush=True,
    )
    if pending:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        ctx = get_context("spawn")
        buffer: list[dict[str, Any]] = []
        completed_now = 0
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_process_task, pending, chunksize=1):
                buffer.append(result)
                completed_now += 1
                if len(buffer) >= batch_size:
                    path = BATCHES / f"fit_batch_{batch_index:05d}.parquet"
                    _write_atomic_parquet(pd.DataFrame(buffer), path)
                    batch_index += 1
                    buffer.clear()
                    print(f"checkpointed {completed_now}/{len(pending)} new fit tasks", flush=True)
            if buffer:
                path = BATCHES / f"fit_batch_{batch_index:05d}.parquet"
                _write_atomic_parquet(pd.DataFrame(buffer), path)
                print(f"checkpointed {completed_now}/{len(pending)} new fit tasks", flush=True)
    result = _existing_fit_results()
    if set(result["task_id"]) != set(task_df["task_id"]):
        raise RuntimeError("Final fit checkpoints do not match task manifest")
    return result.sort_values("task_id").reset_index(drop=True)


def _audit_from_fit(row: pd.Series) -> YangHbetaFitAudit:
    kwargs = {}
    for field in YangHbetaFitAudit.__dataclass_fields__:
        value = row.get(f"yang_{field}")
        if isinstance(value, float) and np.isnan(value):
            value = None
        kwargs[field] = value
    return YangHbetaFitAudit(**kwargs)


def _green_from_fit(row: pd.Series) -> GreenEpochSpectrum:
    return GreenEpochSpectrum(
        wavelength_rest=np.asarray(row["green_wavelength_rest"], dtype=float),
        line_flux=np.asarray(row["green_line_flux"], dtype=float),
        variance=np.asarray(row["green_variance"], dtype=float),
        variance_provenance=np.asarray(row["green_variance_provenance"], dtype=str),
        preprocessing_valid=bool(row.get("green_preprocessing_valid", False)),
        invalid_reason=str(row.get("green_invalid_reason", "missing_fit_result")),
        scale_factor=float(row.get("green_scale_factor", 1.0)),
        scale_provenance=str(row.get("green_scale_provenance", "")),
    )


def _terminal(label: str | None) -> str:
    value = str(label or "").upper()
    if value in {"CL", "NON_CL", "UNCLASSIFIABLE"}:
        return value
    if value == "NON-CL":
        return "NON_CL"
    return "UNCLASSIFIABLE"


def build_outcomes(matrix: pd.DataFrame, fits_df: pd.DataFrame) -> pd.DataFrame:
    lookup = fits_df.set_index("task_id", drop=False)
    applicability = _read_table(APP_PATH)
    app_lookup = {
        (str(r["transition_id"]), str(r["arm"]), int(r["snr_rung"]), str(r["criterion"])): r
        for _, r in applicability.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, c in matrix.sort_values("condition_id").iterrows():
        for realization in range(REALIZATIONS):
            faint_id = f"{c.transition_id}|faint|snr{int(c.snr)}|r{realization:02d}"
            bright_id = (
                f"{c.transition_id}|native_bright"
                if c.arm == "faint_only"
                else f"{c.transition_id}|bright|snr{int(c.snr)}|r{realization:02d}"
            )
            bright = lookup.loc[bright_id]
            faint = lookup.loc[faint_id]
            for criterion in CRITERIA:
                app = app_lookup.get((str(c.transition_id), str(c.arm), int(c.snr), criterion))
                applicable = bool(
                    app is not None
                    and str(_value(app, "applicability", "status", default=""))
                    .upper()
                    .startswith("APPLICABLE")
                    and not bool(_value(app, "unclassifiable_by_design", default=False))
                )
                app_reason = (
                    str(
                        _value(
                            app,
                            "denominator_semantics",
                            "reason",
                            "applicability_reason",
                            default="not_applicable_by_frozen_d093",
                        )
                    )
                    if app is not None
                    else "missing_frozen_applicability"
                )
                base = {
                    "condition_id": c.condition_id,
                    "transition_id": c.transition_id,
                    "object_id": c.object_id,
                    "reference_tier": c.reference_tier,
                    "source_family": c.source_family,
                    "arm": c.arm,
                    "snr": int(c.snr),
                    "realization": realization,
                    "criterion_id": criterion,
                    "bright_task_id": bright_id,
                    "faint_task_id": faint_id,
                    "bright_seed": bright.get("seed"),
                    "faint_seed": faint.get("seed"),
                    "bright_spectrum_array_sha256": bright.get("spectrum_array_sha256", ""),
                    "faint_spectrum_array_sha256": faint.get("spectrum_array_sha256", ""),
                    "applicable": applicable,
                    "fit_completed": False,
                    "fit_valid": False,
                    "classification": "UNCLASSIFIABLE",
                    "reason": app_reason,
                    "measurement_value": np.nan,
                    "measurement_threshold": np.nan,
                }
                if not applicable:
                    rows.append(base)
                    continue
                if bright["job_status"] != "COMPLETED" or faint["job_status"] != "COMPLETED":
                    base["reason"] = "spectrum_fit_job_failed"
                    rows.append(base)
                    continue
                base["fit_completed"] = True
                if criterion == "YANG2024_FINAL":
                    result = apply_yang2024_hbeta(
                        bright=_audit_from_fit(bright), faint=_audit_from_fit(faint)
                    )
                    base["fit_valid"] = not result.is_unclassifiable
                    base["classification"] = _terminal(result.label.value)
                    base["reason"] = result.reason
                    base["measurement_value"] = (
                        result.statistic if result.statistic is not None else np.nan
                    )
                    base["measurement_threshold"] = 0.3
                elif criterion == "GREEN2022_FINAL":
                    measurement = measure_green_pixel_nsigma(
                        _green_from_fit(bright), _green_from_fit(faint)
                    )
                    result = classify_green_pixel_measurement(measurement)
                    base["fit_valid"] = measurement.measurement_available
                    base["classification"] = _terminal(result.label.value)
                    base["reason"] = result.reason
                    base["measurement_value"] = (
                        measurement.nsigma_hbeta if measurement.nsigma_hbeta is not None else np.nan
                    )
                    base["measurement_threshold"] = 3.0
                else:
                    base["reason"] = "visual_final_evidence_unavailable_frozen_non_denominator"
                rows.append(base)
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_OUTCOMES:
        raise RuntimeError(f"Outcome row mismatch: {len(out)} != {EXPECTED_OUTCOMES}")
    if not set(out["classification"]).issubset({"CL", "NON_CL", "UNCLASSIFIABLE"}):
        raise RuntimeError("Non-terminal classifier label detected")
    return out.sort_values(["condition_id", "realization", "criterion_id"]).reset_index(drop=True)


def production_qc(
    matrix: pd.DataFrame, tasks: pd.DataFrame, fits_df: pd.DataFrame, outcomes: pd.DataFrame
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["condition_count_167"] = matrix["condition_id"].nunique() == EXPECTED_CONDITIONS
    checks["condition_realizations_8350"] = (
        len(matrix) * REALIZATIONS == EXPECTED_CONDITION_REALIZATIONS
    )
    checks["fit_inputs_8412"] = len(tasks) == EXPECTED_FIT_INPUTS == len(fits_df)
    checks["outcome_rows_25050"] = len(outcomes) == EXPECTED_OUTCOMES
    checks["all_fit_jobs_completed"] = bool((fits_df["job_status"] == "COMPLETED").all())
    checks["all_outcomes_terminal"] = bool(
        outcomes["classification"].isin(["CL", "NON_CL", "UNCLASSIFIABLE"]).all()
    )
    checks["exact_realizations_per_condition_criterion"] = bool(
        outcomes.groupby(["condition_id", "criterion_id"]).size().eq(REALIZATIONS).all()
    )
    checks["seed_namespace_frozen"] = bool((tasks["seed_namespace"] == SEED_NAMESPACE).all())
    checks["degraded_seeds_unique"] = bool(
        ~tasks.loc[tasks["task_kind"] != "native_bright", "seed"].duplicated().any()
    )
    faint_crn = outcomes.groupby(["transition_id", "snr", "realization"])[
        ["faint_task_id", "faint_seed", "faint_spectrum_array_sha256"]
    ].nunique()
    checks["faint_crn_shared_across_arms"] = bool((faint_crn <= 1).all().all())
    faint_only = outcomes[outcomes["arm"] == "faint_only"]
    checks["faint_only_bright_native_unchanged"] = bool(
        faint_only["bright_task_id"].str.endswith("|native_bright").all()
    )
    expected_unapplicable = int((~outcomes["applicable"]).sum())
    checks["unapplicable_fail_closed"] = bool(
        (outcomes.loc[~outcomes["applicable"], "classification"] == "UNCLASSIFIABLE").all()
    )
    checks["no_missing_task_checksums"] = bool(
        fits_df["spectrum_array_sha256"].astype(str).str.len().eq(64).all()
    )
    checks["no_duplicate_fit_task_ids"] = not fits_df["task_id"].duplicated().any()
    reason_counts = (
        outcomes.loc[outcomes["classification"] == "UNCLASSIFIABLE", "reason"]
        .value_counts(dropna=False)
        .to_dict()
    )
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "conditions": int(matrix["condition_id"].nunique()),
            "condition_realizations": int(len(matrix) * REALIZATIONS),
            "fit_inputs": int(len(fits_df)),
            "fit_jobs_completed": int((fits_df["job_status"] == "COMPLETED").sum()),
            "fit_jobs_failed": int((fits_df["job_status"] != "COMPLETED").sum()),
            "outcome_rows": int(len(outcomes)),
            "unapplicable_outcome_rows": expected_unapplicable,
            "execution_reruns": int(
                json.loads((OUT / "engineering_repair_audit_d094.json").read_text()).get(
                    "affected_tasks_to_rerun", 0
                )
                if (OUT / "engineering_repair_audit_d094.json").exists()
                else 0
            ),
        },
        "fit_status_counts": fits_df["job_status"].value_counts(dropna=False).to_dict(),
        "classifier_status_counts": outcomes.groupby(["criterion_id", "classification"])
        .size()
        .to_dict(),
        "unclassifiable_reason_counts": reason_counts,
        "seed": {
            "base": SEED_BASE,
            "namespace": SEED_NAMESPACE,
            "common_random_numbers": "same faint task/seed/array reused across arms",
        },
    }
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--invalidate-wrong-sdss-mask-checkpoints", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    if (OUT / "FULL_Q1_EXECUTION_COMPLETE.json").exists():
        raise RuntimeError("D-094 is already complete and immutable; refusing to rerun")
    RAW.mkdir(parents=True, exist_ok=True)
    if args.invalidate_wrong_sdss_mask_checkpoints:
        print(json.dumps(invalidate_wrong_sdss_mask_checkpoints(), indent=2))
    matrix, endpoints, tasks = prepare_task_manifest()
    matrix.to_csv(RAW / "frozen_condition_matrix_d093.csv", index=False)
    endpoints.to_csv(RAW / "endpoint_bindings_d094.csv", index=False)
    tasks.to_csv(RAW / "fit_task_manifest_d094.csv", index=False)
    if args.prepare_only:
        print(
            json.dumps(
                {"conditions": len(matrix), "endpoints": len(endpoints), "fit_inputs": len(tasks)},
                indent=2,
            )
        )
        return 0

    fits_df = execute_fits(tasks, args.workers, args.batch_size)
    fit_path = RAW / "spectrum_fit_results_d094.parquet"
    _write_atomic_parquet(fits_df, fit_path)
    outcomes = build_outcomes(matrix, fits_df)
    outcome_path = RAW / "classifier_outcomes_d094.parquet"
    _write_atomic_parquet(outcomes, outcome_path)
    scalar_outcomes = outcomes.drop(columns=[], errors="ignore")
    scalar_outcomes.to_csv(RAW / "classifier_outcomes_d094.csv", index=False)

    checksum_rows = []
    for path in sorted(
        [
            fit_path,
            outcome_path,
            RAW / "classifier_outcomes_d094.csv",
            RAW / "fit_task_manifest_d094.csv",
            RAW / "endpoint_bindings_d094.csv",
            RAW / "frozen_condition_matrix_d093.csv",
        ]
    ):
        checksum_rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    checksums = pd.DataFrame(checksum_rows)
    checksums.to_csv(RAW / "raw_checksums_d094.csv", index=False)

    qc = production_qc(matrix, tasks, fits_df, outcomes)
    qc["raw_checksums"] = checksum_rows
    qc_path = OUT / "production_qc_d094.json"
    qc_path.write_text(json.dumps(_json_safe(qc), indent=2, sort_keys=True) + "\n")
    if qc["status"] != "PASS":
        print(json.dumps(_json_safe(qc), indent=2), file=sys.stderr)
        return 2
    completion = {
        "decision": "FULL_Q1_EXECUTION_COMPLETE",
        "production_qc": "PASS",
        "raw_products_frozen": True,
        "raw_checksum_manifest": str((RAW / "raw_checksums_d094.csv").relative_to(ROOT)),
        "conditions": EXPECTED_CONDITIONS,
        "condition_realizations": EXPECTED_CONDITION_REALIZATIONS,
        "fit_inputs": EXPECTED_FIT_INPUTS,
        "classifier_outcomes": EXPECTED_OUTCOMES,
        "seed_base": SEED_BASE,
        "seed_namespace": SEED_NAMESPACE,
        "realizations": REALIZATIONS,
    }
    (OUT / "FULL_Q1_EXECUTION_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
