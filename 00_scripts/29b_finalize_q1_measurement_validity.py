#!/usr/bin/env python
"""Finalize D-089 saved pilot audits without regenerating spectra or noise."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3sf.criteria.source_verified import apply_yang2024_hbeta  # noqa: E402
from p3sf.fitting.pyqsofit_driver import (  # noqa: E402
    YANG_BROAD_FWHM_RANGE_KMS,
    YangHbetaFitAudit,
)

OUT = ROOT / "05_analysis/q1_design/measurement_validity_d089"


def _load_summary_function():
    path = ROOT / "00_scripts/29_q1_measurement_validity.py"
    spec = importlib.util.spec_from_file_location("q1_measurement_validity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load D-089 pilot module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def _audit_from_row(row: pd.Series) -> YangHbetaFitAudit:
    values: dict[str, object] = {}
    for field in fields(YangHbetaFitAudit):
        value = row[field.name]
        if field.name in {
            "broad_component_amplitude",
            "broad_component_width",
            "broad_flux",
            "residual_rms_local",
        } and pd.isna(value):
            value = None
        elif field.name == "invalid_reason" and pd.isna(value):
            value = ""
        values[field.name] = value
    return YangHbetaFitAudit(**values)  # type: ignore[arg-type]


def _enforce_saved_source_width_contract(audits: pd.DataFrame) -> pd.DataFrame:
    result = audits.copy()
    width = pd.to_numeric(result.broad_component_width, errors="coerce")
    upper = YANG_BROAD_FWHM_RANGE_KMS[1]
    outside = (
        result.broad_component_present.astype(bool)
        & width.notna()
        & ((width < YANG_BROAD_FWHM_RANGE_KMS[0]) | ((width > upper) & ~np.isclose(width, upper, rtol=1.0e-5)))
    )
    reason = "BROAD_COMPONENT_WIDTH_OUTSIDE_YANG_SOURCE_RANGE"
    for index in result.index[outside]:
        old = "" if pd.isna(result.at[index, "invalid_reason"]) else str(result.at[index, "invalid_reason"])
        parts = [part for part in old.split("|") if part]
        if reason not in parts:
            parts.append(reason)
        result.at[index, "invalid_reason"] = "|".join(parts)
        result.at[index, "fit_valid"] = False
        result.at[index, "valid_nondetection"] = False
    return result


def _unique_audit(
    audits: pd.DataFrame,
    *,
    transition_id: str,
    arm: str,
    rung: float | None = None,
    realization: int | None = None,
    fit_role: str | None = None,
) -> YangHbetaFitAudit:
    selected = audits[(audits.transition_id == transition_id) & (audits.arm == arm)]
    if rung is not None:
        selected = selected[selected.snr_rung == rung]
    if realization is not None:
        selected = selected[selected.realization == realization]
    if fit_role is not None:
        selected = selected[selected.fit_role == fit_role]
    if len(selected) != 1:
        raise RuntimeError(f"expected one Yang audit, found {len(selected)}")
    return _audit_from_row(selected.iloc[0])


def _reclassify_yang(stability: pd.DataFrame, audits: pd.DataFrame) -> pd.DataFrame:
    result = stability.copy()
    for index, row in result[result.criterion == "YANG2024_FINAL"].iterrows():
        object_id = str(row.transition_id)
        rung = float(row.snr_rung)
        realization = int(row.realization)
        faint = _unique_audit(
            audits, transition_id=object_id, arm="shared", rung=rung,
            realization=realization,
        )
        if row.arm == "faint_only":
            bright = _unique_audit(
                audits, transition_id=object_id, arm="native", fit_role="native_bright"
            )
        else:
            bright = _unique_audit(
                audits, transition_id=object_id, arm="matched", rung=rung,
                realization=realization,
            )
        classification = apply_yang2024_hbeta(bright=bright, faint=faint)
        result.at[index, "label"] = str(classification.label)
        result.at[index, "measurement"] = classification.statistic
        result.at[index, "measurement_available"] = not classification.is_unclassifiable
        result.at[index, "invalid_reason"] = classification.reason
    return result


def _convergence(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary[[
        "transition_id", "tier", "snr_rung", "arm", "criterion",
        "prefix_recovery_M40", "prefix_recovery_M50",
        "prefix_unclassifiable_M40", "prefix_unclassifiable_M50",
    ]].copy()
    result["abs_delta_recovery_M50_M40"] = (
        result.prefix_recovery_M50 - result.prefix_recovery_M40
    ).abs()
    result["abs_delta_unclassifiable_M50_M40"] = (
        result.prefix_unclassifiable_M50 - result.prefix_unclassifiable_M40
    ).abs()
    both_recovery_undefined = (
        result.prefix_recovery_M40.isna() & result.prefix_recovery_M50.isna()
    )
    result["recovery_rule_pass"] = (
        result.abs_delta_recovery_M50_M40.le(0.10) | both_recovery_undefined
    )
    result["unclassifiable_rule_pass"] = result.abs_delta_unclassifiable_M50_M40.le(0.10)
    result["group_pass"] = result.recovery_rule_pass & result.unclassifiable_rule_pass
    return result


def main() -> int:
    audits = _enforce_saved_source_width_contract(
        pd.read_csv(OUT / "yang_fit_audit_rows.csv")
    )
    stability = _reclassify_yang(
        pd.read_csv(OUT / "stability_rows.csv"), audits
    )
    summarize = _load_summary_function()
    summary = summarize(stability)
    convergence = _convergence(summary)
    audits.to_csv(OUT / "yang_fit_audit_rows.csv", index=False)
    audits[audits.arm == "native"].to_csv(OUT / "yang_native_fit_audit.csv", index=False)
    stability.to_csv(OUT / "stability_rows.csv", index=False)
    summary.to_csv(OUT / "stability_summary.csv", index=False)
    convergence.to_csv(OUT / "convergence_m40_m50.csv", index=False)
    if len(summary) != 16 or not convergence.group_pass.all():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
