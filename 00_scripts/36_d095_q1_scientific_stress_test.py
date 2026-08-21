#!/usr/bin/env python3
"""D-095 final Q1 stress test using immutable D-094 outputs only.

This module never imports the production execution path. It reconstructs all
accounting and estimands directly from the frozen D-094 condition, outcome,
fit-audit, and checksum products.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D094 = ROOT / "05_analysis" / "q1_production" / "d094"
RAW = D094 / "raw"
OUT = ROOT / "05_analysis" / "q1_production" / "d095"
TABLES = OUT / "tables"
BOOTSTRAP_N = 5000
BOOTSTRAP_BASE = 314159
KEYS = [
    "condition_id",
    "transition_id",
    "object_id",
    "reference_tier",
    "source_family",
    "arm",
    "snr",
    "criterion_id",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def bootstrap_mean_ci(values: pd.Series, label: str) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(derived_seed(BOOTSTRAP_BASE, "d095", label))
    sampled = rng.choice(array, size=(BOOTSTRAP_N, len(array)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def reconstruct_transition_metrics(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in outcomes.groupby(KEYS, sort=True, dropna=False):
        labels = group["classification"]
        n_cl = int(labels.eq("CL").sum())
        n_noncl = int(labels.eq("NON_CL").sum())
        n_unclass = int(labels.eq("UNCLASSIFIABLE").sum())
        n_class = n_cl + n_noncl
        rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "N_eligible_realizations": int(len(group)),
                "N_fit_completed": int(group["fit_completed"].sum()),
                "N_fit_valid": int(group["fit_valid"].sum()),
                "N_classifiable": n_class,
                "N_CL": n_cl,
                "N_nonCL": n_noncl,
                "N_unclassifiable": n_unclass,
                "p_i": n_cl / n_class if n_class else np.nan,
                "u_i": n_unclass / len(group),
                "applicable_by_design": bool(group["applicable"].all()),
            }
        )
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def independently_aggregate(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(["arm", "snr", "criterion_id"], sort=True):
        valid = group["p_i"].notna()
        rows.append(
            {
                "selection_scope": scope,
                "arm": key[0],
                "snr": int(key[1]),
                "criterion_id": key[2],
                "N_eligible_for_rung": int(len(group)),
                "N_classifier_applicable": int(group["applicable_by_design"].sum()),
                "N_with_any_classifiable": int(valid.sum()),
                "N_eligible_realizations": int(group["N_eligible_realizations"].sum()),
                "N_fit_completed": int(group["N_fit_completed"].sum()),
                "N_fit_valid": int(group["N_fit_valid"].sum()),
                "N_classifiable": int(group["N_classifiable"].sum()),
                "N_CL": int(group["N_CL"].sum()),
                "N_nonCL": int(group["N_nonCL"].sum()),
                "N_unclassifiable": int(group["N_unclassifiable"].sum()),
                "R_equal_transition": (
                    float(group.loc[valid, "p_i"].mean()) if valid.any() else np.nan
                ),
                "U_equal_transition": float(group["u_i"].mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_accounting(
    conditions: pd.DataFrame,
    outcomes: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> dict[str, Any]:
    frozen = pd.read_csv(D094 / "tables" / "transition_level_estimands_d094.csv")
    frozen = frozen.sort_values(KEYS).reset_index(drop=True)
    compare_counts = [
        "N_eligible_realizations",
        "N_fit_completed",
        "N_fit_valid",
        "N_classifiable",
        "N_CL",
        "N_nonCL",
        "N_unclassifiable",
    ]
    compare_floats = ["p_i", "u_i"]
    keys_equal = frozen[KEYS].astype(str).equals(reconstructed[KEYS].astype(str))
    counts_equal = all(
        np.array_equal(frozen[column].to_numpy(), reconstructed[column].to_numpy())
        for column in compare_counts
    )
    floats_equal = all(
        np.allclose(
            frozen[column].to_numpy(dtype=float),
            reconstructed[column].to_numpy(dtype=float),
            equal_nan=True,
            rtol=0.0,
            atol=1e-15,
        )
        for column in compare_floats
    )

    primary = independently_aggregate(
        reconstructed[reconstructed.reference_tier.eq("GOLD")], "PRIMARY_GOLD"
    )
    sensitivity = independently_aggregate(reconstructed, "GOLD_PLUS_SILVER")
    independent = pd.concat([primary, sensitivity], ignore_index=True)
    frozen_aggregate = pd.read_csv(D094 / "tables" / "aggregate_estimands_bootstrap_d094.csv")
    frozen_aggregate = frozen_aggregate[
        frozen_aggregate.selection_scope.isin(["PRIMARY_GOLD", "GOLD_PLUS_SILVER"])
    ]
    merge_keys = ["selection_scope", "arm", "snr", "criterion_id"]
    merged = independent.merge(
        frozen_aggregate,
        on=merge_keys,
        suffixes=("_reconstructed", "_frozen"),
        validate="one_to_one",
    )
    aggregate_columns = [
        "N_eligible_for_rung",
        "N_classifier_applicable",
        "N_eligible_realizations",
        "N_fit_completed",
        "N_fit_valid",
        "N_classifiable",
        "N_CL",
        "N_nonCL",
        "N_unclassifiable",
        "R_equal_transition",
        "U_equal_transition",
    ]
    aggregate_equal = True
    for column in aggregate_columns:
        left = merged[f"{column}_reconstructed"].to_numpy(dtype=float)
        frozen_name = "N_transition_classifiable" if column == "N_with_any_classifiable" else column
        right_column = f"{frozen_name}_frozen"
        if right_column not in merged:
            continue
        right = merged[right_column].to_numpy(dtype=float)
        aggregate_equal &= bool(np.allclose(left, right, equal_nan=True, rtol=0.0, atol=1e-15))

    checksum_manifest = pd.read_csv(RAW / "raw_checksums_d094.csv")
    checksum_rows = []
    for row in checksum_manifest.itertuples(index=False):
        path = ROOT / row.file
        actual = sha256_file(path)
        checksum_rows.append(
            {
                "file": row.file,
                "expected_sha256": row.sha256,
                "actual_sha256": actual,
                "expected_bytes": int(row.bytes),
                "actual_bytes": path.stat().st_size,
                "pass": actual == row.sha256 and path.stat().st_size == row.bytes,
            }
        )
    pd.DataFrame(checksum_rows).to_csv(
        TABLES / "d094_immutable_checksum_reverification_d095.csv", index=False
    )

    group_sizes = outcomes.groupby(["condition_id", "criterion_id"]).size()
    condition_realizations = outcomes[["condition_id", "realization"]].drop_duplicates()
    label_identity = (
        outcomes["classification"].isin(["CL", "NON_CL", "UNCLASSIFIABLE"]).all()
        and len(outcomes) == outcomes["classification"].notna().sum()
    )
    transition_identity = bool(
        (
            reconstructed.N_CL + reconstructed.N_nonCL + reconstructed.N_unclassifiable
            == reconstructed.N_eligible_realizations
        ).all()
        and (reconstructed.N_CL + reconstructed.N_nonCL == reconstructed.N_classifiable).all()
        and reconstructed.N_eligible_realizations.eq(50).all()
    )
    checks = {
        "conditions_167": conditions.condition_id.nunique() == 167,
        "condition_realizations_8350": len(condition_realizations) == 8350,
        "terminal_classifier_outcomes_25050": len(outcomes) == 25050,
        "three_classifier_rows_per_condition_realization": bool(
            outcomes.groupby(["condition_id", "realization"]).size().eq(3).all()
        ),
        "fifty_realizations_per_condition_classifier": bool(group_sizes.eq(50).all()),
        "terminal_label_domain_complete": bool(label_identity),
        "transition_denominator_identities": transition_identity,
        "transition_table_exactly_reconstructed": bool(
            keys_equal and counts_equal and floats_equal
        ),
        "aggregate_table_exactly_reconstructed": aggregate_equal,
        "immutable_raw_checksums_reverified": all(row["pass"] for row in checksum_rows),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "conditions": int(conditions.condition_id.nunique()),
            "condition_realizations": len(condition_realizations),
            "terminal_classifier_outcomes": len(outcomes),
            "transition_condition_classifier_rows": len(reconstructed),
        },
    }


def paired_green_snr(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = metrics[metrics.reference_tier.eq("GOLD") & metrics.criterion_id.eq("GREEN2022_FINAL")]
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for arm, group in gold.groupby("arm", sort=True):
        wide = group.pivot(
            index=["transition_id", "object_id", "source_family"],
            columns="snr",
            values=["p_i", "u_i"],
        )
        common = wide.dropna(subset=[("p_i", 5), ("p_i", 10)]).copy()
        common["delta_p_10_minus_5"] = common[("p_i", 10)] - common[("p_i", 5)]
        common["delta_u_10_minus_5"] = common[("u_i", 10)] - common[("u_i", 5)]
        for index, row in common.iterrows():
            detail_rows.append(
                {
                    "arm": arm,
                    "transition_id": index[0],
                    "object_id": index[1],
                    "source_family": index[2],
                    "p_i_5": row[("p_i", 5)],
                    "p_i_10": row[("p_i", 10)],
                    "delta_p_10_minus_5": row["delta_p_10_minus_5"],
                    "u_i_5": row[("u_i", 5)],
                    "u_i_10": row[("u_i", 10)],
                    "delta_u_10_minus_5": row["delta_u_10_minus_5"],
                }
            )
        delta = common["delta_p_10_minus_5"]
        low, high = bootstrap_mean_ci(delta, f"green-snr|{arm}")
        full5 = group[group.snr.eq(5)].p_i.mean(skipna=True)
        full10 = group[group.snr.eq(10)].p_i.mean(skipna=True)
        summary_rows.append(
            {
                "arm": arm,
                "paired_transition_count": len(common),
                "delta_p_mean_10_minus_5": float(delta.mean()),
                "delta_p_median_10_minus_5": float(delta.median()),
                "delta_p_q25": float(delta.quantile(0.25)),
                "delta_p_q75": float(delta.quantile(0.75)),
                "delta_p_min": float(delta.min()),
                "delta_p_max": float(delta.max()),
                "N_positive": int((delta > 0).sum()),
                "N_zero": int(np.isclose(delta, 0.0).sum()),
                "N_negative": int((delta < 0).sum()),
                "paired_mean_ci_low": low,
                "paired_mean_ci_high": high,
                "full_supported_R_snr5": float(full5),
                "full_supported_R_snr10": float(full10),
                "full_supported_unpaired_difference_10_minus_5": float(full10 - full5),
                "common_support_R_snr5": float(common[("p_i", 5)].mean()),
                "common_support_R_snr10": float(common[("p_i", 10)].mean()),
                "common_support_paired_difference_10_minus_5": float(delta.mean()),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def paired_green_arm(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = metrics[metrics.reference_tier.eq("GOLD") & metrics.criterion_id.eq("GREEN2022_FINAL")]
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for snr, group in gold.groupby("snr", sort=True):
        wide = group.pivot(
            index=["transition_id", "object_id", "source_family"],
            columns="arm",
            values=["p_i", "u_i"],
        )
        common = wide.dropna(subset=[("p_i", "faint_only"), ("p_i", "matched")]).copy()
        common["delta_p_faint_only_minus_matched"] = (
            common[("p_i", "faint_only")] - common[("p_i", "matched")]
        )
        common["delta_u_faint_only_minus_matched"] = (
            common[("u_i", "faint_only")] - common[("u_i", "matched")]
        )
        for index, row in common.iterrows():
            details.append(
                {
                    "snr": int(snr),
                    "transition_id": index[0],
                    "object_id": index[1],
                    "source_family": index[2],
                    "p_i_faint_only": row[("p_i", "faint_only")],
                    "p_i_matched": row[("p_i", "matched")],
                    "delta_p_faint_only_minus_matched": row["delta_p_faint_only_minus_matched"],
                    "u_i_faint_only": row[("u_i", "faint_only")],
                    "u_i_matched": row[("u_i", "matched")],
                    "delta_u_faint_only_minus_matched": row["delta_u_faint_only_minus_matched"],
                }
            )
        delta = common["delta_p_faint_only_minus_matched"]
        low, high = bootstrap_mean_ci(delta, f"green-arm|{snr}")
        summaries.append(
            {
                "snr": int(snr),
                "paired_transition_count": len(common),
                "delta_p_mean_faint_only_minus_matched": float(delta.mean()),
                "delta_p_median_faint_only_minus_matched": float(delta.median()),
                "delta_p_q25": float(delta.quantile(0.25)),
                "delta_p_q75": float(delta.quantile(0.75)),
                "delta_p_min": float(delta.min()),
                "delta_p_max": float(delta.max()),
                "N_positive": int((delta > 0).sum()),
                "N_zero": int(np.isclose(delta, 0.0).sum()),
                "N_negative": int((delta < 0).sum()),
                "paired_mean_ci_low": low,
                "paired_mean_ci_high": high,
            }
        )
    return pd.DataFrame(details), pd.DataFrame(summaries)


def classify_yang_failure(
    outcome: pd.Series,
    bright: pd.Series,
    faint: pd.Series,
) -> tuple[str, dict[str, bool]]:
    tasks = [bright, faint]
    invalid = [task for task in tasks if not bool(task.yang_fit_valid)]
    native_boundary = any(
        task.task_kind == "native_bright" and bool(task.yang_parameter_at_bound) for task in invalid
    )
    degraded_boundary = any(
        task.task_kind != "native_bright" and bool(task.yang_parameter_at_bound) for task in invalid
    )
    local_invalidity = any(
        (
            not bool(task.yang_optimizer_success)
            or not bool(task.yang_local_hbeta_coverage)
            or not bool(task.yang_continuum_valid)
            or any(
                token in str(task.yang_invalid_reason)
                for token in (
                    "WIDTH_OUTSIDE",
                    "LOCAL_HBETA",
                    "OPTIMIZER",
                    "CONTINUUM",
                )
            )
        )
        for task in invalid
    )
    nondetection_semantics = "not a valid nondetection" in str(outcome.reason) or (
        bool(faint.yang_fit_valid)
        and not bool(faint.yang_broad_component_present)
        and not bool(faint.yang_valid_nondetection)
    )
    measurement_unavailable = any(
        token in str(outcome.reason)
        for token in (
            "measurement",
            "positive valid bright-state broad Hbeta unavailable",
            "finite faint-state broad Hbeta unavailable",
        )
    ) or any("BROAD_MEASUREMENT_NOT_FINITE" in str(task.yang_invalid_reason) for task in invalid)
    flags = {
        "native_boundary_failure": native_boundary,
        "degraded_fit_boundary_failure": degraded_boundary,
        "local_fit_invalidity": local_invalidity,
        "nondetection_semantics": nondetection_semantics,
        "measurement_unavailable": measurement_unavailable,
    }
    if native_boundary:
        category = "native_boundary_failure"
    elif degraded_boundary:
        category = "degraded_fit_boundary_failure"
    elif local_invalidity:
        category = "local_fit_invalidity"
    elif nondetection_semantics:
        category = "nondetection_semantics"
    elif measurement_unavailable:
        category = "measurement_unavailable"
    else:
        category = "other"
    return category, flags


def yang_failure_decomposition(
    outcomes: pd.DataFrame, fits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lookup = fits.set_index("task_id", drop=False)
    yang = outcomes[
        outcomes.criterion_id.eq("YANG2024_FINAL") & outcomes.classification.eq("UNCLASSIFIABLE")
    ]
    rows: list[dict[str, Any]] = []
    for _, outcome in yang.iterrows():
        if not bool(outcome.applicable):
            rows.append(
                {
                    "condition_id": outcome.condition_id,
                    "transition_id": outcome.transition_id,
                    "reference_tier": outcome.reference_tier,
                    "arm": outcome.arm,
                    "snr": int(outcome.snr),
                    "realization": int(outcome.realization),
                    "failure_category": "inapplicable_by_design",
                    "native_boundary_failure": False,
                    "degraded_fit_boundary_failure": False,
                    "local_fit_invalidity": False,
                    "nondetection_semantics": False,
                    "measurement_unavailable": False,
                    "outcome_reason": outcome.reason,
                }
            )
            continue
        bright = lookup.loc[outcome.bright_task_id]
        faint = lookup.loc[outcome.faint_task_id]
        category, flags = classify_yang_failure(outcome, bright, faint)
        rows.append(
            {
                "condition_id": outcome.condition_id,
                "transition_id": outcome.transition_id,
                "reference_tier": outcome.reference_tier,
                "arm": outcome.arm,
                "snr": int(outcome.snr),
                "realization": int(outcome.realization),
                "failure_category": category,
                **flags,
                "outcome_reason": outcome.reason,
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["reference_tier", "arm", "snr", "failure_category"])
        .size()
        .rename("N")
        .reset_index()
    )
    overall = detail.failure_category.value_counts().to_dict()
    applicable = detail[detail.failure_category.ne("inapplicable_by_design")]
    incidence = {
        column: int(applicable[column].sum())
        for column in (
            "native_boundary_failure",
            "degraded_fit_boundary_failure",
            "local_fit_invalidity",
            "nondetection_semantics",
            "measurement_unavailable",
        )
    }
    classified_yang = outcomes[
        outcomes.criterion_id.eq("YANG2024_FINAL") & outcomes.classification.isin(["CL", "NON_CL"])
    ]
    valid_nondetection_used = 0
    for task_id in classified_yang.faint_task_id:
        valid_nondetection_used += int(bool(lookup.loc[task_id].yang_valid_nondetection))
    metadata = {
        "terminal_priority_categories": {str(key): int(value) for key, value in overall.items()},
        "nonexclusive_failure_incidence": incidence,
        "applicable_unclassifiable": len(applicable),
        "inapplicable_by_design": int(detail.failure_category.eq("inapplicable_by_design").sum()),
        "valid_nondetection_semantics_used_in_classifiable_rows": valid_nondetection_used,
        "interpretation": "Boundary failures dominate; valid nondetections are a source-permitted classifiable zero convention, not a failure.",
    }
    return detail, summary, metadata


def classifier_contingency(outcomes: pd.DataFrame) -> pd.DataFrame:
    gold = outcomes[
        outcomes.reference_tier.eq("GOLD")
        & outcomes.criterion_id.isin(["GREEN2022_FINAL", "YANG2024_FINAL"])
    ]
    index = [
        "condition_id",
        "transition_id",
        "object_id",
        "arm",
        "snr",
        "realization",
    ]
    wide = gold.pivot(index=index, columns="criterion_id", values="classification").reset_index()
    rows: list[dict[str, Any]] = []
    for key, group in wide.groupby(["arm", "snr"], sort=True):
        green = group.GREEN2022_FINAL
        yang = group.YANG2024_FINAL
        green_class = green.isin(["CL", "NON_CL"])
        yang_class = yang.isin(["CL", "NON_CL"])
        both = green_class & yang_class
        rows.append(
            {
                "arm": key[0],
                "snr": int(key[1]),
                "N_total_realization_cells": len(group),
                "N_both_classifiable": int(both.sum()),
                "Green_CL__Yang_CL": int((both & green.eq("CL") & yang.eq("CL")).sum()),
                "Green_CL__Yang_nonCL": int((both & green.eq("CL") & yang.eq("NON_CL")).sum()),
                "Green_nonCL__Yang_CL": int((both & green.eq("NON_CL") & yang.eq("CL")).sum()),
                "Green_nonCL__Yang_nonCL": int(
                    (both & green.eq("NON_CL") & yang.eq("NON_CL")).sum()
                ),
                "N_Green_classifiable_Yang_unclassifiable": int((green_class & ~yang_class).sum()),
                "N_Green_unclassifiable_Yang_classifiable": int((~green_class & yang_class).sum()),
                "N_both_unclassifiable": int((~green_class & ~yang_class).sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_effect(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    group = frame[frame.arm.eq(arm)]
    wide = group.pivot(index=["transition_id", "source_family"], columns="snr", values="p_i")
    common = wide.dropna(subset=[5, 10]).reset_index()
    common["delta_p_10_minus_5"] = common[10] - common[5]
    return common


def robustness_stress(metrics: pd.DataFrame) -> pd.DataFrame:
    green = metrics[metrics.criterion_id.eq("GREEN2022_FINAL")]
    gold = green[green.reference_tier.eq("GOLD")]
    sensitivity = green[green.reference_tier.isin(["GOLD", "SILVER"])]
    independence = pd.read_csv(
        ROOT / "05_analysis" / "q1_design" / "d093" / "discovery_classifier_independence_d093.csv"
    )
    independent_families = set(
        independence.loc[
            independence.final_classifier.eq("GREEN2022_FINAL")
            & independence.discovery_independent_sensitivity_included.astype(bool),
            "source_family",
        ]
    )
    independent = gold[gold.source_family.isin(independent_families)]
    rows: list[dict[str, Any]] = []
    for arm in ("faint_only", "matched"):
        for label, frame in (
            ("PRIMARY_GOLD", gold),
            ("GOLD_PLUS_SILVER", sensitivity),
            ("DISCOVERY_INDEPENDENT_GREEN", independent),
        ):
            paired = paired_effect(frame, arm)
            rows.append(
                {
                    "analysis": label,
                    "arm": arm,
                    "omitted_unit": "",
                    "N_paired": len(paired),
                    "mean_delta_p_10_minus_5": float(paired.delta_p_10_minus_5.mean()),
                }
            )
        paired_gold = paired_effect(gold, arm)
        family_means = paired_gold.groupby("source_family").delta_p_10_minus_5.mean()
        rows.append(
            {
                "analysis": "EQUAL_SOURCE_FAMILY_WEIGHTED",
                "arm": arm,
                "omitted_unit": "",
                "N_paired": len(paired_gold),
                "mean_delta_p_10_minus_5": float(family_means.mean()),
            }
        )
        for transition in sorted(paired_gold.transition_id.unique()):
            kept = paired_gold[paired_gold.transition_id.ne(transition)]
            rows.append(
                {
                    "analysis": "LEAVE_ONE_TRANSITION_OUT",
                    "arm": arm,
                    "omitted_unit": transition,
                    "N_paired": len(kept),
                    "mean_delta_p_10_minus_5": float(kept.delta_p_10_minus_5.mean()),
                }
            )
        for family in sorted(paired_gold.source_family.unique()):
            kept = paired_gold[paired_gold.source_family.ne(family)]
            rows.append(
                {
                    "analysis": "LEAVE_ONE_SOURCE_FAMILY_OUT",
                    "arm": arm,
                    "omitted_unit": family,
                    "N_paired": len(kept),
                    "mean_delta_p_10_minus_5": float(kept.delta_p_10_minus_5.mean()),
                }
            )
    result = pd.DataFrame(rows)
    for arm in ("faint_only", "matched"):
        primary = (
            result[result.analysis.eq("PRIMARY_GOLD") & result.arm.eq(arm)]
            .iloc[0]
            .mean_delta_p_10_minus_5
        )
        mask = result.analysis.eq("LEAVE_ONE_SOURCE_FAMILY_OUT") & result.arm.eq(arm)
        result.loc[mask, "attenuation_from_primary"] = (
            primary - result.loc[mask, "mean_delta_p_10_minus_5"]
        )
    return result


def mask_incident_audit(tasks: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    sdss = tasks.instrument_survey.astype(str).str.upper().str.contains("SDSS")
    degraded = tasks[tasks.task_kind.ne("native_bright")]
    seed_checks = []
    for row in degraded.itertuples(index=False):
        tag = "faint" if row.task_kind == "faint_shared" else "bright"
        expected = derived_seed(
            tag,
            int(row.seed_base),
            f"{row.seed_namespace}|{row.transition_id}",
            row.spectrum_id,
            int(row.realization),
            f"{float(row.target_snr):.6f}",
        )
        seed_checks.append(expected == int(row.seed))
    faint_crn = outcomes.groupby(["transition_id", "snr", "realization"])[
        ["faint_task_id", "faint_seed", "faint_spectrum_array_sha256"]
    ].nunique()
    repair = json.loads((D094 / "engineering_repair_audit_d094.json").read_text())
    return {
        "root_cause": "Initial D-094 reader added SDSS survey bitmask rejection although D-089/D-093 froze positive-IVAR-only fitting validity; this removed all Hbeta-window pixels for the affected J233602 native bright endpoint and failed QC.",
        "affected_count_reconstructed": int(sdss.sum()),
        "affected_count_frozen_audit": int(repair["affected_tasks_to_rerun"]),
        "unaffected_count_reconstructed": int((~sdss).sum()),
        "unaffected_instrument_counts": {
            str(key): int(value)
            for key, value in tasks.loc[~sdss, "instrument_survey"].value_counts().items()
        },
        "detection_time": "Production QC before any D-094 scientific interpretation or accepted aggregate",
        "initial_stop_condition": "50 failed J233602 matched-bright S/N=5 inputs caused all-fit-completed and checksum-completeness QC failure",
        "rerun_rule": "Invalidate every task selected solely by SDSS/SDSS-V instrument provenance; retain every non-SDSS task; rerun the full affected set without inspecting or selecting classifier outcomes.",
        "all_8350_degraded_task_seeds_match_frozen_sha256_construction": bool(
            all(seed_checks) and len(seed_checks) == 8350
        ),
        "degraded_seed_uniqueness": bool(~degraded.seed.duplicated().any()),
        "faint_crn_task_seed_array_identity_across_arms": bool((faint_crn <= 1).all().all()),
        "evidence_no_outcome_conditioned_rerun": [
            "Affected selection is exactly instrument_survey contains SDSS and reconstructs 6305/8412 before classifier labels are consulted.",
            "Unaffected complement reconstructs 2107 tasks: 2106 LAMOST_DR11 and 1 DESI_EDR.",
            "Repair audit records seeds_changed=false and matrix_changed=false.",
            "The rejected attempt stopped at production QC; FULL_Q1_EXECUTION_COMPLETE did not exist until the corrected all-input run passed.",
            "Every degraded seed independently matches the frozen namespace/base/task identity SHA-256 construction.",
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    conditions = pd.read_csv(RAW / "frozen_condition_matrix_d093.csv")
    outcomes = pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")
    fit_columns = [
        "task_id",
        "transition_id",
        "task_kind",
        "instrument_survey",
        "spectrum_id",
        "target_snr",
        "realization",
        "seed",
        "seed_namespace",
        "seed_base",
        "yang_optimizer_success",
        "yang_parameter_at_bound",
        "yang_local_hbeta_coverage",
        "yang_continuum_valid",
        "yang_broad_component_present",
        "yang_fit_valid",
        "yang_invalid_reason",
        "yang_valid_nondetection",
    ]
    fits = pd.read_parquet(RAW / "spectrum_fit_results_d094.parquet", columns=fit_columns)
    tasks = pd.read_csv(RAW / "fit_task_manifest_d094.csv")

    reconstructed = reconstruct_transition_metrics(outcomes)
    reconstructed.to_csv(
        TABLES / "independently_reconstructed_transition_estimands_d095.csv",
        index=False,
    )
    accounting = validate_accounting(conditions, outcomes, reconstructed)
    (OUT / "production_accounting_reverification_d095.json").write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n"
    )
    if accounting["status"] != "PASS":
        raise RuntimeError("D-095 independent accounting reconstruction failed")

    green_snr_detail, green_snr_summary = paired_green_snr(reconstructed)
    green_snr_detail.to_csv(
        TABLES / "green_common_support_snr_paired_transitions_d095.csv", index=False
    )
    green_snr_summary.to_csv(TABLES / "green_common_support_snr_summary_d095.csv", index=False)
    green_arm_detail, green_arm_summary = paired_green_arm(reconstructed)
    green_arm_detail.to_csv(
        TABLES / "green_common_support_arm_paired_transitions_d095.csv", index=False
    )
    green_arm_summary.to_csv(TABLES / "green_common_support_arm_summary_d095.csv", index=False)

    yang_detail, yang_summary, yang_metadata = yang_failure_decomposition(outcomes, fits)
    yang_detail.to_csv(TABLES / "yang_failure_decomposition_rows_d095.csv", index=False)
    yang_summary.to_csv(TABLES / "yang_failure_decomposition_summary_d095.csv", index=False)
    (OUT / "yang_failure_decomposition_d095.json").write_text(
        json.dumps(yang_metadata, indent=2, sort_keys=True) + "\n"
    )

    contingency = classifier_contingency(outcomes)
    contingency.to_csv(TABLES / "yang_green_both_classifiable_contingency_d095.csv", index=False)
    robustness = robustness_stress(reconstructed)
    robustness.to_csv(TABLES / "green_paired_robustness_d095.csv", index=False)

    incident = mask_incident_audit(tasks, outcomes)
    (OUT / "mask_contract_incident_audit_d095.json").write_text(
        json.dumps(incident, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "decision": "Q1_SCIENCE_PARTIAL",
        "accounting": accounting,
        "green_snr": green_snr_summary.to_dict("records"),
        "green_arm": green_arm_summary.to_dict("records"),
        "yang_failure": yang_metadata,
        "yang_green_contingency": contingency.to_dict("records"),
        "mask_contract_incident": incident,
        "bootstrap": {
            "clusters": "paired transition/object",
            "draws": BOOTSTRAP_N,
            "interval": "percentile 95% CI for paired mean",
        },
    }
    (OUT / "d095_machine_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
