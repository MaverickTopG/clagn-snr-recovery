#!/usr/bin/env python3
"""Analyze frozen D-094 Q1 products at the transition/object level."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_analysis" / "q1_production" / "d094"
RAW = OUT / "raw"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
BOOTSTRAP_N = 5000
BOOTSTRAP_BASE = 314159


def _seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def transition_metrics(outcomes: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "condition_id",
        "transition_id",
        "object_id",
        "reference_tier",
        "source_family",
        "arm",
        "snr",
        "criterion_id",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in outcomes.groupby(keys, sort=True, dropna=False):
        labels = group["classification"]
        n_cl = int(labels.eq("CL").sum())
        n_noncl = int(labels.eq("NON_CL").sum())
        n_unclass = int(labels.eq("UNCLASSIFIABLE").sum())
        n_class = n_cl + n_noncl
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "N_eligible_realizations": len(group),
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
    result = pd.DataFrame(rows)
    if not result["N_eligible_realizations"].eq(50).all():
        raise RuntimeError("Transition summaries do not all contain frozen M=50")
    return result


def _bootstrap_equal_transition(group: pd.DataFrame, label: str) -> dict[str, float]:
    object_groups = [g for _, g in group.groupby("object_id", sort=True)]
    if not object_groups:
        return {"R_ci_low": np.nan, "R_ci_high": np.nan, "U_ci_low": np.nan, "U_ci_high": np.nan}
    rng = np.random.default_rng(_seed(BOOTSTRAP_BASE, label))
    r_values = np.empty(BOOTSTRAP_N)
    u_values = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        sampled = rng.integers(0, len(object_groups), size=len(object_groups))
        frame = pd.concat([object_groups[i] for i in sampled], ignore_index=True)
        r_values[b] = frame["p_i"].mean(skipna=True)
        u_values[b] = frame["u_i"].mean()
    finite_r = r_values[np.isfinite(r_values)]
    return {
        "R_ci_low": float(np.quantile(finite_r, 0.025)) if len(finite_r) else np.nan,
        "R_ci_high": float(np.quantile(finite_r, 0.975)) if len(finite_r) else np.nan,
        "U_ci_low": float(np.quantile(u_values, 0.025)),
        "U_ci_high": float(np.quantile(u_values, 0.975)),
    }


def aggregate_metrics(
    metrics: pd.DataFrame,
    scope: str,
    n_reference: int | dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(["arm", "snr", "criterion_id"], sort=True):
        arm, snr, criterion = key
        classifiable = group["p_i"].notna()
        row = {
            "selection_scope": scope,
            "arm": arm,
            "snr": int(snr),
            "criterion_id": criterion,
            "N_reference": int(
                n_reference[criterion] if isinstance(n_reference, dict) else n_reference
            ),
            "N_eligible_for_rung": int(len(group)),
            "N_classifier_applicable": int(group["applicable_by_design"].sum()),
            "N_transition_eligible": int(len(group)),
            "N_transition_classifiable": int(classifiable.sum()),
            "N_source_families": int(group["source_family"].nunique()),
            "N_eligible_realizations": int(group["N_eligible_realizations"].sum()),
            "N_fit_completed": int(group["N_fit_completed"].sum()),
            "N_fit_valid": int(group["N_fit_valid"].sum()),
            "N_classifiable": int(group["N_classifiable"].sum()),
            "N_CL": int(group["N_CL"].sum()),
            "N_nonCL": int(group["N_nonCL"].sum()),
            "N_unclassifiable": int(group["N_unclassifiable"].sum()),
            "R_equal_transition": float(group.loc[classifiable, "p_i"].mean())
            if classifiable.any()
            else np.nan,
            "U_equal_transition": float(group["u_i"].mean()),
            "raw_realization_recovery_fraction_descriptive": (
                float(group["N_CL"].sum() / group["N_classifiable"].sum())
                if group["N_classifiable"].sum()
                else np.nan
            ),
            "bootstrap_clusters": "object_id",
            "bootstrap_n": BOOTSTRAP_N,
        }
        row.update(_bootstrap_equal_transition(group, f"{scope}|{arm}|{snr}|{criterion}"))
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_out(metrics: pd.DataFrame, unit: str) -> pd.DataFrame:
    column = "transition_id" if unit == "transition" else "source_family"
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(["arm", "snr", "criterion_id"], sort=True):
        for omitted in sorted(group[column].unique()):
            kept = group[group[column] != omitted]
            valid = kept["p_i"].notna()
            rows.append(
                {
                    "robustness": "LEAVE_ONE_TRANSITION_OUT"
                    if unit == "transition"
                    else "LEAVE_ONE_SOURCE_FAMILY_OUT",
                    "arm": key[0],
                    "snr": int(key[1]),
                    "criterion_id": key[2],
                    "omitted_unit": omitted,
                    "N_transition_eligible": len(kept),
                    "N_transition_classifiable": int(valid.sum()),
                    "R_equal_transition": float(kept.loc[valid, "p_i"].mean())
                    if valid.any()
                    else np.nan,
                    "U_equal_transition": float(kept["u_i"].mean()) if len(kept) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def equal_family(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    keys = ["arm", "snr", "criterion_id"]
    for key, group in metrics.groupby(keys, sort=True):
        local = []
        for family, fg in group.groupby("source_family", sort=True):
            valid = fg["p_i"].notna()
            row = {
                "arm": key[0],
                "snr": int(key[1]),
                "criterion_id": key[2],
                "source_family": family,
                "N_transition_eligible": len(fg),
                "N_transition_classifiable": int(valid.sum()),
                "family_R": float(fg.loc[valid, "p_i"].mean()) if valid.any() else np.nan,
                "family_U": float(fg["u_i"].mean()),
            }
            family_rows.append(row)
            local.append(row)
        ld = pd.DataFrame(local)
        aggregate_rows.append(
            {
                "arm": key[0],
                "snr": int(key[1]),
                "criterion_id": key[2],
                "N_families_eligible": len(ld),
                "N_families_classifiable": int(ld["family_R"].notna().sum()),
                "R_equal_family": float(ld["family_R"].mean(skipna=True))
                if ld["family_R"].notna().any()
                else np.nan,
                "U_equal_family": float(ld["family_U"].mean()),
            }
        )
    return pd.DataFrame(family_rows), pd.DataFrame(aggregate_rows)


def paired_comparison(
    metrics: pd.DataFrame, column: str, values: tuple[Any, Any], name: str
) -> pd.DataFrame:
    index = ["transition_id", "object_id", "reference_tier", "source_family", "criterion_id"]
    other = "arm" if column == "snr" else "snr"
    index.append(other)
    selected = metrics[metrics[column].isin(values)]
    wide = selected.pivot_table(
        index=index, columns=column, values=["p_i", "u_i", "N_classifiable"], aggfunc="first"
    )
    rows = []
    for idx, r in wide.iterrows():
        if not all(
            (metric, value) in wide.columns
            for metric in ("p_i", "u_i", "N_classifiable")
            for value in values
        ):
            continue
        a, b = values
        if pd.isna(r[("u_i", a)]) or pd.isna(r[("u_i", b)]):
            continue
        base = dict(zip(index, idx if isinstance(idx, tuple) else (idx,), strict=True))
        rows.append(
            {
                **base,
                "comparison": name,
                f"p_{a}": r[("p_i", a)],
                f"p_{b}": r[("p_i", b)],
                "delta_p_second_minus_first": r[("p_i", b)] - r[("p_i", a)]
                if pd.notna(r[("p_i", a)]) and pd.notna(r[("p_i", b)])
                else np.nan,
                f"u_{a}": r[("u_i", a)],
                f"u_{b}": r[("u_i", b)],
                "delta_u_second_minus_first": r[("u_i", b)] - r[("u_i", a)],
                f"N_classifiable_{a}": int(r[("N_classifiable", a)]),
                f"N_classifiable_{b}": int(r[("N_classifiable", b)]),
            }
        )
    return pd.DataFrame(rows)


def criterion_agreement(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = outcomes[outcomes["criterion_id"].isin(["YANG2024_FINAL", "GREEN2022_FINAL"])]
    index = [
        "condition_id",
        "transition_id",
        "object_id",
        "reference_tier",
        "source_family",
        "arm",
        "snr",
        "realization",
    ]
    wide = subset.pivot(index=index, columns="criterion_id", values="classification").reset_index()
    both = wide[
        wide["YANG2024_FINAL"].isin(["CL", "NON_CL"])
        & wide["GREEN2022_FINAL"].isin(["CL", "NON_CL"])
    ].copy()
    both["agreement"] = both["YANG2024_FINAL"] == both["GREEN2022_FINAL"]
    both["direction"] = both["YANG2024_FINAL"] + "__" + both["GREEN2022_FINAL"]
    transition_rows = []
    for key, g in both.groupby(
        ["transition_id", "object_id", "reference_tier", "source_family", "arm", "snr"], sort=True
    ):
        transition_rows.append(
            {
                **dict(
                    zip(
                        [
                            "transition_id",
                            "object_id",
                            "reference_tier",
                            "source_family",
                            "arm",
                            "snr",
                        ],
                        key,
                        strict=True,
                    )
                ),
                "N_both_classifiable": len(g),
                "N_agree": int(g["agreement"].sum()),
                "N_disagree": int((~g["agreement"]).sum()),
                "disagreement_i": float((~g["agreement"]).mean()),
                "N_YANG_CL_GREEN_NONCL": int(g["direction"].eq("CL__NON_CL").sum()),
                "N_YANG_NONCL_GREEN_CL": int(g["direction"].eq("NON_CL__CL").sum()),
            }
        )
    transition = pd.DataFrame(transition_rows)
    aggregate_rows = []
    for scope, frame in [
        ("PRIMARY_GOLD", transition[transition.reference_tier.eq("GOLD")]),
        ("GOLD_PLUS_SILVER", transition),
    ]:
        for key, g in frame.groupby(["arm", "snr"], sort=True):
            aggregate_rows.append(
                {
                    "selection_scope": scope,
                    "arm": key[0],
                    "snr": int(key[1]),
                    "N_transition_both_classifiable": len(g),
                    "N_both_classifiable_realizations": int(g["N_both_classifiable"].sum()),
                    "N_disagree_realizations": int(g["N_disagree"].sum()),
                    "disagreement_equal_transition": float(g["disagreement_i"].mean()),
                    "N_YANG_CL_GREEN_NONCL": int(g["N_YANG_CL_GREEN_NONCL"].sum()),
                    "N_YANG_NONCL_GREEN_CL": int(g["N_YANG_NONCL_GREEN_CL"].sum()),
                }
            )
    return transition, pd.DataFrame(aggregate_rows)


def make_figures(aggregate: pd.DataFrame, metrics: pd.DataFrame, agreement: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = aggregate[
        (aggregate.selection_scope == "PRIMARY_GOLD")
        & aggregate.criterion_id.isin(["YANG2024_FINAL", "GREEN2022_FINAL"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, criterion in zip(axes, ["YANG2024_FINAL", "GREEN2022_FINAL"], strict=True):
        data = primary[primary.criterion_id.eq(criterion)]
        for arm, marker in [("faint_only", "o"), ("matched", "s")]:
            d = data[data.arm.eq(arm)].sort_values("snr")
            ax.errorbar(
                d.snr,
                d.R_equal_transition,
                yerr=[d.R_equal_transition - d.R_ci_low, d.R_ci_high - d.R_equal_transition],
                marker=marker,
                capsize=3,
                label=arm,
            )
        ax.set(title=criterion, xlabel="Target S/N", xticks=[5, 10], ylim=(-0.05, 1.05))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Equal-transition recovery R")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "primary_gold_recovery.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, criterion in zip(axes, ["YANG2024_FINAL", "GREEN2022_FINAL"], strict=True):
        data = primary[primary.criterion_id.eq(criterion)]
        for arm, marker in [("faint_only", "o"), ("matched", "s")]:
            d = data[data.arm.eq(arm)].sort_values("snr")
            ax.errorbar(
                d.snr,
                d.U_equal_transition,
                yerr=[d.U_equal_transition - d.U_ci_low, d.U_ci_high - d.U_equal_transition],
                marker=marker,
                capsize=3,
                label=arm,
            )
        ax.set(title=criterion, xlabel="Target S/N", xticks=[5, 10], ylim=(-0.05, 1.05))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Equal-transition unclassifiable fraction U")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "primary_gold_unclassifiable.png", dpi=180)
    plt.close(fig)

    gold = metrics[
        (metrics.reference_tier == "GOLD")
        & metrics.criterion_id.isin(["YANG2024_FINAL", "GREEN2022_FINAL"])
    ]
    gold = gold.assign(
        cell=gold.criterion_id.str.replace("_FINAL", "")
        + "|"
        + gold.arm
        + "|"
        + gold.snr.astype(str)
    )
    heat = gold.pivot(index="transition_id", columns="cell", values="p_i")
    heat = heat.reindex(sorted(heat.columns), axis=1)
    fig, ax = plt.subplots(figsize=(10, max(8, len(heat) * 0.20)))
    image = ax.imshow(heat.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(heat.index)), heat.index, fontsize=5)
    ax.set_title("Gold transition-level recovery p_i (blank = unclassifiable/unsupported)")
    fig.colorbar(image, ax=ax, label="p_i")
    fig.tight_layout()
    fig.savefig(FIGURES / "gold_transition_recovery_heatmap.png", dpi=180)
    plt.close(fig)

    if len(agreement):
        g = agreement[agreement.selection_scope.eq("PRIMARY_GOLD")]
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, marker in [("faint_only", "o"), ("matched", "s")]:
            d = g[g.arm.eq(arm)].sort_values("snr")
            ax.plot(d.snr, d.disagreement_equal_transition, marker=marker, label=arm)
        ax.set(
            xlabel="Target S/N",
            ylabel="Yang–Green disagreement (equal-transition)",
            xticks=[5, 10],
            ylim=(-0.05, 1.05),
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "yang_green_disagreement.png", dpi=180)
        plt.close(fig)


def main() -> int:
    completion = OUT / "FULL_Q1_EXECUTION_COMPLETE.json"
    if not completion.exists() or json.loads(completion.read_text()).get("production_qc") != "PASS":
        raise RuntimeError("Raw D-094 production products are not frozen with PASS QC")
    TABLES.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")
    metrics = transition_metrics(outcomes)
    metrics.to_csv(TABLES / "transition_level_estimands_d094.csv", index=False)

    primary = metrics[metrics.reference_tier.eq("GOLD")]
    sensitivity = metrics[metrics.reference_tier.isin(["GOLD", "SILVER"])]
    aggregate = pd.concat(
        [
            aggregate_metrics(primary, "PRIMARY_GOLD", 58),
            aggregate_metrics(sensitivity, "GOLD_PLUS_SILVER", 62),
        ],
        ignore_index=True,
    )

    independence = pd.read_csv(
        ROOT / "05_analysis" / "q1_design" / "d093" / "discovery_classifier_independence_d093.csv"
    )
    independence = independence.rename(columns={"final_classifier": "criterion_id"})
    independent_keys = independence.loc[
        independence.discovery_independent_sensitivity_included.astype(bool),
        ["source_family", "criterion_id"],
    ].drop_duplicates()
    independent = primary.merge(independent_keys, on=["source_family", "criterion_id"], how="inner")
    gold_reference = pd.read_csv(
        ROOT / "05_analysis" / "q1_design" / "d093" / "final_gold_sample_d093.csv"
    )
    independent_reference_counts: dict[str, int] = {}
    for criterion in independent_keys["criterion_id"].unique():
        families = set(
            independent_keys.loc[independent_keys.criterion_id.eq(criterion), "source_family"]
        )
        independent_reference_counts[str(criterion)] = int(
            gold_reference.source_family.isin(families).sum()
        )
    aggregate = pd.concat(
        [
            aggregate,
            aggregate_metrics(
                independent, "DISCOVERY_INDEPENDENT_GOLD", independent_reference_counts
            ),
        ],
        ignore_index=True,
    )
    aggregate.to_csv(TABLES / "aggregate_estimands_bootstrap_d094.csv", index=False)

    loto = leave_one_out(primary, "transition")
    losfo = leave_one_out(primary, "source_family")
    loto.to_csv(TABLES / "leave_one_transition_out_d094.csv", index=False)
    losfo.to_csv(TABLES / "leave_one_source_family_out_d094.csv", index=False)
    family, equal = equal_family(primary)
    family.to_csv(TABLES / "source_family_estimands_d094.csv", index=False)
    equal.to_csv(TABLES / "equal_source_family_weighted_d094.csv", index=False)

    rung = paired_comparison(sensitivity, "snr", (5, 10), "SNR_10_MINUS_5")
    arms = paired_comparison(
        sensitivity, "arm", ("faint_only", "matched"), "MATCHED_MINUS_FAINT_ONLY"
    )
    rung.to_csv(TABLES / "within_transition_snr_5_vs_10_d094.csv", index=False)
    arms.to_csv(TABLES / "within_transition_arm_comparison_d094.csv", index=False)
    agreement_transition, agreement_aggregate = criterion_agreement(outcomes)
    agreement_transition.to_csv(TABLES / "yang_green_agreement_transition_d094.csv", index=False)
    agreement_aggregate.to_csv(TABLES / "yang_green_agreement_aggregate_d094.csv", index=False)

    reasons = (
        outcomes[outcomes.classification.eq("UNCLASSIFIABLE")]
        .groupby(["criterion_id", "reason"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
        .sort_values(["criterion_id", "N"], ascending=[True, False])
    )
    reasons.to_csv(TABLES / "unclassifiable_reasons_d094.csv", index=False)
    make_figures(aggregate, metrics, agreement_aggregate)

    headline = aggregate[
        aggregate.selection_scope.eq("PRIMARY_GOLD")
        & aggregate.criterion_id.isin(["YANG2024_FINAL", "GREEN2022_FINAL"])
    ].sort_values(["criterion_id", "arm", "snr"])
    summary = {
        "production_qc": "PASS",
        "scientific_outcome": "PARTIAL",
        "rationale": "Green provides a complete scalar-classifier result; Yang is heavily realization-unclassifiable; MacLeod final remains prospectively non-denominator without visual adjudication.",
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_cluster": "object_id",
        "primary_gold_validated": 58,
        "gold_plus_silver_validated": 62,
        "matrix_gold_transitions_with_at_least_one_supported_condition": int(
            primary.transition_id.nunique()
        ),
        "headline": headline.to_dict("records"),
        "tables": sorted(str(p.relative_to(ROOT)) for p in TABLES.glob("*.csv")),
        "figures": sorted(str(p.relative_to(ROOT)) for p in FIGURES.glob("*.png")),
    }
    (OUT / "scientific_summary_d094.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
