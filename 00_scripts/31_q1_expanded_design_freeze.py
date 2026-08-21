#!/usr/bin/env python
"""D-091 expanded native-S/N support and unexecuted Q1 matrix freeze."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_analysis/q1_design"
CANDIDATE_RUNGS = (5, 10, 15, 20, 30, 40)
FINAL_RUNGS = (5, 10)
M = 50
FINAL_CLASSIFIERS = ("YANG2024_FINAL", "GREEN2022_FINAL", "MACLEOD2019_FINAL")

NEW_FAMILIES = {
    "SDSSJ015957.64+003310.4": "LaMassa2015",
    "SDSSJ012648.08-083948.0": "Ruan2016",
    "SDSSJ101152.98+544206.4": "Runnoe2016",
    "SDSSJ000236.25-002724.8": "Potts2021",
    "SDSSJ135855.83+493414.2": "Potts2021",
    "SDSSJ021359.79+004226.81": "Green2022",
}
OLD_FAMILIES = {
    "SDSSJ082942.66+415436.8": "Potts2021",
    "SDSSJ102152.34+464515.6": "MacLeod2016",
    "SDSSJ105325.40+302419.34": "Green2022",
    "SDSSJ233602.98+001728.7": "Ruan2016",
    "SDSSJ012256.19-000252.68": "MacLeod2016",
    "SDSSJ081319.34+460849.5": "Yang2018",
    "SDSSJ105058.42+241351.18": "Green2022",
    "SDSSJ222132.41-010928.70": "MacLeod2016",
}


def new_support_rows() -> pd.DataFrame:
    qc = pd.read_csv(OUT / "native_qc_d091.csv")
    validity = pd.read_csv(OUT / "added_gold_validity_d091.csv")
    eligible = set(validity.loc[validity.q1_eligible, "transition_id"])
    rows: list[dict[str, object]] = []
    for object_id, group in qc[qc.object_id.isin(eligible)].groupby("object_id", sort=False):
        bright = group[group.role == "bright"].iloc[0]
        faint = group[group.role == "faint"].iloc[0]
        row: dict[str, object] = {
            "transition_id": object_id,
            "reference_tier": "GOLD",
            "source_family": NEW_FAMILIES[object_id],
            "bright_science_record_id": f"sdss:{bright.spectrum_id}",
            "faint_science_record_id": f"sdss:{faint.spectrum_id}",
            "bright_snr_continuum_5100": bright.snr_continuum_5100,
            "bright_snr_hbeta": bright.snr_hbeta_window,
            "faint_snr_continuum_5100": faint.snr_continuum_5100,
            "faint_snr_hbeta": faint.snr_hbeta_window,
        }
        for rung in CANDIDATE_RUNGS:
            row[f"faint_only_support_{rung}"] = bool(faint.snr_hbeta_window >= rung)
            row[f"matched_support_{rung}"] = bool(
                min(bright.snr_hbeta_window, faint.snr_hbeta_window) >= rung
            )
        rows.append(row)
    return pd.DataFrame(rows)


def old_support_rows() -> pd.DataFrame:
    old = pd.read_csv(OUT / "native_snr_support_d088.csv")
    old.insert(2, "source_family", old.transition_id.map(OLD_FAMILIES))
    return old


def support_summary(support: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rung in CANDIDATE_RUNGS:
        gold = support[support.reference_tier == "GOLD"]
        rows.append({
            "snr_rung": rung,
            "gold_n": len(gold),
            "gold_faint_only": int(gold[f"faint_only_support_{rung}"].sum()),
            "gold_matched": int(gold[f"matched_support_{rung}"].sum()),
            "gold_plus_silver_n": len(support),
            "gold_plus_silver_faint_only": int(support[f"faint_only_support_{rung}"].sum()),
            "gold_plus_silver_matched": int(support[f"matched_support_{rung}"].sum()),
            "d091_disposition": (
                "FINAL_RETAINED" if rung in FINAL_RUNGS
                else "UNDERPOWERED_EXCLUDED" if rung < 40
                else "UNREACHABLE_REMOVED"
            ),
        })
    return pd.DataFrame(rows)


def build_matrix(support: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in support.itertuples():
        for arm in ("faint_only", "matched"):
            for rung in FINAL_RUNGS:
                if not bool(getattr(record, f"{arm}_support_{rung}")):
                    continue
                rows.append({
                    "transition_id": record.transition_id,
                    "reference_tier": record.reference_tier,
                    "q1_set": "PRIMARY_Q1" if record.reference_tier == "GOLD" else "SENSITIVITY_INCREMENT",
                    "source_family": record.source_family,
                    "bright_science_record_id": record.bright_science_record_id,
                    "faint_science_record_id": record.faint_science_record_id,
                    "arm": arm,
                    "snr_rung": rung,
                    "bright_epoch_action": "UNCHANGED" if arm == "faint_only" else "DEGRADE_TO_RUNG",
                    "faint_epoch_action": "DEGRADE_TO_RUNG",
                    "object_specific_crn": True,
                    "mc_realizations": M,
                    "seed_namespace": "p3sf:q1:full:v1",
                    "grid_status": "FINAL_FROZEN_D091_UNEXECUTED",
                    "execution_authorized": False,
                    "spectra_generated": False,
                })
    return pd.DataFrame(rows).sort_values(
        ["transition_id", "arm", "snr_rung"]
    ).reset_index(drop=True)


def applicability(support: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    old_yang = pd.read_csv(
        OUT / "measurement_validity_d089/yang_native_fit_audit.csv"
    )
    new_yang = pd.read_csv(OUT / "yang_native_fit_audit_d091.csv").rename(
        columns={"object": "transition_id"}
    )
    new_yang["fit_role"] = "native_" + new_yang.role
    native = pd.concat(
        [old_yang, new_yang], ignore_index=True, sort=False
    )
    rows: list[dict[str, object]] = []
    for record in support.itertuples():
        object_id = record.transition_id
        bright = native[
            (native.transition_id == object_id) & (native.fit_role == "native_bright")
        ].iloc[0]
        static_coverage_failure = not bool(bright.local_hbeta_coverage)
        arms = sorted(matrix.loc[matrix.transition_id == object_id, "arm"].unique())
        for arm in arms:
            if static_coverage_failure:
                yang_status = "UNCLASSIFIABLE_STATIC_NATIVE_BRIGHT_HBETA_COVERAGE"
            elif arm == "faint_only" and not bool(bright.fit_valid):
                yang_status = "UNCLASSIFIABLE_NATIVE_BRIGHT_FIT_INVALID"
            else:
                yang_status = "APPLICABLE_FAIL_CLOSED_PER_REALIZATION"
            for criterion, status, denominator in (
                ("YANG2024_FINAL", yang_status, "FINAL_CLASSIFIER_DENOMINATOR_IF_CLASSIFIABLE"),
                ("GREEN2022_FINAL", "APPLICABLE_NATIVE_SDSS_VARIANCE_SUPPORTED", "FINAL_CLASSIFIER_DENOMINATOR_IF_CLASSIFIABLE"),
                ("MACLEOD2019_FINAL", "UNCLASSIFIABLE_NO_PROSPECTIVE_VISUAL_EVIDENCE", "FINAL_CLASSIFIER_ROW_NON_DENOMINATOR_WHEN_UNCLASSIFIABLE"),
                ("MACLEOD2016_SEARCH", "SEARCH_PROTOCOL_ONLY", "NON_DENOMINATOR"),
                ("GUO_QUICK_SEARCH", "SEARCH_STAGE_ONLY_NOT_GUO_FINAL", "NON_DENOMINATOR"),
                ("POTTS_VILLFORTH_SEARCH", "SEARCH_PROTOCOL_ONLY", "NON_DENOMINATOR"),
            ):
                rows.append({
                    "transition_id": object_id,
                    "reference_tier": record.reference_tier,
                    "arm": arm,
                    "criterion": criterion,
                    "applicability": status,
                    "denominator_semantics": denominator,
                })
    return pd.DataFrame(rows)


def robustness_freeze(support: pd.DataFrame) -> pd.DataFrame:
    gold = support[support.reference_tier == "GOLD"]
    rows = [{
        "summary": "LEAVE_ONE_TRANSITION_OUT",
        "unit": row.transition_id,
        "source_family": row.source_family,
        "n_gold_retained": len(gold) - 1,
        "execution_status": "FROZEN_UNEXECUTED",
    } for row in gold.itertuples()]
    for family, group in gold.groupby("source_family", sort=True):
        rows.append({
            "summary": "LEAVE_ONE_SOURCE_FAMILY_OUT",
            "unit": family,
            "source_family": family,
            "n_gold_retained": len(gold) - len(group),
            "execution_status": "FROZEN_UNEXECUTED",
        })
    return pd.DataFrame(rows)


def main() -> int:
    support = pd.concat([old_support_rows(), new_support_rows()], ignore_index=True)
    support = support.sort_values(["reference_tier", "transition_id"]).reset_index(drop=True)
    summary = support_summary(support)
    matrix = build_matrix(support)
    criteria = applicability(support, matrix)
    robustness = robustness_freeze(support)

    support.to_csv(OUT / "expanded_native_snr_support_d091.csv", index=False)
    summary.to_csv(OUT / "expanded_rung_support_d091.csv", index=False)
    matrix.to_csv(OUT / "proposed_expanded_final_q1_matrix_d091_unexecuted.csv", index=False)
    criteria.to_csv(OUT / "expanded_q1_classifier_applicability_d091.csv", index=False)
    robustness.to_csv(OUT / "expanded_q1_robustness_freeze_d091.csv", index=False)

    primary = support[support.reference_tier == "GOLD"]
    families = (
        primary.groupby("source_family").size().rename("n_gold_transitions").reset_index()
    )
    families.to_csv(OUT / "expanded_q1_source_family_composition_d091.csv", index=False)

    unique_spectrum_fit_inputs = len(matrix) * M + len(support)
    unique_classifier_specific_fits = unique_spectrum_fit_inputs * 2
    workload = pd.DataFrame([{
        "primary_transition_count": len(primary),
        "sensitivity_transition_count": len(support),
        "final_snr_grid": "5;10",
        "primary_condition_count": int((matrix.reference_tier == "GOLD").sum()),
        "sensitivity_increment_condition_count": int((matrix.reference_tier == "SILVER").sum()),
        "faint_only_condition_count": int((matrix.arm == "faint_only").sum()),
        "matched_condition_count": int((matrix.arm == "matched").sum()),
        "total_condition_count": len(matrix),
        "mc_realizations": M,
        "condition_realization_count": len(matrix) * M,
        "unique_reusable_spectrum_fit_inputs": unique_spectrum_fit_inputs,
        "yang_fit_count": unique_spectrum_fit_inputs,
        "green_preprocessing_fit_count": unique_spectrum_fit_inputs,
        "unique_reusable_fit_count": unique_classifier_specific_fits,
        "final_classifier_rows_per_condition_realization": len(FINAL_CLASSIFIERS),
        "classifier_evaluation_count": len(matrix) * M * len(FINAL_CLASSIFIERS),
        "execution_status": "UNEXECUTED",
    }])
    workload.to_csv(OUT / "expanded_q1_workload_d091.csv", index=False)

    if len(primary) != 10 or len(support) != 14:
        raise RuntimeError("D-091 expanded Q1 reference counts changed unexpectedly")
    if set(summary.loc[summary.d091_disposition == "FINAL_RETAINED", "snr_rung"]) != set(FINAL_RUNGS):
        raise RuntimeError("D-091 final grid mismatch")
    if matrix.execution_authorized.any() or matrix.spectra_generated.any():
        raise RuntimeError("D-091 matrix must remain unexecuted")
    print(summary.to_string(index=False))
    print(f"primary transitions: {len(primary)}")
    print(f"sensitivity transitions: {len(support)}")
    print(f"conditions: {len(matrix)} ({int((matrix.arm == 'faint_only').sum())} faint_only, {int((matrix.arm == 'matched').sum())} matched)")
    print(f"condition-realizations: {len(matrix) * M}")
    print(f"unique reusable spectrum-fit inputs: {unique_spectrum_fit_inputs}")
    print(f"unique reusable classifier-specific fits: {unique_classifier_specific_fits}")
    print(f"classifier evaluations: {len(matrix) * M * len(FINAL_CLASSIFIERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
