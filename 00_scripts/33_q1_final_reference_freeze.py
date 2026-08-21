#!/usr/bin/env python
"""D-093: freeze the final validated Q1 sample and unexecuted design.

Consumes only D-091 frozen records and D-093 native-validation outputs. It never opens a
spectrum, generates a noise realization, runs a fit, or evaluates a classifier outcome.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_analysis/q1_design"
OUT = BASE / "d093"
RUNGS = (5, 10, 15, 20, 30, 40)
FINAL_RUNGS = (5, 10)
M = 50
CLASSIFIERS = ("YANG2024_FINAL", "GREEN2022_FINAL", "MACLEOD2019_FINAL")

OLD_DIRECTIONS = {
    "SDSSJ000236.25-002724.8": "turn_off",
    "SDSSJ012648.08-083948.0": "turn_off",
    "SDSSJ015957.64+003310.4": "turn_off",
    "SDSSJ021359.79+004226.81": "turn_off",
    "SDSSJ082942.66+415436.8": "turn_on",
    "SDSSJ101152.98+544206.4": "turn_off",
    "SDSSJ102152.34+464515.6": "turn_off",
    "SDSSJ105325.40+302419.34": "turn_on",
    "SDSSJ135855.83+493414.2": "turn_on",
    "SDSSJ233602.98+001728.7": "turn_on",
}

DISCOVERY = {
    "LaMassa2015": {
        "method": "serendipitous archival repeat spectroscopy",
        "yang": False,
        "green": False,
        "visual": True,
        "photometric": True,
        "automated": False,
    },
    "Ruan2016": {
        "method": "photometric change plus repeat spectroscopy and difference spectrum",
        "yang": False,
        "green": False,
        "visual": True,
        "photometric": True,
        "automated": False,
    },
    "Runnoe2016": {
        "method": "TDSS repeat spectroscopy with continuum and Balmer confirmation",
        "yang": False,
        "green": False,
        "visual": True,
        "photometric": True,
        "automated": False,
    },
    "MacLeod2016": {
        "method": "photometric variability selection plus repeat spectral confirmation",
        "yang": False,
        "green": False,
        "visual": True,
        "photometric": True,
        "automated": False,
    },
    "Potts2021": {
        "method": "repeat-spectrum difference preselection plus visual confirmation",
        "yang": False,
        "green": True,
        "visual": True,
        "photometric": False,
        "automated": True,
    },
    "Green2022": {
        "method": "Green Hbeta pixel-significance selection plus visual confirmation",
        "yang": False,
        "green": True,
        "visual": True,
        "photometric": False,
        "automated": True,
    },
    "Yang2018": {
        "method": "heterogeneous repeat-spectroscopy and photometric CLAGN selection",
        "yang": True,
        "green": False,
        "visual": True,
        "photometric": True,
        "automated": True,
    },
    "Dong2025_SDSS_LAMOST": {
        "method": "automated spectral fitting followed by visual and photometric confirmation",
        "yang": True,
        "green": True,
        "visual": True,
        "photometric": True,
        "automated": True,
    },
    "Zeltyn2024_SDSSV": {
        "method": "repeat spectroscopy plus visual confirmation and light-curve/follow-up support",
        "yang": True,
        "green": True,
        "visual": True,
        "photometric": True,
        "automated": True,
    },
    "Yang2025_turn_on": {
        "method": "optical/MIR variability preselection plus spectroscopic confirmation",
        "yang": True,
        "green": True,
        "visual": True,
        "photometric": True,
        "automated": True,
    },
}


def _new_support() -> pd.DataFrame:
    validity = pd.read_csv(OUT / "new_gold_validity_d093.csv")
    qc = pd.read_csv(OUT / "native_qc_d093.csv")
    manifest = pd.read_csv(OUT / "endpoint_acquisition_manifest_d093.csv")
    rows: list[dict[str, Any]] = []
    for valid in validity[validity.q1_eligible].itertuples(index=False):
        group = qc[qc.transition_id == valid.transition_id]
        bright = group[group.physical_role == "bright"].iloc[0]
        faint = group[group.physical_role == "faint"].iloc[0]
        instruments = manifest[manifest.transition_id == valid.transition_id]
        bright_instrument = instruments.loc[
            instruments.physical_role == "bright", "instrument_survey"
        ].item()
        faint_instrument = instruments.loc[
            instruments.physical_role == "faint", "instrument_survey"
        ].item()
        row: dict[str, Any] = {
            "transition_id": valid.transition_id,
            "reference_tier": "GOLD",
            "source_family": valid.source_family,
            "event": valid.event,
            "reference_origin": "D093_NEW_VALIDATED",
            "bright_instrument": bright_instrument,
            "faint_instrument": faint_instrument,
            "instrument_pair": f"{bright_instrument}/{faint_instrument}",
            "bright_snr_hbeta": bright.native_hbeta_window_snr,
            "faint_snr_hbeta": faint.native_hbeta_window_snr,
            "bright_snr_continuum": bright.continuum_snr,
            "faint_snr_continuum": faint.continuum_snr,
            "yang_applicable": bool(valid.yang_instrument_domain_valid),
            "green_applicable": bool(valid.green_instrument_domain_valid),
        }
        for rung in RUNGS:
            row[f"faint_only_support_{rung}"] = bool(faint.native_hbeta_window_snr >= rung)
            row[f"matched_support_{rung}"] = bool(
                min(bright.native_hbeta_window_snr, faint.native_hbeta_window_snr) >= rung
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _old_support() -> pd.DataFrame:
    old = pd.read_csv(BASE / "expanded_native_snr_support_d091.csv")
    applicability = pd.read_csv(BASE / "expanded_q1_classifier_applicability_d091.csv")
    rows: list[dict[str, Any]] = []
    for record in old.itertuples(index=False):
        app = applicability[applicability.transition_id == record.transition_id]
        row: dict[str, Any] = {
            "transition_id": record.transition_id,
            "reference_tier": record.reference_tier,
            "source_family": record.source_family,
            "event": OLD_DIRECTIONS.get(record.transition_id, "sensitivity_event"),
            "reference_origin": "D091_VALIDATED",
            "bright_instrument": "SDSS_LEGACY",
            "faint_instrument": "SDSS_LEGACY",
            "instrument_pair": "SDSS_LEGACY/SDSS_LEGACY",
            "bright_snr_hbeta": record.bright_snr_hbeta,
            "faint_snr_hbeta": record.faint_snr_hbeta,
            "bright_snr_continuum": record.bright_snr_continuum_5100,
            "faint_snr_continuum": record.faint_snr_continuum_5100,
            "yang_applicable": bool(
                app[
                    (app.criterion == "YANG2024_FINAL")
                    & app.applicability.str.startswith("APPLICABLE")
                ].shape[0]
            ),
            "green_applicable": True,
        }
        for rung in RUNGS:
            row[f"faint_only_support_{rung}"] = bool(
                getattr(record, f"faint_only_support_{rung}")
            )
            row[f"matched_support_{rung}"] = bool(
                getattr(record, f"matched_support_{rung}")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _support_summary(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gold = sample[sample.reference_tier == "GOLD"]
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("ALL_FINAL_GOLD", gold),
        ("YANG_APPLICABLE_FINAL_GOLD", gold[gold.yang_applicable]),
        ("GREEN_APPLICABLE_FINAL_GOLD", gold[gold.green_applicable]),
    ]
    scopes.extend(
        (f"SOURCE_FAMILY:{family}", group)
        for family, group in gold.groupby("source_family", sort=True)
    )
    for scope, frame in scopes:
        for arm in ("faint_only", "matched"):
            for rung in RUNGS:
                supported = int(frame[f"{arm}_support_{rung}"].sum())
                rows.append(
                    {
                        "scope": scope,
                        "arm": arm,
                        "snr_rung": rung,
                        "n_total": len(frame),
                        "n_supported": supported,
                        "support_fraction": supported / len(frame) if len(frame) else 0.0,
                        "d093_disposition": (
                            "FINAL_RETAINED_PRIOR_COMPARABLE"
                            if rung in FINAL_RUNGS
                            else "REPORTED_NOT_PROMOTED_COMPARABILITY"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _independence_matrix(families: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in sorted(families):
        info = DISCOVERY[family]
        for classifier in ("YANG2024_FINAL", "GREEN2022_FINAL"):
            related = bool(info["yang"] if classifier.startswith("YANG") else info["green"])
            if family == "Green2022" and classifier == "GREEN2022_FINAL":
                status = "DIRECTLY_CRITERION_SELECTED"
            elif related:
                status = "PARTIALLY_METHOD_OVERLAPPING"
            else:
                status = "DISCOVERY_INDEPENDENT"
            rows.append(
                {
                    "source_family": family,
                    "discovery_method": info["method"],
                    "yang_related_selection": bool(info["yang"]),
                    "green_related_selection": bool(info["green"]),
                    "visual_confirmation": bool(info["visual"]),
                    "photometric_preselection": bool(info["photometric"]),
                    "automated_spectral_preselection": bool(info["automated"]),
                    "final_classifier": classifier,
                    "classifier_independence_status": status,
                    "discovery_independent_sensitivity_included": (
                        status != "DIRECTLY_CRITERION_SELECTED"
                    ),
                    "assignment_basis": "FROZEN_BEFORE_Q1_OUTCOMES_D093",
                }
            )
    return pd.DataFrame(rows)


def _matrix(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in sample.itertuples(index=False):
        for arm in ("faint_only", "matched"):
            for rung in FINAL_RUNGS:
                if not bool(getattr(record, f"{arm}_support_{rung}")):
                    continue
                rows.append(
                    {
                        "transition_id": record.transition_id,
                        "reference_tier": record.reference_tier,
                        "q1_set": (
                            "PRIMARY_Q1"
                            if record.reference_tier == "GOLD"
                            else "SENSITIVITY_INCREMENT"
                        ),
                        "source_family": record.source_family,
                        "instrument_pair": record.instrument_pair,
                        "arm": arm,
                        "snr_rung": rung,
                        "bright_epoch_action": (
                            "UNCHANGED" if arm == "faint_only" else "DEGRADE_TO_RUNG"
                        ),
                        "faint_epoch_action": "DEGRADE_TO_RUNG",
                        "object_specific_crn": True,
                        "mc_realizations": M,
                        "seed_namespace": "p3sf:q1:full:v1",
                        "grid_status": "FINAL_FROZEN_D093_UNEXECUTED",
                        "execution_authorized": False,
                        "spectra_generated": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["reference_tier", "transition_id", "arm", "snr_rung"]
    ).reset_index(drop=True)


def _applicability(
    sample: pd.DataFrame, matrix: pd.DataFrame, independence: pd.DataFrame
) -> pd.DataFrame:
    old = pd.read_csv(BASE / "expanded_q1_classifier_applicability_d091.csv")
    rows: list[dict[str, Any]] = []
    for condition in matrix.itertuples(index=False):
        record = sample[sample.transition_id == condition.transition_id].iloc[0]
        for classifier in CLASSIFIERS:
            if classifier == "MACLEOD2019_FINAL":
                status = "UNCLASSIFIABLE_NO_PROSPECTIVE_VISUAL_EVIDENCE"
                denominator = "FINAL_CLASSIFIER_ROW_NON_DENOMINATOR_WHEN_UNCLASSIFIABLE"
            elif record.reference_origin == "D091_VALIDATED":
                match = old[
                    (old.transition_id == condition.transition_id)
                    & (old.arm == condition.arm)
                    & (old.criterion == classifier)
                ].iloc[0]
                status = match.applicability
                denominator = match.denominator_semantics
            else:
                applicable = bool(
                    record.yang_applicable
                    if classifier == "YANG2024_FINAL"
                    else record.green_applicable
                )
                status = (
                    "APPLICABLE_FAIL_CLOSED_PER_REALIZATION"
                    if classifier == "YANG2024_FINAL" and applicable
                    else "APPLICABLE_SUPPORTED_VARIANCE"
                    if applicable
                    else "UNCLASSIFIABLE_INSTRUMENT_DOMAIN"
                )
                denominator = "FINAL_CLASSIFIER_DENOMINATOR_IF_CLASSIFIABLE"
            independent = independence[
                (independence.source_family == condition.source_family)
                & (independence.final_classifier == classifier)
            ]
            sensitivity_included = (
                True
                if classifier == "MACLEOD2019_FINAL"
                else bool(independent.discovery_independent_sensitivity_included.item())
            )
            rows.append(
                {
                    "transition_id": condition.transition_id,
                    "reference_tier": condition.reference_tier,
                    "source_family": condition.source_family,
                    "arm": condition.arm,
                    "snr_rung": condition.snr_rung,
                    "criterion": classifier,
                    "applicability": status,
                    "denominator_semantics": denominator,
                    "discovery_independent_sensitivity_included": sensitivity_included,
                    "unclassifiable_by_design": status.startswith("UNCLASSIFIABLE"),
                }
            )
    return pd.DataFrame(rows)


def _robustness(gold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "summary": "PRIMARY_TRANSITION_EQUAL",
            "unit": "ALL_TRANSITIONS",
            "n_gold_retained": len(gold),
            "formula": "mean_i(p_i); each transition has weight 1/N",
            "role": "PRIMARY_ESTIMAND",
            "execution_status": "FROZEN_UNEXECUTED",
        },
        {
            "summary": "EQUAL_SOURCE_FAMILY_WEIGHTED",
            "unit": "ALL_SOURCE_FAMILIES",
            "n_gold_retained": len(gold),
            "formula": "mean_family(mean_transition_within_family(p_i))",
            "role": "PRESPECIFIED_ROBUSTNESS_NOT_PRIMARY",
            "execution_status": "FROZEN_UNEXECUTED",
        },
    ]
    for family, group in gold.groupby("source_family", sort=True):
        rows.append(
            {
                "summary": "LEAVE_ONE_SOURCE_FAMILY_OUT",
                "unit": family,
                "n_gold_retained": len(gold) - len(group),
                "formula": f"primary transition-equal mean excluding {family}",
                "role": "PRESPECIFIED_ROBUSTNESS_NOT_PRIMARY",
                "execution_status": "FROZEN_UNEXECUTED",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    old = _old_support()
    new = _new_support()
    sample = pd.concat([old, new], ignore_index=True).sort_values(
        ["reference_tier", "transition_id"]
    )
    gold = sample[sample.reference_tier == "GOLD"]
    support = _support_summary(sample)
    independence = _independence_matrix(set(sample.source_family))
    matrix = _matrix(sample)
    applicability = _applicability(sample, matrix, independence)
    robustness = _robustness(gold)
    identity = pd.read_csv(OUT / "endpoint_identity_validation_d093.csv")
    identity_valid_pairs = int(
        identity.groupby("transition_id").endpoint_identity_pass.all().sum()
    )

    gold.to_csv(OUT / "final_gold_sample_d093.csv", index=False)
    sample.to_csv(OUT / "final_sensitivity_sample_d093.csv", index=False)
    support.to_csv(OUT / "final_snr_support_d093.csv", index=False)
    independence.to_csv(OUT / "discovery_classifier_independence_d093.csv", index=False)
    matrix.to_csv(OUT / "final_q1_matrix_d093_unexecuted.csv", index=False)
    applicability.to_csv(OUT / "final_classifier_applicability_d093.csv", index=False)
    robustness.to_csv(OUT / "source_family_robustness_d093.csv", index=False)

    condition_realizations = len(matrix) * M
    reusable_spectrum_fits = condition_realizations + len(sample)
    workload = pd.DataFrame(
        [
            {
                "validated_pre_d092_gold": 10,
                "prospective_new_gold": 54,
                "acquired_frozen_pairs": 54,
                "identity_valid_exact_pairs": identity_valid_pairs,
                "endpoint_valid_new_gold": len(new),
                "final_gold_n": len(gold),
                "final_sensitivity_n": len(sample),
                "final_grid": "5;10",
                "primary_condition_count": int((matrix.reference_tier == "GOLD").sum()),
                "sensitivity_increment_condition_count": int(
                    (matrix.reference_tier == "SILVER").sum()
                ),
                "total_condition_count": len(matrix),
                "condition_realizations": condition_realizations,
                "reusable_spectrum_fit_inputs": reusable_spectrum_fits,
                "yang_fit_inputs": reusable_spectrum_fits,
                "green_preprocessing_fit_inputs": reusable_spectrum_fits,
                "yang_condition_evaluations": condition_realizations,
                "green_condition_evaluations": condition_realizations,
                "unclassifiable_by_design_cells": int(
                    applicability.unclassifiable_by_design.sum()
                ),
                "mc_realizations": M,
                "execution_status": "UNEXECUTED",
            }
        ]
    )
    workload.to_csv(OUT / "final_q1_workload_d093.csv", index=False)

    if len(old) != 14 or len(new) != 48 or len(gold) != 58 or len(sample) != 62:
        raise RuntimeError("D-093 final reference counts changed")
    if matrix.execution_authorized.any() or matrix.spectra_generated.any():
        raise RuntimeError("D-093 matrix must remain unexecuted")
    print(workload.to_string(index=False))
    print(
        support[support.scope.isin({"ALL_FINAL_GOLD", "YANG_APPLICABLE_FINAL_GOLD", "GREEN_APPLICABLE_FINAL_GOLD"})].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
