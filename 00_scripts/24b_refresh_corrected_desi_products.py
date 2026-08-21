#!/usr/bin/env python
"""Build criterion-ready recovery products without running a final classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.calibration.support import CalibrationSupport  # noqa: E402
from p3sf.config import project_root  # noqa: E402


def main() -> int:
    root = project_root()
    output = root / "06_fitting/recovery_17gr"
    manifest = pd.read_csv(root / "03_spectra/raw_desi/manifest_slice.csv")
    qc = pd.read_csv(root / "03_spectra/qc/qc_verdicts_slice.csv")
    fits = pd.read_csv(output / "pyqsofit_corrected_desi.csv")
    records = manifest.merge(
        qc[["spectrum_id", "snr_hbeta_window", "snr_continuum_5100",
            "continuum_level_5100", "qc_pass"]], on="spectrum_id", validate="one_to_one",
    )
    refreshed = records[[
        "object_id", "spectrum_id", "program", "redshift", "continuum_level_5100",
        "snr_continuum_5100", "snr_hbeta_window", "qc_pass",
    ]].rename(columns={"spectrum_id": "science_record_id"}).merge(
        fits, on=["object_id", "science_record_id", "program", "redshift"],
        validate="one_to_one", suffixes=("_qc", "_fit"),
    )
    if len(refreshed) != 8:
        raise RuntimeError("refreshed recovery product requires exactly eight fit rows")

    is_j082 = refreshed.object_id == "SDSSJ082942.66+415436.8"
    C = CalibrationSupport
    refreshed["continuum_5100_support"] = str(C.CALIBRATED)
    refreshed["broad_hbeta_point_support"] = str(C.CALIBRATED)
    refreshed["broad_hbeta_uncertainty_support"] = str(C.CALIBRATED)
    refreshed["broad_hbeta_detectability_support"] = str(C.CALIBRATED)
    refreshed["broad_hbeta_detected_primary"] = refreshed.broad_hbeta_detected.astype(
        "boolean"
    )
    refreshed.loc[is_j082, "broad_hbeta_point_support"] = str(
        C.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED
    )
    refreshed.loc[is_j082, "broad_hbeta_uncertainty_support"] = str(
        C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION
    )
    refreshed.loc[is_j082, "broad_hbeta_detectability_support"] = str(
        C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION
    )
    refreshed.loc[is_j082, "broad_hbeta_detected_primary"] = pd.NA
    refreshed["stored_fit_uncertainty_semantics"] = (
        "NATIVE_IVAR_NON_MCMC; LOCAL_GATE_C_SUPPORT_MUST_BE_APPLIED_BY_DEPENDENCY"
    )
    refreshed.to_csv(output / "refreshed_desi_measurements.csv", index=False)

    accounting_rows: list[dict[str, object]] = []
    dependencies = (
        ("continuum_5100", C.CALIBRATED),
        ("broad_hbeta_flux_point", C.CALIBRATED),
        ("broad_hbeta_significance", C.CALIBRATED),
        ("macleod_green", C.DEPENDENCY_NOT_YET_DEFINABLE),
        ("potts_villforth", C.DEPENDENCY_NOT_YET_DEFINABLE),
        ("full_spectrum_decomposition_uncertainty", C.OUT_OF_DOMAIN),
    )
    for row in refreshed.itertuples():
        for dependency, default_support in dependencies:
            support = default_support
            fit_valid = row.status == "ok"
            if dependency == "broad_hbeta_flux_point":
                fit_valid = fit_valid and np.isfinite(row.broad_hbeta_flux)
                if is_j082.loc[row.Index]:
                    support = C.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED
            elif dependency == "broad_hbeta_significance":
                fit_valid = fit_valid and np.isfinite(row.broad_hbeta_snr)
                if is_j082.loc[row.Index]:
                    support = C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION
            elif dependency == "continuum_5100":
                fit_valid = np.isfinite(row.continuum_level_5100)
            classifiable = bool(fit_valid and support.classifiable)
            accounting_rows.append({
                "object_id": row.object_id, "science_record_id": row.science_record_id,
                "program": row.program, "criterion_or_input": dependency,
                "calibration_support": str(support), "n_eligible": 1,
                "n_fit_completed": int(row.fit_completed),
                "n_fit_valid": int(fit_valid), "n_classifiable": int(classifiable),
                "final_classifier_run": False,
            })
    accounting = pd.DataFrame(accounting_rows)
    if not (
        (accounting.n_classifiable <= accounting.n_fit_valid)
        & (accounting.n_fit_valid <= accounting.n_fit_completed)
        & (accounting.n_fit_completed <= accounting.n_eligible)
    ).all():
        raise RuntimeError("recovery denominator nesting violated")
    accounting.to_csv(output / "criterion_input_accounting.csv", index=False)

    impact = pd.read_csv(output / "old_vs_corrected_fit_impact.csv")
    if "old_frac_host_5100_pyqsofit_native" in impact:
        impact["old_f_host_cont"] = impact["old_frac_host_5100_pyqsofit_native"]
    stability = pd.read_csv(root / "06_fitting/validation/host_stability_slice.csv")
    stability = stability[stability.survey == "DESI"][["object_id", "fhost_baseline"]]
    stability = stability.drop_duplicates("object_id")
    impact = impact.merge(stability, on="object_id", how="left", validate="many_to_one")
    impact["old_frac_host_5100_pyqsofit_native"] = impact["old_f_host_cont"]
    impact["old_f_host_cont"] = impact["fhost_baseline"]
    impact = impact.drop(columns="fhost_baseline")
    impact["old_f_host_cont_definition"] = "COMMON_RECONSTRUCTED_5080_5120"
    impact["new_f_host_cont_definition"] = "COMMON_RECONSTRUCTED_5080_5120"
    impact["old_broad_hbeta_availability"] = "OLD_NOT_PERSISTED"
    impact["old_residual_availability"] = "OLD_NOT_PERSISTED"
    impact["new_broad_hbeta_uncertainty_semantics"] = "NATIVE_IVAR_NON_MCMC"
    impact["new_broad_hbeta_formal_error_availability"] = (
        "NEW_NATIVE_FORMAL_ERROR_NOT_EMITTED"
    )
    impact.to_csv(output / "old_vs_corrected_fit_impact.csv", index=False)
    print(f"written {len(refreshed)} refreshed measurements and {len(accounting)} accounting rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
