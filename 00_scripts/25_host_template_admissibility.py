#!/usr/bin/env python
"""Step 17H for the corrected eight-record DESI recovery block.

The decision uses structural agreement only.  pPXF objective values and
residuals are copied into the evidence table but no universal delta-chi-square
cut is applied: Gate C does not calibrate the full fitted wavelength domain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

ROBUST = "HOST_TEMPLATE_ROBUST"
SENSITIVE = "HOST_TEMPLATE_MODEL_SENSITIVE"
LOW_IDENTIFIABILITY = "HOST_TEMPLATE_LOW_IDENTIFIABILITY"
NO_PRIMARY = "NO_PRIMARY_HOST_TEMPLATE"

# The optimizer emits values around 1e-36 when a non-negative component is on
# its numerical zero boundary.  This tolerance distinguishes that numerical
# encoding from a fitted component; it is not an astrophysical detection cut.
NUMERICAL_BOUNDARY = 1e-30


def boundary(value) -> bool:
    return pd.isna(value) or float(value) <= NUMERICAL_BOUNDARY


def shape_agreement(component_dir: Path, index: int):
    pyq_path = component_dir / f"pyqsofit_host_{index:02d}.npz"
    ppxf_path = component_dir / f"ppxf_{index:02d}_XSL_M2.npz"
    if not pyq_path.exists() or not ppxf_path.exists():
        return None, None, None
    pyq = np.load(pyq_path)
    ppxf = np.load(ppxf_path)
    grid = np.arange(4000.0, 5500.0, 2.0)

    def normalized(archive, key):
        wave, values = archive["wavelength"], archive[key]
        window = (wave >= 5080.0) & (wave <= 5120.0)
        level = float(np.median(values[window]))
        if window.sum() < 5 or level <= 0:
            return None
        return np.interp(grid, wave, values / level, left=np.nan, right=np.nan)

    a, b = normalized(pyq, "host"), normalized(ppxf, "stellar")
    if a is None or b is None:
        return None, None, None
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 50:
        return None, None, None
    difference = a[good] - b[good]
    return (
        float(np.corrcoef(a[good], b[good])[0, 1]),
        float(np.median(np.abs(difference))),
        float(np.sqrt(np.mean(difference**2))),
    )


def main() -> int:
    root = project_root()
    output = root / "06_fitting/recovery_17gr"
    pyq = pd.read_csv(output / "pyqsofit_corrected_desi.csv")
    ppxf = pd.read_csv(output / "ppxf_corrected_desi_models.csv")
    if len(pyq) != 8:
        raise RuntimeError("17H requires the complete eight-record 17G-R PyQSOFit table")

    rows = []
    for index, q in enumerate(pyq.itertuples(), 1):
        block = ppxf[
            (ppxf.science_record_id == q.science_record_id) & (ppxf.library == "XSL")
        ]
        primary = block[block.model.isin(["M1b", "M2", "M3"])]
        singles = block[block.model == "M1a"]
        emiles = ppxf[
            (ppxf.science_record_id == q.science_record_id)
            & (ppxf.library == "E-MILES")
        ].iloc[0]
        if len(primary) != 3 or len(singles) != 6:
            raise RuntimeError(f"incomplete frozen pPXF family for {q.science_record_id}")
        if not primary.reconstruction_identity_pass.fillna(False).all():
            raise RuntimeError(f"reconstruction hard gate failed for {q.science_record_id}")

        primary_boundary = [boundary(v) for v in primary.stellar_weight_sum]
        single_boundary = [boundary(v) for v in singles.stellar_weight_sum]
        pyq_nonzero = q.host_output_state == "OUTPUT_VALID_NONZERO"
        all_primary_zero = all(primary_boundary)
        all_primary_nonzero = not any(primary_boundary)
        all_single_zero = all(single_boundary)
        states_disagree = pyq_nonzero != all_primary_nonzero
        correlation, median_abs, rms_shape = shape_agreement(output / "components", index)

        # Frozen-category assignment from structural evidence.  Objective
        # closeness never appears in this branch logic.
        if not pyq_nonzero and all_primary_zero:
            label = NO_PRIMARY
            reason = "PyQSOFit declined and every physical XSL M1b/M2/M3 model is at zero"
        elif pyq_nonzero and all_primary_zero:
            label = LOW_IDENTIFIABILITY
            reason = "PyQSOFit host exists but every physical XSL model is at zero"
        elif all_single_zero or states_disagree:
            label = SENSITIVE
            reason = (
                "host state changes across frozen continuum families while the reviewed "
                "localized Hbeta residuals remain descriptively similar; no universal "
                "objective or residual cutoff was applied"
            )
        elif pyq_nonzero and all_primary_nonzero and q.host_reconstruction_identity_pass:
            label = ROBUST
            reason = "nonzero reconstructed host is structurally stable across frozen families"
        else:
            label = LOW_IDENTIFIABILITY
            reason = "host evidence is insufficient for a stable primary representation"

        best_single = singles.loc[singles.chi2_descriptive.idxmin()]
        bank = primary[primary.model == "M1b"].iloc[0]
        rows.append({
            "object_id": q.object_id, "science_record_id": q.science_record_id,
            "program": q.program, "admissibility_class": label,
            "admissible_as_fixed_primary_injection_template": label == ROBUST,
            "reason": reason, "pyqsofit_host_state": q.host_output_state,
            "pyqsofit_f_host_cont": q.host_f_host_cont,
            "pyqsofit_decline_reason": q.decline_reason,
            "pyqsofit_reconstruction_pass": q.host_reconstruction_identity_pass,
            "xsl_reconstruction_all_pass": True,
            "xsl_m1b_f_host_cont": bank.f_host_cont,
            "xsl_m2_f_host_cont": primary[primary.model == "M2"].iloc[0].f_host_cont,
            "xsl_m3_f_host_cont": primary[primary.model == "M3"].iloc[0].f_host_cont,
            "xsl_primary_all_numerical_zero": all_primary_zero,
            "xsl_m1a_all_numerical_zero": all_single_zero,
            "xsl_best_m1a_slope_by_objective": best_single.slope,
            "xsl_best_m1a_f_host_cont": best_single.f_host_cont,
            "xsl_m1b_objective_descriptive": bank.chi2_descriptive,
            "xsl_best_m1a_objective_descriptive": best_single.chi2_descriptive,
            "xsl_m1b_rms_stellar_features": bank.rms_stellar_features,
            "xsl_best_m1a_rms_stellar_features": best_single.rms_stellar_features,
            "xsl_m1b_rms_hbeta": bank.rms_hbeta,
            "xsl_best_m1a_rms_hbeta": best_single.rms_hbeta,
            "xsl_m1b_over_best_m1a_rms_hbeta_descriptive": (
                bank.rms_hbeta / best_single.rms_hbeta
            ),
            "comparability_adjudication": (
                "QUALITATIVE_LOCAL_RESIDUAL_REVIEW_NO_UNIVERSAL_THRESHOLD"
                if label == SENSITIVE else "NOT_APPLICABLE"
            ),
            "stellar_alignment_status": bank.stellar_alignment_status,
            "stellar_alignment_n_measured": bank.stellar_alignment_n_measured,
            "pyqsofit_xsl_host_shape_correlation": correlation,
            "pyqsofit_xsl_host_shape_median_abs_difference": median_abs,
            "pyqsofit_xsl_host_shape_rms_difference": rms_shape,
            "emiles_status": emiles.status,
            "full_spectrum_uncertainty_status": "UNCERTAINTY_CALIBRATION_OUT_OF_DOMAIN",
            "universal_delta_chi2_threshold_applied": False,
            "universal_residual_ratio_threshold_applied": False,
        })

    result = pd.DataFrame(rows)
    result.to_csv(output / "host_template_admissibility_17h.csv", index=False)
    missing = result[~result.admissible_as_fixed_primary_injection_template][[
        "object_id", "science_record_id", "program", "admissibility_class", "reason"
    ]]
    missing.to_csv(output / "spectra_without_admissible_primary_host.csv", index=False)
    print(result.admissibility_class.value_counts().to_string())
    print(f"admissible fixed primary templates: {int(result.admissible_as_fixed_primary_injection_template.sum())}/8")
    print(f"written -> {output / 'host_template_admissibility_17h.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
