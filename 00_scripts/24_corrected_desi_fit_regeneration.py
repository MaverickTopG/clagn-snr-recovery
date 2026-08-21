#!/usr/bin/env python
"""17G-R: regenerate invalidated fit products for the corrected eight DESI records.

This is recovery, not tuning.  The PyQSOFit production and host-audit settings,
pPXF M0--M3 ladder, M1a fine slope grid, and M1b bank are copied from the
already frozen implementations.  Native IVAR is used throughout.  Global pPXF
chi-square is persisted only as a descriptive objective.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.calibration.support import CalibrationSupport  # noqa: E402
from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.fitting.components import from_ppxf, from_pyqsofit  # noqa: E402
from p3sf.fitting.desi_recovery import (  # noqa: E402
    read_corrected_desi,
    residual_summary,
    resolution_fwhm_angstrom,
    run_pyqsofit_host_audit,
)
from p3sf.fitting.ppxf_host import DEFAULT_POWERLAW_SLOPES, decompose  # noqa: E402
from p3sf.fitting.pyqsofit_driver import _diagnostics_from_result, _oiii5007_from_fit  # noqa: E402
from p3sf.qc.alignment import check_rest_frame, check_stellar_alignment  # noqa: E402

FIT_RANGE = (3700.0, 7000.0)
LADDER = {
    "M0": dict(include_powerlaw=False, include_feii=False, include_lines=False,
               include_balmer=False, mask_emission=True),
    "M1b": dict(include_powerlaw=True, include_feii=False, include_lines=True,
                include_balmer=False, mask_emission=False),
    "M2": dict(include_powerlaw=True, include_feii=True, include_lines=True,
               include_balmer=False, mask_emission=False),
    "M3": dict(include_powerlaw=True, include_feii=True, include_lines=True,
               include_balmer=True, mask_emission=False),
}


def finite(mapping: dict[str, object], name: str, *, positive: bool = False):
    try:
        value = float(mapping[name])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or (positive and value <= 0):
        return None
    return value


def run_production_pyq(spectrum, redshift: float, vendor: Path):
    from pyqsofit.PyQSOFit import QSOFit

    good = spectrum.good
    continuum_snr = float(np.median(spectrum.flux[good] / spectrum.error[good]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = QSOFit(
                spectrum.wavelength[good], spectrum.flux[good], spectrum.error[good],
                redshift, path=str(vendor),
            )
            fit.Fit(
                name=None, nsmooth=1, deredden=False, reject_badpix=False,
                decompose_host=True, host_prior=False, npca_gal=5, npca_qso=10,
                Fe_uv_op=True, poly=True, BC=False, linefit=True, MCMC=False,
                plot_fig=False, save_fig=False, save_result=False,
            )
        line = dict(zip(fit.line_result_name, fit.line_result, strict=True))
        cont = dict(zip(fit.conti_result_name, fit.conti_result, strict=True))
        diagnostic = _diagnostics_from_result(line, continuum_snr=continuum_snr)
        oiii_flux, oiii_snr = _oiii5007_from_fit(fit)
        row = asdict(diagnostic)
        row.update(
            l5100=finite(cont, "L5100", positive=True),
            frac_host_5100=finite(cont, "frac_host_5100"),
            oiii5007_flux=oiii_flux,
            oiii5007_snr=oiii_snr,
            production_decomposed=bool(getattr(fit, "decomposed", False)),
        )
        model_flux = np.asarray(fit.f_conti_model) + np.asarray(fit.f_line_model)
        rest = np.asarray(fit.wave) / (1.0 + redshift)
        row.update({f"production_{k}": v for k, v in residual_summary(
            rest, np.asarray(fit.flux), model_flux
        ).items()})
        return fit, row
    except Exception as error:  # noqa: BLE001
        return None, {
            "status": "unusable", "production_fit_exception":
            f"{type(error).__name__}: {str(error)[:160]}",
            "production_decomposed": False,
        }


def template_lsf_verdict(sps_file: Path, rest_lam: np.ndarray, fwhm_rest: np.ndarray):
    archive = np.load(sps_file)
    template_lam = np.asarray(archive["lam"], dtype=float)
    template_fwhm = np.asarray(archive["fwhm"], dtype=float)
    inside = (rest_lam >= FIT_RANGE[0]) & (rest_lam <= FIT_RANGE[1])
    lam = rest_lam[inside]
    gal = fwhm_rest[inside]
    temp = np.interp(lam, template_lam, template_fwhm, left=np.nan, right=np.nan)
    finite_mask = np.isfinite(gal) & np.isfinite(temp)
    incompatible = gal[finite_mask] < temp[finite_mask]
    critical = (lam[finite_mask] >= 4700.0) & (lam[finite_mask] <= 5100.0)
    return {
        "lsf_n_pixels": int(finite_mask.sum()),
        "lsf_incompatible_fraction": float(np.mean(incompatible)),
        "lsf_hbeta_incompatible_fraction": (
            float(np.mean(incompatible[critical])) if critical.sum() else None
        ),
        "lsf_min_galaxy_fwhm": float(np.min(gal[finite_mask])),
        "lsf_median_galaxy_fwhm": float(np.median(gal[finite_mask])),
        "lsf_median_template_fwhm": float(np.median(temp[finite_mask])),
        "lsf_compatible": bool(not incompatible.any()),
    }


def ppxf_record(result, row, library: str, model_name: str, slope=None):
    record = {
        "object_id": row.object_id, "science_record_id": row.spectrum_id,
        "program": row.program, "library": library, "model": model_name,
        "slope": slope, "status": result.status,
        "objective_semantics": "DESCRIPTIVE_NATIVE_IVAR_NOT_GLOBAL_CALIBRATED_CHI2",
        "detail": result.detail,
    }
    if result.status != "ok" or result.fit is None:
        return record, None
    model = from_ppxf(
        result.fit, result.wavelength, result.n_stellar_templates,
        result.nonstellar_names, method=f"ppxf-{library}-{model_name}",
        native_f_host=result.f_host_5100,
    )
    check = model.check_reconstruction()
    record.update(
        reconstruction_identity_pass=check.passed,
        reconstruction_relative_difference=check.relative_difference,
        chi2_descriptive=result.chi2,
        stellar_weight_sum=result.diagnostics.get("stellar_weight_sum"),
        **residual_summary(result.wavelength, result.fit.galaxy, result.fit.bestfit),
    )
    if check.passed:
        record.update(model.canonical_fractions(tuple(load_config().snr_metrics.continuum_5100)))
    return record, model


def j082_support_table() -> pd.DataFrame:
    C = CalibrationSupport
    rows = [
        ("continuum_5100", C.CALIBRATED, "T3-supported; no T2-B dependency"),
        ("delta_l5100", C.CALIBRATED, "T3-supported; no T2-B dependency"),
        ("broad_hbeta_flux_point", C.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED,
         "corrected-spectrum point estimate retained; T2-B 50-55 uncertainty unsupported"),
        ("broad_hbeta_flux_uncertainty", C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION,
         "requires unsupported T2-B 50-55 uncertainty calibration"),
        ("broad_hbeta_significance", C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION,
         "requires unsupported T2-B 50-55 uncertainty calibration"),
        ("difference_spectrum_flux_point", C.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED,
         "point estimate may be retained; calibrated difference uncertainty unavailable"),
        ("difference_spectrum_significance", C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION,
         "T2-dependent significance unsupported"),
        ("yang_flux_ratio", C.POINT_ESTIMATE_AVAILABLE_UNCERTAINTY_UNSUPPORTED,
         "verified point-flux dependency retained without calibrated significance"),
        ("guo_quick_screen", C.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION,
         "variance-based Hbeta screen requires calibrated uncertainty"),
        ("macleod_green", C.DEPENDENCY_NOT_YET_DEFINABLE,
         "exact source implementation not yet frozen"),
        ("potts_villforth", C.DEPENDENCY_NOT_YET_DEFINABLE,
         "exact source implementation not yet frozen"),
        ("full_spectrum_decomposition_uncertainty", C.OUT_OF_DOMAIN,
         "Gate C does not calibrate the full pPXF/PyQSOFit domain"),
    ]
    return pd.DataFrame([
        {"object_id": "SDSSJ082942.66+415436.8", "quantity_or_criterion": quantity,
         "support_status": str(status), "point_estimate_available": status.point_estimate_available,
         "classifiable_from_calibration_support": status.classifiable, "reason": reason}
        for quantity, status, reason in rows
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test prefix only")
    args = parser.parse_args()
    root = project_root()
    cfg = load_config()
    output = root / "06_fitting" / "recovery_17gr"
    arrays = output / "components"
    output.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(root / "03_spectra/raw_desi/manifest_slice.csv")
    qc = pd.read_csv(root / "03_spectra/qc/qc_verdicts_slice.csv")
    records = manifest.merge(
        qc[["spectrum_id", "snr_hbeta_window", "snr_continuum_5100",
            "continuum_level_5100", "qc_pass"]], on="spectrum_id", validate="one_to_one",
    )
    if len(records) != 8 or records.spectrum_id.nunique() != 8:
        raise RuntimeError("17G-R requires exactly eight unique corrected DESI science records")
    if not records.qc_pass.all():
        raise RuntimeError("17G-R refuses a corrected record that did not pass frozen QC")
    if args.limit is not None:
        records = records.head(args.limit)

    vendor = root / "06_fitting/pyqsofit/vendor/PyQSOFit"
    feii = vendor / "src/pyqsofit/fe_optical.txt"
    libraries = {
        "XSL": root / "06_fitting/alternative_fit/sps_libraries/spectra_xsl_9.0.npz",
        "E-MILES": root / "06_fitting/alternative_fit/sps_libraries/spectra_emiles_9.0.npz",
    }
    pyq_rows: list[dict[str, object]] = []
    ppxf_rows: list[dict[str, object]] = []

    for i, row in enumerate(records.itertuples(), 1):
        print(f"[{i}/{len(records)}] {row.object_id} {row.program} {row.spectrum_id}", flush=True)
        spectrum = read_corrected_desi(root / row.path)
        host_fit, host = run_pyqsofit_host_audit(spectrum, row.redshift, vendor)
        prod_fit, prod = run_production_pyq(spectrum, row.redshift, vendor)
        pyq = {
            "object_id": row.object_id, "science_record_id": row.spectrum_id,
            "program": row.program, "redshift": row.redshift,
            "fit_completed": host_fit is not None and prod_fit is not None,
            "native_ivar": True, "global_k_applied": False,
            **host, **prod,
        }
        if host_fit is not None and bool(getattr(host_fit, "decomposed", False)):
            model = from_pyqsofit(host_fit, row.redshift, native_f_host=getattr(host_fit, "frac_host_5100", None))
            check = model.check_reconstruction()
            pyq["host_reconstruction_identity_pass"] = check.passed
            if check.passed:
                fractions = model.canonical_fractions(tuple(cfg.snr_metrics.continuum_5100))
                pyq["host_f_host_cont"] = fractions["f_host_cont"]
                pyq["host_f_host_pseudo"] = None  # PCA QSO cannot isolate pseudocontinuum
            np.savez_compressed(
                arrays / f"pyqsofit_host_{i:02d}.npz", wavelength=model.wavelength,
                host=model.get("stellar"), qso=model.get("agn_smooth"), bestfit=model.bestfit,
            )
        else:
            pyq["host_reconstruction_identity_pass"] = None
            pyq["host_f_host_cont"] = None
            pyq["host_f_host_pseudo"] = None
        if prod_fit is not None:
            np.savez_compressed(
                arrays / f"pyqsofit_production_{i:02d}.npz",
                wavelength=np.asarray(prod_fit.wave) / (1 + row.redshift),
                flux=np.asarray(prod_fit.flux), continuum=np.asarray(prod_fit.f_conti_model),
                lines=np.asarray(prod_fit.f_line_model),
            )
        pyq_rows.append(pyq)

        fwhm_obs = resolution_fwhm_angstrom(spectrum.wavelength, spectrum.resolution)
        rest_lam = spectrum.wavelength[spectrum.good] / (1.0 + row.redshift)
        fwhm_rest = fwhm_obs[spectrum.good] / (1.0 + row.redshift)
        alignment_rest = check_rest_frame(
            rest_lam, spectrum.flux[spectrum.good], spectrum.error[spectrum.good]
        )
        alignment_stellar = check_stellar_alignment(
            rest_lam, spectrum.flux[spectrum.good], spectrum.error[spectrum.good]
        )
        for library, sps_file in libraries.items():
            verdict = template_lsf_verdict(sps_file, rest_lam, fwhm_rest)
            if not verdict["lsf_compatible"]:
                ppxf_rows.append({
                    "object_id": row.object_id, "science_record_id": row.spectrum_id,
                    "program": row.program, "library": library, "model": "ALL_FROZEN",
                    "status": "LSF_INCOMPATIBLE_NOT_RUN",
                    "objective_semantics": "NOT_APPLICABLE",
                    "rest_alignment_status": str(alignment_rest.status),
                    "stellar_alignment_status": str(alignment_stellar.status),
                    **verdict,
                })
                continue
            fwhm_gal = {"lam": rest_lam, "fwhm": fwhm_rest}
            model_runs = list(LADDER.items()) + [
                ("M1a", dict(include_powerlaw=True, include_feii=False, include_lines=True,
                             include_balmer=False, mask_emission=False,
                             powerlaw_slopes=(slope,)))
                for slope in DEFAULT_POWERLAW_SLOPES
            ]
            # One cache per science record: every family sees the identical LSF,
            # while no SPS object is reused across records with a different LSF.
            sps_cache: dict[str, Any] = {}
            for run_index, (model_name, switches) in enumerate(model_runs):
                slope = switches.get("powerlaw_slopes", (None,))[0]
                result = decompose(
                    spectrum.wavelength[spectrum.good], spectrum.flux[spectrum.good],
                    spectrum.error[spectrum.good], row.redshift, sps_file=sps_file,
                    fhost_window=tuple(cfg.snr_metrics.continuum_5100), fwhm_gal=fwhm_gal,
                    feii_path=feii, fit_range=FIT_RANGE, **switches,
                    sps_cache=sps_cache,
                )
                record, model = ppxf_record(result, row, library, model_name, slope)
                record.update(
                    rest_alignment_status=str(alignment_rest.status),
                    rest_alignment_offset_angstrom=alignment_rest.offset_angstrom,
                    stellar_alignment_status=str(alignment_stellar.status),
                    stellar_alignment_n_measured=alignment_stellar.n_measured,
                    stellar_alignment_median_offset_kms=alignment_stellar.median_offset_kms,
                    **verdict,
                )
                ppxf_rows.append(record)
                if model is not None and model_name in {"M0", "M1b", "M2", "M3"}:
                    np.savez_compressed(
                        arrays / f"ppxf_{i:02d}_{library.replace('-', '')}_{model_name}.npz",
                        wavelength=model.wavelength, bestfit=model.bestfit,
                        stellar=model.get("stellar"), agn_smooth=model.get("agn_smooth"),
                        feii=model.get("feii"), balmer=model.get("balmer_cont"),
                        lines=model.get("lines"),
                    )
                if run_index % 10 == 0:
                    print(f"  {library} {model_name} {slope}: {result.status}", flush=True)

    pyq_frame = pd.DataFrame(pyq_rows)
    ppxf_frame = pd.DataFrame(ppxf_rows)
    pyq_frame.to_csv(output / "pyqsofit_corrected_desi.csv", index=False)
    ppxf_frame.to_csv(output / "ppxf_corrected_desi_models.csv", index=False)
    j082_support_table().to_csv(output / "j082_criterion_support.csv", index=False)

    old = pd.read_csv(root / "06_fitting/validation/host_output_audit_slice.csv")
    old = old[old.survey == "DESI"].copy()
    impact_rows = []
    for new in pyq_frame.itertuples():
        prior = old[old.object_id == new.object_id]
        invalid_overwrite = new.object_id == "SDSSJ161711.42+063833.4"
        old_row = prior.iloc[0] if len(prior) == 1 and not invalid_overwrite else None
        old_status = (
            "INVALID_OLD_PRODUCT_FILE_OVERWRITE" if invalid_overwrite
            else (old_row.host_output_state if old_row is not None else "OLD_BASELINE_UNAVAILABLE")
        )
        impact_rows.append({
            "object_id": new.object_id, "science_record_id": new.science_record_id,
            "program": new.program, "old_fit_status": old_status,
            "new_fit_status": new.status, "old_decomposition_status": old_status,
            "new_decomposition_status": new.host_output_state,
            "old_broad_hbeta_flux": None, "new_broad_hbeta_flux": new.broad_hbeta_flux,
            "old_broad_hbeta_flux_error_native": None,
            "new_broad_hbeta_flux_error_native": new.broad_hbeta_flux_error,
            "old_f_host_cont": (old_row.frac_host_5100 if old_row is not None else None),
            "new_f_host_cont": new.host_f_host_cont,
            "old_f_host_pseudo": None, "new_f_host_pseudo": new.host_f_host_pseudo,
            "old_residual_metrics": "NOT_PERSISTED",
            "new_rms_global": getattr(new, "production_rms_global", None),
            "new_rms_stellar_features": getattr(new, "production_rms_stellar_features", None),
            "new_rms_hbeta": getattr(new, "production_rms_hbeta", None),
            "reconstruction_identity_pass": new.host_reconstruction_identity_pass,
            "material_domain_differences": (
                "old malformed camera concatenation; corrected official camera merge; "
                "unique full science ID; native IVAR; modern RESOLUTION; "
                + ("old J161 file overwritten by DARK product" if invalid_overwrite else
                   "old broad-line and residual values not persisted")
            ),
        })
    pd.DataFrame(impact_rows).to_csv(output / "old_vs_corrected_fit_impact.csv", index=False)

    refreshed = records[[
        "object_id", "spectrum_id", "program", "redshift", "continuum_level_5100",
        "snr_continuum_5100", "snr_hbeta_window",
    ]].rename(columns={"spectrum_id": "science_record_id"}).merge(
        pyq_frame, on=["object_id", "science_record_id", "program", "redshift"],
        validate="one_to_one", suffixes=("_qc", "_fit"),
    )
    refreshed["n_eligible"] = 1
    refreshed["n_fit_completed"] = refreshed.fit_completed.astype(int)
    refreshed["n_fit_valid"] = (refreshed.status == "ok").astype(int)
    refreshed["n_classifiable"] = refreshed["n_fit_valid"]
    is_j082 = refreshed.object_id == "SDSSJ082942.66+415436.8"
    refreshed.loc[is_j082, "broad_hbeta_uncertainty_support"] = str(
        CalibrationSupport.UNCLASSIFIABLE_UNCERTAINTY_CALIBRATION
    )
    refreshed.loc[is_j082, "n_classifiable"] = 0
    refreshed.loc[~is_j082, "broad_hbeta_uncertainty_support"] = str(
        CalibrationSupport.CALIBRATED
    )
    refreshed.to_csv(output / "refreshed_desi_measurements.csv", index=False)
    print(f"written {len(pyq_frame)} PyQSOFit rows and {len(ppxf_frame)} pPXF rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
