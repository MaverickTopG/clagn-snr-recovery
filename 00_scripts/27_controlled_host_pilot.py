#!/usr/bin/env python
"""Execute only the frozen D-083 40-cell controlled-host pilot."""

from __future__ import annotations

import hashlib
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.counterfactual.host import (  # noqa: E402
    PairFluxBasis,
    added_star_scale_for_target,
    fixed_continuum_snr_intervention,
    realized_added_star_fraction,
    reference_anchored_pair_amplitude,
)
from p3sf.counterfactual.stellar import transform_xsl_to_desi  # noqa: E402
from p3sf.fitting.desi_recovery import (  # noqa: E402
    DesiSpectrum,
    read_corrected_desi,
    residual_summary,
)
from p3sf.fitting.pyqsofit_driver import (  # noqa: E402
    _diagnostics_from_result,
    _oiii5007_from_fit,
)

WINDOW = (5080.0, 5130.0)
EXPECTED_SHAPES = {"XSL_H1", "XSL_H2", "XSL_H3"}
EXPECTED_RUNGS = {0.0, 0.1, 0.5, 0.85}
PAIR_ARM = "PAIR_J161_FAINT_ANCHORED"


@dataclass(frozen=True)
class Baseline:
    science_record_id: str
    object_id: str
    path: Path
    redshift: float
    mjd: float
    physical_role: str
    data: DesiSpectrum
    mask: np.ndarray
    spectrum: Spectrum


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def window_median(wavelength, values, redshift, good) -> float:
    rest = wavelength / (1.0 + redshift)
    inside = (rest >= WINDOW[0]) & (rest <= WINDOW[1]) & good & np.isfinite(values)
    if inside.sum() < 5:
        raise ValueError("insufficient frozen-window coverage")
    return float(np.median(values[inside]))


def load_baselines(root: Path) -> dict[str, Baseline]:
    manifest = pd.read_csv(root / "03_spectra/raw_desi/manifest_slice.csv")
    wanted = {
        "desi:iron:healpix-coadd:main:bright:hp18789:tid39627796707803702:zpix158456325139248767426053014070": "faint",
        "desi:iron:healpix-coadd:main:bright:hp31535:tid39627951624425454:zpix158456325139248767580969635822": "bright",
        "desi:iron:healpix-coadd:main:dark:hp31535:tid39627951624425454:zpix237684487653513105174513586158": "faint",
    }
    output: dict[str, Baseline] = {}
    for science_id, physical_role in wanted.items():
        match = manifest[manifest.spectrum_id == science_id]
        if len(match) != 1:
            raise RuntimeError(f"baseline manifest lookup failed for {science_id}")
        row = match.iloc[0]
        path = root / str(row.path)
        data = read_corrected_desi(path)
        with fits.open(path) as hdul:
            mask = np.asarray(hdul["MASK"].data).copy()
        combined_mask = ((mask != 0) | (~data.good)).astype(np.int64)
        spectrum = Spectrum(
            data.wavelength, data.flux, data.error, float(row.redshift),
            mask=combined_mask,
            meta={"science_record_id": science_id, "authoritative_corrected_baseline": True},
        )
        output[science_id] = Baseline(
            science_id, str(row.object_id), path, float(row.redshift),
            float(row.mjd_effective), physical_role, data, mask, spectrum,
        )
    return output


def load_sdss_comparator(root: Path) -> tuple[Spectrum, float]:
    path = root / "03_spectra/raw_sdss/SDSSJ233602.98+001728.7__early__spec-0385-51783-0360.fits"
    with fits.open(path) as hdul:
        table = hdul[1].data
        wave = 10.0 ** np.asarray(table["loglam"], dtype=float)
        flux = np.asarray(table["flux"], dtype=float)
        ivar = np.asarray(table["ivar"], dtype=float)
        mask = (ivar <= 0).astype(np.int64)
    error = np.full(ivar.shape, np.nan)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    return Spectrum(wave, flux, error, 0.2428296, mask=mask), 51783.0


def run_frozen_fit(spectrum: Spectrum, vendor: Path):
    """Exact production settings from 24_corrected_desi_fit_regeneration.py."""
    from pyqsofit.PyQSOFit import QSOFit

    good = spectrum.good
    continuum_snr = float(np.median(spectrum.flux[good] / spectrum.error[good]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = QSOFit(
                spectrum.wavelength[good], spectrum.flux[good], spectrum.error[good],
                spectrum.redshift, path=str(vendor),
            )
            fit.Fit(
                name=None, nsmooth=1, deredden=False, reject_badpix=False,
                decompose_host=True, host_prior=False, npca_gal=5, npca_qso=10,
                Fe_uv_op=True, poly=True, BC=False, linefit=True, MCMC=False,
                plot_fig=False, save_fig=False, save_result=False,
            )
        line = dict(zip(fit.line_result_name, fit.line_result, strict=True))
        cont = dict(zip(fit.conti_result_name, fit.conti_result, strict=True))
        row = asdict(_diagnostics_from_result(line, continuum_snr=continuum_snr))
        row["fit_completed"] = True
        row["fit_valid"] = row["status"] == "ok"
        row["production_decomposed"] = bool(getattr(fit, "decomposed", False))
        for key in ("L5100", "frac_host_5100"):
            try:
                value = float(cont[key])
            except (KeyError, TypeError, ValueError):
                value = np.nan
            row[key.lower()] = value if np.isfinite(value) else None
        oiii_flux, oiii_snr = _oiii5007_from_fit(fit)
        row["oiii5007_flux"] = oiii_flux
        row["oiii5007_snr"] = oiii_snr
        model = np.asarray(fit.f_conti_model) + np.asarray(fit.f_line_model)
        rest = np.asarray(fit.wave) / (1.0 + spectrum.redshift)
        row.update({f"production_{key}": value for key, value in residual_summary(
            rest, np.asarray(fit.flux), model
        ).items()})
        return row
    except Exception as error:  # noqa: BLE001
        return {
            "status": "unusable", "fit_completed": False, "fit_valid": False,
            "production_fit_exception": f"{type(error).__name__}: {str(error)[:160]}",
        }


def assert_matrix(matrix: pd.DataFrame) -> None:
    if len(matrix) != 40 or matrix.pilot_arm_id.nunique() != 4:
        raise RuntimeError("pilot matrix is not the frozen 40-cell/four-arm design")
    if not matrix.groupby("pilot_arm_id").size().eq(10).all():
        raise RuntimeError("every frozen pilot arm must contain exactly 10 cells")
    if set(matrix.requested_f_added_star_reference) != EXPECTED_RUNGS:
        raise RuntimeError("pilot rung set changed")
    if set(matrix.loc[matrix.host_shape_id != "NONE", "host_shape_id"]) != EXPECTED_SHAPES:
        raise RuntimeError("pilot XSL ensemble changed")
    zero = matrix.requested_f_added_star_reference.eq(0.0)
    if not (matrix.loc[zero, "host_shape_id"] == "NONE").all():
        raise RuntimeError("zero-rung shape must be NONE")


def yang_label(bright: dict, faint: dict) -> tuple[str, float | None, str]:
    """D-020/D-021 Yang 2024 protocol: faint/bright broad-line flux < 0.3."""
    if not bright.get("fit_valid") or not faint.get("fit_valid"):
        return "unclassifiable", None, "one or both frozen fits invalid"
    bright_flux = bright.get("broad_hbeta_flux")
    faint_flux = faint.get("broad_hbeta_flux")
    if bright_flux is None or not np.isfinite(bright_flux) or bright_flux <= 0:
        return "unclassifiable", None, "positive physical-bright broad-Hbeta flux unavailable"
    faint_protocol = 0.0 if not faint.get("broad_hbeta_detected") else faint_flux
    if faint_protocol is None or not np.isfinite(faint_protocol) or faint_protocol < 0:
        return "unclassifiable", None, "physical-faint protocol flux unavailable"
    ratio = float(faint_protocol / bright_flux)
    return ("CL" if ratio < 0.3 else "non-CL"), ratio, "Yang faint non-detection=0 protocol"


def main() -> int:
    root = project_root()
    cfg = load_config()
    if tuple(cfg.snr_metrics.continuum_5100) != WINDOW:
        raise RuntimeError("frozen configuration no longer uses rest 5080-5130 A")
    output = root / "05_analysis/host_shape_pilot/execution"
    products = output / "spectra"
    output.mkdir(parents=True, exist_ok=True)
    products.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(root / "05_analysis/host_shape_pilot/revised_pilot_matrix.csv")
    assert_matrix(matrix)
    matrix = matrix.copy()
    matrix.insert(0, "cell_id", [f"HC{i:03d}" for i in range(1, len(matrix) + 1)])
    baselines = load_baselines(root)
    sdss_bright, sdss_mjd = load_sdss_comparator(root)
    vendor = root / "06_fitting/pyqsofit/vendor/PyQSOFit"
    archive = np.load(root / "06_fitting/alternative_fit/sps_libraries/spectra_xsl_9.0.npz")
    ensemble = np.load(root / "05_analysis/host_shape_pilot/proposed_xsl_host_shape_ensemble_intrinsic.npz")
    template_wave = np.asarray(ensemble["wavelength_rest_intrinsic"], dtype=float)
    template_fwhm = np.asarray(archive["fwhm"], dtype=float)
    expected = pd.read_csv(root / "06_fitting/recovery_17gr/pyqsofit_corrected_desi.csv")
    transforms: dict[tuple[str, str], tuple[np.ndarray, dict]] = {}
    for science_id, baseline in baselines.items():
        for shape in sorted(EXPECTED_SHAPES):
            transformed, diagnostic = transform_xsl_to_desi(
                template_wave, np.asarray(ensemble[shape], dtype=float), template_fwhm,
                baseline.data.wavelength, baseline.data.resolution,
                redshift=baseline.redshift, losvd_sigma_kms=150.0,
            )
            transforms[(science_id, shape)] = (transformed, asdict(diagnostic))

    comparator_fits: dict[str, dict] = {
        "sdss_j233_bright": run_frozen_fit(sdss_bright, vendor)
    }
    cell_rows: list[dict] = []
    measurement_rows: list[dict] = []
    classification_rows: list[dict] = []

    for row in matrix.itertuples(index=False):
        reference = baselines[row.reference_science_record_id]
        is_pair = row.pilot_arm_id == PAIR_ARM
        requested = float(row.requested_f_added_star_reference)
        shape = str(row.host_shape_id)
        targets = [reference]
        if is_pair:
            targets.append(baselines[row.other_science_record_id])
        amplitudes: dict[str, float] = {}
        realized: dict[str, float] = {}
        if requested == 0.0:
            for target in targets:
                amplitudes[target.science_record_id] = 0.0
                realized[target.science_record_id] = 0.0
        elif is_pair:
            faint_h = transforms[(reference.science_record_id, shape)][0]
            bright = baselines[row.other_science_record_id]
            bright_h = transforms[(bright.science_record_id, shape)][0]
            pair = reference_anchored_pair_amplitude(
                reference.data.wavelength, reference.data.flux, faint_h,
                bright.data.wavelength, bright.data.flux, bright_h,
                reference_redshift=reference.redshift, other_redshift=bright.redshift,
                requested_f_added_star_reference=requested, window_rest=WINDOW,
                flux_basis=PairFluxBasis.SAME_SURVEY_SAME_APERTURE,
            )
            amplitudes[reference.science_record_id] = pair.shared_scale
            amplitudes[bright.science_record_id] = pair.shared_scale
            realized[reference.science_record_id] = pair.realized_f_added_star_reference
            realized[bright.science_record_id] = pair.realized_f_added_star_other
        else:
            host = transforms[(reference.science_record_id, shape)][0]
            scale = added_star_scale_for_target(
                reference.data.wavelength, reference.data.flux, host,
                redshift=reference.redshift, requested_f_added_star=requested,
                window_rest=WINDOW,
            )
            amplitudes[reference.science_record_id] = scale
            realized[reference.science_record_id] = realized_added_star_fraction(
                reference.data.wavelength, reference.data.flux, scale * host,
                redshift=reference.redshift, window_rest=WINDOW,
            )

        fit_by_role: dict[str, dict] = {}
        normalization_pass = True
        noop_pass = True
        shared_pass = True
        for target in targets:
            transformed = (
                np.zeros_like(target.data.flux)
                if requested == 0.0 else transforms[(target.science_record_id, shape)][0]
            )
            added = amplitudes[target.science_record_id] * transformed
            intervention = fixed_continuum_snr_intervention(
                target.spectrum, added, window_rest=WINDOW
            )
            spectrum = intervention.spectrum
            achieved_fraction = realized_added_star_fraction(
                target.data.wavelength, target.data.flux, added,
                redshift=target.redshift, window_rest=WINDOW,
            )
            anchored = target.science_record_id == reference.science_record_id
            identity_error = abs(achieved_fraction - requested) if anchored else None
            cell_normalization_pass = bool(not anchored or identity_error < 1e-9)
            normalization_pass &= cell_normalization_pass
            same_wavelength = np.array_equal(spectrum.wavelength, target.data.wavelength)
            same_mask = np.array_equal(spectrum.mask, target.spectrum.mask)
            same_resolution = np.array_equal(target.data.resolution, read_corrected_desi(target.path).resolution)
            if requested == 0.0:
                local_noop = bool(
                    spectrum is target.spectrum
                    and same_wavelength and np.array_equal(spectrum.flux, target.data.flux)
                    and np.array_equal(spectrum.error, target.data.error, equal_nan=True)
                    and same_mask and same_resolution
                )
                noop_pass &= local_noop
            else:
                local_noop = True
            fit = run_frozen_fit(spectrum, vendor) if cell_normalization_pass else {
                "status": "HOST_NORMALIZATION_IDENTITY_FAIL", "fit_completed": False,
                "fit_valid": False,
            }
            role = target.physical_role
            fit_by_role[role] = fit
            baseline_continuum = window_median(
                target.data.wavelength, target.data.flux, target.redshift, target.data.good
            )
            added_continuum = window_median(
                target.data.wavelength, added, target.redshift, target.data.good
            )
            final_continuum = window_median(
                target.data.wavelength, spectrum.flux, target.redshift, target.data.good
            )
            product = products / f"{row.cell_id}__{role}.npz"
            np.savez_compressed(
                product, wavelength=spectrum.wavelength, flux=spectrum.flux,
                error=spectrum.error, mask=spectrum.mask,
                resolution=target.data.resolution, added_stellar_flux=added,
                good=spectrum.good,
            )
            measurement = {
                "cell_id": row.cell_id, "pilot_arm_id": row.pilot_arm_id,
                "semantics": row.semantics, "object_id": row.object_id,
                "science_record_id": target.science_record_id, "physical_epoch_role": role,
                "template_id": shape, "requested_f_added_star": requested,
                "realized_f_added_star": achieved_fraction,
                "scalar_a": amplitudes[target.science_record_id],
                "baseline_continuum": baseline_continuum,
                "added_host_continuum": added_continuum,
                "final_continuum": final_continuum,
                "normalization_identity_error": identity_error,
                "normalization_identity_pass": cell_normalization_pass,
                "baseline_continuum_snr": intervention.baseline_snr,
                "achieved_continuum_snr": intervention.achieved_snr,
                "fixed_snr_error": intervention.achieved_snr - intervention.baseline_snr,
                "uncertainty_scale": intervention.uncertainty_scale,
                "random_noise_generated": False,
                "wavelength_unchanged": same_wavelength, "mask_unchanged": same_mask,
                "resolution_unchanged": same_resolution, "noop_identity_pass": local_noop,
                "product_path": str(product.relative_to(root)),
                "product_sha256_arrays": sha256_arrays(
                    spectrum.wavelength, spectrum.flux, spectrum.error,
                    np.asarray(spectrum.mask), target.data.resolution,
                ),
                **fit,
            }
            if requested != 0.0:
                measurement.update({
                    f"transform_{key}": value
                    for key, value in transforms[(target.science_record_id, shape)][1].items()
                })
            measurement_rows.append(measurement)

            if requested == 0.0:
                baseline_fit = expected[expected.science_record_id == target.science_record_id].iloc[0]
                comparisons = []
                for field in (
                    "broad_hbeta_flux", "broad_hbeta_fwhm_kms", "broad_hbeta_ew",
                    "broad_hbeta_snr", "production_rms_global",
                    "production_rms_stellar_features", "production_rms_hbeta",
                ):
                    got, prior = fit.get(field), baseline_fit.get(field)
                    comparisons.append(
                        (pd.isna(got) and pd.isna(prior))
                        or (got is not None and np.isclose(float(got), float(prior), rtol=1e-12, atol=1e-12))
                    )
                noop_pass &= bool(all(comparisons))

        if is_pair:
            shared_pass = amplitudes[targets[0].science_record_id] == amplitudes[targets[1].science_record_id]
        else:
            other_id = row.other_science_record_id
            if row.pilot_arm_id == "DIAG_J233_FAINT":
                fit_by_role["bright"] = comparator_fits["sdss_j233_bright"]
            else:
                other = baselines[other_id]
                baseline_key = f"baseline_{other.science_record_id}"
                if baseline_key not in comparator_fits:
                    comparator_fits[baseline_key] = run_frozen_fit(other.spectrum, vendor)
                fit_by_role[other.physical_role] = comparator_fits[baseline_key]

        yang, yang_stat, yang_reason = yang_label(fit_by_role["bright"], fit_by_role["faint"])
        classification_rows.append({
            "cell_id": row.cell_id, "pilot_arm_id": row.pilot_arm_id,
            "template_id": shape, "requested_f_added_star": requested,
            "criterion": "yang_2024_flux_ratio", "support_status": "SOURCE_RULE_FROZEN_D020_D021",
            "label": yang, "statistic": yang_stat, "threshold": 0.3, "reason": yang_reason,
        })
        for criterion, reason in (
            ("guo_quick_screen", "source rule named, but exact pilot max-flux implementation is not frozen"),
            ("macleod_green", "exact source implementation not yet frozen"),
            ("potts_villforth", "exact source implementation not yet frozen"),
        ):
            classification_rows.append({
                "cell_id": row.cell_id, "pilot_arm_id": row.pilot_arm_id,
                "template_id": shape, "requested_f_added_star": requested,
                "criterion": criterion, "support_status": "DEPENDENCY_NOT_YET_DEFINABLE",
                "label": "unclassifiable", "statistic": None, "threshold": None,
                "reason": reason,
            })
        fits_valid = all(bool(value.get("fit_valid")) for value in fit_by_role.values())
        operator_pass = normalization_pass and noop_pass and shared_pass
        cell_rows.append({
            **row._asdict(), "cell_status": "COMPLETED" if operator_pass else "OPERATOR_FAIL",
            "operator_pass": operator_pass, "normalization_pass": normalization_pass,
            "noop_pass": noop_pass, "shared_amplitude_pass": shared_pass,
            "faint_requested_f_added_star": requested if is_pair else None,
            "faint_realized_f_added_star": realized.get(targets[0].science_record_id) if is_pair else None,
            "bright_realized_f_added_star": realized.get(targets[1].science_record_id) if is_pair else None,
            "shared_a": amplitudes[targets[0].science_record_id] if is_pair else None,
            "all_pair_fits_valid": fits_valid,
        })
        print(f"[{row.cell_id}] {row.pilot_arm_id} {shape} f={requested}: "
              f"operator={'PASS' if operator_pass else 'FAIL'} fits={'valid' if fits_valid else 'invalid'}",
              flush=True)

    cells = pd.DataFrame(cell_rows)
    measurements = pd.DataFrame(measurement_rows)
    classifications = pd.DataFrame(classification_rows)
    cells.to_csv(output / "pilot_cell_execution.csv", index=False)
    measurements.to_csv(output / "pilot_spectrum_measurements.csv", index=False)
    classifications.to_csv(output / "pilot_classification_outcomes.csv", index=False)

    accounting = pd.DataFrame([{
        "n_matrix_cells": len(cells),
        "n_operator_completed": int(cells.operator_pass.sum()),
        "n_operator_failed": int((~cells.operator_pass).sum()),
        "n_spectrum_products": len(measurements),
        "n_fit_completed": int(measurements.fit_completed.fillna(False).sum()),
        "n_fit_valid": int(measurements.fit_valid.fillna(False).sum()),
        "n_fit_invalid": int((~measurements.fit_valid.fillna(False)).sum()),
        "n_verified_criterion_rows": int(
            classifications.support_status.eq("SOURCE_RULE_FROZEN_D020_D021").sum()
        ),
        "n_classifiable": int(
            classifications.label.isin(["CL", "non-CL"]).sum()
        ),
        "n_cl": int(classifications.label.eq("CL").sum()),
        "n_non_cl": int(classifications.label.eq("non-CL").sum()),
        "n_unclassifiable": int(classifications.label.eq("unclassifiable").sum()),
    }])
    accounting.to_csv(output / "pilot_execution_accounting.csv", index=False)

    nonzero = measurements[measurements.requested_f_added_star.fillna(0) > 0]
    contrast_rows = []
    for keys, group in nonzero.groupby(["pilot_arm_id", "physical_epoch_role", "requested_f_added_star"]):
        if set(group.template_id) != EXPECTED_SHAPES:
            continue
        contrast_rows.append({
            "pilot_arm_id": keys[0], "physical_epoch_role": keys[1],
            "requested_f_added_star": keys[2],
            "realized_f_added_star_min": group.realized_f_added_star.min(),
            "realized_f_added_star_max": group.realized_f_added_star.max(),
            "final_continuum_min": group.final_continuum.min(),
            "final_continuum_max": group.final_continuum.max(),
            "broad_hbeta_flux_min": group.broad_hbeta_flux.min(),
            "broad_hbeta_flux_max": group.broad_hbeta_flux.max(),
            "broad_hbeta_ew_min": group.broad_hbeta_ew.min(),
            "broad_hbeta_ew_max": group.broad_hbeta_ew.max(),
            "broad_hbeta_snr_min": group.broad_hbeta_snr.min(),
            "broad_hbeta_snr_max": group.broad_hbeta_snr.max(),
            "rms_hbeta_min": group.production_rms_hbeta.min(),
            "rms_hbeta_max": group.production_rms_hbeta.max(),
            "fit_statuses": "|".join(sorted(set(group.status.astype(str)))),
        })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "pilot_shape_contrasts.csv", index=False)

    monotonic_rows = []
    for keys, group in measurements.groupby(["pilot_arm_id", "physical_epoch_role", "template_id"]):
        if keys[2] == "NONE":
            continue
        group = group.sort_values("requested_f_added_star")
        for metric in ("broad_hbeta_flux", "broad_hbeta_ew", "broad_hbeta_snr", "production_rms_hbeta"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if values.size >= 3:
                difference = np.diff(values)
                monotonic = bool(np.all(difference >= 0) or np.all(difference <= 0))
            else:
                monotonic = False
            monotonic_rows.append({
                "pilot_arm_id": keys[0], "physical_epoch_role": keys[1],
                "template_id": keys[2], "metric": metric,
                "n_finite": int(values.size), "monotonic": monotonic,
                "nonmonotonic": bool(values.size >= 3 and not monotonic),
                "operator_failure": False,
            })
    pd.DataFrame(monotonic_rows).to_csv(output / "pilot_nonmonotonicity.csv", index=False)

    operator_fail = bool((~cells.operator_pass).any() or (~measurements.fit_valid.fillna(False)).any())
    verified = classifications[classifications.criterion == "yang_2024_flux_ratio"]
    label_dependent = any(
        group.label.nunique() > 1
        for _, group in verified[verified.requested_f_added_star > 0].groupby(
            ["pilot_arm_id", "requested_f_added_star"]
        )
    )
    fit_dependent = any(
        group.fit_valid.nunique() > 1
        for _, group in nonzero.groupby(
            ["pilot_arm_id", "physical_epoch_role", "requested_f_added_star"]
        )
    )
    if operator_fail:
        verdict = "HOST_PILOT_OPERATOR_FAIL"
    elif label_dependent or fit_dependent:
        verdict = "HOST_PILOT_PASS_SHAPE_DEPENDENT"
    else:
        verdict = "HOST_PILOT_PASS_SHAPE_STABLE"
    gates = pd.DataFrame([
        {"gate": "matrix_exact", "passed": len(cells) == 40, "detail": "40 cells; four arms x 10"},
        {"gate": "normalization", "passed": bool(cells.normalization_pass.all()), "detail": "anchored abs error <1e-9"},
        {"gate": "noop", "passed": bool(cells.noop_pass.all()), "detail": "arrays/resolution/measurements reproduced"},
        {"gate": "resolution", "passed": bool(not measurements.get("transform_convolution_clipped", pd.Series(False)).fillna(False).any()), "detail": "no XSL convolution clipping"},
        {"gate": "fixed_snr", "passed": bool((measurements.fixed_snr_error.abs() < 1e-9).all()), "detail": "deterministic; no random draw"},
        {"gate": "shared_pair_a", "passed": bool(cells[cells.pilot_arm_id == PAIR_ARM].shared_amplitude_pass.all()), "detail": "one scalar stored and reused"},
        {"gate": "fit_pipeline", "passed": bool(measurements.fit_valid.fillna(False).all()), "detail": "frozen settings; failures retained"},
        {"gate": "outcome_accounting", "passed": bool(len(classifications) == 160), "detail": "CL/non-CL/unclassifiable retained"},
    ])
    gates.to_csv(output / "pilot_operator_gates.csv", index=False)
    (output / "pilot_verdict.txt").write_text(verdict + "\n")
    print(f"VERDICT={verdict}")
    print(accounting.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
