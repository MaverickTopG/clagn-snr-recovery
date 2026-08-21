#!/usr/bin/env python
"""Design-only XSL host-shape audit for the prospective Q2 pilot.

Reads already-saved 17G-R component arrays.  It does not construct an injected
spectrum, run a classifier, or inspect a counterfactual recovery outcome.
"""

from __future__ import annotations

import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

NORMALIZATION_WINDOW = (5080.0, 5130.0)
SHAPE_WINDOW = (4000.0, 5500.0)
HBETA_SHAPE_WINDOW = (4800.0, 4920.0)
FAMILIES = ("M1b", "M2", "M3")

# Selected prospectively from exact XSL grid coordinates to span continuum and
# absorption morphology.  Labels describe library coordinates; they are not
# inferred stellar-population properties of any CLAGN host.
ENSEMBLE = (
    ("XSL_H1", 0.5011872336272715, 0.0, "younger_strong_balmer_library_shape"),
    ("XSL_H2", 2.5118864315095824, -0.6, "intermediate_metal_poor_library_shape"),
    ("XSL_H3", 7.943282347242821, 0.2, "older_metal_rich_library_shape"),
)
F_ADDED_STAR_RUNGS = (0.0, 0.1, 0.5, 0.85)


def revised_pilot_matrix() -> pd.DataFrame:
    """Return the frozen design cells; never load or construct spectral flux."""
    j233 = "desi:iron:healpix-coadd:main:bright:hp18789:tid39627796707803702:zpix158456325139248767426053014070"
    j161_bright = "desi:iron:healpix-coadd:main:bright:hp31535:tid39627951624425454:zpix158456325139248767580969635822"
    j161_faint = "desi:iron:healpix-coadd:main:dark:hp31535:tid39627951624425454:zpix237684487653513105174513586158"
    arms = (
        ("DIAG_J233_FAINT", "SPECTRUM_DIAGNOSTIC", "SDSSJ233602.98+001728.7", j233,
         "desi_faint", "sdss:plate-mjd-fiber:385-51783-360", "CROSS_SURVEY_APERTURE_MISMATCH", False),
        ("DIAG_J161_BRIGHT", "SPECTRUM_DIAGNOSTIC", "SDSSJ161711.42+063833.4", j161_bright,
         "desi_bright", j161_faint, "SAME_SURVEY_SAME_APERTURE", False),
        ("DIAG_J161_FAINT", "SPECTRUM_DIAGNOSTIC", "SDSSJ161711.42+063833.4", j161_faint,
         "desi_faint", j161_bright, "SAME_SURVEY_SAME_APERTURE", False),
        ("PAIR_J161_FAINT_ANCHORED", "PAIR_CONSISTENT", "SDSSJ161711.42+063833.4", j161_faint,
         "desi_faint", j161_bright, "SAME_SURVEY_SAME_APERTURE", True),
    )
    rows = []
    for arm_id, semantics, object_id, anchor_id, anchor_role, other_id, basis, shared in arms:
        for requested in F_ADDED_STAR_RUNGS:
            shapes = ("NONE",) if requested == 0.0 else tuple(row[0] for row in ENSEMBLE)
            for shape in shapes:
                rows.append({
                    "pilot_arm_id": arm_id,
                    "semantics": semantics,
                    "object_id": object_id,
                    "reference_science_record_id": anchor_id,
                    "reference_epoch_role": anchor_role,
                    "other_science_record_id": other_id,
                    "flux_aperture_basis": basis,
                    "shared_absolute_amplitude_across_pair": shared,
                    "host_shape_id": shape,
                    "requested_f_added_star_reference": requested,
                    "requested_added_star_to_baseline_ratio_reference": (
                        requested / (1.0 - requested)
                    ),
                    "normalization_window_rest_angstrom": "5080-5130",
                    "fixed_continuum_snr_is_intervention": True,
                    "pilot_probability_estimand": False,
                    "spectra_generated": False,
                })
    frame = pd.DataFrame(rows)
    if len(frame) != 40:
        raise RuntimeError(f"revised pilot matrix must contain 40 design cells, got {len(frame)}")
    return frame


def median_window(wave, values, window):
    inside = (wave >= window[0]) & (wave <= window[1]) & np.isfinite(values)
    if inside.sum() < 5:
        raise ValueError(f"insufficient support in {window}")
    return float(np.median(values[inside]))


def normalize(wave, values):
    level = median_window(wave, values, NORMALIZATION_WINDOW)
    if level <= 0:
        raise ValueError("non-positive XSL host normalization")
    return np.asarray(values, dtype=float) / level


def hbeta_absorption_ew(wave, shape):
    line = (wave >= 4830.0) & (wave <= 4890.0)
    blue = median_window(wave, shape, (4800.0, 4820.0))
    red = median_window(wave, shape, (4900.0, 4920.0))
    baseline = blue + (red - blue) * (wave[line] - 4810.0) / 100.0
    return float(np.trapezoid(1.0 - shape[line] / baseline, wave[line]))


def shape_metrics(wave, values):
    shape = normalize(wave, values)
    return {
        "dn4000_narrow": (
            median_window(wave, shape, (4000.0, 4100.0))
            / median_window(wave, shape, (3850.0, 3950.0))
        ),
        "continuum_4000_over_5100": (
            median_window(wave, shape, (4000.0, 4100.0))
            / median_window(wave, shape, NORMALIZATION_WINDOW)
        ),
        "hbeta_absorption_ew_angstrom": hbeta_absorption_ew(wave, shape),
        "mgb_band_over_sidebands": (
            median_window(wave, shape, (5150.0, 5190.0))
            / np.mean([
                median_window(wave, shape, (5100.0, 5130.0)),
                median_window(wave, shape, (5220.0, 5250.0)),
            ])
        ),
    }


def common_shape(wave, values, grid):
    normalized = normalize(wave, values)
    return np.interp(grid, wave, normalized, left=np.nan, right=np.nan)


def pair_metrics(a, b, grid):
    good = np.isfinite(a) & np.isfinite(b)
    hbeta = good & (grid >= HBETA_SHAPE_WINDOW[0]) & (grid <= HBETA_SHAPE_WINDOW[1])
    difference = a[good] - b[good]
    return {
        "n_common_pixels": int(good.sum()),
        "shape_correlation": float(np.corrcoef(a[good], b[good])[0, 1]),
        "median_abs_shape_difference": float(np.median(np.abs(difference))),
        "rms_shape_difference": float(np.sqrt(np.mean(difference**2))),
        "rms_hbeta_shape_difference": float(np.sqrt(np.mean((a[hbeta] - b[hbeta]) ** 2))),
    }


def main() -> int:
    root = project_root()
    recovery = root / "06_fitting/recovery_17gr"
    output = root / "05_analysis/host_shape_pilot"
    output.mkdir(parents=True, exist_ok=True)
    admissibility = pd.read_csv(recovery / "host_template_admissibility_17h.csv")
    fit_order = pd.read_csv(recovery / "pyqsofit_corrected_desi.csv").reset_index()
    sensitive = admissibility[
        admissibility.admissibility_class == "HOST_TEMPLATE_MODEL_SENSITIVE"
    ]
    if len(sensitive) != 5:
        raise RuntimeError("host-shape design requires the accepted five model-sensitive rows")

    grid = np.arange(SHAPE_WINDOW[0], SHAPE_WINDOW[1] + 0.1, 2.0)
    shape_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for row in sensitive.itertuples():
        match = fit_order[fit_order.science_record_id == row.science_record_id]
        index = int(match.iloc[0]["index"]) + 1
        shapes: dict[str, np.ndarray] = {}
        metrics_by_family: dict[str, dict[str, float]] = {}
        fhosts: dict[str, float] = {}
        for family in FAMILIES:
            archive = np.load(recovery / "components" / f"ppxf_{index:02d}_XSL_{family}.npz")
            wave = np.asarray(archive["wavelength"], dtype=float)
            stellar = np.asarray(archive["stellar"], dtype=float)
            shape = common_shape(wave, stellar, grid)
            metrics = shape_metrics(wave, stellar)
            fhost = float(getattr(row, f"xsl_{family.lower()}_f_host_cont"))
            digest = hashlib.sha256(np.round(shape, 10).tobytes()).hexdigest()
            shapes[family] = shape
            metrics_by_family[family] = metrics
            fhosts[family] = fhost
            shape_rows.append({
                "object_id": row.object_id, "science_record_id": row.science_record_id,
                "program": row.program, "family": family,
                "reconstruction_identity_pass": True, "f_host_cont": fhost,
                "normalized_shape_sha256_round10": digest, **metrics,
            })

        object_pairs = []
        for first, second in combinations(FAMILIES, 2):
            evidence = pair_metrics(shapes[first], shapes[second], grid)
            record = {
                "object_id": row.object_id, "science_record_id": row.science_record_id,
                "program": row.program, "family_a": first, "family_b": second,
                "abs_hbeta_ew_difference": abs(
                    metrics_by_family[first]["hbeta_absorption_ew_angstrom"]
                    - metrics_by_family[second]["hbeta_absorption_ew_angstrom"]
                ),
                "abs_dn4000_difference": abs(
                    metrics_by_family[first]["dn4000_narrow"]
                    - metrics_by_family[second]["dn4000_narrow"]
                ),
                "abs_continuum_color_difference": abs(
                    metrics_by_family[first]["continuum_4000_over_5100"]
                    - metrics_by_family[second]["continuum_4000_over_5100"]
                ),
                "abs_f_host_cont_difference": abs(fhosts[first] - fhosts[second]),
                **evidence,
            }
            pair_rows.append(record)
            object_pairs.append(record)
        distinct = len({
            hashlib.sha256(np.round(shape, 10).tobytes()).hexdigest()
            for shape in shapes.values()
        })
        summaries.append({
            "object_id": row.object_id, "science_record_id": row.science_record_id,
            "program": row.program, "n_valid_xsl_family_shapes": len(FAMILIES),
            "n_numerically_distinct_shapes_round10": distinct,
            "f_host_cont_min": min(fhosts.values()), "f_host_cont_max": max(fhosts.values()),
            "f_host_cont_range": max(fhosts.values()) - min(fhosts.values()),
            "max_pairwise_rms_shape_difference": max(r["rms_shape_difference"] for r in object_pairs),
            "max_pairwise_rms_hbeta_shape_difference": max(r["rms_hbeta_shape_difference"] for r in object_pairs),
            "max_pairwise_abs_hbeta_ew_difference": max(r["abs_hbeta_ew_difference"] for r in object_pairs),
            "max_pairwise_abs_dn4000_difference": max(r["abs_dn4000_difference"] for r in object_pairs),
            "global_objective_used_for_selection": False,
        })

    pd.DataFrame(shape_rows).to_csv(output / "empirical_xsl_family_shapes.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output / "empirical_xsl_pairwise_shape_differences.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "empirical_xsl_shape_envelope.csv", index=False)

    xsl_path = root / "06_fitting/alternative_fit/sps_libraries/spectra_xsl_9.0.npz"
    xsl = np.load(xsl_path)
    wave = np.asarray(xsl["lam"], dtype=float)
    ages, metals = np.asarray(xsl["ages"]), np.asarray(xsl["metals"])
    templates = np.asarray(xsl["templates"])
    ensemble_rows = []
    saved = {"wavelength_rest_intrinsic": wave}
    ensemble_shapes: dict[str, np.ndarray] = {}
    for host_id, requested_age, requested_metal, label in ENSEMBLE:
        age_index = int(np.argmin(np.abs(ages - requested_age)))
        metal_index = int(np.argmin(np.abs(metals - requested_metal)))
        if not np.isclose(ages[age_index], requested_age) or not np.isclose(
            metals[metal_index], requested_metal
        ):
            raise RuntimeError(f"candidate {host_id} is not an exact XSL grid coordinate")
        shape = normalize(wave, templates[:, age_index, metal_index])
        saved[host_id] = shape
        ensemble_shapes[host_id] = common_shape(wave, shape, grid)
        in_fit = (wave >= 3700.0) & (wave <= 7000.0)
        ensemble_rows.append({
            "host_shape_id": host_id, "xsl_age_gyr": ages[age_index],
            "xsl_metallicity_dex": metals[metal_index], "descriptive_library_label": label,
            "selection_used_recovery_outcomes": False,
            "normalization_window_rest": "5080-5130_A",
            "normalization_median": median_window(wave, shape, NORMALIZATION_WINDOW),
            "intrinsic_xsl_fwhm_median_3700_7000": float(np.median(xsl["fwhm"][in_fit])),
            "target_resolution_status": "TARGET_SPECIFIC_XSL_TO_DATA_CONVOLUTION_REQUIRED",
            **shape_metrics(wave, shape),
        })
    np.savez_compressed(output / "proposed_xsl_host_shape_ensemble_intrinsic.npz", **saved)
    pd.DataFrame(ensemble_rows).to_csv(output / "proposed_xsl_host_shape_ensemble.csv", index=False)
    ensemble_pairs = []
    ensemble_metrics = {row["host_shape_id"]: row for row in ensemble_rows}
    for first, second in combinations(ensemble_shapes, 2):
        ensemble_pairs.append({
            "host_shape_a": first, "host_shape_b": second,
            "abs_hbeta_ew_difference": abs(
                ensemble_metrics[first]["hbeta_absorption_ew_angstrom"]
                - ensemble_metrics[second]["hbeta_absorption_ew_angstrom"]
            ),
            "abs_dn4000_difference": abs(
                ensemble_metrics[first]["dn4000_narrow"]
                - ensemble_metrics[second]["dn4000_narrow"]
            ),
            "abs_continuum_color_difference": abs(
                ensemble_metrics[first]["continuum_4000_over_5100"]
                - ensemble_metrics[second]["continuum_4000_over_5100"]
            ),
            **pair_metrics(ensemble_shapes[first], ensemble_shapes[second], grid),
        })
    pd.DataFrame(ensemble_pairs).to_csv(
        output / "proposed_xsl_host_shape_ensemble_pairwise.csv", index=False
    )

    # Three corrected records, selected from QC/17H provenance before any pilot
    # classification result: one supported low-S/N physical-faint model-sensitive
    # record and the supported corrected bright/faint no-primary pair.
    pilot_ids = [
        "desi:iron:healpix-coadd:main:bright:hp18789:tid39627796707803702:zpix158456325139248767426053014070",
        "desi:iron:healpix-coadd:main:bright:hp31535:tid39627951624425454:zpix158456325139248767580969635822",
        "desi:iron:healpix-coadd:main:dark:hp31535:tid39627951624425454:zpix237684487653513105174513586158",
    ]
    pilot = admissibility[admissibility.science_record_id.isin(pilot_ids)][[
        "object_id", "science_record_id", "program", "admissibility_class"
    ]].copy()
    rationale = {
        pilot_ids[0]: "physical faint; lowest corrected-DESI Hbeta S/N; model-sensitive; Gate-C supported",
        pilot_ids[1]: "physical bright member of the only corrected-DESI internal pair; no primary host; Gate-C supported",
        pilot_ids[2]: "physical faint member of the same corrected-DESI internal pair; no primary host; Gate-C supported",
    }
    pilot["selection_rationale"] = pilot.science_record_id.map(rationale)
    pilot["selection_used_pilot_classification_outcomes"] = False
    pilot.to_csv(output / "proposed_pilot_records.csv", index=False)
    revised_pilot_matrix().to_csv(output / "revised_pilot_matrix.csv", index=False)
    print(f"wrote design audit for {len(sensitive)} model-sensitive records")
    print(
        f"proposed ensemble: {len(ensemble_rows)} exact XSL shapes; "
        f"pilot records: {len(pilot)}; design cells: 40"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
