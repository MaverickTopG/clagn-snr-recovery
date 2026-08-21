#!/usr/bin/env python
"""D-089 Yang/Green native audit and bounded Q1 stability pilot only."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.counterfactual.pair_snr import PairCondition, degrade_pair  # noqa: E402
from p3sf.criteria.green_pixel import (  # noqa: E402
    VarianceProvenance,
    classify_green_pixel_measurement,
    fit_green_epoch_line_spectrum_pyqsofit,
    measure_green_pixel_nsigma,
)
from p3sf.criteria.source_verified import apply_yang2024_hbeta  # noqa: E402
from p3sf.fitting.pyqsofit_driver import (  # noqa: E402
    YANG_BROAD_FWHM_RANGE_KMS,
    YangHbetaFitAudit,
    fit_yang_hbeta_audit_pyqsofit,
)

PILOT_M = 50
PILOT_OBJECTS = {
    "SDSSJ082942.66+415436.8": "GOLD_HIGH_NATIVE_SN",
    "SDSSJ012256.19-000252.68": "SILVER_LOW_NATIVE_SN",
}
PILOT_RUNGS = (5.0, 10.0)
# Reuse D-088 exactly; no new noise draw may replace the observed instability.
PILOT_NAMESPACE = "p3sf:q1:d088:stability:v1"
GREEN_SCALE_PROVENANCE = "NO_RESCALE_EXACT_SDSS_PAIR_NO_SOURCE_OUTLIER_TRIGGER"


def read_sdss(path: Path, redshift: float, spectrum_id: str) -> Spectrum:
    with fits.open(path) as hdul:
        data = hdul[1].data
        wavelength = 10.0 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        ivar = np.asarray(data["ivar"], dtype=float)
    error = np.full(ivar.shape, np.nan)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    return Spectrum(
        wavelength, flux, error, redshift, mask=(~positive).astype(int),
        meta={"spectrum_id": spectrum_id},
    )


def coordinates(object_id: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"SDSSJ(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)",
        object_id,
    )
    if match is None:
        raise ValueError(f"cannot parse IAU coordinates: {object_id}")
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra = 15.0 * (float(hh) + float(mm) / 60.0 + float(ss) / 3600.0)
    dec = float(dd) + float(dm) / 60.0 + float(ds) / 3600.0
    return ra, dec if sign == "+" else -dec


def fit_yang(spectrum: Spectrum, vendor: Path) -> YangHbetaFitAudit:
    _, audit = fit_yang_hbeta_audit_pyqsofit(
        spectrum.wavelength, spectrum.flux, spectrum.error,
        redshift=spectrum.redshift, path=str(vendor),
    )
    return audit


def fit_green(spectrum: Spectrum, vendor: Path, object_id: str):
    ra, dec = coordinates(object_id)
    return fit_green_epoch_line_spectrum_pyqsofit(
        spectrum.wavelength, spectrum.flux, spectrum.error,
        redshift=spectrum.redshift, path=vendor,
        variance_provenance=VarianceProvenance.NATIVE_SDSS_SUPPORTED,
        ra=ra, dec=dec, scale_factor=1.0,
        scale_provenance=GREEN_SCALE_PROVENANCE,
    )


def load_pair(
    root: Path, matrix: pd.DataFrame, qc: pd.DataFrame, object_id: str
) -> tuple[pd.Series, dict[str, Spectrum]]:
    design = matrix[matrix.transition_id == object_id].iloc[0]
    spectra: dict[str, Spectrum] = {}
    for role in ("bright", "faint"):
        spectrum_id = str(design[f"{role}_science_record_id"]).removeprefix("sdss:")
        match = qc[(qc.object_id == object_id) & (qc.spectrum_id.astype(str) == spectrum_id)]
        if len(match) != 1:
            raise RuntimeError(f"exact historical {role} lookup failed for {object_id}")
        record = match.iloc[0]
        spectra[role] = read_sdss(
            root / str(record.path), float(record.redshift), spectrum_id
        )
    return design, spectra


def audit_row(
    *,
    object_id: str,
    tier: str,
    spectrum_id: str,
    fit_role: str,
    arm: str,
    rung: float | None,
    realization: int | None,
    audit: YangHbetaFitAudit,
) -> dict[str, object]:
    return {
        "transition_id": object_id,
        "tier": tier,
        "spectrum_id": spectrum_id,
        "fit_role": fit_role,
        "arm": arm,
        "snr_rung": rung,
        "realization": realization,
        **asdict(audit),
    }


def prefix_recovery(group: pd.DataFrame, n: int) -> float | None:
    prefix = group.sort_values("realization").iloc[:n]
    classifiable = prefix.label.isin(["CL", "non-CL"])
    if not classifiable.any():
        return None
    return float((prefix.loc[classifiable, "label"] == "CL").mean())


def prefix_unclassifiable(group: pd.DataFrame, n: int) -> float:
    prefix = group.sort_values("realization").iloc[:n]
    return float((prefix.label.astype(str) == "unclassifiable").mean())


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    keys = ["transition_id", "tier", "selection_role", "snr_rung", "arm", "criterion"]
    for key, group in rows.groupby(keys, sort=True):
        labels = group.label.astype(str)
        measurement = pd.to_numeric(group.measurement, errors="coerce")
        finite = measurement[np.isfinite(measurement)]
        output.append({
            **dict(zip(keys, key, strict=True)),
            "n_realizations": len(group),
            "n_CL": int((labels == "CL").sum()),
            "n_nonCL": int((labels == "non-CL").sum()),
            "n_unclassifiable": int((labels == "unclassifiable").sum()),
            "classifiable_fraction": float(labels.isin(["CL", "non-CL"]).mean()),
            "prefix_recovery_M5": prefix_recovery(group, 5),
            "prefix_recovery_M10": prefix_recovery(group, 10),
            "prefix_recovery_M20": prefix_recovery(group, 20),
            "prefix_recovery_M40": prefix_recovery(group, 40),
            "prefix_recovery_M50": prefix_recovery(group, 50),
            "prefix_unclassifiable_M20": prefix_unclassifiable(group, 20),
            "prefix_unclassifiable_M40": prefix_unclassifiable(group, 40),
            "prefix_unclassifiable_M50": prefix_unclassifiable(group, 50),
            "measurement_mean": float(finite.mean()) if len(finite) else None,
            "measurement_sd": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "measurement_median": float(finite.median()) if len(finite) else None,
            "measurement_q10": float(finite.quantile(0.1)) if len(finite) else None,
            "measurement_q90": float(finite.quantile(0.9)) if len(finite) else None,
        })
    return pd.DataFrame(output)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--extend-from", type=int, default=0,
        help="reuse already persisted lower-prefix rows and generate only later realizations",
    )
    args = parser.parse_args()
    if args.extend_from not in {0, 20}:
        raise ValueError("D-089 permits only a fresh run or the frozen M20-to-M50 extension")
    root = project_root()
    cfg = load_config()
    if cfg.snr_grid != [5, 10] or cfg.snr_experiment.binding_metric != "hbeta_window":
        raise RuntimeError("D-089 requires the frozen D-088 [5,10] Hbeta grid")
    if (
        cfg.fitting.broad_fwhm_min_kms,
        cfg.fitting.yang_broad_fwhm_max_kms,
    ) != YANG_BROAD_FWHM_RANGE_KMS:
        raise RuntimeError("Yang code and frozen source FWHM domain disagree")
    window = tuple(cfg.snr_metrics.hbeta_window)
    output = root / "05_analysis/q1_design/measurement_validity_d089"
    output.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(root / "05_analysis/q1_design/proposed_final_q1_matrix_d088_unexecuted.csv")
    qc = pd.read_csv(root / "03_spectra/qc/qc_verdicts_slice.csv")
    vendor = root / "06_fitting/pyqsofit/vendor/PyQSOFit"
    exact_objects = list(dict.fromkeys(matrix.transition_id))
    pair_cache: dict[str, tuple[pd.Series, dict[str, Spectrum]]] = {}
    audit_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    if args.extend_from == 20:
        old_audits = pd.read_csv(output / "yang_fit_audit_rows.csv")
        old_stability = pd.read_csv(output / "stability_rows.csv")
        if set(old_stability.realization.unique()) != set(range(20)):
            raise RuntimeError("persisted prefix is not the exact completed M=20 pilot")
        audit_rows.extend(
            old_audits[old_audits.arm != "native"].to_dict(orient="records")
        )
        stability_rows.extend(old_stability.to_dict(orient="records"))

    # Audit every exact native endpoint in the locked GOLD+SILVER Q1 set.
    native_yang: dict[tuple[str, str], YangHbetaFitAudit] = {}
    for object_id in exact_objects:
        design, spectra = load_pair(root, matrix, qc, object_id)
        pair_cache[object_id] = (design, spectra)
        for role in ("bright", "faint"):
            audit = fit_yang(spectra[role], vendor)
            native_yang[(object_id, role)] = audit
            audit_rows.append(audit_row(
                object_id=object_id, tier=str(design.reference_tier),
                spectrum_id=str(spectra[role].meta["spectrum_id"]),
                fit_role=f"native_{role}", arm="native", rung=None,
                realization=None, audit=audit,
            ))
        print(f"native Yang audit {object_id}", flush=True)

    for object_id, selection_role in PILOT_OBJECTS.items():
        design, spectra = pair_cache[object_id]
        tier = str(design.reference_tier)
        native_bright_yang = native_yang[(object_id, "bright")]
        native_bright_green = fit_green(spectra["bright"], vendor, object_id)
        for rung in PILOT_RUNGS:
            for realization in range(args.extend_from, PILOT_M):
                faint_only = degrade_pair(
                    spectra["bright"], spectra["faint"],
                    PairCondition("faint_only", snr_faint=rung),
                    window_rest=window, base_seed=cfg.random_seed, pair_id=object_id,
                    realization=realization, seed_namespace=PILOT_NAMESPACE,
                )
                matched = degrade_pair(
                    spectra["bright"], spectra["faint"],
                    PairCondition("matched", snr_faint=rung, snr_bright=rung),
                    window_rest=window, base_seed=cfg.random_seed, pair_id=object_id,
                    realization=realization, seed_namespace=PILOT_NAMESPACE,
                )
                if not np.array_equal(
                    faint_only.faint_spectrum.flux, matched.faint_spectrum.flux
                ):
                    raise RuntimeError("common-random-number identity failed across arms")

                faint_yang = fit_yang(faint_only.faint_spectrum, vendor)
                matched_bright_yang = fit_yang(matched.bright_spectrum, vendor)
                audit_rows.extend((
                    audit_row(
                        object_id=object_id, tier=tier,
                        spectrum_id=str(spectra["faint"].meta["spectrum_id"]),
                        fit_role="pilot_faint_shared_across_arms", arm="shared",
                        rung=rung, realization=realization, audit=faint_yang,
                    ),
                    audit_row(
                        object_id=object_id, tier=tier,
                        spectrum_id=str(spectra["bright"].meta["spectrum_id"]),
                        fit_role="pilot_matched_bright", arm="matched",
                        rung=rung, realization=realization, audit=matched_bright_yang,
                    ),
                ))

                faint_green = fit_green(faint_only.faint_spectrum, vendor, object_id)
                matched_bright_green = fit_green(matched.bright_spectrum, vendor, object_id)
                for arm, bright_yang, bright_green in (
                    ("faint_only", native_bright_yang, native_bright_green),
                    ("matched", matched_bright_yang, matched_bright_green),
                ):
                    yang = apply_yang2024_hbeta(bright=bright_yang, faint=faint_yang)
                    stability_rows.append({
                        "transition_id": object_id, "tier": tier,
                        "selection_role": selection_role, "snr_rung": rung,
                        "arm": arm, "criterion": "YANG2024_FINAL",
                        "realization": realization, "label": str(yang.label),
                        "measurement": yang.statistic,
                        "measurement_available": not yang.is_unclassifiable,
                        "invalid_reason": yang.reason,
                    })
                    green_measurement = measure_green_pixel_nsigma(
                        bright_green, faint_green
                    )
                    green = classify_green_pixel_measurement(green_measurement)
                    stability_rows.append({
                        "transition_id": object_id, "tier": tier,
                        "selection_role": selection_role, "snr_rung": rung,
                        "arm": arm, "criterion": "GREEN2022_FINAL",
                        "realization": realization, "label": str(green.label),
                        "measurement": green_measurement.nsigma_hbeta,
                        "measurement_available": green_measurement.measurement_available,
                        "invalid_reason": green_measurement.invalid_reason or green.reason,
                    })
                print(
                    f"{object_id} rung={rung:g} realization={realization + 1}/{PILOT_M}",
                    flush=True,
                )

    audits = pd.DataFrame(audit_rows)
    stability = pd.DataFrame(stability_rows)
    audits.to_csv(output / "yang_fit_audit_rows.csv", index=False)
    audits[audits.arm == "native"].to_csv(output / "yang_native_fit_audit.csv", index=False)
    stability.to_csv(output / "stability_rows.csv", index=False)
    summarize(stability).to_csv(output / "stability_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
