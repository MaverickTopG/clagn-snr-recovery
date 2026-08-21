#!/usr/bin/env python
"""Run only the prospectively frozen D-088 Q1 realization-stability pilot.

The pilot spectra exist in memory only.  No degraded FITS product is written,
and no row is admitted to a final-classifier denominator because the fit-boundary
audit and the source-protocol Green measurement are not yet implemented.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.counterfactual.pair_snr import PairCondition, degrade_pair  # noqa: E402
from p3sf.counterfactual.snr import measure_snr  # noqa: E402
from p3sf.fitting.pyqsofit_driver import (  # noqa: E402
    BroadHbetaDiagnostics,
    fit_broad_hbeta_diagnostics_pyqsofit,
)

PILOT_M = 20
PILOT_OBJECTS = {
    "SDSSJ082942.66+415436.8": "GOLD_HIGH_NATIVE_SN",
    "SDSSJ012256.19-000252.68": "SILVER_LOW_NATIVE_SN",
}
PILOT_RUNGS = (5.0, 10.0)
PILOT_NAMESPACE = "p3sf:q1:d088:stability:v1"


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


def run_fit(spectrum: Spectrum, vendor: Path) -> BroadHbetaDiagnostics:
    good = spectrum.good
    return fit_broad_hbeta_diagnostics_pyqsofit(
        spectrum.wavelength[good], spectrum.flux[good], spectrum.error[good],
        redshift=spectrum.redshift, path=str(vendor),
    )


def yang_engineering_label(
    bright: BroadHbetaDiagnostics, faint: BroadHbetaDiagnostics
) -> tuple[str, float | None]:
    """Diagnostic-only D-020/D-021 value, never a final D-088 label."""
    if bright.status != "ok" or faint.status != "ok":
        return "unclassifiable", None
    bright_flux = bright.broad_hbeta_flux
    faint_flux = faint.broad_hbeta_flux
    if bright_flux is None or not np.isfinite(bright_flux) or bright_flux <= 0:
        return "unclassifiable", None
    protocol_faint = 0.0 if not faint.broad_hbeta_detected else faint_flux
    if protocol_faint is None or not np.isfinite(protocol_faint) or protocol_faint < 0:
        return "unclassifiable", None
    ratio = float(protocol_faint / bright_flux)
    return ("CL" if ratio < 0.3 else "non-CL"), ratio


def prefix_fraction(labels: pd.Series, n: int) -> float | None:
    prefix = labels.iloc[:n]
    decided = prefix.isin(["CL", "non-CL"])
    return float((prefix[decided] == "CL").mean()) if decided.any() else None


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    keys = ["transition_id", "tier", "selection_role", "snr_rung", "arm"]
    for key, group in rows.sort_values("realization").groupby(keys, sort=True):
        transition_id, tier, selection_role, rung, arm = key
        ratios = pd.to_numeric(group.yang_engineering_ratio, errors="coerce")
        finite = ratios[np.isfinite(ratios)]
        labels = group.yang_engineering_label.astype(str)
        output.append({
            "transition_id": transition_id,
            "tier": tier,
            "selection_role": selection_role,
            "snr_rung": rung,
            "arm": arm,
            "n_realizations": len(group),
            "faint_achieved_sn_mean": group.faint_achieved_sn.mean(),
            "faint_achieved_sn_sd": group.faint_achieved_sn.std(ddof=1),
            "bright_achieved_sn_mean": group.bright_achieved_sn.mean(),
            "engineering_fit_ok_fraction": group.engineering_fit_ok.mean(),
            "yang_engineering_ratio_median": finite.median() if len(finite) else None,
            "yang_engineering_ratio_q10": finite.quantile(0.1) if len(finite) else None,
            "yang_engineering_ratio_q90": finite.quantile(0.9) if len(finite) else None,
            "yang_engineering_distinct_labels": "|".join(sorted(set(labels))),
            "yang_engineering_cl_fraction_m5": prefix_fraction(labels, 5),
            "yang_engineering_cl_fraction_m10": prefix_fraction(labels, 10),
            "yang_engineering_cl_fraction_m20": prefix_fraction(labels, 20),
            "yang_final_unclassifiable_fraction": 1.0,
            "green_final_unclassifiable_fraction": 1.0,
            "final_denominator_admissible": False,
        })
    return pd.DataFrame(output)


def main() -> int:
    root = project_root()
    cfg = load_config()
    if cfg.snr_experiment.binding_metric != "hbeta_window":
        raise RuntimeError("D-088 pilot requires frozen hbeta_window binding metric")
    window = tuple(cfg.snr_metrics.hbeta_window)
    output = root / "05_analysis/q1_design/realization_stability_d088"
    output.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(root / "05_analysis/q1_design/proposed_q1_matrix_d087_unexecuted.csv")
    qc = pd.read_csv(root / "03_spectra/qc/qc_verdicts_slice.csv")
    vendor = root / "06_fitting/pyqsofit/vendor/PyQSOFit"
    rows: list[dict[str, object]] = []

    for object_id, selection_role in PILOT_OBJECTS.items():
        design = matrix[matrix.transition_id == object_id].iloc[0]
        tier = str(design.reference_tier)
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
        native_bright_fit = run_fit(spectra["bright"], vendor)
        for rung in PILOT_RUNGS:
            for realization in range(PILOT_M):
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
                faint_fit = run_fit(faint_only.faint_spectrum, vendor)
                matched_bright_fit = run_fit(matched.bright_spectrum, vendor)
                for arm, result, bright_fit in (
                    ("faint_only", faint_only, native_bright_fit),
                    ("matched", matched, matched_bright_fit),
                ):
                    label, ratio = yang_engineering_label(bright_fit, faint_fit)
                    bright_achieved = (
                        result.bright.achieved_snr if result.bright is not None
                        else measure_snr(result.bright_spectrum, window)
                    )
                    row = {
                        "transition_id": object_id,
                        "tier": tier,
                        "selection_role": selection_role,
                        "snr_rung": rung,
                        "arm": arm,
                        "realization": realization,
                        "seed_namespace": PILOT_NAMESPACE,
                        "faint_seed": result.faint.seed,
                        "bright_seed": result.bright.seed if result.bright is not None else None,
                        "faint_achieved_sn": result.faint.achieved_snr,
                        "bright_achieved_sn": bright_achieved,
                        "engineering_fit_ok": (
                            bright_fit.status == "ok" and faint_fit.status == "ok"
                        ),
                        "yang_engineering_label": label,
                        "yang_engineering_ratio": ratio,
                        "yang_final_label": "unclassifiable",
                        "yang_final_reason": "FIT_BOUNDARY_VALIDITY_AUDIT_NOT_IMPLEMENTED",
                        "green_final_label": "unclassifiable",
                        "green_final_reason": "SOURCE_PROTOCOL_PIXEL_NSIGMA_PREPROCESSING_NOT_IMPLEMENTED",
                    }
                    for prefix, diagnostic in (("bright", bright_fit), ("faint", faint_fit)):
                        values = asdict(diagnostic)
                        for name in (
                            "status", "broad_hbeta_flux", "broad_hbeta_ew",
                            "broad_hbeta_snr", "broad_hbeta_detected", "l5100",
                        ):
                            row[f"{prefix}_{name}"] = values[name]
                    rows.append(row)
                print(
                    f"{object_id} rung={rung:g} realization={realization + 1}/{PILOT_M}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "realization_rows.csv", index=False)
    summarise(frame).to_csv(output / "stability_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
