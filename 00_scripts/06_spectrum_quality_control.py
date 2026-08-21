#!/usr/bin/env python
"""Stage 7 — deterministic quality control and the two-epoch S/N census.

Applies the frozen QC rules to every retrieved spectrum, SDSS and DESI alike,
and reports the census that decision D-010 needs. Nothing here inspects a line
flux or a transition: QC must be decidable before any outcome is known, or the
sample depends on the answer.

**Two S/N metrics, never one (D-010).** A rest-frame 5100 A continuum S/N and a
local Hbeta-window metric. The Hbeta metric is the one that actually governs
broad-line detectability, and the two can disagree — a spectrum can have a
respectable continuum while the Hbeta region sits on a skyline or a chip gap.
No conclusion may rest on a single definition.

**The census is per epoch role, not per pair.** A pair does not have one S/N: it
has SNR_bright and SNR_faint, and the faint-state S/N is partly coupled to the
physical transition itself. Reporting a single pair-level number would hide the
asymmetry the primary Q1 arm is built to measure.

Bright and faint are assigned here by *continuum* level, not by line strength —
using the line would let the outcome choose the labels.

Usage
-----
    uv run python 00_scripts/06_spectrum_quality_control.py --slice
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.qc.zwarning import Disposition, classify_zwarning  # noqa: E402

SCRIPT = "06_spectrum_quality_control.py"


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def read_sdss(path: Path, redshift: float) -> Spectrum:
    with fits.open(path) as hdul:
        data = hdul[1].data
        wavelength = 10.0 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        ivar = np.asarray(data["ivar"], dtype=float)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    return Spectrum(wavelength, flux, error, float(redshift), mask=(ivar <= 0).astype(int))


def read_desi(path: Path, redshift: float) -> Spectrum:
    with fits.open(path) as hdul:
        wavelength = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
        flux = np.asarray(hdul["FLUX"].data, dtype=float)
        ivar = np.asarray(hdul["IVAR"].data, dtype=float)
        mask = np.asarray(hdul["MASK"].data)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    combined = ((mask != 0) | (ivar <= 0)).astype(int)
    return Spectrum(wavelength, flux, error, float(redshift), mask=combined)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def window_snr(spectrum: Spectrum, window: tuple[float, float]) -> float:
    """Median per-pixel S/N inside a rest-frame window.

    Median rather than mean so a few bad pixels or an unmasked skyline cannot
    set the value for the whole window.
    """
    rest = spectrum.rest_wavelength()
    inside = (rest >= window[0]) & (rest <= window[1]) & spectrum.good
    if inside.sum() < 5:
        return float("nan")
    return float(np.median(spectrum.flux[inside] / spectrum.error[inside]))


def continuum_level(spectrum: Spectrum, window: tuple[float, float]) -> float:
    rest = spectrum.rest_wavelength()
    inside = (rest >= window[0]) & (rest <= window[1]) & spectrum.good
    if inside.sum() < 5:
        return float("nan")
    return float(np.median(spectrum.flux[inside]))


def assess(spectrum: Spectrum, row: pd.Series, survey: str) -> dict[str, object]:
    cfg = load_config()
    rest = spectrum.rest_wavelength()
    line_lo, line_hi = cfg.windows.hbeta_region

    in_line = (rest >= line_lo) & (rest <= line_hi)
    n_line = int(in_line.sum())
    masked_fraction = float((~spectrum.good & in_line).sum() / n_line) if n_line else 1.0
    covers = bool(rest.min() <= line_lo and rest.max() >= line_hi)

    snr_continuum = window_snr(spectrum, tuple(cfg.snr_metrics.continuum_5100))
    snr_hbeta = window_snr(spectrum, tuple(cfg.snr_metrics.hbeta_window))

    # Bit-aware ZWARNING (D-016). A blind nonzero cut would discard broad-line
    # AGN for MANY_OUTLIERS and would bite hardest on the dim epochs that carry
    # the transition. Only FAIL excludes; REVIEW is surfaced for adjudication
    # and never silently drops out.
    zwarn = int(row.zwarning) if survey == "SDSS" else int(row.zwarn)
    warning = classify_zwarning(zwarn) if survey == "SDSS" else None

    failures: list[str] = []
    if not covers:
        failures.append("HBETA_OUT_OF_RANGE")
    if n_line == 0 or masked_fraction > cfg.qc.max_masked_fraction_in_line_region:
        failures.append("MASKED_HBETA")
    # The binding metric is the Hbeta window, since that is what governs
    # broad-line detectability; the continuum metric is reported alongside.
    if not np.isfinite(snr_hbeta) or snr_hbeta < cfg.qc.min_continuum_snr:
        failures.append("LOW_CONTINUUM_SNR")
    if warning is not None and warning.disposition is Disposition.FAIL:
        failures.append("ZWARNING_SET")

    # Mixed-survey QC retains ``mjd`` as an explicitly documented scalar
    # ordering/display coordinate. For DESI this is the D-077 arithmetic mean
    # of contributing EXP_FIBERMAP MJDs, never a claim of a single-night epoch.
    mjd_effective = (
        float(row.mjd_effective)
        if survey == "DESI" and hasattr(row, "mjd_effective")
        else float(row.mjd)
    )
    result: dict[str, object] = {
        "object_id": row.object_id,
        "survey": survey,
        "spectrum_id": str(row.spectrum_id),
        "mjd": mjd_effective,
        "redshift": float(row.redshift),
        "zwarn": zwarn,
        "zwarning_bits": "|".join(warning.bit_names) if warning else "",
        "zwarning_disposition": str(warning.disposition) if warning else "n/a",
        "zwarning_threatens": ",".join(warning.threatens) if warning else "",
        "zwarning_needs_review": bool(warning.requires_manual_review) if warning else False,
        # Retained so every major result can be re-run on strict-zero spectra
        # alone and reported as a robustness check.
        "strict_zwarning_pass": bool(warning.strict_pass) if warning else (zwarn == 0),
        "n_pixels": int(spectrum.wavelength.size),
        "rest_min": float(rest.min()),
        "rest_max": float(rest.max()),
        "covers_hbeta": covers,
        "masked_fraction_hbeta": masked_fraction,
        "snr_continuum_5100": snr_continuum,
        "snr_hbeta_window": snr_hbeta,
        "continuum_level_5100": continuum_level(spectrum, tuple(cfg.snr_metrics.continuum_5100)),
        "qc_pass": not failures,
        "qc_failures": ";".join(failures),
        "path": str(row.path),
    }
    if survey == "DESI":
        result.update(
            {
                "mjd_min": float(row.mjd_min),
                "mjd_effective": float(row.mjd_effective),
                "mjd_max": float(row.mjd_max),
                "n_exp": int(row.n_exp),
                "n_night": int(row.n_night),
                "program": str(row.program),
            }
        )
    return result


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------


def build_census(verdicts: pd.DataFrame) -> pd.DataFrame:
    """One row per object: the bright and faint epoch, and their S/N separately.

    Bright/faint is assigned by continuum level at 5100 A. Using the broad-line
    flux would let the outcome pick the labels, which is exactly the circularity
    the preregistration forbids.
    """
    cfg = load_config()
    rows: list[dict[str, object]] = []

    for object_id, group in verdicts.groupby("object_id"):
        usable = group[group.qc_pass & group.continuum_level_5100.notna()]
        if len(usable) < 2:
            rows.append(
                {
                    "object_id": object_id,
                    "n_usable_epochs": len(usable),
                    "pairable": False,
                }
            )
            continue

        ordered = usable.sort_values("continuum_level_5100", ascending=False)
        bright, faint = ordered.iloc[0], ordered.iloc[-1]

        rows.append(
            {
                "object_id": object_id,
                "n_usable_epochs": len(usable),
                "pairable": True,
                "bright_survey": bright.survey,
                "faint_survey": faint.survey,
                "same_instrument": bright.survey == faint.survey,
                "bright_mjd": bright.mjd,
                "faint_mjd": faint.mjd,
                "snr_bright_continuum": bright.snr_continuum_5100,
                "snr_faint_continuum": faint.snr_continuum_5100,
                "snr_bright_hbeta": bright.snr_hbeta_window,
                "snr_faint_hbeta": faint.snr_hbeta_window,
                "continuum_ratio": bright.continuum_level_5100 / faint.continuum_level_5100,
            }
        )

    census = pd.DataFrame(rows)
    if census.empty or "snr_faint_hbeta" not in census:
        return census

    grid = cfg.snr_grid
    for metric, column in (("hbeta", "snr_faint_hbeta"), ("continuum", "snr_faint_continuum")):
        census[f"faint_highest_rung_{metric}"] = [
            max([g for g in grid if g <= s], default=np.nan) if np.isfinite(s) else np.nan
            for s in census[column]
        ]
    return census.sort_values("snr_faint_hbeta", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    root = project_root()
    suffix = "_slice" if args.slice else ""

    records: list[dict[str, object]] = []

    sdss_manifest = root / "03_spectra" / "raw_sdss" / f"manifest{suffix}.csv"
    if sdss_manifest.exists():
        for row in pd.read_csv(sdss_manifest).itertuples():
            records.append(assess(read_sdss(root / row.path, row.redshift), row, "SDSS"))

    desi_manifest = root / "03_spectra" / "raw_desi" / f"manifest{suffix}.csv"
    if desi_manifest.exists():
        frame = pd.read_csv(desi_manifest, dtype={"targetid": str, "spectrum_id": str})
        for row in frame.itertuples():
            records.append(assess(read_desi(root / row.path, row.redshift), row, "DESI"))

    verdicts = pd.DataFrame(records)
    qc_dir = root / "03_spectra" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    verdicts.to_csv(qc_dir / f"qc_verdicts{suffix}.csv", index=False)

    census = build_census(verdicts)
    census.to_csv(qc_dir / f"snr_census{suffix}.csv", index=False)

    print("=" * 78)
    print("QC VERDICTS")
    print("=" * 78)
    for survey, group in verdicts.groupby("survey"):
        print(f"{survey:5s}  {len(group):>3} spectra, {int(group.qc_pass.sum()):>3} pass, "
              f"{int((group.zwarn != 0).sum()):>2} with nonzero ZWARN")

    sdss = verdicts[verdicts.survey == "SDSS"]
    if not sdss.empty:
        print()
        print("ZWARNING disposition (D-016, bit-aware):")
        for disposition, group in sdss.groupby("zwarning_disposition"):
            bits = sorted({b for row in group.zwarning_bits if row for b in row.split("|")})
            print(f"  {disposition:<7} {len(group):>3}  {'|'.join(bits) if bits else '(clean)'}")
        strict = int(sdss.strict_zwarning_pass.sum())
        print(f"  strict ZWARNING == 0: {strict}/{len(sdss)} "
              f"(robustness re-run uses these only)")
        review = sdss[sdss.zwarning_needs_review]
        if not review.empty:
            print(f"  awaiting manual review: {len(review)}")
            for row in review.itertuples():
                print(f"    {row.object_id} MJD {row.mjd:.0f}  {row.zwarning_bits} "
                      f"-> threatens {row.zwarning_threatens}")
    for code in ("HBETA_OUT_OF_RANGE", "MASKED_HBETA", "LOW_CONTINUUM_SNR"):
        n = int(verdicts.qc_failures.str.contains(code).sum())
        if n:
            print(f"  failing {code:<20} {n}")

    print()
    print("S/N by metric and survey (median):")
    print(
        verdicts.groupby("survey")[["snr_continuum_5100", "snr_hbeta_window"]]
        .median()
        .to_string(float_format=lambda v: f"{v:.1f}")
    )
    both = verdicts.dropna(subset=["snr_continuum_5100", "snr_hbeta_window"])
    if len(both) > 2:
        ratio = both.snr_hbeta_window / both.snr_continuum_5100
        print(f"\nHbeta-window / continuum S/N ratio: median {ratio.median():.2f}, "
              f"range {ratio.min():.2f}-{ratio.max():.2f}")

    print()
    print("=" * 78)
    print("TWO-EPOCH CENSUS (D-010: bright and faint reported separately)")
    print("=" * 78)
    pairable = census[census.pairable] if "pairable" in census else pd.DataFrame()
    print(f"objects with a usable pair: {len(pairable)} / {len(census)}")
    if not pairable.empty:
        show = pairable[[
            "object_id", "bright_survey", "faint_survey", "same_instrument",
            "snr_bright_hbeta", "snr_faint_hbeta", "faint_highest_rung_hbeta",
        ]]
        print()
        print(show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

        print()
        print("Primary Q1 arm degrades the FAINT epoch only, so the faint-state S/N")
        print("sets the reachable rungs. Bright epochs stay at native quality.")
        print()
        for rung in cfg.snr_grid:
            n_faint = int((pairable.snr_faint_hbeta >= rung).sum())
            n_both = int(
                ((pairable.snr_faint_hbeta >= rung) & (pairable.snr_bright_hbeta >= rung)).sum()
            )
            print(
                f"  S/N {rung:>3.0f}:  faint-only arm {n_faint:>2}/{len(pairable)}"
                f"   |  matched-both arm {n_both:>2}/{len(pairable)}"
            )
        print()
        print(f"same-instrument pairs: {int(pairable.same_instrument.sum())} / {len(pairable)}")
    print()
    print(f"written -> {(qc_dir / f'snr_census{suffix}.csv').relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
