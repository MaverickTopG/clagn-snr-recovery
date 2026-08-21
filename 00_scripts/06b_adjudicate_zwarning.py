#!/usr/bin/env python
"""Stage 7b — Hβ-specific adjudication of REVIEW-flagged spectra (D-016, D-032).

A survey warning is an *object-level* statement. This experiment is about Hβ.
Translating one into the other without checking is how a good epoch gets thrown
away — or a bad one kept.

``NEGATIVE_EMISSION`` (bit 6) is the sharp case: SDSS raises it when *any* of
C IV, C III], Mg II, Hβ or Hα satisfies ``LINEAREA + 3*LINEAREA_ERR < 0``. So
the flag alone does not implicate Hβ, and the first job is to find the line that
actually triggered it. That information is in the spZline table carried inside
the SDSS spec file itself (HDU 3), which is the very table the flag was computed
from.

Unmeasured lines are excluded from trigger consideration: SDSS writes
``LINEAREA = 0`` with ``LINEAREA_ERR = -1`` for lines outside coverage, which
would satisfy the inequality arithmetically while meaning nothing.

Nothing here decides Gold membership. It decides whether an epoch may enter the
primary Hβ experiment at all.

Usage
-----
    uv run python 00_scripts/06b_adjudicate_zwarning.py --slice
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.qc.region import ADJUDICATION_COLUMNS, HbetaAdjudication, RegionVerdict  # noqa: E402

SCRIPT = "06b_adjudicate_zwarning.py"

# Lines SDSS considers for the NEGATIVE_EMISSION test.
NEGATIVE_EMISSION_LINES = {"C_IV 1549", "C_III] 1908", "Mg_II 2799", "H_beta", "H_alpha"}


def _native(array: np.ndarray) -> np.ndarray:
    """FITS tables are big-endian; pandas refuses them on little-endian hosts."""
    return array.astype(array.dtype.newbyteorder("="))


def read_spzline(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        if "SPZLINE" not in hdul:
            return pd.DataFrame()
        data = hdul["SPZLINE"].data
        frame = pd.DataFrame(
            {
                "line": [str(n).strip() for n in data["LINENAME"]],
                "wave": _native(data["LINEWAVE"]),
                "area": _native(data["LINEAREA"]),
                "area_err": _native(data["LINEAREA_ERR"]),
                "chi2": _native(data["LINECHI2"]),
            }
        )
    # LINEAREA_ERR <= 0 marks a line that was not measured at all.
    frame["measured"] = frame.area_err > 0
    frame["negative_trigger"] = frame.measured & (frame.area + 3.0 * frame.area_err < 0)
    return frame


def hbeta_region_checks(path: Path, redshift: float) -> dict[str, object]:
    """Measure the Hβ region directly, independently of any survey flag."""
    cfg = load_config()
    low, high = cfg.windows.hbeta_region

    with fits.open(path) as hdul:
        data = hdul["COADD"].data
        wavelength = 10.0 ** _native(data["loglam"])
        flux = _native(data["flux"])
        ivar = _native(data["ivar"])
        model = _native(data["model"]) if "model" in data.columns.names else None

    rest = wavelength / (1.0 + redshift)
    window = (rest >= low) & (rest <= high)
    n_window = int(window.sum())
    if n_window == 0:
        return {"hbeta_pixel_mask_ok": False, "hbeta_ivar_ok": False,
                "hbeta_local_residual_ok": False, "n_window_pixels": 0}

    bad = ivar[window] <= 0
    masked_fraction = float(bad.mean())

    finite_ivar = ivar[window][~bad]
    ivar_ok = bool(finite_ivar.size > 0 and np.all(np.isfinite(finite_ivar)))

    residual_ok: bool | None = None
    residual_scatter: float | None = None
    if model is not None:
        good = window & (ivar > 0)
        if good.sum() > 20:
            standardised = (flux[good] - model[good]) * np.sqrt(ivar[good])
            residual_scatter = float(np.std(standardised))
            # Well-behaved pixels scatter near 1. Wide latitude here: this is a
            # pathology check, not the uncertainty calibration of Gate C.
            residual_ok = bool(0.3 < residual_scatter < 3.0)

    return {
        "hbeta_pixel_mask_ok": bool(masked_fraction <= cfg.qc.max_masked_fraction_in_line_region),
        "hbeta_masked_fraction": masked_fraction,
        "hbeta_ivar_ok": ivar_ok,
        "hbeta_local_residual_ok": residual_ok,
        "hbeta_residual_scatter": residual_scatter,
        "n_window_pixels": n_window,
    }


def adjudicate(row: pd.Series, root: Path) -> tuple[HbetaAdjudication, pd.DataFrame]:
    path = root / row.path
    lines = read_spzline(path)
    checks = hbeta_region_checks(path, float(row.redshift))

    trigger_line = trigger_area = trigger_err = None
    if not lines.empty:
        candidates = lines[lines.negative_trigger & lines.line.isin(NEGATIVE_EMISSION_LINES)]
        if not candidates.empty:
            # Most negative significance is the one that raised the flag.
            worst = candidates.assign(sig=candidates.area / candidates.area_err).sort_values("sig")
            first = worst.iloc[0]
            trigger_line, trigger_area, trigger_err = (
                str(first.line), float(first.area), float(first.area_err)
            )

    hbeta_row = lines[lines.line == "H_beta"] if not lines.empty else pd.DataFrame()
    hbeta_significance = (
        float(hbeta_row.iloc[0].area / hbeta_row.iloc[0].area_err)
        if not hbeta_row.empty and hbeta_row.iloc[0].area_err > 0
        else None
    )

    record = HbetaAdjudication(
        object_id=str(row.object_id),
        mjd=float(row.mjd),
        zwarning=int(row.zwarn),
        trigger_line=trigger_line,
        trigger_linearea=trigger_area,
        trigger_linearea_err=trigger_err,
        hbeta_pixel_mask_ok=checks["hbeta_pixel_mask_ok"],  # type: ignore[arg-type]
        hbeta_ivar_ok=checks["hbeta_ivar_ok"],  # type: ignore[arg-type]
        hbeta_local_residual_ok=checks["hbeta_local_residual_ok"],  # type: ignore[arg-type]
        # Set by the Gate B fit-validation stage; deliberately unset here so an
        # unchecked box cannot read as a pass.
        hbeta_pyqsofit_valid=None,
        redshift_secure=bool(int(row.zwarn) & 0b100100 == 0),  # no SMALL_DELTA_CHI2 / Z_FITLIMIT
        reviewer_note=(
            f"spZline Hbeta area significance {hbeta_significance:.1f} sigma"
            if hbeta_significance is not None
            else "spZline Hbeta measurement unavailable"
        ),
    )
    return record, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    root = project_root()
    suffix = "_slice" if args.slice else ""
    qc_dir = root / "03_spectra" / "qc"

    verdicts = pd.read_csv(qc_dir / f"qc_verdicts{suffix}.csv")
    flagged = verdicts[
        (verdicts.survey == "SDSS") & verdicts.zwarning_needs_review.fillna(False).astype(bool)
    ]

    if flagged.empty:
        print("no spectra require Hbeta-specific adjudication")
        return 0

    records: list[dict[str, object]] = []
    for row in flagged.itertuples():
        record, lines = adjudicate(row, root)
        records.append(record.to_row())

        print("=" * 74)
        print(f"{record.object_id}  MJD {record.mjd:.0f}  ZWARNING={record.zwarning} "
              f"[{row.zwarning_bits}]")
        print("=" * 74)

        if not lines.empty:
            considered = lines[lines.line.isin(NEGATIVE_EMISSION_LINES)]
            print("\nLines SDSS tests for NEGATIVE_EMISSION:")
            for line in considered.itertuples():
                if not line.measured:
                    print(f"  {line.line:<14} not measured (LINEAREA_ERR <= 0)")
                    continue
                sig = line.area / line.area_err
                mark = "  <-- TRIGGER" if line.negative_trigger else ""
                print(f"  {line.line:<14} area {line.area:>12.4g} +/- {line.area_err:<11.4g}"
                      f" ({sig:+6.1f} sigma){mark}")

        print(f"\ntrigger line          {record.trigger_line}")
        print(f"trigger is Hbeta      {record.trigger_is_hbeta}")
        print(f"Hbeta pixels ok       {record.hbeta_pixel_mask_ok}")
        print(f"Hbeta ivar ok         {record.hbeta_ivar_ok}")
        print(f"Hbeta residual ok     {record.hbeta_local_residual_ok}")
        print(f"Hbeta PyQSOFit valid  {record.hbeta_pyqsofit_valid}  (set at Gate B)")
        print(f"redshift secure       {record.redshift_secure}")
        print(f"note                  {record.reviewer_note}")
        print(f"\nrecommended           {record.recommended_disposition()}")
        print(f"final                 {record.final_hbeta_disposition}  <-- requires a reviewer")

    frame = pd.DataFrame(records, columns=list(ADJUDICATION_COLUMNS))
    destination = qc_dir / f"hbeta_adjudication{suffix}.csv"
    frame.to_csv(destination, index=False)
    print()
    print(f"{len(frame)} record(s) -> {destination.relative_to(root)}")
    pending = int((frame.final_hbeta_disposition == str(RegionVerdict.PENDING)).sum())
    print(f"{pending} still PENDING; these epochs are blocked from the primary Hbeta experiment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
