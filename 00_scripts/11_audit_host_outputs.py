#!/usr/bin/env python
"""Stage 11 / Phase 5A — mechanical audit of PyQSOFit host outputs.

**No astrophysics here.** This stage establishes only what the software did,
against the exact vendored PyQSOFit build. Interpreting a zero host as a
scientific measurement before knowing whether zero is how this version
represents a declined decomposition would be a category error.

What the vendored build actually does (``PyQSOFit.py`` ``decompose_host_qso``):

* ``self.host`` is initialised to ``np.zeros(len(wave))`` and ``self.decomposed``
  to ``True`` *before* any fitting;
* on any of its decline conditions it sets ``self.decomposed = False`` and
  **leaves host as zeros**;
* only on success does it assign ``self.host = datacube[3]``, ``self.qso =
  datacube[4]``, and populate ``host_result`` with ``SN_host``,
  ``rchi2_decomp``, ``frac_host_4200``, ``frac_host_5100``, ``Dn4000``.

So a zero host array is ambiguous on its own, and ``self.decomposed`` is the
authoritative flag. Its decline conditions are: more than 10% of pixels with
negative host or negative QSO, or ``median(host) < 0.01 * median(|flux|)``, or
``median(data - qso) < 0``.

Mechanical states emitted here:

    DECOMPOSITION_NOT_RUN   host decomposition was not requested
    DECOMPOSITION_FAILED    ran, declined by PyQSOFit's own conditions
    OUTPUT_MISSING          expected attribute absent
    OUTPUT_NONFINITE        NaN or inf in the host model
    OUTPUT_VALID_ZERO       ran, succeeded, host identically zero
    OUTPUT_VALID_NONZERO    ran, succeeded, host has non-zero flux

Scientific states (HOST_DETECTED / HOST_CONSISTENT_WITH_ZERO /
HOST_DECOMPOSITION_FAILED) are **not** assigned here. Those are statistical
claims and need the stability and uncertainty work of Phase 5B.

Usage
-----
    uv run python 00_scripts/11_audit_host_outputs.py --slice
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

SCRIPT = "11_audit_host_outputs.py"

# Emission lines that must NOT appear in a stellar host template. If the host
# component carries line flux, raising the host fraction in Q2 would change the
# line spectrum too, confounding host dilution with a line-flux change.
LINE_PROBES = {
    "Hbeta_broad_wing": (4790.0, 4930.0),
    "OIII_5007": (4995.0, 5020.0),
    "OIII_4959": (4948.0, 4970.0),
    "Halpha_NII": (6540.0, 6590.0),
    "FeII_optical": (5150.0, 5350.0),
}


def read_spectrum(path: Path, survey: str, redshift: float):
    from astropy.io import fits

    with fits.open(path) as hdul:
        if survey == "SDSS":
            data = hdul[1].data
            wavelength = 10.0 ** np.asarray(data["loglam"], dtype=float)
            flux = np.asarray(data["flux"], dtype=float)
            ivar = np.asarray(data["ivar"], dtype=float)
        else:
            wavelength = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
            flux = np.asarray(hdul["FLUX"].data, dtype=float)
            ivar = np.asarray(hdul["IVAR"].data, dtype=float)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    good = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    return wavelength[good], flux[good], error[good], float(redshift)


def host_result_value(fit, name: str) -> float | None:
    """Pull one named quantity out of PyQSOFit's host_result array."""
    names = getattr(fit, "host_result_name", None)
    values = getattr(fit, "host_result", None)
    if names is None or values is None or len(names) == 0:
        return None
    matches = np.where(np.asarray(names) == name)[0]
    if matches.size == 0:
        return None
    value = float(np.asarray(values)[matches[0]])
    return value if np.isfinite(value) else None


def line_leakage(fit, redshift: float) -> dict[str, float | None]:
    """Fraction of the host model's flux sitting inside line windows.

    A stellar continuum template should carry stellar absorption, not nebular
    emission. Measuring this now prevents a Q2 experiment that silently changes
    the line spectrum while claiming to change only the host fraction.
    """
    host = getattr(fit, "host", None)
    wave = getattr(fit, "wave", None)
    if host is None or wave is None or np.size(host) == 0:
        return dict.fromkeys(LINE_PROBES, None)

    host = np.asarray(host, dtype=float)
    rest = np.asarray(wave, dtype=float) / (1.0 + redshift)

    out: dict[str, float | None] = {}
    for name, (low, high) in LINE_PROBES.items():
        window = (rest >= low) & (rest <= high)
        if window.sum() < 5:
            out[name] = None
            continue
        # Excess over a local straight-line continuum through the window edges.
        edge = max(3, window.sum() // 6)
        inside = host[window]
        baseline = np.linspace(np.median(inside[:edge]), np.median(inside[-edge:]), inside.size)
        excess = float(np.trapezoid(inside - baseline, rest[window]))
        scale = float(np.trapezoid(np.abs(baseline), rest[window]))
        out[name] = excess / scale if scale > 0 else None
    return out


def audit_one(row: pd.Series, root: Path) -> dict[str, object]:
    from pyqsofit.PyQSOFit import QSOFit

    vendor = str(root / "06_fitting" / "pyqsofit" / "vendor" / "PyQSOFit")
    wavelength, flux, error, redshift = read_spectrum(
        root / row.path, row.survey, row.redshift
    )

    record: dict[str, object] = {
        "object_id": row.object_id,
        "survey": row.survey,
        "spectrum_id": str(row.spectrum_id),
        "mjd": row.mjd,
        "redshift": redshift,
        "snr_hbeta_window": row.snr_hbeta_window,
        # Observed continuum, carried through so bright/faint roles never depend
        # on a fitted quantity that is undefined when decomposition declines.
        "continuum_level_5100": row.continuum_level_5100,
        "host_requested": True,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = QSOFit(wavelength, flux, error, redshift, path=vendor)
            fit.Fit(
                name=str(row.object_id), deredden=True, decompose_host=True,
                Fe_uv_op=True, linefit=True, MC=False, save_result=False,
                plot_fig=False, save_fig=False, verbose=False,
            )
    except Exception as error_:  # noqa: BLE001
        record.update(
            host_decomposition_executed=False,
            host_output_state="DECOMPOSITION_NOT_RUN",
            fit_exception=f"{type(error_).__name__}: {error_}",
        )
        return record

    record["fit_exception"] = ""
    record["host_decomposition_executed"] = True

    # The authoritative flag, not a truthiness check on `host`.
    decomposed = bool(getattr(fit, "decomposed", False))
    host = getattr(fit, "host", None)
    qso = getattr(fit, "qso", None)

    host_present = host is not None and np.size(host) > 0
    host_finite = bool(host_present and np.all(np.isfinite(np.asarray(host, dtype=float))))
    host_nonzero = bool(host_present and np.any(np.asarray(host, dtype=float) != 0.0))

    if not host_present:
        state = "OUTPUT_MISSING"
    elif not host_finite:
        state = "OUTPUT_NONFINITE"
    elif not decomposed:
        state = "DECOMPOSITION_FAILED"
    elif host_nonzero:
        state = "OUTPUT_VALID_NONZERO"
    else:
        state = "OUTPUT_VALID_ZERO"

    host_array = np.asarray(host, dtype=float) if host_present else np.array([])
    qso_array = np.asarray(qso, dtype=float) if qso is not None and np.size(qso) else np.array([])
    wave_out = np.asarray(getattr(fit, "wave", []), dtype=float)

    record.update(
        pyqsofit_decomposed_flag=decomposed,
        host_output_present=host_present,
        host_output_finite=host_finite,
        host_output_nonzero=host_nonzero,
        host_output_state=state,
        host_coefficients_present=bool(np.size(getattr(fit, "host_result", [])) > 0),
        host_model_on_expected_grid=bool(
            host_present and wave_out.size == host_array.size
        ),
        n_host_pixels=int(host_array.size),
        host_model_integrated_flux=(
            float(np.trapezoid(host_array, wave_out)) if host_array.size == wave_out.size else None
        ),
        qso_model_integrated_flux=(
            float(np.trapezoid(qso_array, wave_out)) if qso_array.size == wave_out.size else None
        ),
        host_negative_pixel_fraction=(
            float(np.mean(host_array < 0)) if host_array.size else None
        ),
        # PyQSOFit's own host diagnostics, populated only on success.
        frac_host_5100=host_result_value(fit, "frac_host_5100"),
        frac_host_4200=host_result_value(fit, "frac_host_4200"),
        SN_host=host_result_value(fit, "SN_host"),
        rchi2_decomp=host_result_value(fit, "rchi2_decomp"),
        Dn4000=host_result_value(fit, "Dn4000"),
    )
    record.update({f"leak_{k}": v for k, v in line_leakage(fit, redshift).items()})
    return record


def pair_pattern(frame: pd.DataFrame) -> pd.DataFrame:
    """Bright/faint host pattern per object.

    Roles come from the **observed continuum level**, never from the fitted QSO
    model: the QSO model is undefined when decomposition declines, so ordering
    on it would sort every failed epoch into the faint slot regardless of how
    bright it actually was, manufacturing a spurious bright/faint asymmetry.
    """
    rows: list[dict[str, object]] = []
    for object_id, group in frame.groupby("object_id"):
        if len(group) < 2:
            rows.append({"object_id": object_id, "n_epochs": len(group), "pattern": "SINGLE_EPOCH"})
            continue
        if group.continuum_level_5100.isna().all():
            rows.append({"object_id": object_id, "n_epochs": len(group),
                         "pattern": "ROLES_UNDETERMINED"})
            continue
        ordered = group.sort_values("continuum_level_5100", ascending=False, na_position="last")
        bright, faint = ordered.iloc[0], ordered.iloc[-1]

        def has_host(record: pd.Series) -> bool:
            return record.host_output_state == "OUTPUT_VALID_NONZERO"

        if any(r.host_output_state in {"DECOMPOSITION_NOT_RUN", "OUTPUT_MISSING",
                                       "OUTPUT_NONFINITE"} for r in (bright, faint)):
            pattern = "DECOMPOSITION_FAILURE"
        elif has_host(bright) and has_host(faint):
            pattern = "BOTH_PRESENT"
        elif has_host(faint) and not has_host(bright):
            pattern = "BRIGHT_ABSENT_FAINT_PRESENT"
        elif has_host(bright) and not has_host(faint):
            pattern = "BRIGHT_PRESENT_FAINT_ABSENT"
        else:
            pattern = "BOTH_ABSENT"

        rows.append({
            "object_id": object_id,
            "n_epochs": len(group),
            "pattern": pattern,
            "bright_survey": bright.survey,
            "faint_survey": faint.survey,
            "same_instrument": bright.survey == faint.survey,
            "bright_state": bright.host_output_state,
            "faint_state": faint.host_output_state,
            "bright_frac_host_5100": bright.frac_host_5100,
            "faint_frac_host_5100": faint.frac_host_5100,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    root = project_root()
    suffix = "_slice" if args.slice else ""
    qc_dir = root / "03_spectra" / "qc"

    verdicts = pd.read_csv(qc_dir / f"qc_verdicts{suffix}.csv")
    passing = verdicts[verdicts.qc_pass]
    print(f"auditing {len(passing)} QC-passing spectra\n")

    records = [audit_one(row, root) for row in passing.itertuples()]
    frame = pd.DataFrame(records)

    out_dir = root / "06_fitting" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"host_output_audit{suffix}.csv", index=False)

    print("=" * 78)
    print("PHASE 5A — MECHANICAL HOST OUTPUT STATES")
    print("=" * 78)
    print(frame.host_output_state.value_counts().to_string())
    print()
    show = frame[[
        "object_id", "survey", "snr_hbeta_window", "pyqsofit_decomposed_flag",
        "host_output_nonzero", "host_output_state", "frac_host_5100", "SN_host",
    ]].sort_values(["object_id", "survey"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print()
    print("=" * 78)
    print("HOST-COMPONENT LINE LEAKAGE (fractional excess over local baseline)")
    print("=" * 78)
    print("A stellar template should carry absorption, not emission. Leakage here")
    print("would mean raising f_host in Q2 also changes the line spectrum.")
    leak_columns = [c for c in frame.columns if c.startswith("leak_")]
    valid = frame[frame.host_output_state == "OUTPUT_VALID_NONZERO"]
    if not valid.empty:
        print()
        print(valid[leak_columns].describe().loc[["count", "mean", "min", "max"]]
              .to_string(float_format=lambda v: f"{v:.4f}"))
    else:
        print("\n(no valid non-zero host models to test)")

    print()
    print("=" * 78)
    print("PAIRED BRIGHT / FAINT HOST PATTERN")
    print("=" * 78)
    pattern = pair_pattern(frame)
    pattern.to_csv(out_dir / f"host_pair_pattern{suffix}.csv", index=False)
    print(pattern.pattern.value_counts().to_string())
    print()
    print(pattern.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print()
    print("NOTE: no scientific host state is assigned here. HOST_DETECTED and")
    print("HOST_CONSISTENT_WITH_ZERO are statistical claims requiring the Phase 5B")
    print("stability and uncertainty work, and Q2 eligibility is not decided yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
