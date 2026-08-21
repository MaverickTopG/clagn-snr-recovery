#!/usr/bin/env python
"""Stage 18 / Gate C1-C2 — native repeat-residual diagnostics in W only.

**Diagnostics only.** No `k_sigma` is fitted, no `sigma_corrected` is created,
no z-width is forced to unity, nothing outside W is touched.

C1 semantics, enforced here and nowhere overridden:

    ivar_native   immutable, exactly as DESI supplied
    sigma_native  1/sqrt(ivar_native), only where ivar > 0 AND mask == 0
    sigma_corrected  DOES NOT EXIST

Nonzero DESI spectral mask pixels are excluded outright.

C2a  robust central width of the normalised difference
C2b  wavelength-correlated repeatability within W

Cross-bin pairs are retained with both exposure bins ``(b1, b2)``. They are
never collapsed to a single pair-level S/N by min, mean or geometric mean:
``k_sigma`` would act on each exposure's uncertainty, not on an invented
pair-level quantity.

Independence: targets are the astrophysical units, but targets sharing a
petal-night share calibration environment. All intervals resample at
NIGHT/cluster level — never pixels, never exposure pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

W = (5000.0, 5200.0)
LABELS = ["00-10", "10-20", "20-30", "30-50", "50+"]
EDGES = [0, 10, 20, 30, 50, np.inf]
N_BOOT = 2000
SEED = 314159


def robust_width(z: np.ndarray) -> float | None:
    """Predeclared robust scale: half the 16-84 percentile span.

    Robust rather than the standard deviation because outlying pixels — cosmic
    rays, imperfectly masked sky — would otherwise set the answer. Not forced
    to unity by any step.
    """
    z = z[np.isfinite(z)]
    if z.size < 50:
        return None
    lo, hi = np.percentile(z, [16, 84])
    return float((hi - lo) / 2.0)


def autocorrelation(residual: np.ndarray, max_lag: int = 20) -> np.ndarray:
    r = residual - np.nanmean(residual)
    denom = np.nansum(r * r)
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    return np.array([np.nansum(r[: len(r) - k] * r[k:]) / denom for k in range(max_lag + 1)])


def main() -> int:
    from astropy.io import fits

    root = project_root()
    final = root / "02_catalogs" / "final"

    eligible = set(pd.read_csv(final / "c0_eligible_targets.csv", dtype={"targetid": str}).targetid)
    manifest = pd.read_csv(final / "c0_acquisition_manifest.csv")

    rows: list[dict] = []
    ac_store: list[np.ndarray] = []

    for entry in manifest.itertuples():
        path = root / entry.local_path
        if not path.exists():
            continue
        with fits.open(path, memmap=False) as hdul:
            fm = hdul["FIBERMAP"].data
            lam = np.asarray(hdul["B_WAVELENGTH"].data, dtype=float)
            flux = np.asarray(hdul["B_FLUX"].data, dtype=float)
            ivar = np.asarray(hdul["B_IVAR"].data, dtype=float)
            mask = np.asarray(hdul["B_MASK"].data)
            tid = np.asarray(fm["TARGETID"]).astype(str)
            night = int(entry.night)

            in_w = (lam >= W[0]) & (lam <= W[1])
            order = np.argsort(tid, kind="stable")
            for target in np.unique(tid):
                if target not in eligible:
                    continue
                idx = order[np.searchsorted(tid[order], target, side="left"):
                            order.size if False else np.searchsorted(tid[order], target, side="right")]
                if idx.size != 2:
                    continue
                i, j = idx[0], idx[1]

                # C1: valid only where ivar > 0 AND mask == 0, both exposures.
                good = (in_w & (ivar[i] > 0) & (mask[i] == 0) & (ivar[j] > 0) & (mask[j] == 0)
                        & np.isfinite(flux[i]) & np.isfinite(flux[j]))
                if good.sum() < 50:
                    continue

                f1, f2 = flux[i][good], flux[j][good]
                s1 = 1.0 / np.sqrt(ivar[i][good])
                s2 = 1.0 / np.sqrt(ivar[j][good])
                snr1 = float(np.median(np.abs(f1) / s1))
                snr2 = float(np.median(np.abs(f2) / s2))

                # PRIMARY: no renormalisation.
                z_raw = (f1 - f2) / np.sqrt(s1**2 + s2**2)
                # ROBUSTNESS: one scalar per pair, nothing wavelength dependent.
                a = float(np.median(f1) / np.median(f2)) if np.median(f2) != 0 else np.nan
                z_scaled = ((f1 - a * f2) / np.sqrt(s1**2 + (a**2) * s2**2)
                            if np.isfinite(a) and a > 0 else np.full_like(z_raw, np.nan))

                b1 = LABELS[int(np.digitize(snr1, EDGES[1:-1]))]
                b2 = LABELS[int(np.digitize(snr2, EDGES[1:-1]))]
                rows.append({
                    "targetid": target, "night": night, "tileid": int(entry.tileid),
                    "petal": int(entry.petal), "cluster": entry.cluster_id,
                    "snr1": snr1, "snr2": snr2, "bin1": b1, "bin2": b2,
                    "same_bin": b1 == b2, "n_pixels": int(good.sum()),
                    "width_raw": robust_width(z_raw),
                    "width_scaled": robust_width(z_scaled),
                    "scalar_a": a,
                })
                if b1 == b2:
                    ac_store.append(autocorrelation(z_raw))

    frame = pd.DataFrame(rows)
    frame.to_csv(final / "c2_native_residuals.csv", index=False)
    print(f"pairs measured: {len(frame)}  targets {frame.targetid.nunique()}  "
          f"clusters {frame.cluster.nunique()}  nights {frame.night.nunique()}\n")

    # ---- C2a: width by same-bin regime, night-level bootstrap ----------------
    rng = np.random.default_rng(SEED)
    nights = frame.night.unique()
    print("=" * 74)
    print("C2a  NATIVE z WIDTH BY SAME-BIN S/N REGIME (night-level resampling)")
    print("=" * 74)
    print(f"{'bin':<8}{'targets':>9}{'nights':>8}{'width_raw':>11}{'95% CI':>20}{'width_scaled':>14}")
    for lab in LABELS:
        cell = frame[frame.same_bin & (frame.bin1 == lab)]
        if len(cell) < 20:
            print(f"{lab:<8}{len(cell):>9}{cell.night.nunique():>8}   insufficient support")
            continue
        point = float(np.median(cell.width_raw.dropna()))
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(nights, size=nights.size, replace=True)
            vals = pd.concat([cell[cell.night == n].width_raw for n in pick]).dropna()
            if len(vals):
                boots.append(float(np.median(vals)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        scaled = float(np.median(cell.width_scaled.dropna())) if cell.width_scaled.notna().any() else np.nan
        print(f"{lab:<8}{cell.targetid.nunique():>9}{cell.night.nunique():>8}"
              f"{point:>11.3f}   [{lo:.3f}, {hi:.3f}]{scaled:>14.3f}")

    # ---- C2b: wavelength correlation ----------------------------------------
    print()
    print("=" * 74)
    print("C2b  WAVELENGTH-CORRELATED REPEATABILITY WITHIN W")
    print("=" * 74)
    if ac_store:
        ac = np.nanmedian(np.vstack(ac_store), axis=0)
        print("median residual autocorrelation by pixel lag:")
        print("  lag :  " + "  ".join(f"{k:>5d}" for k in range(0, 11)))
        print("  rho :  " + "  ".join(f"{v:>5.2f}" for v in ac[:11]))
        beyond = float(np.nanmax(np.abs(ac[1:11])))
        verdict = ("SCALAR_INFLATION_INSUFFICIENT" if beyond > 0.10
                   else "no strong correlated structure detected at lags 1-10")
        print(f"\n  max |rho| over lags 1-10: {beyond:.3f}  ->  {verdict}")
    print()
    print("No k_sigma fitted. No sigma_corrected created. Nothing outside W touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
