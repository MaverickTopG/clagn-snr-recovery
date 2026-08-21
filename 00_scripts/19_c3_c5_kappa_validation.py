#!/usr/bin/env python
"""Stage 19 / Gate C2-C5 — integrity checks, k_empirical, and validation.

**Nothing is adopted here.** `k_empirical` is what the repeat data estimate;
`k_adopted` is a later decision and is not made. No production
`sigma_corrected`, no IVAR overwrite, no smooth surface, nothing outside W.

The four factors come from the **same-bin** populations only. The 255 cross-bin
targets are deliberately excluded from estimation and held back as validation —
their residual calibration has not informed the factors they will test.
Cross-bin validation uses each exposure's own regime,

    z_corr = (f1 - f2) / sqrt(k_a^2 sigma_1^2 + k_b^2 sigma_2^2)

so no pair-level S/N is invented.
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
FIT_BINS = LABELS[:4]                      # >50 is EMPIRICAL_HIGH_SN_UNSUPPORTED
N_BOOT = 2000
SEED = 314159
KNOWN_ISSUE = {(20211212, 9)}              # documented DESI calibration problem


def robust_width(z: np.ndarray) -> float | None:
    """The estimator frozen in C2. Unchanged here."""
    z = z[np.isfinite(z)]
    if z.size < 50:
        return None
    lo, hi = np.percentile(z, [16, 84])
    return float((hi - lo) / 2.0)


def autocorr(r: np.ndarray, max_lag: int) -> np.ndarray:
    r = r - np.nanmean(r)
    d = np.nansum(r * r)
    if d <= 0:
        return np.full(max_lag + 1, np.nan)
    return np.array([np.nansum(r[: len(r) - k] * r[k:]) / d for k in range(max_lag + 1)])


def collect(root: Path):
    """Per-pair residual vectors in W, keeping both exposure bins."""
    from astropy.io import fits

    final = root / "02_catalogs" / "final"
    eligible = set(pd.read_csv(final / "c0_eligible_targets.csv",
                               dtype={"targetid": str}).targetid)
    manifest = pd.read_csv(final / "c0_acquisition_manifest.csv")

    out = []
    for entry in manifest.itertuples():
        path = root / entry.local_path
        if not path.exists():
            continue
        with fits.open(path, memmap=False) as hdul:
            fm = hdul["FIBERMAP"].data
            lam = np.asarray(hdul["B_WAVELENGTH"].data, float)
            flux = np.asarray(hdul["B_FLUX"].data, float)
            ivar = np.asarray(hdul["B_IVAR"].data, float)
            mask = np.asarray(hdul["B_MASK"].data)
            tid = np.asarray(fm["TARGETID"]).astype(str)
            in_w = (lam >= W[0]) & (lam <= W[1])
            srt = np.argsort(tid, kind="stable")
            for t in np.unique(tid):
                if t not in eligible:
                    continue
                lo = np.searchsorted(tid[srt], t, "left")
                hi = np.searchsorted(tid[srt], t, "right")
                if hi - lo != 2:
                    continue
                i, j = srt[lo], srt[hi - 1]
                good = (in_w & (ivar[i] > 0) & (mask[i] == 0) & (ivar[j] > 0)
                        & (mask[j] == 0) & np.isfinite(flux[i]) & np.isfinite(flux[j]))
                if good.sum() < 50:
                    continue
                f1, f2 = flux[i][good], flux[j][good]
                s1 = 1 / np.sqrt(ivar[i][good])
                s2 = 1 / np.sqrt(ivar[j][good])
                n1 = float(np.median(np.abs(f1) / s1))
                n2 = float(np.median(np.abs(f2) / s2))
                out.append({
                    "targetid": t, "night": int(entry.night), "petal": int(entry.petal),
                    "b1": LABELS[int(np.digitize(n1, EDGES[1:-1]))],
                    "b2": LABELS[int(np.digitize(n2, EDGES[1:-1]))],
                    "d": f1 - f2, "s1": s1, "s2": s2,
                })
    return out


def main() -> int:
    root = project_root()
    pairs = collect(root)
    frame = pd.DataFrame([{k: v for k, v in p.items() if k not in ("d", "s1", "s2")}
                          for p in pairs])
    nights = sorted(frame.night.unique())

    # ---------------- known-issue cluster check ----------------
    print("=" * 74)
    print("C2 INTEGRITY — DESI known-issue cluster cross-reference")
    print("=" * 74)
    clusters = set(zip(frame.night, frame.petal, strict=True))
    hits = clusters & KNOWN_ISSUE
    print(f"selected (NIGHT, PETAL): {sorted(clusters)}")
    print(f"checked against documented issue {sorted(KNOWN_ISSUE)}")
    print(f"-> {'FLAGGED ' + str(sorted(hits)) if hits else 'PASS (no overlap)'}")

    # ---------------- covariance: broader lags + per night ----------------
    print()
    print("=" * 74)
    print("C2b EXTENDED — autocorrelation over broader lags, and per night")
    print("=" * 74)
    same = [p for p in pairs if p["b1"] == p["b2"]]
    max_lag = 60
    pooled = np.nanmedian(np.vstack([
        autocorr(p["d"] / np.sqrt(p["s1"] ** 2 + p["s2"] ** 2), max_lag) for p in same
    ]), axis=0)
    print("pooled |rho| max by lag range:")
    for name, sl in (("1-10", slice(1, 11)), ("11-30", slice(11, 31)), ("31-60", slice(31, 61))):
        print(f"   lags {name:<6} max|rho| = {np.nanmax(np.abs(pooled[sl])):.4f}")

    print("\nper-night max|rho| over lags 1-60 (opposite-sign nights cannot cancel here):")
    per_night = {}
    for n in nights:
        sub = [p for p in same if p["night"] == n]
        if len(sub) < 20:
            continue
        ac = np.nanmedian(np.vstack([
            autocorr(p["d"] / np.sqrt(p["s1"] ** 2 + p["s2"] ** 2), max_lag) for p in sub
        ]), axis=0)
        per_night[n] = float(np.nanmax(np.abs(ac[1:])))
        print(f"   {n}: {per_night[n]:.4f}   (n={len(sub)})")
    if per_night:
        v = np.array(list(per_night.values()))
        print(f"\n   distribution: min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f}")
        print("   (no threshold declared post hoc; distribution reported as measured)")

    # ---------------- C3: k_empirical from same-bin only ----------------
    print()
    print("=" * 74)
    print("C3  k_empirical — same-bin populations only, piecewise constant")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    k_emp: dict[str, float] = {}
    print(f"{'bin':<8}{'targets':>9}{'nights':>8}{'k_empirical':>13}{'95% CI (night-level)':>24}")
    for lab in FIT_BINS:
        sub = [p for p in same if p["b1"] == lab]
        if len(sub) < 20:
            print(f"{lab:<8}{len(sub):>9}   insufficient")
            continue
        widths = pd.DataFrame({
            "night": [p["night"] for p in sub],
            "w": [robust_width(p["d"] / np.sqrt(p["s1"] ** 2 + p["s2"] ** 2)) for p in sub],
        }).dropna()
        k_emp[lab] = float(np.median(widths.w))
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(nights, size=len(nights), replace=True)
            vals = pd.concat([widths[widths.night == n].w for n in pick])
            if len(vals):
                boots.append(float(np.median(vals)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{lab:<8}{len(sub):>9}{widths.night.nunique():>8}{k_emp[lab]:>13.3f}"
              f"      [{lo:.3f}, {hi:.3f}]")
    print(f"{'50+':<8}{'—':>9}   EMPIRICAL_HIGH_SN_UNSUPPORTED (not fitted)")
    print("\nNo factor forced to 1. No monotonicity imposed. No smooth curve fitted.")

    # ---------------- C4/C5: held-out cross-bin validation ----------------
    print()
    print("=" * 74)
    print("C4  CROSS-BIN VALIDATION (held out; never used to estimate k)")
    print("=" * 74)
    cross = [p for p in pairs if p["b1"] != p["b2"]]
    print(f"{'cell':<18}{'targets':>9}{'native width':>14}{'corrected width':>17}")
    for cell in sorted({tuple(sorted((p["b1"], p["b2"]))) for p in cross}):
        sub = [p for p in cross if tuple(sorted((p["b1"], p["b2"]))) == cell]
        if not (cell[0] in k_emp and cell[1] in k_emp):
            print(f"{cell[0]+' x '+cell[1]:<18}{len(sub):>9}   skipped (bin not fitted)")
            continue
        nat, cor = [], []
        for p in sub:
            nat.append(robust_width(p["d"] / np.sqrt(p["s1"] ** 2 + p["s2"] ** 2)))
            ka, kb = k_emp[p["b1"]], k_emp[p["b2"]]
            cor.append(robust_width(p["d"] / np.sqrt((ka * p["s1"]) ** 2 + (kb * p["s2"]) ** 2)))
        nat = [x for x in nat if x]
        cor = [x for x in cor if x]
        flag = "   <- small, do not overinterpret" if len(sub) < 20 else ""
        print(f"{cell[0]+' x '+cell[1]:<18}{len(sub):>9}{np.median(nat):>14.3f}"
              f"{np.median(cor):>17.3f}{flag}")
    print("\nCorrected width not forced to 1.")

    # ---------------- C5: leave-one-night-out ----------------
    print()
    print("=" * 74)
    print("C5  LEAVE-ONE-NIGHT-OUT (predictive CV, not an untouched holdout)")
    print("=" * 74)
    print(f"{'bin':<8}{'k spread over folds':>22}{'held-out corrected width':>28}")
    for lab in FIT_BINS:
        sub = [p for p in same if p["b1"] == lab]
        if len(sub) < 20:
            continue
        ks, ws = [], []
        for n in nights:
            tr = [p for p in sub if p["night"] != n]
            te = [p for p in sub if p["night"] == n]
            if len(tr) < 20 or len(te) < 5:
                continue
            k = float(np.median([w for w in (
                robust_width(p["d"] / np.sqrt(p["s1"] ** 2 + p["s2"] ** 2)) for p in tr) if w]))
            ks.append(k)
            ws += [w for w in (robust_width(
                p["d"] / np.sqrt((k * p["s1"]) ** 2 + (k * p["s2"]) ** 2)) for p in te) if w]
        if ks:
            print(f"{lab:<8}   {min(ks):.3f} - {max(ks):.3f}        "
                  f"median {np.median(ws):.3f}  [{np.percentile(ws,16):.3f}, {np.percentile(ws,84):.3f}]")
    print()
    print("k_empirical reported. k_adopted NOT decided. No sigma_corrected written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
