#!/usr/bin/env python
"""Stage 20 / Gate C2.5 — two falsification tests before any k is adopted.

**A. Endogenous-selection coupling.** S/N bin assignment and z both currently use
the same W pixels, so the variable defining the bin is mathematically coupled to
the residual it selects. Cross-fit *within* W: assign S/N from one disjoint half,
measure z-width on the other, then swap. W is unchanged and the frozen
``S/N_pix(W)`` remains the scientific axis — this only asks whether the trend
survives when the two quantities stop sharing pixels.

**B. Stellar-line / wavelength-registration contamination.** These calibrators are
stars and W is a low-*spikiness* window, not a featureless one. A sub-pixel
exposure-to-exposure wavelength shift produces residuals proportional to dF/dlam
that grow with S/N and can mimic underestimated IVAR. Diagnostic only: no
spectrum is shifted, aligned or modified.

Nothing is fitted or adopted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

W = (5000.0, 5200.0)
MID = 5100.0                                # disjoint halves of W
LABELS = ["00-10", "10-20", "20-30", "30-50", "50+"]
EDGES = [0, 10, 20, 30, 50, np.inf]
FIT_BINS = LABELS[:4]


def robust_width(z: np.ndarray) -> float | None:
    z = z[np.isfinite(z)]
    if z.size < 30:
        return None
    lo, hi = np.percentile(z, [16, 84])
    return float((hi - lo) / 2.0)


def label_of(snr: float) -> str:
    return LABELS[int(np.digitize(snr, EDGES[1:-1]))]


def main() -> int:
    from astropy.io import fits

    root = project_root()
    final = root / "02_catalogs" / "final"
    eligible = set(pd.read_csv(final / "c0_eligible_targets.csv",
                               dtype={"targetid": str}).targetid)
    manifest = pd.read_csv(final / "c0_acquisition_manifest.csv")

    rows: list[dict] = []
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
            lower, upper = in_w & (lam < MID), in_w & (lam >= MID)
            srt = np.argsort(tid, kind="stable")

            for t in np.unique(tid):
                if t not in eligible:
                    continue
                lo_i = np.searchsorted(tid[srt], t, "left")
                hi_i = np.searchsorted(tid[srt], t, "right")
                if hi_i - lo_i != 2:
                    continue
                i, j = srt[lo_i], srt[hi_i - 1]
                ok = ((ivar[i] > 0) & (mask[i] == 0) & (ivar[j] > 0) & (mask[j] == 0)
                      & np.isfinite(flux[i]) & np.isfinite(flux[j]))

                def part(sel, i=i, j=j, ok=ok, flux=flux, ivar=ivar, lam=lam):
                    g = sel & ok
                    if g.sum() < 30:
                        return None
                    f1, f2 = flux[i][g], flux[j][g]
                    s1, s2 = 1 / np.sqrt(ivar[i][g]), 1 / np.sqrt(ivar[j][g])
                    return {
                        "lam": lam[g], "f1": f1, "f2": f2, "s1": s1, "s2": s2,
                        "snr1": float(np.median(np.abs(f1) / s1)),
                        "snr2": float(np.median(np.abs(f2) / s2)),
                        "z": (f1 - f2) / np.sqrt(s1**2 + s2**2),
                    }

                a, b = part(lower), part(upper)
                if a is None or b is None:
                    continue

                # ---- B: gradient / wavelength-shift diagnostic over full W ----
                g = in_w & ok
                f1, f2 = flux[i][g], flux[j][g]
                s1, s2 = 1 / np.sqrt(ivar[i][g]), 1 / np.sqrt(ivar[j][g])
                ref = 0.5 * (f1 + f2)
                grad = np.gradient(ref, lam[g])
                diff = f1 - f2
                denom = float(np.sum(grad * grad))
                delta_lam = float(np.sum(diff * grad) / denom) if denom > 0 else np.nan
                corr = (float(np.corrcoef(diff, grad)[0, 1])
                        if np.std(grad) > 0 and np.std(diff) > 0 else np.nan)
                z_full = diff / np.sqrt(s1**2 + s2**2)
                absg = np.abs(grad)
                cut = np.nanpercentile(absg, 50)
                low_g, high_g = absg <= cut, absg > cut

                snr_full = float(np.median(np.abs(f1) / s1))
                rows.append({
                    "targetid": t, "night": int(entry.night),
                    "bin_full": label_of(snr_full),
                    # cross-fit: assign from one half, measure on the other
                    "bin_from_lower": label_of(a["snr1"]),
                    "width_upper": robust_width(b["z"]),
                    "bin_from_upper": label_of(b["snr1"]),
                    "width_lower": robust_width(a["z"]),
                    # gradient diagnostics
                    "delta_lam": delta_lam, "corr_diff_grad": corr,
                    "width_lowgrad": robust_width(z_full[low_g]),
                    "width_highgrad": robust_width(z_full[high_g]),
                    "width_sub_lower": robust_width(a["z"]),
                    "width_sub_upper": robust_width(b["z"]),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(final / "c2_5_falsification.csv", index=False)
    print(f"pairs: {len(frame)}   nights: {frame.night.nunique()}\n")

    print("=" * 74)
    print("A. CROSS-FITTED S/N ASSIGNMENT (bin and residual use disjoint pixels)")
    print("=" * 74)
    print(f"{'bin':<8}{'same-pixel k':>14}{'assign lower/measure upper':>29}"
          f"{'assign upper/measure lower':>29}")
    for lab in FIT_BINS:
        same = frame[frame.bin_full == lab]
        xa = frame[frame.bin_from_lower == lab]
        xb = frame[frame.bin_from_upper == lab]

        def med(s, col):
            v = s[col].dropna()
            return float(np.median(v)) if len(v) >= 20 else np.nan

        k0 = med(same, "width_sub_lower")
        k0b = med(same, "width_sub_upper")
        base = np.nanmean([k0, k0b])
        print(f"{lab:<8}{base:>14.3f}{med(xa,'width_upper'):>29.3f}"
              f"{med(xb,'width_lower'):>29.3f}   (n={len(xa)}/{len(xb)})")

    print()
    print("=" * 74)
    print("B. WAVELENGTH-SHIFT / STELLAR-GRADIENT DIAGNOSTIC")
    print("=" * 74)
    d = frame.delta_lam.dropna()
    print(f"inferred delta_lambda: median {np.median(d):+.5f} A   "
          f"16-84% [{np.percentile(d,16):+.4f}, {np.percentile(d,84):+.4f}]")
    c = frame.corr_diff_grad.dropna()
    print(f"corr(f1-f2, dF/dlam): median {np.median(c):+.4f}   "
          f"16-84% [{np.percentile(c,16):+.4f}, {np.percentile(c,84):+.4f}]")
    print()
    print(f"{'bin':<8}{'width low-grad':>16}{'width high-grad':>17}"
          f"{'delta_lam med':>15}{'sub-W lower':>13}{'sub-W upper':>13}")
    for lab in FIT_BINS:
        s = frame[frame.bin_full == lab]
        if len(s) < 20:
            continue
        print(f"{lab:<8}{np.median(s.width_lowgrad.dropna()):>16.3f}"
              f"{np.median(s.width_highgrad.dropna()):>17.3f}"
              f"{np.median(s.delta_lam.dropna()):>15.5f}"
              f"{np.median(s.width_sub_lower.dropna()):>13.3f}"
              f"{np.median(s.width_sub_upper.dropna()):>13.3f}")
    print()
    print("Diagnostic only. No spectrum shifted, aligned or modified. Nothing adopted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
