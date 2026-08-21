#!/usr/bin/env python
"""Stage 21 / Gate C6-B — targeted tiled extension of the uncertainty calibration.

Domain set by **science requirements**, not cleanliness. The tiling is a
deterministic 200 A grid anchored at W's upper edge, covering the 5294-6878 A
observed union of the Paper-3 Hbeta and cont5100 windows. No tile was chosen or
rejected for how its noise behaves — if part of the Hbeta domain is
instrumentally worse, Gate C must find that rather than route around it.

Arm identity is explicit. DESI B spans ~3600-5800 A and R ~5760-7620 A, so the
extension crosses the transition. The W result is in B and is not assumed to
transfer. The overlap tile is evaluated **separately per arm**, never stitched
before its uncertainty behaviour is known.

Local S/N per tile is the calibration coordinate. ``S/N_pix(W)`` was frozen for
C0 acquisition and is *not* the right coordinate for an R-arm tile.

Products: an empirical ``k(tile, S/N)`` lookup table. No spline, no polynomial,
no smooth S/N law, no B/R/Z surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

# FROZEN before any new z-width was inspected: 200 A, anchored at W's upper edge.
TILES = [(5200.0 + 200 * i, 5400.0 + 200 * i) for i in range(9)]
TILE_NAMES = [f"T{i+1}" for i in range(9)]
LABELS = ["00-10", "10-20", "20-30", "30-50", "50+"]
EDGES = [0, 10, 20, 30, 50, np.inf]
MIN_PAIRS = 20
MIN_NIGHTS = 3
N_BOOT = 1000
SEED = 314159


def robust_width(z: np.ndarray) -> float | None:
    z = z[np.isfinite(z)]
    if z.size < 30:
        return None
    lo, hi = np.percentile(z, [16, 84])
    return float((hi - lo) / 2.0)


def label_of(s: float) -> str:
    return LABELS[int(np.digitize(s, EDGES[1:-1]))]


def main() -> int:
    from astropy.io import fits

    root = project_root()
    final = root / "02_catalogs" / "final"
    eligible = set(pd.read_csv(final / "c0_eligible_targets.csv",
                               dtype={"targetid": str}).targetid)
    manifest = pd.read_csv(final / "c0_acquisition_manifest.csv")

    rows: list[dict] = []
    arm_cov: dict[tuple[str, str], tuple[float, float]] = {}

    for entry in manifest.itertuples():
        path = root / entry.local_path
        if not path.exists():
            continue
        with fits.open(path, memmap=False) as hdul:
            fm = hdul["FIBERMAP"].data
            tid = np.asarray(fm["TARGETID"]).astype(str)
            srt = np.argsort(tid, kind="stable")
            arms = {}
            for arm in ("B", "R"):
                if f"{arm}_WAVELENGTH" not in hdul:
                    continue
                arms[arm] = (
                    np.asarray(hdul[f"{arm}_WAVELENGTH"].data, float),
                    np.asarray(hdul[f"{arm}_FLUX"].data, float),
                    np.asarray(hdul[f"{arm}_IVAR"].data, float),
                    np.asarray(hdul[f"{arm}_MASK"].data),
                )
            for arm, (lam, *_ ) in arms.items():
                for name, (lo, hi) in zip(TILE_NAMES, TILES, strict=True):
                    n = int(((lam >= lo) & (lam < hi)).sum())
                    if n:
                        arm_cov[(name, arm)] = (float(lam.min()), float(lam.max()))

            for t in np.unique(tid):
                if t not in eligible:
                    continue
                a = np.searchsorted(tid[srt], t, "left")
                b = np.searchsorted(tid[srt], t, "right")
                if b - a != 2:
                    continue
                i, j = srt[a], srt[b - 1]
                for arm, (lam, flux, ivar, mask) in arms.items():
                    ok = ((ivar[i] > 0) & (mask[i] == 0) & (ivar[j] > 0) & (mask[j] == 0)
                          & np.isfinite(flux[i]) & np.isfinite(flux[j]))
                    for name, (lo, hi) in zip(TILE_NAMES, TILES, strict=True):
                        g = ok & (lam >= lo) & (lam < hi)
                        if g.sum() < 30:
                            continue
                        f1, f2 = flux[i][g], flux[j][g]
                        s1 = 1 / np.sqrt(ivar[i][g])
                        s2 = 1 / np.sqrt(ivar[j][g])
                        # LOCAL tile S/N, not W-S/N.
                        n1 = float(np.median(np.abs(f1) / s1))
                        n2 = float(np.median(np.abs(f2) / s2))
                        z = (f1 - f2) / np.sqrt(s1**2 + s2**2)
                        ref = 0.5 * (f1 + f2)
                        grad = np.gradient(ref, lam[g])
                        den = float(np.sum(grad * grad))
                        rows.append({
                            "tile": name, "arm": arm, "targetid": t,
                            "night": int(entry.night),
                            "b1": label_of(n1), "b2": label_of(n2),
                            "same_bin": label_of(n1) == label_of(n2),
                            "width": robust_width(z), "n_pix": int(g.sum()),
                            "delta_lam": (float(np.sum((f1 - f2) * grad) / den)
                                          if den > 0 else np.nan),
                        })

    frame = pd.DataFrame(rows)
    frame.to_csv(final / "c6b_tile_residuals.csv", index=False)

    print("=" * 92)
    print("1. FROZEN TILE / ARM MAP")
    print("=" * 92)
    print(f"{'tile':<6}{'range':>14}{'arms with pixels':>20}   science use")
    sci = {"T1": "-", "T2": "Hb(J082942)", "T3": "Hb(J082942)+cont",
           "T4": "Hb x5", "T5": "Hb x6 + cont x2", "T6": "Hb x5 + cont x4",
           "T7": "Hb x2 + cont x2", "T8": "Hb(J012256)+cont", "T9": "-"}
    for name, (lo, hi) in zip(TILE_NAMES, TILES, strict=True):
        a = sorted({arm for (t_, arm) in arm_cov if t_ == name})
        print(f"{name:<6}{f'{lo:.0f}-{hi:.0f}':>14}{','.join(a) if a else 'none':>20}   {sci[name]}")

    print()
    print("=" * 92)
    print("2. OCCUPANCY per tile x arm x local S/N  (same-bin pairs / nights)")
    print("=" * 92)
    print(f"{'tile':<6}{'arm':<4}" + "".join(f"{lab:>16}" for lab in LABELS))
    support: dict[tuple[str, str, str], bool] = {}
    for name in TILE_NAMES:
        for arm in ("B", "R"):
            sub = frame[(frame.tile == name) & (frame.arm == arm) & frame.same_bin]
            if sub.empty:
                continue
            cells = []
            for lab in LABELS:
                c = sub[sub.b1 == lab]
                ok = len(c) >= MIN_PAIRS and c.night.nunique() >= MIN_NIGHTS
                support[(name, arm, lab)] = ok
                cells.append(f"{len(c)}/{c.night.nunique()}" + ("" if ok else "*"))
            print(f"{name:<6}{arm:<4}" + "".join(f"{c:>16}" for c in cells))
    print("  * = EMPIRICAL_CELL_UNSUPPORTED (<20 pairs or <3 nights)")

    print()
    print("=" * 92)
    print("3. k_empirical(tile, arm, local S/N) — supported cells only")
    print("=" * 92)
    rng = np.random.default_rng(SEED)
    out = []
    print(f"{'tile':<6}{'arm':<4}" + "".join(f"{lab:>17}" for lab in LABELS))
    for name in TILE_NAMES:
        for arm in ("B", "R"):
            sub = frame[(frame.tile == name) & (frame.arm == arm) & frame.same_bin]
            if sub.empty:
                continue
            cells = []
            for lab in LABELS:
                if not support.get((name, arm, lab)):
                    cells.append("—")
                    continue
                c = sub[sub.b1 == lab].dropna(subset=["width"])
                k = float(np.median(c.width))
                nights = c.night.unique()
                bs = [float(np.median(pd.concat(
                    [c[c.night == n].width for n in rng.choice(nights, nights.size, True)])))
                    for _ in range(N_BOOT)]
                lo_, hi_ = np.percentile(bs, [2.5, 97.5])
                cells.append(f"{k:.3f}[{lo_:.2f},{hi_:.2f}]")
                out.append({"tile": name, "arm": arm, "sn_bin": lab, "k_empirical": k,
                            "ci_lo": lo_, "ci_hi": hi_, "n_pairs": len(c),
                            "n_nights": c.night.nunique()})
            print(f"{name:<6}{arm:<4}" + "".join(f"{c:>17}" for c in cells))
    pd.DataFrame(out).to_csv(final / "c6b_k_empirical.csv", index=False)

    print()
    print("=" * 92)
    print("5. REGISTRATION / GRADIENT FALSIFICATION per tile")
    print("=" * 92)
    print(f"{'tile':<6}{'arm':<4}{'median delta_lam (A)':>22}{'16-84%':>26}")
    for name in TILE_NAMES:
        for arm in ("B", "R"):
            d = frame[(frame.tile == name) & (frame.arm == arm)].delta_lam.dropna()
            if len(d) < 20:
                continue
            print(f"{name:<6}{arm:<4}{np.median(d):>22.5f}"
                  f"{f'[{np.percentile(d,16):+.4f}, {np.percentile(d,84):+.4f}]':>26}")

    print()
    print("No smooth surface fitted. k_empirical unclamped. k_adopted not constructed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
