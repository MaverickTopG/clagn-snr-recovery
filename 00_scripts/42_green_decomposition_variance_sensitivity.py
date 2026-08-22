#!/usr/bin/env python
"""Stage 42 -- how much unmodelled variance would overturn the Green S/N result?

The Green denominator carries propagated spectral-pixel variance only. The
continuum/host/Fe II/Balmer-continuum model is fitted to each noisy realization,
and no uncertainty in that fitted model enters V^B_j or V^F_j. A referee can
reasonably ask whether that omission, which grows as the spectrum degrades,
could manufacture the reported S/N contrast.

The decomposition covariance was never stored, so the missing term cannot be
computed. What *can* be done from the frozen products is a breakdown analysis:
the stored per-pixel line spectra and variances reproduce G exactly, so the
combined variance can be inflated by a factor (1+f) and every downstream
quantity recomputed. The question then becomes how large f would have to be
before the conclusion changes.

Two sweeps are run.

    symmetric   the same f at both rungs. This asks whether the sign reverses
                under a uniformly understated variance. It does not, over the
                range tested: the faint-only contrast stays positive through
                f=24 and reaches zero at f=99, where recovery has collapsed at
                both rungs. At f=1 it is +0.223.

    asymmetric  f applied at S/N 5 only, or more at S/N 5 than at S/N 10. This
                is the physically motivated direction: the decomposition is
                least constrained where the spectrum is noisiest, so any real
                omission is larger at the lower rung.

What this bounds and what it does not: f scales the combined variance
multiplicatively and uniformly across pixels within an epoch. Real decomposition
error is correlated across wavelength and may carry structure -- a continuum
tilt, an Fe II mismatch -- that a scalar inflation does not reproduce. This
analysis therefore bounds one perturbation family, not every possible one, and
it does not turn G into a calibrated significance.

Nothing frozen is modified. No spectrum is refitted. At f=0 the script must
reproduce the frozen production values exactly or it aborts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.criteria.green_pixel import (  # noqa: E402
    GreenEpochSpectrum,
    _rebin_two_angstrom,
    _running_median_16,
)

ROOT = project_root()
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
OUT = ROOT / "05_analysis" / "q1_production" / "d094" / "tables"

GREEN_THRESHOLD = 3.0
# Frozen production values this script must reproduce at f = 0.
FROZEN = {"faint_only": (27, 0.2896296296296296), "matched": (25, 0.2592)}

SYMMETRIC = [0.0, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 9.00, 24.00, 99.00]
ASYMMETRIC = [(0.25, 0.0), (0.50, 0.0), (1.00, 0.0), (2.00, 0.0), (4.00, 0.0),
              (0.50, 0.25), (1.00, 0.50), (2.00, 1.00)]


def _read_outcomes() -> pd.DataFrame:
    """The repository ships CSV; the public checkout ships parquet."""
    csv = RAW / "classifier_outcomes_d094.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")


def _read_pixels() -> pd.DataFrame:
    """Prefer the full product; fall back to the verified window extract.

    `green_pixel_window_d094.parquet` holds the same pixels the statistic can
    reach, for the same epochs, and is checked at build time to rebin
    bitwise-identically (Stage 43). It exists because the full product is
    675 MB and cannot be redistributed with the code.
    """
    full = RAW / "spectrum_fit_results_d094.parquet"
    if full.exists():
        return pd.read_parquet(full).set_index("task_id")
    window = RAW / "green_pixel_window_d094.parquet"
    if window.exists():
        print(f"using window extract {window.name} ({window.stat().st_size / 1e6:.1f} MB)")
        return pd.read_parquet(window).set_index("task_id")
    raise SystemExit(
        "Neither spectrum_fit_results_d094.parquet nor green_pixel_window_d094.parquet "
        "is present; Stage 42 cannot run."
    )


def _load() -> tuple[pd.DataFrame, dict[str, tuple | None]]:
    fits = _read_pixels()
    outcomes = _read_outcomes()
    green = outcomes[
        (outcomes.criterion_id == "GREEN2022_FINAL")
        & outcomes.applicable
        & (outcomes.reference_tier == "GOLD")
    ].copy()

    def rebin(task_id: str):
        row = fits.loc[task_id]
        return _rebin_two_angstrom(
            GreenEpochSpectrum(
                wavelength_rest=np.asarray(row.green_wavelength_rest, dtype=float),
                line_flux=np.asarray(row.green_line_flux, dtype=float),
                variance=np.asarray(row.green_variance, dtype=float),
                variance_provenance=np.asarray(row.green_variance_provenance, dtype=str),
                preprocessing_valid=bool(row.green_preprocessing_valid),
                invalid_reason=str(row.green_invalid_reason or ""),
                scale_factor=float(row.green_scale_factor),
                scale_provenance=str(row.green_scale_provenance),
            )
        )

    epochs = sorted(set(green.bright_task_id) | set(green.faint_task_id))
    return green, {task: rebin(task) for task in epochs}


def paired_means(
    green: pd.DataFrame, cache: dict, f_at_5: float, f_at_10: float
) -> dict[str, tuple[int, float, float]]:
    """Equal-transition mean of p_i(10)-p_i(5) on paired common support."""
    rows = []
    for record in green.itertuples():
        bright, faint = cache[record.bright_task_id], cache[record.faint_task_id]
        if bright is None or faint is None:
            continue
        inflation = f_at_5 if record.snr == 5 else f_at_10
        denominator = np.sqrt((bright[2] + faint[2]) * (1.0 + inflation))
        smoothed = _running_median_16((bright[1] - faint[1]) / denominator)
        statistic = float((smoothed - smoothed[0]).max())
        rows.append((record.transition_id, record.arm, record.snr, statistic >= GREEN_THRESHOLD))

    frame = pd.DataFrame(rows, columns=["transition", "arm", "snr", "cl"])
    result = {}
    for arm in ("faint_only", "matched"):
        recovery = frame[frame.arm == arm].groupby(["transition", "snr"]).cl.mean().unstack()
        delta = (recovery[10] - recovery[5]).dropna()
        result[arm] = (len(delta), float(delta.mean()), float(delta.median()))
    return result


def main() -> int:
    green, cache = _load()
    print(f"GOLD applicable Green rows: {len(green)}   distinct epochs: {len(cache)}")

    baseline = paired_means(green, cache, 0.0, 0.0)
    for arm, (n_expected, mean_expected) in FROZEN.items():
        n_got, mean_got, _ = baseline[arm]
        if n_got != n_expected or abs(mean_got - mean_expected) > 1e-12:
            print(f"ABORT: f=0 does not reproduce frozen {arm}: "
                  f"got N={n_got} mean={mean_got!r}, expected N={n_expected} mean={mean_expected!r}")
            return 1
    print("gate: f=0 reproduces the frozen paired means exactly")

    records = []
    for f in SYMMETRIC:
        res = paired_means(green, cache, f, f)
        for arm, (n, mean, median) in res.items():
            records.append(dict(sweep="symmetric", f_at_snr5=f, f_at_snr10=f, arm=arm,
                                N_paired=n, mean_delta_p=mean, median_delta_p=median))
    for f5, f10 in ASYMMETRIC:
        res = paired_means(green, cache, f5, f10)
        for arm, (n, mean, median) in res.items():
            records.append(dict(sweep="asymmetric", f_at_snr5=f5, f_at_snr10=f10, arm=arm,
                                N_paired=n, mean_delta_p=mean, median_delta_p=median))

    table = pd.DataFrame(records)
    destination = OUT / "green_decomposition_variance_sensitivity_d094.csv"
    table.to_csv(destination, index=False)

    print(f"\n{'sweep':<11}{'f@5':>7}{'f@10':>7}{'arm':>12}{'N':>5}{'mean dp':>10}")
    for r in table.itertuples():
        print(f"{r.sweep:<11}{r.f_at_snr5:>7.2f}{r.f_at_snr10:>7.2f}{r.arm:>12}"
              f"{r.N_paired:>5d}{r.mean_delta_p:>+10.4f}")

    sym = table[table.sweep == "symmetric"]
    print(f"\nsymmetric sweep: minimum mean delta_p over all f and both arms = "
          f"{sym.mean_delta_p.min():+.4f}  (no sign reversal: {bool((sym.mean_delta_p >= 0).all())})")
    asym = table[(table.sweep == "asymmetric") & (table.f_at_snr10 == 0.0)]
    for arm in ("faint_only", "matched"):
        a = asym[asym.arm == arm]
        base = baseline[arm][1]
        print(f"{arm:>12}: S/N-5-only inflation moves {base:+.4f} -> "
              f"{a.mean_delta_p.min():+.4f} .. {a.mean_delta_p.max():+.4f}")
    print(f"\nwrote {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
