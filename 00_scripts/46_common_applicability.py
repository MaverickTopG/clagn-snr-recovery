#!/usr/bin/env python
"""Stage 46 -- outcome geometry with protocol applicability held fixed.

Table 3 reports each protocol over the transitions it can be posed for, and
those sets are not identical: a few transitions are applicable to one protocol
and not the other. A referee can reasonably ask whether the gap between the two
operational yields is partly a difference in membership rather than in
behaviour.

This stage removes that possibility. For each arm and rung it takes the
transitions applicable under BOTH protocols, restricts BOTH to exactly that set,
and recomputes the three-state decomposition, the conditional recovery R and the
operational yield Y with the transition as the unit.

It is a re-tabulation of frozen classification records. Nothing is refitted, no
frozen output is modified, and the comparison is between these two operational
implementations rather than between the published methods they follow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

ROOT = project_root()
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
OUT = ROOT / "05_analysis" / "q1_production" / "d094" / "tables"

CLASSIFIED = ["CL", "NON_CL"]
PROTOCOLS = [("GREEN2022_FINAL", "Green-statistic"), ("YANG2024_FINAL", "Yang-ratio")]
CONDITIONS = [("faint_only", "Faint-only"), ("matched", "Matched")]


def _read_outcomes() -> pd.DataFrame:
    csv = RAW / "classifier_outcomes_d094.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")


def common_applicability(outcomes: pd.DataFrame, gold: set[str]) -> pd.DataFrame:
    rows = []
    for arm, arm_label in CONDITIONS:
        for snr in (5, 10):
            applicable = {}
            for criterion, label in PROTOCOLS:
                applicable[label] = outcomes[
                    outcomes.criterion_id.eq(criterion)
                    & outcomes.transition_id.isin(gold)
                    & outcomes.arm.eq(arm)
                    & outcomes.snr.eq(snr)
                    & outcomes.applicable
                ]
            shared = sorted(
                set(applicable["Green-statistic"].transition_id)
                & set(applicable["Yang-ratio"].transition_id)
            )
            for label, cell in applicable.items():
                only = len(set(cell.transition_id) - set(shared))
                cell = cell[cell.transition_id.isin(shared)].copy()
                cell["is_cl"] = cell.classification.eq("CL")
                cell["is_classified"] = cell.classification.isin(CLASSIFIED)
                n_app = len(cell)
                n_cl = int(cell.is_cl.sum())
                n_noncl = int((cell.is_classified & ~cell.is_cl).sum())
                by_tr = cell.groupby("transition_id")
                rows.append({
                    "arm": arm_label, "snr": snr, "protocol": label,
                    "N_tr_common": len(shared), "N_tr_dropped": only,
                    "n_applicable": n_app, "n_cl": n_cl, "n_noncl": n_noncl,
                    "n_invalid": n_app - n_cl - n_noncl,
                    "R_conditional": by_tr.apply(
                        lambda g: g.loc[g.is_classified, "is_cl"].mean(), include_groups=False
                    ).mean(),
                    "Y_operational": by_tr.is_cl.mean().mean(),
                    "invalid_fraction": by_tr.apply(
                        lambda g: 1.0 - g.is_classified.mean(), include_groups=False
                    ).mean(),
                })
    frame = pd.DataFrame(rows)
    assert (frame.n_cl + frame.n_noncl + frame.n_invalid).equals(frame.n_applicable)
    # Both protocols must now sit on identical membership and identical row counts.
    for _key, group in frame.groupby(["arm", "snr"]):
        assert group.N_tr_common.nunique() == 1
        assert group.n_applicable.nunique() == 1
    green = frame[frame.protocol.eq("Green-statistic")]
    assert green.n_invalid.eq(0).all(), "Green invalidity must remain exactly zero"
    return frame


def main() -> int:
    outcomes = _read_outcomes()
    gold = set(outcomes.loc[outcomes.reference_tier.eq("GOLD"), "transition_id"])
    frame = common_applicability(outcomes, gold)
    frame.to_csv(OUT / "common_applicability_d094.csv", index=False)

    print("Outcome geometry on transitions applicable under both protocols")
    print(frame[["arm", "snr", "protocol", "N_tr_common", "N_tr_dropped", "n_applicable",
                 "R_conditional", "Y_operational", "invalid_fraction"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    yang = frame[frame.protocol.eq("Yang-ratio")]
    print(f"\nGreen invalid fraction: {frame[frame.protocol.eq('Green-statistic')].invalid_fraction.max():.3f} "
          f"in every condition")
    print(f"Yang invalid fraction : {yang.invalid_fraction.min():.3f}--{yang.invalid_fraction.max():.3f}")
    print(f"Yang R/Y ratio        : {(yang.R_conditional / yang.Y_operational).min():.2f}"
          f"--{(yang.R_conditional / yang.Y_operational).max():.2f}")
    print(f"\nwrote {(OUT / 'common_applicability_d094.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
