#!/usr/bin/env python
"""Stage 40 / D-100 — descriptive strata requested by the referee.

Four questions, all answerable from frozen D-094/D-095 output without touching a
spectrum:

1. Is the paired Green effect carried by one instrument combination? Most pairs
   are cross-instrument and no resolution homogenization is applied, so a
   direction that lived in a single combination would be an instrumental result
   wearing astrophysical clothes.
2. Is the Yang-ratio validity collapse concentrated in one instrument or source
   family? If it were, it would be a fit-instrument interaction rather than a
   property of the protocol under degradation.
3. How large is the Monte Carlo contribution to the *aggregate* paired mean, as
   opposed to the worst-case single-transition Bernoulli bound?
4. Do turn-off and turn-on transitions behave differently? Exploratory only; the
   experiment was never powered to answer it.

None of these is a new experiment. Strata are small and are reported as
descriptive, with counts attached so nobody reads a three-object cell as an
estimate.

One trap: the ``delta_p_10_minus_5`` column in the D-095 paired CSV is a corrupt
legacy serialization that embedded a pandas Series repr. Deltas are rebuilt from
the ``p_i`` columns, exactly as the manuscript asset builder does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

ROOT = project_root()
D094 = ROOT / "05_analysis" / "q1_production" / "d094"
D095 = ROOT / "05_analysis" / "q1_production" / "d095" / "tables"
D100 = ROOT / "05_analysis" / "derived"
OUT = D100 / "tables"

PRETTY = {
    "SDSS_LEGACY": "SDSS", "SDSSV_DR19": "SDSS-V",
    "LAMOST_DR11": "LAMOST", "DESI_EDR": "DESI",
}


def pretty_pair(pair: str) -> str:
    bright, faint = pair.split("/")
    return f"{PRETTY.get(bright, bright)}/{PRETTY.get(faint, faint)}"


def paired_deltas() -> pd.DataFrame:
    """Per-transition paired differences, rebuilt from the p_i columns."""
    frame = pd.read_csv(D095 / "green_common_support_snr_paired_transitions_d095.csv")
    frame["delta"] = frame.p_i_10.astype(float) - frame.p_i_5.astype(float)
    manifest = pd.read_csv(D100 / "reference_transition_manifest.csv")
    return frame.merge(
        manifest[["transition_id", "instrument_pair", "transition_direction"]],
        on="transition_id", how="left", validate="many_to_one",
    )


def stratify(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    for arm, group in frame.groupby("arm", sort=True):
        for value, cell in group.groupby(column, sort=True):
            rows.append({
                "arm": arm, "stratum_kind": label, "stratum": value,
                "n_transitions": len(cell),
                "mean_delta": float(cell.delta.mean()),
                "median_delta": float(cell.delta.median()),
                "min_delta": float(cell.delta.min()),
                "max_delta": float(cell.delta.max()),
            })
    return pd.DataFrame(rows)


def monte_carlo_contribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Binomial MC error on p_i, propagated through the equal-transition mean.

        Var(mean) = (1/N^2) * sum_i [ p5(1-p5)/n5 + p10(1-p10)/n10 ]

    The two rungs are summed with no covariance term. That is a fact about the
    design, not a convenience: common random numbers are keyed on the
    transition, the realization index *and* the target S/N, so a faint draw is
    shared between the two arms at a fixed rung -- which is what makes the arm
    comparison paired -- but never between rungs. Conditional on the fixed
    archival spectrum the two rungs are independent draws.

    The referee asked which of these two situations obtained, so the claim is
    checked against the frozen task manifest here rather than asserted. The arm
    contrast is the opposite case: it *does* share the faint draw, so this
    formula would not apply to it and no analogous figure is published.

    This is the aggregate analogue of the single-transition sqrt(0.25/50) worst
    case of 0.071.
    """
    shared, total = _rung_seed_overlap()
    if shared:
        raise RuntimeError(
            f"{shared}/{total} transition-realization pairs share a faint seed across "
            "rungs; the independent-binomial propagation below would understate the "
            "Monte Carlo error"
        )
    estimands = pd.read_csv(D094 / "tables" / "transition_level_estimands_d094.csv")
    green = estimands[
        estimands.criterion_id.eq("GREEN2022_FINAL") & estimands.reference_tier.eq("GOLD")
    ].copy()
    green["p"] = green.N_CL / green.N_classifiable
    rows = []
    for arm, group in frame.groupby("arm", sort=True):
        supported = set(group.transition_id)
        subset = green[green.arm.eq(arm) & green.transition_id.isin(supported)]
        wide = subset.pivot_table(
            index="transition_id", columns="snr", values=["p", "N_classifiable"]
        )
        p5, p10 = wide[("p", 5)], wide[("p", 10)]
        n5, n10 = wide[("N_classifiable", 5)], wide[("N_classifiable", 10)]
        variance = p5 * (1 - p5) / n5 + p10 * (1 - p10) / n10
        n_tr = len(wide)
        se = float(np.sqrt(variance.sum()) / n_tr)
        boot = pd.read_csv(D095 / "green_common_support_snr_summary_d095.csv").set_index("arm")
        half = float(
            (boot.loc[arm].paired_mean_ci_high - boot.loc[arm].paired_mean_ci_low) / 2.0
        )
        rows.append({
            "arm": arm, "n_transitions": n_tr,
            "mean_delta": float((p10 - p5).mean()),
            "aggregate_monte_carlo_se": se,
            "bootstrap_ci_half_width": half,
            "mc_share_of_bootstrap_width": se / half,
            "worst_case_single_transition_se": float(np.sqrt(0.25 / 50)),
            "covariance_term_included": False,
            "independence_basis": (
                "CRN keys on (transition, realization, target S/N): faint draws are "
                "shared across arms at fixed rung, never across rungs"
            ),
            "rung_seed_pairs_checked": total,
            "rung_seed_pairs_shared": shared,
        })
    return pd.DataFrame(rows)


def _rung_seed_overlap() -> tuple[int, int]:
    """How many transition-realization pairs reuse a faint seed between rungs.

    Zero is the designed answer. Returning the denominator too keeps the check
    honest: a manifest that silently lost its S/N 10 rows would otherwise pass
    by having nothing left to compare.
    """
    manifest = pd.read_csv(D094 / "raw" / "fit_task_manifest_d094.csv")
    faint = manifest[manifest.task_kind.eq("faint_shared")]
    wide = faint.pivot_table(
        index=["transition_id", "realization"], columns="target_snr",
        values="seed", aggfunc="first",
    ).dropna()
    return int((wide[5.0] == wide[10.0]).sum()), int(len(wide))


def yang_failure_strata() -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = pd.read_csv(D094 / "raw" / "classifier_outcomes_d094.csv")
    manifest = pd.read_csv(D100 / "reference_transition_manifest.csv")
    yang = outcomes[
        outcomes.criterion_id.eq("YANG2024_FINAL") & outcomes.reference_tier.eq("GOLD")
    ]
    applicable = yang[yang.applicable].merge(
        manifest[["transition_id", "instrument_pair"]], on="transition_id", how="left",
    )
    applicable = applicable.assign(invalid=~applicable.fit_valid)

    def summarise(column: str, label: str) -> pd.DataFrame:
        grouped = applicable.groupby(column).agg(
            n_transitions=("transition_id", "nunique"),
            n_applicable_rows=("invalid", "size"),
            n_invalid_rows=("invalid", "sum"),
        ).reset_index()
        grouped["invalid_fraction"] = grouped.n_invalid_rows / grouped.n_applicable_rows
        grouped.insert(0, "stratum_kind", label)
        return grouped.rename(columns={column: "stratum"})

    return summarise("instrument_pair", "instrument_pair"), summarise("source_family", "source_family")


def yang_boundary_mechanism() -> pd.DataFrame:
    """Split applicable-invalid rows by which boundary mechanism was recorded.

    The stored flag for a parameter at a bound merges scale, width and centroid
    contacts, so those three cannot be told apart without refitting. The two
    *mechanisms* can be: a fitted parameter pinned at an allowed limit, and an
    integrated profile whose width falls outside the source FWHM domain. This is
    a re-tabulation of the frozen reason strings; nothing is refitted.
    """
    outcomes = pd.read_csv(D094 / "raw" / "classifier_outcomes_d094.csv")
    yang = outcomes[outcomes.criterion_id.eq("YANG2024_FINAL") & outcomes.applicable]
    invalid = yang[~yang.classification.isin(["CL", "NON_CL"])]
    at_bound = invalid.reason.str.contains("PARAMETER_AT_BOUND", regex=False)
    width_out = invalid.reason.str.contains("WIDTH_OUTSIDE", regex=False)
    rows = [
        ("parameter_at_bound_only", int((at_bound & ~width_out).sum())),
        ("width_domain_only", int((~at_bound & width_out).sum())),
        ("both_mechanisms", int((at_bound & width_out).sum())),
        ("neither_mechanism", int((~at_bound & ~width_out).sum())),
    ]
    frame = pd.DataFrame(rows, columns=["mechanism", "n_rows"])
    assert int(frame.n_rows.sum()) == len(invalid) == 4456
    # Cross-cutting count: rows carrying at least one boundary diagnostic flag,
    # regardless of which mutually exclusive failure category they were assigned.
    frame = pd.concat(
        [frame, pd.DataFrame([("any_boundary_flag", int((at_bound | width_out).sum()))],
                             columns=["mechanism", "n_rows"])],
        ignore_index=True,
    )
    return frame


def latex_table(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines = [
        r"\begin{deluxetable*}{llrrrr}",
        rf"\tablecaption{{{caption}\label{{{label}}}}}",
        r"\tablehead{\colhead{Arm} & \colhead{Stratum} & \colhead{$N_{\rm tr}$} & "
        r"\colhead{Mean} & \colhead{Median} & \colhead{Range}}",
        r"\startdata",
    ]
    for row in frame.itertuples(index=False):
        stratum = pretty_pair(row.stratum) if "/" in str(row.stratum) else str(row.stratum).replace("_", " ")
        lines.append(
            f"{row.arm.replace('_','-')} & {stratum} & {row.n_transitions} & "
            f"{row.mean_delta:+.3f} & {row.median_delta:+.3f} & "
            f"{row.min_delta:+.2f} to {row.max_delta:+.2f} " + r"\\"
        )
    lines += [
        r"\enddata",
        r"\tablecomments{Descriptive strata of the paired common-support differences "
        r"$p_i(10)-p_i(5)$. Several strata contain only three or four transitions and are "
        r"shown to expose composition, not to estimate a stratum effect.}",
        r"\end{deluxetable*}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = paired_deltas()
    assert frame.groupby("arm").size().to_dict() == {"faint_only": 27, "matched": 25}
    assert frame.instrument_pair.notna().all() and frame.transition_direction.notna().all()

    instrument = stratify(frame, "instrument_pair", "instrument_pair")
    direction = stratify(frame, "transition_direction", "transition_direction")
    combined = pd.concat([instrument, direction], ignore_index=True)
    combined.to_csv(OUT / "paired_green_strata_d100.csv", index=False)
    # One table, two groupings: they share columns and a reader compares them.
    latex_table(
        combined, OUT / "tablea9_paired_strata.tex",
        "Paired Green-statistic S/N effect by instrument pair and transition direction",
        "tab:instrstrata",
    )

    mc = monte_carlo_contribution(frame)
    mc.to_csv(OUT / "aggregate_monte_carlo_error_d100.csv", index=False)

    mechanism = yang_boundary_mechanism()
    mechanism.to_csv(OUT / "yang_boundary_mechanism_d100.csv", index=False)

    by_instrument, by_family = yang_failure_strata()
    yang = pd.concat([by_instrument, by_family], ignore_index=True)
    yang.to_csv(OUT / "yang_failure_strata_d100.csv", index=False)
    assert int(by_instrument.n_invalid_rows.sum()) == int(by_family.n_invalid_rows.sum()) == 4075

    lines = [
        r"\begin{deluxetable*}{lllrrr}",
        r"\tablecaption{Yang-ratio/PyQSOFit measurement invalidity by instrument pair and "
        r"source family\label{tab:yangstrata}}",
        r"\tablehead{\colhead{Grouping} & \colhead{Stratum} & \colhead{$N_{\rm tr}$} & "
        r"\colhead{Applicable rows} & \colhead{Invalid} & \colhead{Fraction}}",
        r"\startdata",
    ]
    for kind, block in (("Instrument pair", by_instrument), ("Source family", by_family)):
        for row in block.sort_values("invalid_fraction").itertuples(index=False):
            stratum = pretty_pair(row.stratum) if "/" in str(row.stratum) else str(row.stratum).replace("_", " ")
            lines.append(
                f"{kind} & {stratum} & {row.n_transitions} & {row.n_applicable_rows} & "
                f"{row.n_invalid_rows} & {row.invalid_fraction:.3f} " + r"\\"
            )
    lines += [
        r"\enddata",
        r"\tablecomments{Primary reference set only, so totals are 4075 invalid of 6250 "
        r"applicable rows. Invalidity is high in every stratum, so it is not localized to "
        r"one instrument or one source family. The single DESI transition contributes one "
        r"row group and is shown for completeness.}",
        r"\end{deluxetable*}",
    ]
    (OUT / "tablea11_yang_strata.tex").write_text("\n".join(lines) + "\n")

    print("=" * 92)
    print("D-100 | referee-requested descriptive strata (frozen outputs only)")
    print("=" * 92)
    print("1. Paired Green effect by instrument pair")
    for row in instrument.itertuples(index=False):
        print(f"   {row.arm:<12}{pretty_pair(row.stratum):<16}n={row.n_transitions:<3}"
              f"mean={row.mean_delta:+.3f}  median={row.median_delta:+.3f}")
    print("   -> positive in every stratum, both arms")
    print()
    print("2. Yang invalidity by source family")
    for row in by_family.sort_values("invalid_fraction").itertuples(index=False):
        print(f"   {str(row.stratum).replace('_',' '):<24}{row.invalid_fraction:.3f}"
              f"  ({row.n_invalid_rows}/{row.n_applicable_rows})")
    print("   -> pervasive, not localized")
    print()
    print("3. Aggregate Monte Carlo contribution")
    for row in mc.itertuples(index=False):
        print(f"   {row.arm:<12}MC SE={row.aggregate_monte_carlo_se:.4f}  "
              f"bootstrap half-width={row.bootstrap_ci_half_width:.3f}  "
              f"ratio={row.mc_share_of_bootstrap_width:.2f}")
    print()
    print("4. Direction split (exploratory)")
    for row in direction.itertuples(index=False):
        print(f"   {row.arm:<12}{row.stratum:<10}n={row.n_transitions:<3}"
              f"mean={row.mean_delta:+.3f}  median={row.median_delta:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
