"""Per-rung attrition accounting: eligible, fit, classifiable.

Three counts, reported separately at every rung and in every arm, because they
diverge and the divergence is itself a result:

- ``n_eligible`` — pairs whose native S/N allows this rung at all.
- ``n_fit_succeeded`` — of those, pairs where both epochs decomposed.
- ``n_classifiable`` — of those, pairs a given criterion could actually decide.

Reporting only the last would invite the obvious question of where the rest
went. At S/N 5 a rung might have 100 eligible, 92 fit, 78 classifiable; the 8
fit failures and 14 undecidable pairs are data, not noise. Preregistration §8
already forbids collapsing ``unclassifiable`` into ``non-CL``; this module is
where that commitment becomes countable.

The counts are nested by construction — every classifiable pair was fit, and
every fit pair was eligible — and the invariant is enforced rather than assumed,
since a violation means a pair was counted at a rung it never reached.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class RungCounts:
    """Attrition at one (arm, rung), optionally for one criterion."""

    arm: str
    rung: float
    n_eligible: int
    n_fit_succeeded: int
    n_classifiable: int
    criterion: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.n_classifiable <= self.n_fit_succeeded <= self.n_eligible:
            raise ValueError(
                f"counts must nest as 0 <= classifiable <= fit <= eligible; got "
                f"eligible={self.n_eligible}, fit={self.n_fit_succeeded}, "
                f"classifiable={self.n_classifiable} at {self.arm}/{self.rung}"
            )

    @property
    def n_fit_failed(self) -> int:
        return self.n_eligible - self.n_fit_succeeded

    @property
    def n_unclassifiable(self) -> int:
        return self.n_fit_succeeded - self.n_classifiable

    @property
    def fit_success_rate(self) -> float | None:
        """None rather than a sentinel when no pair was eligible."""
        return self.n_fit_succeeded / self.n_eligible if self.n_eligible else None

    @property
    def classifiable_rate(self) -> float | None:
        """Fraction of *eligible* pairs that reached a decision.

        Deliberately against ``n_eligible``, not ``n_fit_succeeded``: dividing by
        the survivors would hide fit failures, which is the survivorship bias
        preregistration §8 rules out.
        """
        return self.n_classifiable / self.n_eligible if self.n_eligible else None

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row.update(
            n_fit_failed=self.n_fit_failed,
            n_unclassifiable=self.n_unclassifiable,
            fit_success_rate=self.fit_success_rate,
            classifiable_rate=self.classifiable_rate,
        )
        return row


def accounting_table(counts: Iterable[RungCounts]) -> pd.DataFrame:
    """Tidy frame, one row per (arm, rung, criterion)."""
    rows = [c.to_row() for c in counts]
    if not rows:
        return pd.DataFrame(
            columns=[
                "arm", "rung", "criterion", "n_eligible", "n_fit_succeeded",
                "n_classifiable", "n_fit_failed", "n_unclassifiable",
                "fit_success_rate", "classifiable_rate",
            ]
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["arm", "criterion", "rung"], na_position="first").reset_index(
        drop=True
    )


def summarise_attrition(
    results: pd.DataFrame,
    *,
    arm_column: str = "arm",
    rung_column: str = "rung",
    object_column: str = "object_id",
    fit_ok_column: str = "fit_succeeded",
    label_column: str = "label",
    criterion_column: str | None = "criterion",
    unclassifiable_label: str = "unclassifiable",
    role_column: str | None = None,
    final_classifier_role: str = "FINAL_CLASSIFIER",
) -> pd.DataFrame:
    """Build the accounting table from per-pair outcome records.

    ``results`` carries one row per (object, arm, rung, criterion, realization).
    Objects are counted **distinctly**: Monte Carlo realizations are not
    independent AGN, so counting rows would inflate every N by the realization
    count and misrepresent the sample size.
    """
    if role_column is not None:
        if role_column not in results.columns:
            raise KeyError(f"role column {role_column!r} is absent")
        # Search/confirmation protocols have their own accounting.  Including
        # them here would turn a candidate-search pass into a classifier datum.
        results = results.loc[results[role_column] == final_classifier_role].copy()

    group_columns = [arm_column, rung_column]
    if criterion_column and criterion_column in results.columns:
        group_columns.append(criterion_column)

    counts: list[RungCounts] = []
    for keys, group in results.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        mapping: dict[str, object] = dict(zip(group_columns, key_values, strict=True))

        eligible = group[object_column].nunique()
        fit = group.loc[group[fit_ok_column], object_column].nunique()
        decided = group.loc[
            group[fit_ok_column] & (group[label_column] != unclassifiable_label),
            object_column,
        ].nunique()

        counts.append(
            RungCounts(
                arm=str(mapping[arm_column]),
                rung=float(mapping[rung_column]),  # type: ignore[arg-type]
                n_eligible=int(eligible),
                n_fit_succeeded=int(fit),
                n_classifiable=int(decided),
                criterion=(
                    str(mapping[criterion_column])
                    if criterion_column and criterion_column in mapping
                    else None
                ),
            )
        )
    return accounting_table(counts)


def format_attrition(frame: pd.DataFrame) -> str:
    """Human-readable block for reports, showing where every object went."""
    if frame.empty:
        return "(no records)"
    lines = [
        f"{'arm':<12}{'rung':>6}{'criterion':>22}{'elig':>6}{'fit':>6}{'class':>7}"
        f"{'failed':>8}{'unclass':>9}"
    ]
    for row in frame.itertuples():
        criterion = row.criterion if isinstance(row.criterion, str) else "-"
        lines.append(
            f"{row.arm:<12}{row.rung:>6.0f}{criterion:>22}{row.n_eligible:>6}"
            f"{row.n_fit_succeeded:>6}{row.n_classifiable:>7}"
            f"{row.n_fit_failed:>8}{row.n_unclassifiable:>9}"
        )
    return "\n".join(lines)


__all__ = ["RungCounts", "accounting_table", "format_attrition", "summarise_attrition"]
