"""Object-level Q2 estimands and denominator accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Q2Counts:
    """Nested distinct-transition counts for one (f, H, criterion) cell."""

    f_added_star: float
    host_shape_id: str
    criterion: str
    n_eligible: int
    n_fit_completed: int
    n_fit_valid: int
    n_classifiable: int
    n_cl: int
    n_noncl: int
    n_unclassifiable: int

    def __post_init__(self) -> None:
        nested = (
            0
            <= self.n_classifiable
            <= self.n_fit_valid
            <= self.n_fit_completed
            <= self.n_eligible
        )
        if not nested:
            raise ValueError("Q2 counts must nest: classifiable <= valid <= completed <= eligible")
        if self.n_cl + self.n_noncl != self.n_classifiable:
            raise ValueError("N_CL + N_nonCL must equal N_classifiable")
        if self.n_classifiable + self.n_unclassifiable != self.n_eligible:
            raise ValueError("N_classifiable + N_unclassifiable must equal N_eligible")

    @property
    def empirical_recovery_fraction(self) -> float | None:
        """R(f,H,k) = N_CL / N_classifiable; undefined with no decisions."""
        return self.n_cl / self.n_classifiable if self.n_classifiable else None

    @property
    def unclassifiable_fraction(self) -> float | None:
        """U(f,H,k) = N_unclassifiable / N_eligible."""
        return self.n_unclassifiable / self.n_eligible if self.n_eligible else None

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["R_empirical_recovery"] = self.empirical_recovery_fraction
        row["U_unclassifiable"] = self.unclassifiable_fraction
        return row


def summarise_q2_outcomes(results: pd.DataFrame) -> pd.DataFrame:
    """Count transitions, never shapes/realizations/spectra as independent AGN.

    Input rows may repeat an object because of fits or realizations. Within one
    condition and criterion, each object must nevertheless have a consistent
    terminal status; duplicated rows cannot increase any count.
    """
    required = {
        "object_id",
        "f_added_star",
        "host_shape_id",
        "criterion",
        "fit_completed",
        "fit_valid",
        "label",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Q2 outcome table missing columns: {sorted(missing)}")

    keys = ["f_added_star", "host_shape_id", "criterion"]
    rows: list[dict[str, object]] = []
    for key, group in results.groupby(keys, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        per_object = group.groupby("object_id", as_index=False).agg(
            fit_completed=("fit_completed", "max"),
            fit_valid=("fit_valid", "max"),
            label=("label", lambda values: _one_terminal_label(set(values))),
        )
        labels = per_object.label
        count = Q2Counts(
            f_added_star=float(str(key_values[0])),
            host_shape_id=str(key_values[1]),
            criterion=str(key_values[2]),
            n_eligible=len(per_object),
            n_fit_completed=int(per_object.fit_completed.sum()),
            n_fit_valid=int(per_object.fit_valid.sum()),
            n_classifiable=int(labels.isin(["CL", "non-CL"]).sum()),
            n_cl=int((labels == "CL").sum()),
            n_noncl=int((labels == "non-CL").sum()),
            n_unclassifiable=int((labels == "unclassifiable").sum()),
        )
        rows.append(count.to_row())
    return pd.DataFrame(rows)


def _one_terminal_label(labels: set[object]) -> str:
    normalized = {str(label) for label in labels}
    if len(normalized) != 1:
        raise ValueError(f"one transition has inconsistent repeated-condition labels: {normalized}")
    label = normalized.pop()
    if label not in {"CL", "non-CL", "unclassifiable"}:
        raise ValueError(f"invalid final-classifier label: {label}")
    return label


__all__ = ["Q2Counts", "summarise_q2_outcomes"]
