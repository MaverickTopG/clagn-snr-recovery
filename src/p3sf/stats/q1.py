"""Object-level Q1 recovery and attrition accounting.

Noise realizations, S/N arms, and classifier evaluations are repeated conditions
on one real transition.  They never increase the independent-unit count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Q1Counts:
    """Nested distinct-transition counts for one ``(s, arm, criterion)`` cell."""

    snr_rung: float
    arm: str
    criterion: str
    n_eligible: int
    n_fit_completed: int
    n_fit_valid: int
    n_classifiable: int
    n_cl: int
    n_noncl: int
    n_unclassifiable: int

    def __post_init__(self) -> None:
        if not (
            0
            <= self.n_classifiable
            <= self.n_fit_valid
            <= self.n_fit_completed
            <= self.n_eligible
        ):
            raise ValueError("Q1 counts must nest: classifiable <= valid <= completed <= eligible")
        if self.n_cl + self.n_noncl != self.n_classifiable:
            raise ValueError("N_CL + N_nonCL must equal N_classifiable")
        if self.n_classifiable + self.n_unclassifiable != self.n_eligible:
            raise ValueError("N_classifiable + N_unclassifiable must equal N_eligible")

    @property
    def empirical_recovery_fraction(self) -> float | None:
        """R(s,a,k) = N_CL / N_classifiable; undefined with no decisions."""
        return self.n_cl / self.n_classifiable if self.n_classifiable else None

    @property
    def unclassifiable_fraction(self) -> float | None:
        """U(s,a,k) = N_unclassifiable / N_eligible."""
        return self.n_unclassifiable / self.n_eligible if self.n_eligible else None

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["R_empirical_recovery"] = self.empirical_recovery_fraction
        row["U_unclassifiable"] = self.unclassifiable_fraction
        return row


def summarise_q1_outcomes(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise terminal labels without counting repeated realizations as AGN."""
    required = {
        "object_id",
        "snr_rung",
        "arm",
        "criterion",
        "fit_completed",
        "fit_valid",
        "label",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Q1 outcome table missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    keys = ["snr_rung", "arm", "criterion"]
    for key, group in results.groupby(keys, dropna=False):
        snr_rung, arm, criterion = key
        per_object = group.groupby("object_id", as_index=False).agg(
            fit_completed=("fit_completed", "max"),
            fit_valid=("fit_valid", "max"),
            label=("label", lambda values: _one_terminal_label(set(values))),
        )
        labels = per_object.label
        rows.append(
            Q1Counts(
                snr_rung=float(str(snr_rung)),
                arm=str(arm),
                criterion=str(criterion),
                n_eligible=len(per_object),
                n_fit_completed=int(per_object.fit_completed.sum()),
                n_fit_valid=int(per_object.fit_valid.sum()),
                n_classifiable=int(labels.isin(["CL", "non-CL"]).sum()),
                n_cl=int((labels == "CL").sum()),
                n_noncl=int((labels == "non-CL").sum()),
                n_unclassifiable=int((labels == "unclassifiable").sum()),
            ).to_row()
        )
    return pd.DataFrame(rows)


def _one_terminal_label(labels: set[object]) -> str:
    normalized = {str(label) for label in labels}
    if len(normalized) != 1:
        raise ValueError(f"one transition has inconsistent repeated-condition labels: {normalized}")
    label = normalized.pop()
    if label not in {"CL", "non-CL", "unclassifiable"}:
        raise ValueError(f"invalid final-classifier label: {label}")
    return label


def summarise_q1_realizations(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate stochastic Q1 outcomes in two stages, preserving object weight.

    Stage one produces one row per transition and condition with
    ``p_i = n_CL_i / n_classifiable_i`` and an unclassifiable-realization
    fraction. Stage two averages those transition-level quantities, never the
    realization rows. Thus an object with more surviving or more interesting
    draws cannot receive more scientific weight.
    """
    required = {"object_id", "snr_rung", "arm", "criterion", "realization", "label"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Q1 realization table missing columns: {sorted(missing)}")
    valid_labels = {"CL", "non-CL", "unclassifiable"}
    observed = set(results["label"].astype(str))
    if not observed <= valid_labels:
        raise ValueError(f"invalid final-classifier labels: {sorted(observed - valid_labels)}")
    duplicate_keys = ["object_id", "snr_rung", "arm", "criterion", "realization"]
    if results.duplicated(duplicate_keys).any():
        raise ValueError("one transition/condition has duplicate realization labels")

    transition_rows: list[dict[str, object]] = []
    group_keys = ["object_id", "snr_rung", "arm", "criterion"]
    for key, group in results.groupby(group_keys, dropna=False):
        object_id, snr_rung, arm, criterion = key
        labels = group["label"].astype(str)
        n_total = len(labels)
        n_cl = int((labels == "CL").sum())
        n_noncl = int((labels == "non-CL").sum())
        n_unclassifiable = int((labels == "unclassifiable").sum())
        n_classifiable = n_cl + n_noncl
        transition_rows.append({
            "object_id": str(object_id),
            "snr_rung": float(str(snr_rung)),
            "arm": str(arm),
            "criterion": str(criterion),
            "n_realizations": n_total,
            "n_classifiable_realizations": n_classifiable,
            "n_cl_realizations": n_cl,
            "n_noncl_realizations": n_noncl,
            "n_unclassifiable_realizations": n_unclassifiable,
            "p_i_recovery": n_cl / n_classifiable if n_classifiable else None,
            "u_i_unclassifiable": n_unclassifiable / n_total if n_total else None,
        })
    transition = pd.DataFrame(transition_rows)

    aggregate_rows: list[dict[str, object]] = []
    for key, group in transition.groupby(["snr_rung", "arm", "criterion"], dropna=False):
        snr_rung, arm, criterion = key
        defined = group.p_i_recovery.notna()
        aggregate_rows.append({
            "snr_rung": float(str(snr_rung)),
            "arm": str(arm),
            "criterion": str(criterion),
            "n_transitions": len(group),
            "n_transitions_with_defined_p": int(defined.sum()),
            "R_equal_object_mean": (
                float(group.loc[defined, "p_i_recovery"].mean()) if defined.any() else None
            ),
            "U_equal_object_mean": float(group.u_i_unclassifiable.mean()),
        })
    return transition, pd.DataFrame(aggregate_rows)


__all__ = ["Q1Counts", "summarise_q1_outcomes", "summarise_q1_realizations"]
