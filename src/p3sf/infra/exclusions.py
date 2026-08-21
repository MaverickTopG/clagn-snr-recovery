"""Append-only exclusion ledger with a frozen reason-code vocabulary.

Every object that leaves the sample leaves a row here. The manuscript must be
able to say "N objects were excluded for predefined reason X"; it must never
have to say "we visually removed several weird spectra".

Codes are validated against ``00_admin/exclusion_reason_codes.yaml`` at write
time, so a typo becomes an error rather than a silently new category.
"""

from __future__ import annotations

import csv
import functools
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from p3sf.config import project_root

LOG_COLUMNS = ("object_id", "stage", "reason_code", "detail", "script", "timestamp")


@dataclass(frozen=True)
class ReasonCode:
    code: str
    stage: str
    description: str


@functools.lru_cache(maxsize=1)
def reason_codes(path: str | Path | None = None) -> dict[str, ReasonCode]:
    """Load the frozen reason-code vocabulary."""
    resolved = Path(path) if path else project_root() / "00_admin" / "exclusion_reason_codes.yaml"
    payload = yaml.safe_load(resolved.read_text())
    out: dict[str, ReasonCode] = {}
    for code, body in payload["codes"].items():
        out[code] = ReasonCode(code=code, stage=body["stage"], description=body["description"])
    return out


def log_path() -> Path:
    return project_root() / "00_admin" / "exclusions_log.csv"


def _ensure_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)


def record_exclusion(
    object_id: str,
    reason_code: str,
    *,
    script: str,
    detail: str = "",
    path: Path | None = None,
) -> None:
    """Append one exclusion. Raises if the code is not in the frozen vocabulary."""
    record_exclusions([(object_id, reason_code, detail)], script=script, path=path)


def record_exclusions(
    rows: Iterable[tuple[str, str, str]],
    *,
    script: str,
    path: Path | None = None,
) -> int:
    """Append many exclusions as ``(object_id, reason_code, detail)`` triples.

    Validates every code before writing anything, so a bad batch does not leave
    a half-written ledger.
    """
    vocabulary = reason_codes()
    materialised = list(rows)

    unknown = sorted({code for _, code, _ in materialised if code not in vocabulary})
    if unknown:
        raise ValueError(
            f"unknown exclusion reason code(s): {unknown}. "
            "Add them to 00_admin/exclusion_reason_codes.yaml with a decisions_log.md entry."
        )

    target = path or log_path()
    _ensure_header(target)

    # Append-only, but idempotent: re-running a script must not duplicate its
    # own rows, or the ledger counts drift every time the pipeline is re-run.
    existing = read_exclusions(target)
    already = (
        set(zip(existing.object_id, existing.reason_code, existing.script, strict=True))
        if not existing.empty
        else set()
    )

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    written = 0
    with target.open("a", newline="") as handle:
        writer = csv.writer(handle)
        for object_id, code, detail in materialised:
            if (object_id, code, script) in already:
                continue
            writer.writerow([object_id, vocabulary[code].stage, code, detail, script, stamp])
            already.add((object_id, code, script))
            written += 1
    return written


def read_exclusions(path: Path | None = None) -> pd.DataFrame:
    target = path or log_path()
    if not target.exists() or target.stat().st_size == 0:
        return pd.DataFrame(columns=list(LOG_COLUMNS))
    return pd.read_csv(target, dtype=str).fillna("")


def exclusion_summary(path: Path | None = None) -> pd.DataFrame:
    """Counts by stage and reason code — the substrate of the sample-flow diagram."""
    frame = read_exclusions(path)
    if frame.empty:
        return pd.DataFrame(columns=["stage", "reason_code", "n_objects"])
    grouped = (
        frame.groupby(["stage", "reason_code"])["object_id"]
        .nunique()
        .reset_index(name="n_objects")
        .sort_values(["stage", "n_objects"], ascending=[True, False])
    )
    return grouped.reset_index(drop=True)


__all__ = [
    "LOG_COLUMNS",
    "ReasonCode",
    "exclusion_summary",
    "log_path",
    "read_exclusions",
    "reason_codes",
    "record_exclusion",
    "record_exclusions",
]
