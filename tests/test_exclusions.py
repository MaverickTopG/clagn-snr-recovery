"""The exclusion ledger and its frozen vocabulary."""

from __future__ import annotations

import pytest
import yaml

from p3sf.config import project_root
from p3sf.infra.exclusions import (
    LOG_COLUMNS,
    exclusion_summary,
    read_exclusions,
    reason_codes,
    record_exclusion,
    record_exclusions,
)


def test_vocabulary_loads_and_is_non_trivial() -> None:
    codes = reason_codes()
    assert len(codes) >= 20
    for expected in (
        "BAD_REDSHIFT",
        "HBETA_OUT_OF_RANGE",
        "MASKED_HBETA",
        "CALIBRATION_FAILURE",
        "DECOMPOSITION_FAILURE",
        "NO_VALID_REFERENCE_STATE",
    ):
        assert expected in codes, f"{expected} missing from the frozen vocabulary"


def test_every_code_declares_a_stage_and_description() -> None:
    for code, entry in reason_codes().items():
        assert entry.stage, f"{code} has no stage"
        assert len(entry.description) > 15, f"{code} description is too thin to be useful"


def test_reference_sample_codes_cover_the_one_directional_degradation_rule() -> None:
    """Degradation only degrades, so 'too faint to degrade' must be an expressible outcome."""
    assert "REFERENCE_SNR_TOO_LOW_TO_DEGRADE" in reason_codes()


def test_host_injection_forbids_foreign_templates_via_a_code() -> None:
    assert "NO_HOST_TEMPLATE" in reason_codes()


def test_recording_writes_a_header_and_a_row(tmp_path) -> None:
    log = tmp_path / "exclusions_log.csv"
    record_exclusion("OBJ1", "BAD_REDSHIFT", script="07_x.py", detail="z=-1", path=log)

    lines = log.read_text().strip().splitlines()
    assert lines[0].split(",") == list(LOG_COLUMNS)
    assert len(lines) == 2

    frame = read_exclusions(log)
    row = frame.iloc[0]
    assert row.object_id == "OBJ1"
    assert row.reason_code == "BAD_REDSHIFT"
    assert row.stage == "catalog"  # taken from the vocabulary, not the caller
    assert row.script == "07_x.py"
    assert row.timestamp


def test_recording_appends_rather_than_truncates(tmp_path) -> None:
    log = tmp_path / "exclusions_log.csv"
    record_exclusion("OBJ1", "BAD_REDSHIFT", script="a.py", path=log)
    record_exclusion("OBJ2", "MASKED_HBETA", script="b.py", path=log)
    assert len(read_exclusions(log)) == 2


def test_unknown_code_is_rejected(tmp_path) -> None:
    log = tmp_path / "exclusions_log.csv"
    with pytest.raises(ValueError, match="unknown exclusion reason code"):
        record_exclusion("OBJ1", "LOOKED_WEIRD", script="a.py", path=log)


def test_a_bad_code_in_a_batch_writes_nothing(tmp_path) -> None:
    """Validation happens before any write, so the ledger is never left half-formed."""
    log = tmp_path / "exclusions_log.csv"
    with pytest.raises(ValueError):
        record_exclusions(
            [("OBJ1", "BAD_REDSHIFT", ""), ("OBJ2", "NOT_A_REAL_CODE", "")],
            script="a.py",
            path=log,
        )
    assert not log.exists() or read_exclusions(log).empty


def test_summary_counts_unique_objects(tmp_path) -> None:
    log = tmp_path / "exclusions_log.csv"
    record_exclusions(
        [
            ("OBJ1", "BAD_REDSHIFT", ""),
            ("OBJ1", "BAD_REDSHIFT", "recorded twice by two scripts"),
            ("OBJ2", "BAD_REDSHIFT", ""),
            ("OBJ3", "MASKED_HBETA", ""),
        ],
        script="a.py",
        path=log,
    )
    summary = exclusion_summary(log)
    counts = dict(zip(summary.reason_code, summary.n_objects, strict=True))
    assert counts["BAD_REDSHIFT"] == 2  # OBJ1 counted once
    assert counts["MASKED_HBETA"] == 1


def test_empty_ledger_reads_as_an_empty_frame() -> None:
    frame = read_exclusions()
    assert list(frame.columns) == list(LOG_COLUMNS)


def test_committed_ledger_only_uses_frozen_codes() -> None:
    """Guards the real ledger, not a fixture."""
    frame = read_exclusions()
    if frame.empty:
        pytest.skip("ledger is empty")
    unknown = set(frame.reason_code) - set(reason_codes())
    assert not unknown, f"ledger contains codes outside the vocabulary: {sorted(unknown)}"


def test_vocabulary_file_has_no_duplicate_codes() -> None:
    """YAML silently keeps the last duplicate key, so check the raw text."""
    path = project_root() / "00_admin" / "exclusion_reason_codes.yaml"
    payload = yaml.safe_load(path.read_text())
    declared = list(payload["codes"])
    raw_keys = [
        line.strip().rstrip(":")
        for line in path.read_text().splitlines()
        if line.startswith("  ") and line.rstrip().endswith(":") and not line.startswith("    ")
    ]
    assert len(raw_keys) == len(set(raw_keys)), "duplicate reason code in the YAML"
    assert set(raw_keys) == set(declared)
