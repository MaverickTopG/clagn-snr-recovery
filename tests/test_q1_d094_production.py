from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_analysis" / "q1_production" / "d094"
RAW = OUT / "raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_d094_completion_and_production_qc_are_exact() -> None:
    completion = json.loads((OUT / "FULL_Q1_EXECUTION_COMPLETE.json").read_text())
    qc = json.loads((OUT / "production_qc_d094.json").read_text())
    assert completion["decision"] == "FULL_Q1_EXECUTION_COMPLETE"
    assert completion["production_qc"] == "PASS"
    assert completion["conditions"] == 167
    assert completion["condition_realizations"] == 8350
    assert completion["fit_inputs"] == 8412
    assert completion["classifier_outcomes"] == 25050
    assert completion["seed_namespace"] == "p3sf:q1:full:v1"
    assert completion["realizations"] == 50
    assert qc["status"] == "PASS"
    assert all(qc["checks"].values())
    assert qc["counts"]["fit_jobs_failed"] == 0
    assert qc["counts"]["execution_reruns"] == 6305


_FULL_TREE = (Path(__file__).resolve().parents[1] / "03_spectra" / "raw_sdss").is_dir()
_needs_full_tree = pytest.mark.skipif(
    not _FULL_TREE,
    reason="requires the full development tree, including raw survey spectra",
)


@_needs_full_tree
def test_d094_raw_products_match_frozen_checksums() -> None:
    checksums = pd.read_csv(RAW / "raw_checksums_d094.csv")
    assert len(checksums) == 6
    for row in checksums.itertuples(index=False):
        path = ROOT / row.file
        assert path.stat().st_size == row.bytes
        assert _sha256(path) == row.sha256


def test_d094_outcome_accounting_and_crn_identity() -> None:
    outcomes = pd.read_parquet(RAW / "classifier_outcomes_d094.parquet")
    assert len(outcomes) == 25050
    assert outcomes["condition_id"].nunique() == 167
    assert set(outcomes["classification"]) <= {"CL", "NON_CL", "UNCLASSIFIABLE"}
    assert outcomes.groupby(["condition_id", "criterion_id"]).size().eq(50).all()
    faint = outcomes.groupby(["transition_id", "snr", "realization"])[
        ["faint_task_id", "faint_seed", "faint_spectrum_array_sha256"]
    ].nunique()
    assert (faint <= 1).all().all()
    faint_only = outcomes[outcomes["arm"] == "faint_only"]
    assert faint_only["bright_task_id"].str.endswith("|native_bright").all()
    assert (outcomes.loc[~outcomes["applicable"], "classification"] == "UNCLASSIFIABLE").all()


def test_d094_transition_first_denominator_identities() -> None:
    metrics = pd.read_csv(OUT / "tables" / "transition_level_estimands_d094.csv")
    assert len(metrics) == 501
    assert metrics["N_eligible_realizations"].eq(50).all()
    assert (
        metrics["N_CL"] + metrics["N_nonCL"] + metrics["N_unclassifiable"]
        == metrics["N_eligible_realizations"]
    ).all()
    assert (metrics["N_CL"] + metrics["N_nonCL"] == metrics["N_classifiable"]).all()

    aggregate = pd.read_csv(OUT / "tables" / "aggregate_estimands_bootstrap_d094.csv")
    primary = aggregate[aggregate["selection_scope"] == "PRIMARY_GOLD"]
    assert primary["N_reference"].eq(58).all()
    sensitivity = aggregate[aggregate["selection_scope"] == "GOLD_PLUS_SILVER"]
    assert sensitivity["N_reference"].eq(62).all()
    macleod = aggregate[aggregate["criterion_id"] == "MACLEOD2019_FINAL"]
    assert macleod["N_transition_classifiable"].eq(0).all()
    assert macleod["U_equal_transition"].eq(1.0).all()
