"""The frozen configuration and the object-level split.

The split tests are the ones that matter scientifically: a leak between strata
would invalidate the locked validation and replication analyses, and a split
that depends on sample order would make the whole thing irreproducible.
"""

from __future__ import annotations

import hashlib

import pytest
import yaml
from pydantic import ValidationError

from p3sf.config import load_config, project_root
from p3sf.infra.splits import assign_splits, bucket_of, stratum_of

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_config_loads_and_is_checksummed() -> None:
    cfg = load_config()
    raw = (project_root() / "00_admin" / "frozen_config.yaml").read_bytes()
    assert cfg.source_sha256 == hashlib.sha256(raw).hexdigest()


def test_config_is_immutable() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.random_seed = 1  # type: ignore[misc]


def test_primary_grids_match_the_preregistration() -> None:
    cfg = load_config()
    assert cfg.primary_line == "Hbeta"
    assert cfg.redshift.primary_max == 0.45
    assert cfg.snr_grid == [5, 10]
    assert cfg.host_fraction_grid == [0.1, 0.3, 0.5, 0.7, 0.85]
    assert cfg.mc_realizations == 50
    assert cfg.random_seed == 314159


def test_snr_experiment_declares_the_three_arms() -> None:
    """D-010: a pair has two S/N values, and the primary arm degrades only the faint one."""
    experiment = load_config().snr_experiment
    assert experiment.primary_arm == "faint_only"
    assert experiment.secondary_arm == "matched"
    assert experiment.exploratory_arm == "joint"
    assert experiment.binding_metric == "hbeta_window"


def test_unreachable_rung_is_removed_not_silently_kept() -> None:
    """No pair in the census reaches S/N 40 under either arm."""
    cfg = load_config()
    assert 40 in cfg.snr_experiment.removed_rungs
    assert 40 not in cfg.snr_grid


def test_underpowered_rungs_are_flagged_rather_than_promoted() -> None:
    """D-088 excludes sparsely supported exact-reference rungs from final Q1."""
    cfg = load_config()
    assert cfg.snr_experiment.underpowered_rungs == [15, 20, 30]
    assert not set(cfg.snr_experiment.underpowered_rungs) & set(cfg.snr_grid)


def test_bootstrap_clusters_on_object() -> None:
    """Monte Carlo realizations are not independent AGN. This must never change."""
    assert load_config().bootstrap.cluster == "object_id"
    assert load_config().bootstrap.n == 5000


def test_exactly_three_primary_contrasts_are_preregistered() -> None:
    cfg = load_config()
    assert [c.id for c in cfg.primary_contrasts] == ["PC1", "PC2", "PC3"]


def test_slice_grid_is_a_subset_of_the_frozen_grids() -> None:
    """The feasibility slice may only sample the frozen grid, never extend it."""
    cfg = load_config()
    assert set(cfg.slice_grid.snr) <= set(cfg.snr_grid)
    assert set(cfg.slice_grid.host_fraction) <= set(cfg.host_fraction_grid)
    assert cfg.slice_grid.mc_realizations <= cfg.mc_realizations


def test_config_rejects_unknown_keys(tmp_path) -> None:
    payload = yaml.safe_load((project_root() / "00_admin" / "frozen_config.yaml").read_text())
    payload["a_key_nobody_declared"] = 1
    bad = tmp_path / "frozen_config.yaml"
    bad.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_config_rejects_a_split_that_does_not_partition(tmp_path) -> None:
    payload = yaml.safe_load((project_root() / "00_admin" / "frozen_config.yaml").read_text())
    payload["splits"]["validation"] = [6, 7, 8]  # 8 now in two strata
    bad = tmp_path / "frozen_config.yaml"
    bad.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError):
        load_config(bad)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_bucket_is_deterministic_across_calls() -> None:
    assert bucket_of("SDSSJ011311.82+013542.4") == bucket_of("SDSSJ011311.82+013542.4")


def test_bucket_is_a_pure_function_of_the_identifier_string() -> None:
    """Known-value test: pins the hash convention so it cannot drift silently."""
    digest = hashlib.sha256(b"SDSSJ011311.82+013542.4").hexdigest()
    assert bucket_of("SDSSJ011311.82+013542.4") == int(digest, 16) % 10


def test_bucket_rejects_empty_and_non_string_identifiers() -> None:
    for bad in ("", None, 12345):
        with pytest.raises(ValueError):
            bucket_of(bad)  # type: ignore[arg-type]


def test_strata_are_disjoint_and_stable_under_reordering() -> None:
    ids = [f"OBJ{n:06d}" for n in range(5000)]
    forward = assign_splits(ids)
    backward = assign_splits(list(reversed(ids)))
    merged = forward.merge(backward, on="object_id", suffixes=("_a", "_b"))
    assert (merged.stratum_a == merged.stratum_b).all()

    groups = {s: set(g.object_id) for s, g in forward.groupby("stratum")}
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (groups[a] & groups[b]), f"{a}/{b} overlap"


def test_split_proportions_are_close_to_60_20_20() -> None:
    ids = [f"OBJ{n:06d}" for n in range(20000)]
    fractions = assign_splits(ids).stratum.value_counts(normalize=True)
    assert abs(fractions["development"] - 0.60) < 0.02
    assert abs(fractions["validation"] - 0.20) < 0.02
    assert abs(fractions["replication"] - 0.20) < 0.02


def test_duplicate_object_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        assign_splits(["A", "B", "A"])


def test_stratum_of_agrees_with_assign_splits() -> None:
    ids = [f"OBJ{n:05d}" for n in range(500)]
    frame = assign_splits(ids)
    assert all(stratum_of(row.object_id) == row.stratum for row in frame.itertuples())


def test_every_bucket_maps_to_exactly_one_stratum() -> None:
    cfg = load_config()
    seen: dict[int, str] = {}
    for n in range(3000):
        oid = f"OBJ{n:05d}"
        b, s = bucket_of(oid, buckets=cfg.splits.buckets), stratum_of(oid)
        if b in seen:
            assert seen[b] == s, f"bucket {b} mapped to both {seen[b]} and {s}"
        seen[b] = s
    assert len(seen) == cfg.splits.buckets, f"only {len(seen)} of {cfg.splits.buckets} buckets seen"
