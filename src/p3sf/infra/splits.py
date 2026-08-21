"""Deterministic, object-level development / validation / replication split.

The scientific unit is the **object**, never the spectrum and never the Monte
Carlo realization. Splitting by spectrum would let two epochs of the same AGN
land in different strata, which is leakage.

The bucket is a pure function of ``object_id``, so the split is stable across
runs, machines, and re-downloads, and does not depend on sample order or size.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

import pandas as pd

from p3sf.config import Config, load_config

Stratum = Literal["development", "validation", "replication"]


def bucket_of(object_id: str, *, buckets: int = 10) -> int:
    """Map an object identifier to a stable bucket in ``[0, buckets)``.

    Uses the full SHA-256 digest as a big integer rather than a truncated
    prefix, so the mapping is uniform and independent of identifier format.
    """
    if not isinstance(object_id, str) or not object_id:
        raise ValueError(f"object_id must be a non-empty string, got {object_id!r}")
    digest = hashlib.sha256(object_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def stratum_of(object_id: str, config: Config | None = None) -> Stratum:
    """Return the frozen stratum for one object."""
    cfg = config or load_config()
    split = cfg.splits
    bucket = bucket_of(object_id, buckets=split.buckets)
    if bucket in split.development:
        return "development"
    if bucket in split.validation:
        return "validation"
    if bucket in split.replication:
        return "replication"
    raise AssertionError(f"bucket {bucket} unassigned; config validation should have caught this")


def assign_splits(
    object_ids: Iterable[str],
    config: Config | None = None,
) -> pd.DataFrame:
    """Return a frame of ``object_id, bucket, stratum`` for the given objects."""
    cfg = config or load_config()
    ids = list(object_ids)
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"object_ids must be unique; repeated: {sorted(duplicates)[:5]}")
    return pd.DataFrame(
        {
            "object_id": ids,
            "bucket": [bucket_of(i, buckets=cfg.splits.buckets) for i in ids],
            "stratum": [stratum_of(i, cfg) for i in ids],
        }
    )


def _selftest() -> int:
    """Verify determinism, disjointness, and approximate 60/20/20 balance."""
    cfg = load_config()
    ids = [f"SDSSJ{n:012d}" for n in range(20_000)]
    first = assign_splits(ids, cfg)
    second = assign_splits(list(reversed(ids)), cfg)

    merged = first.merge(second, on="object_id", suffixes=("_a", "_b"))
    assert (merged.stratum_a == merged.stratum_b).all(), "split depends on input order"
    assert (merged.bucket_a == merged.bucket_b).all(), "bucket depends on input order"

    by_stratum: dict[str, set[str]] = {
        str(s): set(g.object_id) for s, g in first.groupby("stratum")
    }
    strata = sorted(by_stratum)
    for i, a in enumerate(strata):
        for b in strata[i + 1 :]:
            overlap = by_stratum[a] & by_stratum[b]
            assert not overlap, f"{a} and {b} share {len(overlap)} objects"

    fractions = first.stratum.value_counts(normalize=True)
    expected = {"development": 0.60, "validation": 0.20, "replication": 0.20}
    print(f"{'stratum':<14} {'n':>7} {'fraction':>9} {'expected':>9}")
    for stratum, want in expected.items():
        got = float(fractions.get(stratum, 0.0))
        n = int((first.stratum == stratum).sum())
        print(f"{stratum:<14} {n:>7} {got:>9.4f} {want:>9.2f}")
        assert abs(got - want) < 0.02, f"{stratum} fraction {got:.4f} deviates from {want}"

    print("\nsplit selftest PASSED (deterministic, disjoint, balanced)")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
