# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/freeze.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : bf2f4161f87092282c013095ea080b661ef64946ca135005c1acf25a3715cb7f
# =====================================================================
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from p3sf.infra.provenance import canonical_hash, sha256_file


def create_freeze(
    destination: Path,
    *,
    configuration: Path,
    splits: Path,
    model_backend: str,
    literature_mask_date: str,
    power: float,
    minimum_power: float,
) -> tuple[str, Path]:
    if power < minimum_power:
        raise ValueError(f"power {power:.3f} is below required {minimum_power:.3f}")
    allowed_backends = {"censored_measurement_error", "latent_ew_interval_censored"}
    if model_backend not in allowed_backends:
        raise ValueError(f"production freeze requires one of {sorted(allowed_backends)}")
    body: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {"path": str(configuration), "sha256": sha256_file(configuration)},
        "splits": {"path": str(splits), "sha256": sha256_file(splits)},
        "model_backend": model_backend,
        "literature_mask_date": literature_mask_date,
        "power": power,
    }
    freeze_id = canonical_hash(body)
    payload = {**body, "freeze_id": freeze_id, "signature_algorithm": "sha256-canonical-json"}
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{freeze_id}.json"
    if path.exists():
        raise FileExistsError(f"freeze already exists: {freeze_id}")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return freeze_id, path


def claim_test(freeze_path: Path, test_root: Path) -> Path:
    payload = json.loads(freeze_path.read_text())
    freeze_id = str(payload["freeze_id"])
    lock = test_root / freeze_id / "TEST_ACCESSED"
    if lock.exists():
        raise PermissionError(f"test was already accessed for freeze {freeze_id}")
    lock.parent.mkdir(parents=True, exist_ok=False)
    temporary = lock.with_suffix(".tmp")
    temporary.write_text(datetime.now(UTC).isoformat() + "\n")
    temporary.replace(lock)
    return lock
