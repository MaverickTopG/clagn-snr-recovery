# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/provenance.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 351a0ecf3d21a7aa99adb8c4edfbdf868e7b16a7037116fd85e574c4f065ab71
# =====================================================================
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from p3sf import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class RunManifest:
    command: str
    config: dict[str, Any]
    inputs: dict[str, dict[str, Any]]
    run_id: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    exclusions: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: str = "running"


class RunWriter:
    def __init__(self, temporary: Path, final: Path, manifest: RunManifest):
        self.path = temporary
        self.final_path = final
        self.manifest = manifest

    def output_path(self, name: str) -> Path:
        return self.path / name

    def add_output(self, name: str, path: Path, rows: int | None = None) -> None:
        entry: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if rows is not None:
            entry["rows"] = rows
        self.manifest.outputs[name] = entry


@contextmanager
def atomic_run(
    run_root: Path,
    command: str,
    config: dict[str, Any],
    inputs: dict[str, Path],
) -> Iterator[RunWriter]:
    input_meta = {
        key: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in sorted(inputs.items())
    }
    identity = {"command": command, "config": config, "inputs": input_meta, "version": __version__}
    run_id = canonical_hash(identity)[:20]
    final = run_root / command.replace(" ", "_") / run_id
    if final.exists():
        raise FileExistsError(f"immutable run already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=final.parent))
    manifest = RunManifest(command, config, input_meta, run_id)
    writer = RunWriter(temporary, final, manifest)
    try:
        yield writer
        manifest.status = "complete"
        payload = {
            **manifest.__dict__,
            "completed_at": datetime.now(UTC).isoformat(),
            "software": {
                "package_version": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_revision": git_revision(),
            },
            "config_hash": canonical_hash(config),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
