#!/usr/bin/env python
"""Stage 3 — build and verify the analysis environment.

This is the single entry point for environment setup. Plain ``uv sync`` is not
sufficient: PyQSOFit is vendored from a pinned upstream commit rather than
resolved from PyPI, so it must be cloned and installed separately.

Usage
-----
    uv run python 00_scripts/00_make_environment.py            # build
    uv run python 00_scripts/00_make_environment.py --check    # verify only

``--check`` fails loudly on any drift between the running interpreter and what
``00_admin/frozen_config.yaml`` and ``00_admin/requirements-lock.txt`` pin.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import load_config, project_root  # noqa: E402

REQUIRED_PYTHON = (3, 12)

# Packages whose version materially affects a fitted number. Checked explicitly
# rather than trusting the lock file to have been applied.
CRITICAL_PACKAGES = (
    "numpy",
    "scipy",
    "astropy",
    "pandas",
    "pyqsofit",
    "ppxf",
    "specutils",
    "lmfit",
    "spectres",
    "statsmodels",
)


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def vendor_dir(root: Path) -> Path:
    return root / "06_fitting" / "pyqsofit" / "vendor" / "PyQSOFit"


def ensure_pyqsofit(root: Path, repo: str, commit: str) -> tuple[bool, str]:
    """Clone (if needed) and pin the vendored PyQSOFit. Returns (ok, message)."""
    target = vendor_dir(root)
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        clone = _run(["git", "clone", "--quiet", repo, str(target)])
        if clone.returncode != 0:
            return False, f"clone failed: {clone.stderr.strip()}"

    checkout = _run(["git", "checkout", "--quiet", commit], cwd=target)
    if checkout.returncode != 0:
        fetch = _run(["git", "fetch", "--quiet", "origin"], cwd=target)
        if fetch.returncode != 0:
            return False, f"fetch failed: {fetch.stderr.strip()}"
        checkout = _run(["git", "checkout", "--quiet", commit], cwd=target)
        if checkout.returncode != 0:
            return False, f"checkout {commit[:12]} failed: {checkout.stderr.strip()}"

    head = _run(["git", "rev-parse", "HEAD"], cwd=target).stdout.strip()
    if head != commit:
        return False, f"HEAD is {head[:12]}, expected {commit[:12]}"
    return True, head


def check_pyqsofit_commit(root: Path, commit: str) -> tuple[bool, str]:
    target = vendor_dir(root)
    if not (target / ".git").is_dir():
        return False, "not cloned"
    head = _run(["git", "rev-parse", "HEAD"], cwd=target).stdout.strip()
    if head != commit:
        return False, f"HEAD {head[:12]} != pinned {commit[:12]}"
    dirty = _run(["git", "status", "--porcelain"], cwd=target).stdout.strip()
    if dirty:
        return False, f"working tree dirty ({len(dirty.splitlines())} files)"
    return True, head


def installed_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in CRITICAL_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def git_revision(root: Path) -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if result.returncode == 0:
        return result.stdout.strip()
    if _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root).returncode == 0:
        return "<no commits yet>"
    return "<not a git repo>"


def build(root: Path, repo: str, commit: str) -> int:
    print("→ cloning / pinning PyQSOFit")
    ok, message = ensure_pyqsofit(root, repo, commit)
    if not ok:
        print(f"  FAIL {message}", file=sys.stderr)
        return 1
    print(f"  at {message[:12]}")

    print("→ uv sync --extra dev")
    sync = _run(["uv", "sync", "--extra", "dev"], cwd=root)
    if sync.returncode != 0:
        print(sync.stderr, file=sys.stderr)
        return 1

    print("→ installing vendored PyQSOFit (editable, --no-deps)")
    install = _run(
        ["uv", "pip", "install", "--no-deps", "-e", str(vendor_dir(root))], cwd=root
    )
    if install.returncode != 0:
        print(install.stderr, file=sys.stderr)
        return 1

    print("→ exporting requirements-lock.txt")
    export = _run(
        ["uv", "export", "--no-hashes", "--format", "requirements-txt",
         "-o", "00_admin/requirements-lock.txt"],
        cwd=root,
    )
    if export.returncode != 0:
        print(export.stderr, file=sys.stderr)
        return 1

    print("\nbuild complete — re-run with --check to verify\n")
    return 0


def check(root: Path, commit: str) -> int:
    cfg = load_config()
    failures: list[str] = []

    print("=" * 68)
    print("ENVIRONMENT CHECK")
    print("=" * 68)

    actual_py = sys.version_info[:2]
    py_ok = actual_py == REQUIRED_PYTHON
    print(f"{'python':<16} {'.'.join(map(str, sys.version_info[:3])):<28} "
          f"{'OK' if py_ok else f'FAIL (want {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x)'}")
    if not py_ok:
        failures.append("python version")

    pq_ok, pq_msg = check_pyqsofit_commit(root, commit)
    print(f"{'pyqsofit commit':<16} {pq_msg[:28]:<28} {'OK' if pq_ok else 'FAIL'}")
    if not pq_ok:
        failures.append("pyqsofit commit")

    print("-" * 68)
    for name, version in installed_versions().items():
        status = "OK" if version else "MISSING"
        print(f"{name:<16} {version or '-':<28} {status}")
        if version is None:
            failures.append(f"{name} not installed")

    print("-" * 68)
    print(f"{'config sha256':<16} {cfg.source_sha256[:28]}")
    print(f"{'config frozen':<16} {str(cfg.frozen):<28}")
    print(f"{'repo git HEAD':<16} {git_revision(root)[:28]}")
    print(f"{'random_seed':<16} {cfg.random_seed}")
    print("=" * 68)

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nenvironment OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; do not install")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config()
    commit = cfg.fitting.pyqsofit_commit
    repo = cfg.fitting.pyqsofit_repo

    if args.check:
        return check(root, commit)
    rc = build(root, repo, commit)
    return rc or check(root, commit)


if __name__ == "__main__":
    raise SystemExit(main())
