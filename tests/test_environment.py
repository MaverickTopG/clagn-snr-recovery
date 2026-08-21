"""Environment and vendored-code integrity.

These are contract tests, not unit tests: they assert that the analysis is
running against the pinned interpreter and the pinned fitting engine, and that
every vendored module still records where it came from.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from p3sf.config import load_config, project_root

VENDORED = [
    "access/spectra.py",
    "access/datalab.py",
    "access/emfit.py",
    "fitting/pyqsofit_driver.py",
    "fitting/ppxf_driver.py",
    "stats/latent_ew.py",
    "stats/censored.py",
    "infra/provenance.py",
    "infra/freeze.py",
]


def test_python_is_pinned_minor_version() -> None:
    assert sys.version_info[:2] == (3, 12), (
        f"PyQSOFit 2.1.6 is pinned against Python 3.12; running {sys.version_info[:3]}"
    )


def test_all_vendored_modules_import() -> None:
    import p3sf.access.datalab  # noqa: F401
    import p3sf.access.emfit  # noqa: F401
    import p3sf.access.spectra  # noqa: F401
    import p3sf.fitting.ppxf_driver  # noqa: F401
    import p3sf.fitting.pyqsofit_driver  # noqa: F401
    import p3sf.infra.freeze  # noqa: F401
    import p3sf.infra.provenance  # noqa: F401
    import p3sf.stats.censored  # noqa: F401
    import p3sf.stats.latent_ew  # noqa: F401


@pytest.mark.parametrize("relative", VENDORED)
def test_vendored_files_carry_provenance_header(relative: str) -> None:
    path = project_root() / "src" / "p3sf" / relative
    head = path.read_text().split("\n", 12)
    joined = "\n".join(head)
    assert "VENDORED CODE" in joined, f"{relative} lost its provenance header"
    assert re.search(r"upstream rev: [0-9a-f]{40}", joined), f"{relative} has no upstream revision"
    assert re.search(r"sha256\(src\) : [0-9a-f]{64}", joined), f"{relative} has no source checksum"


def test_no_upstream_package_imports_remain() -> None:
    """Vendored code must not reach back into the CLAGN-WISE-ApJ package."""
    offenders = []
    for path in (project_root() / "src" / "p3sf").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.match(r"\s*(from|import)\s+clagn_apj", line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"imports from upstream package remain: {offenders}"


def test_pyqsofit_is_at_the_pinned_commit() -> None:
    cfg = load_config()
    vendor = project_root() / "06_fitting" / "pyqsofit" / "vendor" / "PyQSOFit"
    if not (vendor / ".git").is_dir():
        pytest.skip("PyQSOFit not cloned; run 00_scripts/00_make_environment.py")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=vendor, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head == cfg.fitting.pyqsofit_commit, (
        f"vendored PyQSOFit at {head[:12]}, config pins {cfg.fitting.pyqsofit_commit[:12]}"
    )


def test_pyqsofit_imports_and_reports_pinned_version() -> None:
    import importlib.metadata as md

    from pyqsofit.PyQSOFit import QSOFit  # noqa: F401

    cfg = load_config()
    assert md.version("pyqsofit") == cfg.fitting.pyqsofit_version


_FULL_TREE = (Path(__file__).resolve().parents[1] / "03_spectra" / "raw_sdss").is_dir()
_needs_full_tree = pytest.mark.skipif(
    not _FULL_TREE,
    reason="requires the full development tree, including raw survey spectra",
)


@_needs_full_tree
def test_repository_layout_is_present() -> None:
    root = project_root()
    required = [
        "00_admin", "00_scripts", "01_literature", "02_catalogs/raw",
        "03_spectra/raw_sdss", "03_spectra/raw_desi", "04_reference_sample/gold",
        "05_controls/synthetic_null", "06_fitting/validation",
        "07_counterfactuals/combined", "08_classifiers/published_criteria",
        "09_statistics/primary", "10_figures", "11_tables", "12_paper",
        "13_release/catalog", "14_referee_audit",
    ]
    missing = [d for d in required if not (root / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_raw_survey_data_is_not_redistributed() -> None:
    """Raw survey data must never be committed; it is re-fetchable from manifests.

    Asserted against what the repository actually contains rather than against
    particular ignore patterns, so the guarantee survives a change of wording.
    The only FITS file shipped is the PyQSOFit line-parameter file, which is
    configuration, not observation.
    """
    root = project_root()
    ignore = (root / ".gitignore").read_text()
    assert "03_spectra" in ignore

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    assert not [f for f in tracked if f.startswith("03_spectra/")]
    fits = [f for f in tracked if f.endswith((".fits", ".fits.gz"))]
    assert fits == ["config/pyqsofit_qsopar.fits"], fits
