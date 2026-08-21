"""Validated loader for the frozen analysis configuration.

No script in this project may hardcode an analysis constant. Everything that
could change a result lives in ``00_admin/frozen_config.yaml`` and is read
through :func:`load_config`.

Once ``frozen: true`` is set (by ``29_locked_validation.py``), any further
change to the file must be accompanied by an entry in
``00_admin/decisions_log.md``.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def project_root() -> Path:
    """Repository root, located by walking up to the directory holding 00_admin."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "00_admin" / "frozen_config.yaml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate project root: no ancestor of "
        f"{here} contains 00_admin/frozen_config.yaml"
    )


DEFAULT_CONFIG_PATH = "00_admin/frozen_config.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Redshift(_Base):
    primary_max: float
    primary_min: float = 0.0


class SnrExperiment(_Base):
    """The three Q1 arms and the rung provenance (D-010, D-027)."""

    binding_metric: Literal["hbeta_window", "continuum_5100"]
    primary_arm: Literal["faint_only"]
    secondary_arm: Literal["matched"]
    exploratory_arm: Literal["joint"]
    exploratory_enabled: bool
    exploratory_bright_rungs: list[float]
    underpowered_rungs: list[float]
    removed_rungs: list[float]


class SliceGrid(_Base):
    snr: list[float]
    host_fraction: list[float]
    mc_realizations: int
    n_objects: int


class PrimarySample(_Base):
    cls: Literal["gold", "silver", "gold+silver"] = Field(alias="class")


class Splits(_Base):
    method: Literal["sha256_object_id"]
    buckets: int
    development: list[int]
    validation: list[int]
    replication: list[int]

    @model_validator(mode="after")
    def _partition_is_exact(self) -> Splits:
        assigned = self.development + self.validation + self.replication
        if sorted(assigned) != list(range(self.buckets)):
            raise ValueError(
                f"split buckets must partition 0..{self.buckets - 1} exactly once; got {assigned}"
            )
        return self


class SnrMetrics(_Base):
    """Windows for the two mandated S/N definitions (D-010)."""

    continuum_5100: tuple[float, float]
    hbeta_window: tuple[float, float]


class Windows(_Base):
    hbeta_region: tuple[float, float]
    halpha_region: tuple[float, float]
    continuum_5100: tuple[float, float]
    continuum_blue: tuple[float, float]
    feii_window: tuple[float, float]
    error_audit_windows: list[tuple[float, float]]


class Fitting(_Base):
    engine: Literal["pyqsofit"]
    pyqsofit_repo: str
    pyqsofit_commit: str
    pyqsofit_version: str
    components: list[str]
    broad_fwhm_min_kms: float
    yang_broad_fwhm_max_kms: float
    detection_sigma: float
    upper_limit_sigma: float
    max_iterations: int


class QC(_Base):
    min_continuum_snr: float
    max_masked_fraction_in_line_region: float
    require_hbeta_coverage: bool
    sdss_zwarning_max: int
    desi_zwarn_max: int
    desi_use_fiberstatus: bool


class NoiseModel(_Base):
    renormalise: bool
    stratify_by: list[str]
    snr_bins: list[float]
    surveys: list[str]
    measured_factors: dict[str, float] | None = None


class Bootstrap(_Base):
    n: int
    cluster: Literal["object_id"]


class Contrast(_Base):
    id: str
    description: str


class Multiplicity(_Base):
    primary_contrasts_adjusted: bool
    secondary_adjustment: Literal["holm", "fdr", "none"]


class Regression(_Base):
    smoother: str
    spline_knots: int
    interactions: list[list[str]]


class NSigmaCriterion(_Base):
    threshold: float
    robustness_sweep: list[float]


class FluxRatioCriterion(_Base):
    threshold: float | None
    robustness_sweep: list[float]


class Criteria(_Base):
    primary: list[str]
    nsigma: NSigmaCriterion
    flux_ratio: FluxRatioCriterion


class Config(_Base):
    config_version: int
    frozen: bool
    frozen_at_git_hash: str | None

    primary_line: str
    secondary_lines: list[str]
    redshift: Redshift
    random_seed: int

    snr_grid: list[float]
    snr_experiment: SnrExperiment
    host_fraction_grid: list[float]
    mc_realizations: int
    slice_grid: SliceGrid

    primary_sample: PrimarySample
    splits: Splits

    windows: Windows
    host_fraction_reference_wavelength: float
    snr_metrics: SnrMetrics

    fitting: Fitting
    qc: QC
    noise_model: NoiseModel
    calibration_treatments: list[str]

    bootstrap: Bootstrap
    primary_contrasts: list[Contrast]
    multiplicity: Multiplicity
    regression: Regression
    criteria: Criteria

    # Not part of the YAML; recorded so every run can stamp what it read.
    source_path: Path
    source_sha256: str

    @model_validator(mode="after")
    def _grids_are_sane(self) -> Config:
        if sorted(self.snr_grid) != self.snr_grid:
            raise ValueError("snr_grid must be ascending")
        if sorted(self.host_fraction_grid) != self.host_fraction_grid:
            raise ValueError("host_fraction_grid must be ascending")
        if not all(0.0 <= f < 1.0 for f in self.host_fraction_grid):
            raise ValueError("host fractions must lie in [0, 1)")
        if self.mc_realizations < 1:
            raise ValueError("mc_realizations must be >= 1")
        overlap = set(self.snr_grid) & set(self.snr_experiment.removed_rungs)
        if overlap:
            raise ValueError(f"rungs {sorted(overlap)} are both frozen and marked removed")
        if len(self.primary_contrasts) != 3:
            raise ValueError(
                "exactly three primary contrasts are preregistered; "
                "adding a fourth requires a decisions_log.md entry and a config_version bump"
            )
        return self


@functools.lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    """Load, validate, and checksum the frozen configuration.

    Cached, so repeated calls inside one process return an identical object.
    """
    resolved = Path(path) if path is not None else project_root() / DEFAULT_CONFIG_PATH
    resolved = resolved.resolve()
    raw_bytes = resolved.read_bytes()
    payload: dict[str, Any] = yaml.safe_load(raw_bytes)
    payload["source_path"] = resolved
    payload["source_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return Config.model_validate(payload)


__all__ = ["Config", "load_config", "project_root", "DEFAULT_CONFIG_PATH"]
