"""DESI DR1/Iron science-record identity and camera-combination helpers.

The public functions in this module encode decision D-077.  They deliberately
keep source-record identity, observing-program provenance, and physical epoch
roles separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PRODUCT_TYPE = "healpix-coadd"
EXPECTED_BANDS = ("b", "r", "z")


def parse_catalog_bool(value: object) -> bool:
    """Parse a catalog boolean without Python's truthy-string trap.

    Unknown or missing representations raise instead of being guessed.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"t", "true", "1"}:
            return True
        if normalized in {"f", "false", "0"}:
            return False
    raise ValueError(f"unknown catalog boolean representation: {value!r}")


def _coordinate(value: object) -> str:
    text = str(value).strip().lower()
    if not text or any(char in text for char in ":/\\"):
        raise ValueError(f"unsafe or empty DESI source coordinate: {value!r}")
    return text


@dataclass(frozen=True)
class ScienceRecord:
    """Immutable coordinates of one intended Paper-3 DESI science unit."""

    object_id: str
    specprod: str
    survey: str
    program: str
    healpix: int
    targetid: int
    zpix_id: str | None
    product_type: str = PRODUCT_TYPE

    @property
    def zpix_coordinate(self) -> str:
        return "na" if self.zpix_id is None else _coordinate(self.zpix_id)

    @property
    def spectrum_id(self) -> str:
        return (
            f"desi:{_coordinate(self.specprod)}:{_coordinate(self.product_type)}:"
            f"{_coordinate(self.survey)}:{_coordinate(self.program)}:hp{int(self.healpix)}:"
            f"tid{int(self.targetid)}:zpix{self.zpix_coordinate}"
        )

    @property
    def filename(self) -> str:
        object_coordinate = self.object_id.replace(" ", "")
        if any(char in object_coordinate for char in "/\\"):
            raise ValueError(f"unsafe object ID for filename: {self.object_id!r}")
        return (
            f"{object_coordinate}__desi__{_coordinate(self.specprod)}__"
            f"{_coordinate(self.product_type)}__{_coordinate(self.survey)}__"
            f"{_coordinate(self.program)}__hp{int(self.healpix)}__tid{int(self.targetid)}__"
            f"zpix{self.zpix_coordinate}.fits"
        )


def wavelength_overlap_tolerance(wavelength: np.ndarray, fraction: float = 0.25) -> float:
    """Return an overlap-audit tolerance tied to the nominal physical grid.

    Tiny differences are excluded when estimating the grid step, so a malformed
    near-duplicate cannot define its own (ineffective) tolerance.
    """
    wave = np.sort(np.asarray(wavelength, dtype=float))
    differences = np.diff(wave)
    finite_positive = differences[np.isfinite(differences) & (differences > 0)]
    if finite_positive.size == 0:
        raise ValueError("at least two distinct finite wavelengths are required")
    upper_half = finite_positive[finite_positive >= np.median(finite_positive)]
    nominal_spacing = float(np.median(upper_half))
    return fraction * nominal_spacing


def count_tolerance_duplicates(wavelength: np.ndarray) -> int:
    """Count adjacent samples representing the same nominal grid location."""
    wave = np.sort(np.asarray(wavelength, dtype=float))
    if wave.size < 2:
        return 0
    tolerance = wavelength_overlap_tolerance(wave)
    return int(np.count_nonzero(np.diff(wave) < tolerance))


def merge_cameras_official(
    *,
    wave: dict[str, np.ndarray],
    flux: dict[str, np.ndarray],
    ivar: dict[str, np.ndarray],
    mask: dict[str, np.ndarray],
    resolution_data: dict[str, np.ndarray],
    fibermap: Any,
    exp_fibermap: Any,
) -> dict[str, Any]:
    """Merge B/R/Z with official desispec, preserving one source row.

    This has no concatenate/sort fallback.  Absence or failure of desispec is a
    hard ingest failure because a hand-rolled merge is outside the science
    contract.
    """
    from desispec.coaddition import coadd_cameras
    from desispec.spectra import Spectra

    bands = list(EXPECTED_BANDS)
    spectra = Spectra(
        bands=bands,
        wave={band: np.asarray(wave[band]) for band in bands},
        flux={band: np.asarray(flux[band]) for band in bands},
        ivar={band: np.asarray(ivar[band]) for band in bands},
        mask={band: np.asarray(mask[band]) for band in bands},
        resolution_data={band: np.asarray(resolution_data[band]) for band in bands},
        fibermap=fibermap,
        exp_fibermap=exp_fibermap,
    )
    if spectra.num_spectra() != 1:
        raise ValueError(f"camera merge requires one science row, got {spectra.num_spectra()}")

    merged = coadd_cameras(spectra)
    if merged.num_spectra() != 1:
        raise RuntimeError(f"camera merge changed row count 1 -> {merged.num_spectra()}")
    if merged.exp_fibermap is None or len(merged.exp_fibermap) != len(exp_fibermap):
        raise RuntimeError("camera merge did not preserve complete EXP_FIBERMAP provenance")

    band = merged.bands[0]
    wavelength = np.asarray(merged.wave[band])
    if not np.all(np.diff(wavelength) > 0):
        raise RuntimeError("official camera merge did not produce a strictly increasing grid")
    if count_tolerance_duplicates(wavelength) != 0:
        raise RuntimeError("official camera merge retained tolerance-level wavelength overlaps")

    source_target = int(np.asarray(fibermap["TARGETID"])[0])
    merged_target = int(np.asarray(merged.fibermap["TARGETID"])[0])
    if source_target != merged_target:
        raise RuntimeError(f"TARGETID changed during camera merge: {source_target} -> {merged_target}")

    return {
        "wavelength": wavelength,
        "flux": np.asarray(merged.flux[band][0]),
        "ivar": np.asarray(merged.ivar[band][0]),
        "mask": np.asarray(merged.mask[band][0]),
        "resolution": np.asarray(merged.resolution_data[band][0]),
        "fibermap": merged.fibermap,
        "exp_fibermap": merged.exp_fibermap,
    }


def exposure_summary(exp_fibermap: Any) -> dict[str, float | int]:
    """Summarize exposure provenance without replacing the underlying rows."""
    if exp_fibermap is None or len(exp_fibermap) == 0:
        raise ValueError("EXP_FIBERMAP is required for a DESI coadd science record")
    names = set(exp_fibermap.dtype.names or ())
    required = {"NIGHT", "EXPID", "TILEID", "MJD"}
    missing = required - names
    if missing:
        raise ValueError(f"EXP_FIBERMAP missing required columns: {sorted(missing)}")
    mjds = np.asarray(exp_fibermap["MJD"], dtype=float)
    if not np.all(np.isfinite(mjds)):
        raise ValueError("EXP_FIBERMAP contains non-finite MJD")
    return {
        "mjd_min": float(np.min(mjds)),
        "mjd_effective": float(np.mean(mjds)),
        "mjd_max": float(np.max(mjds)),
        "n_exp": int(len(np.unique(np.asarray(exp_fibermap["EXPID"])) )),
        "n_night": int(len(np.unique(np.asarray(exp_fibermap["NIGHT"])) )),
    }


def relative_product_path(path: Path, root: Path) -> str:
    """Serialize a local product path for the repository manifest."""
    return str(path.resolve().relative_to(root.resolve()))
