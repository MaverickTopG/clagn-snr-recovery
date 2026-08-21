#!/usr/bin/env python
"""D-093: acquire and natively validate only the 108 frozen D-092 endpoints.

The script has no catalogue-search path and accepts no candidate arguments.  It reads the
frozen D-092 manifest, never substitutes an epoch, never degrades a spectrum, and never
computes a CLAGN recovery outcome.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.config import load_config  # noqa: E402
from p3sf.criteria.green_pixel import (  # noqa: E402
    GreenEpochSpectrum,
    VarianceProvenance,
    fit_green_epoch_line_spectrum_pyqsofit,
    measure_green_pixel_nsigma,
)
from p3sf.fitting.pyqsofit_driver import fit_yang_hbeta_audit_pyqsofit  # noqa: E402
from p3sf.ingest.desi import exposure_summary, merge_cameras_official  # noqa: E402
from p3sf.qc.zwarning import Disposition, classify_zwarning  # noqa: E402

INPUT = ROOT / "04_reference_sample/q1_completeness_gold_acquisition_d092.csv"
RAW = ROOT / "03_spectra/raw_q1_d093"
OUT = ROOT / "05_analysis/q1_design/d093"
VENDOR = ROOT / "06_fitting/pyqsofit/vendor/PyQSOFit"
PROCESSING_VERSION = "p3sf-d093"
SDSS_BASE = "https://data.sdss.org/sas/dr17"
VALIS = "https://api.sdss.org/valis"
LAMOST = "https://www.lamost.org/dr11/v2.0/spectrum/fits"
DESI_EDR = "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix"


def _parts(text: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^;]+)", text))


def _coords_from_name(name: str) -> tuple[float, float] | None:
    match = re.search(
        r"J(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)$",
        name,
    )
    if match is None:
        return None
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra = 15.0 * (float(hh) + float(mm) / 60.0 + float(ss) / 3600.0)
    dec = float(dd) + float(dm) / 60.0 + float(ds) / 3600.0
    return ra, dec if sign == "+" else -dec


def _offset(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dx = (ra1 - ra2) * np.cos(np.deg2rad(0.5 * (dec1 + dec2)))
    return float(3600.0 * np.hypot(dx, dec1 - dec2))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, path: Path, *, gunzip: bool = False) -> str:
    if path.exists():
        return "PRESENT_CHECKSUM_REUSED"
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    payload = gzip.decompress(response.content) if gunzip else response.content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return "DOWNLOADED_EXACT"


def _sdss_url(identifier: str) -> tuple[str, str, dict[str, Any]]:
    stem, *fields = identifier.removeprefix("SDSS:").split(";")
    plate, mjd, fiber = map(int, stem.split("-"))
    values = _parts(";".join(fields))
    # The six D-092 Yang identifiers freeze PMF+specObjID but omit the redundant
    # legacy reduction label.  Their exact SDSS-I/II products use run2d=26;
    # specObjID is still checked inside the downloaded product, so this cannot
    # redirect the endpoint to a different observation.
    run2d = values.get("run2d", "26")
    tree = "eboss" if run2d.startswith("v") else "sdss"
    filename = f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"
    url = f"{SDSS_BASE}/{tree}/spectro/redux/{run2d}/spectra/lite/{plate:04d}/{filename}"
    expected = {
        "plate": plate,
        "mjd": mjd,
        "fiber": fiber,
        "specobjid": int(values["specObjID"]),
        "run2d": run2d,
    }
    return url, filename, expected


def _sdssv_url(identifier: str) -> tuple[str, str, dict[str, Any]]:
    values = _parts(identifier)
    expected = {
        "sdss_id": int(values["sdss_id"]),
        "field": int(values["field"]),
        "mjd": int(values["mjd"]),
        "catalogid": int(values["catalogid"]),
        "run2d": values["run2d"],
    }
    kwargs = [
        ("release", "DR19"),
        ("kwargs", f"fieldid={expected['field']}"),
        ("kwargs", f"mjd={expected['mjd']}"),
        ("kwargs", f"catalogid={expected['catalogid']}"),
        ("kwargs", f"run2d={expected['run2d']}"),
    ]
    url = f"{VALIS}/file/specLite/download?{urlencode(kwargs)}"
    filename = (
        f"spec-{expected['field']:06d}-{expected['mjd']}-"
        f"{expected['catalogid']}.fits"
    )
    return url, filename, expected


def _lamost_url(identifier: str) -> tuple[str, str, dict[str, Any]]:
    values = _parts(identifier)
    obsid = int(values["obsid"])
    expected = {
        "obsid": obsid,
        "planid": values["planid"],
        "spid": int(values["spid"]),
        "fiberid": int(values["fiberid"]),
        "lmjd": int(values["LMJD"]),
    }
    return f"{LAMOST}/{obsid}", f"lamost-dr11-{obsid}.fits", expected


def _desi_expected(identifier: str) -> dict[str, Any]:
    values = _parts(identifier)
    return {
        "targetid": int(values["TARGETID"]),
        "zpix_id": values["zpix_id"],
        "survey": values["survey"],
        "program": values["program"],
        "healpix": int(values["healpix"]),
    }


def _desi_url(expected: dict[str, Any]) -> str:
    healpix = int(expected["healpix"])
    return (
        f"{DESI_EDR}/{expected['survey']}/{expected['program']}/{healpix // 100}/"
        f"{healpix}/coadd-{expected['survey']}-{expected['program']}-{healpix}.fits"
    )


def _download_desi(identifier: str, path: Path) -> tuple[str, str, dict[str, Any]]:
    expected = _desi_expected(identifier)
    url = _desi_url(expected)
    if path.exists():
        return "PRESENT_CHECKSUM_REUSED", url, expected
    import fsspec

    handle = fsspec.open(url, block_size=1 << 20, cache_type="readahead").open()
    try:
        with fits.open(handle, memmap=False) as hdul:
            fibermap = hdul["FIBERMAP"].data
            rows = np.where(np.asarray(fibermap["TARGETID"]) == expected["targetid"])[0]
            if len(rows) != 1:
                raise RuntimeError(f"DESI TARGETID row count is {len(rows)}")
            row = int(rows[0])
            source_fibermap = np.asarray(fibermap[row : row + 1])
            exp = np.asarray(
                hdul["EXP_FIBERMAP"].data[
                    hdul["EXP_FIBERMAP"].data["TARGETID"] == expected["targetid"]
                ]
            )
            bands = ("B", "R", "Z")
            merged = merge_cameras_official(
                wave={b.lower(): np.asarray(hdul[f"{b}_WAVELENGTH"].data) for b in bands},
                flux={
                    b.lower(): np.asarray(hdul[f"{b}_FLUX"].section[row : row + 1])
                    for b in bands
                },
                ivar={
                    b.lower(): np.asarray(hdul[f"{b}_IVAR"].section[row : row + 1])
                    for b in bands
                },
                mask={
                    b.lower(): np.asarray(
                        hdul[f"{b}_MASK"].section[row : row + 1], dtype=np.uint32
                    )
                    for b in bands
                },
                resolution_data={
                    b.lower(): np.asarray(hdul[f"{b}_RESOLUTION"].section[row : row + 1])
                    for b in bands
                },
                fibermap=source_fibermap,
                exp_fibermap=exp,
            )
    finally:
        handle.close()
    primary = fits.PrimaryHDU()
    primary.header["SPECPROD"] = "fuji"
    primary.header["SURVEY"] = expected["survey"]
    primary.header["PROGRAM"] = expected["program"]
    primary.header["HEALPIX"] = expected["healpix"]
    primary.header["TARGETID"] = expected["targetid"]
    primary.header["ZPIX_ID"] = expected["zpix_id"]
    hdus = [
        primary,
        fits.BinTableHDU(data=merged["fibermap"], name="FIBERMAP"),
        fits.BinTableHDU(data=merged["exp_fibermap"], name="EXP_FIBERMAP"),
        fits.ImageHDU(data=merged["wavelength"], name="WAVELENGTH"),
        fits.ImageHDU(data=merged["flux"], name="FLUX"),
        fits.ImageHDU(data=merged["ivar"], name="IVAR"),
        fits.ImageHDU(data=merged["mask"], name="MASK"),
        fits.ImageHDU(data=merged["resolution"], name="RESOLUTION"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path)
    return "DOWNLOADED_EXACT", url, expected


def _field(row: Any, names: set[str], *candidates: str, default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in names:
            return row[candidate]
    return default


def _read_sdss(path: Path, expected: dict[str, Any], family: str) -> tuple[Spectrum, dict[str, Any]]:
    with fits.open(path) as hdul:
        data = hdul[1].data
        names = set(hdul[1].columns.names)
        wave = 10.0 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        ivar = np.asarray(data["ivar"], dtype=float)
        bitmask = np.asarray(data["and_mask"], dtype=np.uint64) | np.asarray(
            data["or_mask"], dtype=np.uint64
        )
        table = hdul[2].data[0]
        table_names = set(hdul[2].columns.names)
        header = hdul[0].header
        z = float(_field(table, table_names, "Z", default=np.nan))
        zwarning_raw = _field(table, table_names, "ZWARNING", default=0)
        try:
            zwarning = int(zwarning_raw)
        except (TypeError, ValueError):
            zwarning = 0
        ra = float(_field(table, table_names, "PLUG_RA", "FIBER_RA", "RA", default=np.nan))
        dec = float(_field(table, table_names, "PLUG_DEC", "FIBER_DEC", "DEC", default=np.nan))
        if family == "SDSS_LEGACY":
            actual = {
                "plate": int(_field(table, table_names, "PLATE", default=header.get("PLATEID", -1))),
                "mjd": int(_field(table, table_names, "MJD", default=header.get("MJD", -1))),
                "fiber": int(
                    _field(table, table_names, "FIBERID", default=header.get("FIBERID", -1))
                ),
                "specobjid": int(str(_field(table, table_names, "SPECOBJID", default=-1)).strip()),
                "run2d": str(_field(table, table_names, "RUN2D", default=expected["run2d"])).strip(),
            }
            exact = all(actual[key] == expected[key] for key in expected)
        else:
            actual = {
                "field": int(_field(table, table_names, "FIELD", "PLATE", default=-1)),
                "mjd": int(_field(table, table_names, "MJD", default=-1)),
                "catalogid": int(_field(table, table_names, "CATALOGID", default=-1)),
                "run2d": str(_field(table, table_names, "RUN2D", default=expected["run2d"])).strip(),
            }
            exact = all(actual[key] == expected[key] for key in ("field", "mjd", "catalogid", "run2d"))
        unit = str(hdul[1].columns["flux"].unit or "")
        structure = {"columns": sorted(names), "flux_unit": unit, "actual": actual}
    error = np.full_like(ivar, np.nan, dtype=float)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    # D-091 native QC reports SDSS bit masks but treats positive-IVAR pixels as
    # usable. Preserve that frozen behavior rather than inventing a stricter cut.
    spectrum = Spectrum(wave, flux, error, z, mask=(ivar <= 0).astype(int))
    return spectrum, {
        "source_identifier_exact": bool(exact),
        "ra": ra,
        "dec": dec,
        "z": z,
        "warning": zwarning,
        "mask_nonzero": int(np.count_nonzero(bitmask)),
        "structure": structure,
        "variance_semantics_valid": bool(np.all(ivar[np.isfinite(ivar)] >= 0)),
        "wavelength_convention_valid": bool(np.all(np.diff(wave) > 0)),
    }


def _read_lamost(path: Path, expected: dict[str, Any]) -> tuple[Spectrum, dict[str, Any]]:
    with fits.open(path) as hdul:
        header = hdul[0].header
        data = hdul[1].data[0]
        wave = np.asarray(data["WAVELENGTH"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float)
        ivar = np.asarray(data["IVAR"], dtype=float)
        bitmask = np.asarray(data["ANDMASK"], dtype=np.uint64) | np.asarray(
            data["ORMASK"], dtype=np.uint64
        )
        actual = {
            "obsid": int(header.get("OBSID", -1)),
            "planid": str(header.get("PLANID", "")).strip(),
            "spid": int(header.get("SPID", -1)),
            "fiberid": int(header.get("FIBERID", -1)),
            "lmjd": int(header.get("LMJD", -1)),
        }
        exact = all(actual[key] == expected[key] for key in expected)
        meta = {
            "source_identifier_exact": bool(exact),
            "ra": float(header.get("RA", np.nan)),
            "dec": float(header.get("DEC", np.nan)),
            "z": float(header.get("Z", np.nan)),
            "warning": int(header.get("FIB_MASK", 0)),
            "mask_nonzero": int(np.count_nonzero(bitmask)),
            "mjd": int(header.get("MJD", -1)),
            "lmjd": int(header.get("LMJD", -1)),
            "structure": {
                "actual": actual,
                "vacuum": bool(header.get("VACUUM", False)),
                "dispersion": str(header.get("WFITTYPE", "")),
                "flux_unit": str(hdul[1].columns["FLUX"].unit or ""),
            },
            "variance_semantics_valid": bool(np.all(ivar[np.isfinite(ivar)] >= 0)),
            "wavelength_convention_valid": bool(
                header.get("VACUUM", False) and np.all(np.diff(wave) > 0)
            ),
        }
    error = np.full_like(ivar, np.nan, dtype=float)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    return Spectrum(wave, flux, error, meta["z"], mask=(ivar <= 0).astype(int)), meta


def _read_desi(path: Path, expected: dict[str, Any]) -> tuple[Spectrum, dict[str, Any]]:
    with fits.open(path) as hdul:
        header = hdul[0].header
        fibermap = hdul["FIBERMAP"].data[0]
        names = set(hdul["FIBERMAP"].columns.names)
        exp = hdul["EXP_FIBERMAP"].data
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
        flux = np.asarray(hdul["FLUX"].data, dtype=float)
        ivar = np.asarray(hdul["IVAR"].data, dtype=float)
        mask = np.asarray(hdul["MASK"].data, dtype=np.uint64)
        z = float(_field(fibermap, names, "Z", default=np.nan))
        summary = exposure_summary(exp)
        actual = {
            "targetid": int(fibermap["TARGETID"]),
            "zpix_id": str(header.get("ZPIX_ID", "")),
            "survey": str(header.get("SURVEY", "")),
            "program": str(header.get("PROGRAM", "")),
            "healpix": int(header.get("HEALPIX", -1)),
        }
        exact = all(actual[key] == expected[key] for key in expected)
        meta = {
            "source_identifier_exact": bool(exact),
            "ra": float(_field(fibermap, names, "TARGET_RA", default=np.nan)),
            "dec": float(_field(fibermap, names, "TARGET_DEC", default=np.nan)),
            "z": z,
            "warning": int(_field(fibermap, names, "ZWARN", default=0)),
            "mask_nonzero": int(np.count_nonzero(mask)),
            **summary,
            "structure": {"actual": actual, "resolution_shape": hdul["RESOLUTION"].data.shape},
            "variance_semantics_valid": bool(np.all(ivar[np.isfinite(ivar)] >= 0)),
            "wavelength_convention_valid": bool(np.all(np.diff(wave) > 0)),
        }
    error = np.full_like(ivar, np.nan, dtype=float)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    return Spectrum(wave, flux, error, z, mask=mask), meta


def _zero_flux(epoch: GreenEpochSpectrum) -> GreenEpochSpectrum:
    return GreenEpochSpectrum(
        epoch.wavelength_rest,
        np.zeros_like(epoch.line_flux),
        epoch.variance,
        epoch.variance_provenance,
        epoch.preprocessing_valid,
        epoch.invalid_reason,
        epoch.scale_factor,
        epoch.scale_provenance,
    )


def _snr(spectrum: Spectrum, window: tuple[float, float]) -> float:
    rest = spectrum.rest_wavelength()
    use = (rest >= window[0]) & (rest <= window[1]) & spectrum.good
    return float(np.median(spectrum.flux[use] / spectrum.error[use])) if use.sum() >= 5 else np.nan


def _instrument_family(identifier: str) -> str:
    if identifier.startswith("SDSS:"):
        return "SDSS_LEGACY"
    if identifier.startswith("LAMOST"):
        return "LAMOST_DR11"
    if identifier.startswith("SDSSV"):
        return "SDSSV_DR19"
    if identifier.startswith("DESI"):
        return "DESI_EDR"
    raise ValueError(f"unknown source identifier {identifier}")


def _provenance(family: str) -> VarianceProvenance:
    return {
        "SDSS_LEGACY": VarianceProvenance.NATIVE_SDSS_SUPPORTED,
        "LAMOST_DR11": VarianceProvenance.NATIVE_LAMOST_SUPPORTED,
        "SDSSV_DR19": VarianceProvenance.NATIVE_SDSSV_SUPPORTED,
        "DESI_EDR": VarianceProvenance.NATIVE_DESI_EDR_SUPPORTED,
    }[family]


def _instrument_gate() -> pd.DataFrame:
    common = {
        "rest_frame_conversion": "lambda_rest=lambda_observed/(1+z)",
        "rebinning_behavior": "flux-conserving overlap-weighted 2A rest bins; variance sum(w^2 var)",
        "noise_degradation_operator_compatible": True,
        "yang_contract_compatible": True,
        "green_contract_compatible": True,
        "empirical_uncertainty_correction": "NONE",
    }
    rows = [
        {
            "instrument_family": "SDSS_LEGACY",
            "wavelength_convention": "vacuum log10-lambda Angstrom",
            "flux_density_units": "1e-17 erg s-1 cm-2 Angstrom-1",
            "variance_semantics": "IVAR=inverse variance of FLUX; error=1/sqrt(IVAR)",
            "mask_semantics": "AND_MASK OR OR_MASK; zero is usable",
            "spectral_sampling": "constant 1e-4 log10 Angstrom",
            "resolution_behavior": "per-pixel WDISP retained; Q1 changes noise only",
            **common,
        },
        {
            "instrument_family": "LAMOST_DR11",
            "wavelength_convention": "VACUUM=T; log-linear Angstrom",
            "flux_density_units": "native FLUX density; paired IVAR in same scaling",
            "variance_semantics": "IVAR=inverse variance of FLUX; error=1/sqrt(IVAR)",
            "mask_semantics": "ANDMASK OR ORMASK plus FIB_MASK; zero is usable",
            "spectral_sampling": "constant 1e-4 log10 Angstrom",
            "resolution_behavior": "native low-resolution LSF; no resolution mutation in Q1",
            **common,
        },
        {
            "instrument_family": "SDSSV_DR19",
            "wavelength_convention": "vacuum log10-lambda Angstrom",
            "flux_density_units": "1e-17 erg s-1 cm-2 Angstrom-1",
            "variance_semantics": "IVAR=inverse variance of FLUX; error=1/sqrt(IVAR)",
            "mask_semantics": "AND_MASK OR OR_MASK; zero is usable",
            "spectral_sampling": "BOSS log-lambda coadd grid",
            "resolution_behavior": "per-pixel WDISP retained; Q1 changes noise only",
            **common,
        },
        {
            "instrument_family": "DESI_EDR",
            "wavelength_convention": "vacuum linear Angstrom",
            "flux_density_units": "1e-17 erg s-1 cm-2 Angstrom-1",
            "variance_semantics": "IVAR=inverse variance of FLUX; error=1/sqrt(IVAR)",
            "mask_semantics": "DESI MASK bitfield; zero is usable",
            "spectral_sampling": "official desispec B/R/Z camera coadd grid",
            "resolution_behavior": "official resolution matrix preserved; Q1 changes noise only",
            **common,
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    frozen = pd.read_csv(INPUT)
    if len(frozen) != 54 or not frozen.status.eq("ACQUIRE_VALIDATE_ONLY").all():
        raise RuntimeError("D-093 requires the exact frozen 54-pair D-092 manifest")
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    acquisition: list[dict[str, Any]] = []
    identity: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], tuple[Spectrum, dict[str, Any], str, Path]] = {}

    for record in frozen.itertuples(index=False):
        for number in (1, 2):
            role = str(getattr(record, f"endpoint_{number}_role"))
            declared_mjd = int(getattr(record, f"endpoint_{number}_mjd"))
            identifier = str(getattr(record, f"endpoint_{number}_identifier"))
            family = _instrument_family(identifier)
            directory = RAW / family.lower()
            actual_epoch: float = np.nan
            url = ""
            try:
                if family == "SDSS_LEGACY":
                    url, filename, expected = _sdss_url(identifier)
                    path = directory / f"{record.object}__{role}__{filename}"
                    status = _download(url, path)
                    spectrum, meta = _read_sdss(path, expected, family)
                    actual_epoch = expected["mjd"]
                elif family == "SDSSV_DR19":
                    url, filename, expected = _sdssv_url(identifier)
                    path = directory / f"{record.object}__{role}__{filename}"
                    status = _download(url, path)
                    spectrum, meta = _read_sdss(path, expected, family)
                    actual_epoch = expected["mjd"]
                elif family == "LAMOST_DR11":
                    url, filename, expected = _lamost_url(identifier)
                    path = directory / f"{record.object}__{role}__{filename}"
                    status = _download(url, path, gunzip=True)
                    spectrum, meta = _read_lamost(path, expected)
                    actual_epoch = meta["mjd"]
                else:
                    expected = _desi_expected(identifier)
                    filename = (
                        f"desi-edr-{expected['survey']}-{expected['program']}-"
                        f"hp{expected['healpix']}-tid{expected['targetid']}.fits"
                    )
                    path = directory / f"{record.object}__{role}__{filename}"
                    status, url, expected = _download_desi(identifier, path)
                    spectrum, meta = _read_desi(path, expected)
                    actual_epoch = int(round(float(meta["mjd_effective"])))
                epoch_exact = (
                    declared_mjd == actual_epoch
                    or family == "LAMOST_DR11" and declared_mjd == int(meta["lmjd"])
                    or family == "DESI_EDR"
                    and float(meta["mjd_min"]) <= declared_mjd <= float(meta["mjd_max"])
                )
                name_coords = _coords_from_name(str(record.object))
                name_offset = (
                    _offset(*name_coords, float(meta["ra"]), float(meta["dec"]))
                    if name_coords is not None
                    else np.nan
                )
                name_coordinate_ok = bool(name_coords is None or name_offset <= 1.0)
                endpoint_identity = bool(
                    meta["source_identifier_exact"]
                    and epoch_exact
                    and name_coordinate_ok
                    and meta["wavelength_convention_valid"]
                    and meta["variance_semantics_valid"]
                )
                acquisition_status = status if endpoint_identity else "ACQUIRED_IDENTITY_FAIL"
                cache[(str(record.object), role)] = (spectrum, meta, family, path)
                failure = "" if endpoint_identity else "SOURCE_EPOCH_COORDINATE_OR_PRODUCT_MISMATCH"
            except Exception as error:  # noqa: BLE001 - every frozen endpoint must be recorded
                path = directory / f"{record.object}__{role}__UNAVAILABLE.fits"
                status = "ACQUISITION_FAILED"
                acquisition_status = status
                endpoint_identity = False
                epoch_exact = False
                name_offset = np.nan
                name_coordinate_ok = False
                meta = {
                    "source_identifier_exact": False,
                    "ra": np.nan,
                    "dec": np.nan,
                    "z": np.nan,
                    "warning": -1,
                    "structure": {},
                }
                failure = f"{type(error).__name__}:{error}"
            acquisition.append(
                {
                    "transition_id": record.object,
                    "physical_role": role,
                    "source_family": record.source_family,
                    "discovery_method": {
                        "Dong2025_SDSS_LAMOST": "automated spectral preselection plus visual and photometric confirmation",
                        "Zeltyn2024_SDSSV": "repeat spectroscopy plus visual confirmation and light-curve/follow-up support",
                        "Yang2025_turn_on": "optical/MIR variability preselection plus spectroscopic confirmation",
                    }[record.source_family],
                    "instrument_survey": family,
                    "immutable_source_identifier": identifier,
                    "observation_epoch_declared": declared_mjd,
                    "local_filename": str(path.relative_to(ROOT)) if path.exists() else "",
                    "byte_size": path.stat().st_size if path.exists() else 0,
                    "sha256": _digest(path) if path.exists() else "",
                    "source_url": url,
                    "acquisition_status": acquisition_status,
                    "processing_version": PROCESSING_VERSION,
                }
            )
            identity.append(
                {
                    "transition_id": record.object,
                    "physical_role": role,
                    "instrument_family": family,
                    "immutable_source_identifier": identifier,
                    "declared_epoch": declared_mjd,
                    "actual_epoch": actual_epoch,
                    "source_identifier_exact": meta["source_identifier_exact"],
                    "endpoint_epoch_exact": epoch_exact,
                    "archive_ra": meta["ra"],
                    "archive_dec": meta["dec"],
                    "name_coordinate_offset_arcsec": name_offset,
                    "name_coordinate_consistent": name_coordinate_ok,
                    "product_redshift": meta["z"],
                    "expected_instrument_product": family,
                    "wavelength_convention_valid": meta.get("wavelength_convention_valid", False),
                    "variance_semantics_valid": meta.get("variance_semantics_valid", False),
                    "endpoint_identity_pass": endpoint_identity,
                    "identity_failure_reason": failure,
                }
            )
            print(f"{record.object} {role} {family}: {acquisition_status}", flush=True)

    acquisition_frame = pd.DataFrame(acquisition)
    identity_frame = pd.DataFrame(identity)
    identity_frame["product_redshift"] = identity_frame.groupby(
        "transition_id"
    ).product_redshift.transform(lambda values: values.fillna(values.dropna().iloc[0]) if values.notna().any() else values)
    for _object_id, group in identity_frame.groupby("transition_id"):
        indices = group.index
        if group.archive_ra.notna().all() and group.archive_dec.notna().all():
            pair_offset = _offset(
                float(group.iloc[0].archive_ra),
                float(group.iloc[0].archive_dec),
                float(group.iloc[1].archive_ra),
                float(group.iloc[1].archive_dec),
            )
        else:
            pair_offset = np.nan
        redshift_delta = float(group.product_redshift.max() - group.product_redshift.min())
        identity_frame.loc[indices, "pair_coordinate_offset_arcsec"] = pair_offset
        identity_frame.loc[indices, "pair_coordinate_consistent"] = pair_offset <= 1.0
        identity_frame.loc[indices, "pair_redshift_delta"] = redshift_delta
        identity_frame.loc[indices, "pair_redshift_consistent"] = redshift_delta <= 0.001
        identity_frame.loc[indices, "endpoint_identity_pass"] &= (
            pair_offset <= 1.0 and redshift_delta <= 0.001
        )
        if not (pair_offset <= 1.0 and redshift_delta <= 0.001):
            reasons = identity_frame.loc[indices, "identity_failure_reason"]
            empty_reason = reasons.isna() | reasons.eq("")
            identity_frame.loc[indices[empty_reason], "identity_failure_reason"] = (
                "PAIR_COORDINATE_OR_REDSHIFT_INCONSISTENT"
            )
    failed_keys = set(
        map(
            tuple,
            identity_frame.loc[
                ~identity_frame.endpoint_identity_pass, ["transition_id", "physical_role"]
            ].to_numpy(),
        )
    )
    for key in failed_keys:
        selected = (
            (acquisition_frame.transition_id == key[0])
            & (acquisition_frame.physical_role == key[1])
        )
        acquisition_frame.loc[selected, "acquisition_status"] = "ACQUIRED_IDENTITY_FAIL"
    acquisition_frame.to_csv(OUT / "endpoint_acquisition_manifest_d093.csv", index=False)
    identity_frame.to_csv(OUT / "endpoint_identity_validation_d093.csv", index=False)
    _instrument_gate().to_csv(OUT / "instrument_domain_validation_d093.csv", index=False)

    cfg = load_config()
    qc_rows: list[dict[str, Any]] = []
    yang_rows: list[dict[str, Any]] = []
    green_epochs: dict[tuple[str, str], GreenEpochSpectrum] = {}
    canonical_redshift = identity_frame.groupby("transition_id").product_redshift.first().to_dict()
    for record in frozen.itertuples(index=False):
        for number in (1, 2):
            role = str(getattr(record, f"endpoint_{number}_role"))
            key = (str(record.object), role)
            if key not in cache:
                continue
            spectrum, meta, family, path = cache[key]
            spectrum = Spectrum(
                spectrum.wavelength,
                spectrum.flux,
                spectrum.error,
                float(canonical_redshift[record.object]),
                mask=spectrum.mask,
                meta=spectrum.meta,
            )
            rest = spectrum.rest_wavelength()
            in_hbeta = (rest >= 4700.0) & (rest <= 5100.0)
            n_hbeta = int(in_hbeta.sum())
            masked_fraction = (
                float((~spectrum.good & in_hbeta).sum() / n_hbeta) if n_hbeta else 1.0
            )
            covers = bool(rest.min() <= 4700.0 and rest.max() >= 5100.0)
            snr_hbeta = _snr(spectrum, (4700.0, 5100.0))
            snr_continuum = _snr(spectrum, tuple(cfg.snr_metrics.continuum_5100))
            warning_fail = False
            warning_disposition = "INSTRUMENT_RECORDED_NO_NEW_THRESHOLD"
            if family in {"SDSS_LEGACY", "SDSSV_DR19"}:
                warning = classify_zwarning(int(meta["warning"]))
                warning_fail = warning.disposition is Disposition.FAIL
                warning_disposition = str(warning.disposition)
            failures = []
            if not covers:
                failures.append("HBETA_OUT_OF_RANGE")
            if masked_fraction > cfg.qc.max_masked_fraction_in_line_region:
                failures.append("MASKED_HBETA")
            if not np.isfinite(snr_hbeta) or snr_hbeta < cfg.qc.min_continuum_snr:
                failures.append("LOW_CONTINUUM_SNR")
            if warning_fail:
                failures.append("ZWARNING_SET")
            identity_pass = bool(
                identity_frame.loc[
                    (identity_frame.transition_id == record.object)
                    & (identity_frame.physical_role == role),
                    "endpoint_identity_pass",
                ].item()
            )
            qc_pass = not failures
            qc_rows.append(
                {
                    "transition_id": record.object,
                    "physical_role": role,
                    "source_family": record.source_family,
                    "instrument_family": family,
                    "path": str(path.relative_to(ROOT)),
                    "canonical_redshift": spectrum.redshift,
                    "rest_min": float(rest.min()),
                    "rest_max": float(rest.max()),
                    "covers_rest_4700_5100": covers,
                    "valid_pixel_fraction": float(spectrum.good.mean()),
                    "masked_fraction_hbeta": masked_fraction,
                    "native_warning_value": int(meta["warning"]),
                    "warning_disposition": warning_disposition,
                    "local_hbeta_qc": "PASS" if qc_pass else "FAIL",
                    "native_hbeta_window_snr": snr_hbeta,
                    "continuum_snr": snr_continuum,
                    "variance_provenance": _provenance(family).value,
                    "endpoint_identity_pass": identity_pass,
                    "native_qc_pass": qc_pass,
                    "degradation_source_epoch_available": bool(identity_pass and qc_pass),
                    "qc_failures": ";".join(failures),
                }
            )
            _, yang = fit_yang_hbeta_audit_pyqsofit(
                spectrum.wavelength,
                spectrum.flux,
                spectrum.error,
                redshift=spectrum.redshift,
                path=str(VENDOR),
            )
            yang_rows.append(
                {
                    "transition_id": record.object,
                    "physical_role": role,
                    "instrument_family": family,
                    **asdict(yang),
                }
            )
            coords = _coords_from_name(str(record.object))
            ra, dec = coords if coords is not None else (float(meta["ra"]), float(meta["dec"]))
            green_epochs[key] = fit_green_epoch_line_spectrum_pyqsofit(
                spectrum.wavelength,
                spectrum.flux,
                spectrum.error,
                redshift=spectrum.redshift,
                path=VENDOR,
                variance_provenance=_provenance(family),
                ra=ra,
                dec=dec,
                scale_factor=1.0,
                scale_provenance="NO_RESCALE_EXACT_HISTORICAL_PAIR_D093",
            )
            print(f"native feasibility {record.object} {role}", flush=True)

    qc = pd.DataFrame(qc_rows)
    yang = pd.DataFrame(yang_rows)
    green_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for record in frozen.itertuples(index=False):
        object_id = str(record.object)
        roles = {
            str(record.endpoint_1_role): green_epochs.get((object_id, str(record.endpoint_1_role))),
            str(record.endpoint_2_role): green_epochs.get((object_id, str(record.endpoint_2_role))),
        }
        bright = roles.get("bright")
        faint = roles.get("faint")
        if bright is not None and faint is not None:
            feasibility = measure_green_pixel_nsigma(_zero_flux(bright), _zero_flux(faint))
        else:
            feasibility = None
        for role, epoch in roles.items():
            green_rows.append(
                {
                    "transition_id": object_id,
                    "physical_role": role,
                    "instrument_family": cache[(object_id, role)][2] if (object_id, role) in cache else "",
                    "preprocessing_valid": bool(epoch and epoch.preprocessing_valid),
                    "invalid_reason": epoch.invalid_reason if epoch else "ENDPOINT_UNAVAILABLE",
                    "variance_provenance": (
                        str(epoch.variance_provenance[0]) if epoch and len(epoch.variance_provenance) else ""
                    ),
                    "n_output_pixels": len(epoch.wavelength_rest) if epoch else 0,
                    "pair_zero_flux_feasibility_available": bool(
                        feasibility and feasibility.measurement_available
                    ),
                    "pair_zero_flux_feasibility_reason": (
                        feasibility.invalid_reason if feasibility else "ENDPOINT_UNAVAILABLE"
                    ),
                }
            )
        object_identity = bool(
            len(identity_frame[identity_frame.transition_id == object_id]) == 2
            and identity_frame.loc[
                identity_frame.transition_id == object_id, "endpoint_identity_pass"
            ].all()
        )
        object_qc = bool(
            len(qc[qc.transition_id == object_id]) == 2
            and qc.loc[qc.transition_id == object_id, "native_qc_pass"].all()
        )
        green_valid = bool(feasibility and feasibility.measurement_available)
        yang_group = yang[yang.transition_id == object_id]
        yang_measurement_domain = bool(
            len(yang_group) == 2 and yang_group.local_hbeta_coverage.all()
        )
        endpoint_valid = object_identity and object_qc
        validity_rows.append(
            {
                "transition_id": object_id,
                "source_family": record.source_family,
                "event": record.event,
                "gold_reference_evidence": "GOLD_REFERENCE_EVIDENCE",
                "identity_pair_valid": object_identity,
                "both_endpoints_native_qc_valid": object_qc,
                "q1_native_endpoint_valid": endpoint_valid,
                "yang_instrument_domain_valid": yang_measurement_domain,
                "yang_applicability": (
                    "APPLICABLE_FAIL_CLOSED_PER_FIT" if endpoint_valid and yang_measurement_domain
                    else "UNCLASSIFIABLE_INSTRUMENT_DOMAIN"
                ),
                "green_instrument_domain_valid": green_valid,
                "green_applicability": (
                    "APPLICABLE_SUPPORTED_VARIANCE" if endpoint_valid and green_valid
                    else "UNCLASSIFIABLE_INSTRUMENT_DOMAIN"
                ),
                "q1_eligible": endpoint_valid,
                "technical_exclusion_reason": (
                    "" if endpoint_valid else "IDENTITY_OR_NATIVE_QC_FAIL"
                ),
            }
        )
    qc.to_csv(OUT / "native_qc_d093.csv", index=False)
    yang.to_csv(OUT / "yang_native_fit_feasibility_d093.csv", index=False)
    pd.DataFrame(green_rows).to_csv(OUT / "green_native_feasibility_d093.csv", index=False)
    pd.DataFrame(validity_rows).to_csv(OUT / "new_gold_validity_d093.csv", index=False)
    print(
        f"acquired endpoints={int(acquisition_frame.byte_size.gt(0).sum())}/108; "
        f"endpoint-valid new GOLD={int(pd.DataFrame(validity_rows).q1_eligible.sum())}/54"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
