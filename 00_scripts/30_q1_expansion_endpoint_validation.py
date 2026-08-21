#!/usr/bin/env python
"""D-091 bounded acquisition and native validation for six frozen GOLD events.

This script intentionally knows only the twelve D-090 endpoint identifiers.  It
does not search for alternatives, degrade spectra, or evaluate a CLAGN label.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astroquery.sdss import SDSS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3sf.access.spectra import Spectrum  # noqa: E402
from p3sf.criteria.green_pixel import (  # noqa: E402
    GreenEpochSpectrum,
    VarianceProvenance,
    fit_green_epoch_line_spectrum_pyqsofit,
    measure_green_pixel_nsigma,
)
from p3sf.fitting.pyqsofit_driver import (  # noqa: E402
    fit_yang_hbeta_audit_pyqsofit,
)

PROCESSING_VERSION = "p3sf-d091"
RELEASE = "DR17"
SAS_BASE = "https://data.sdss.org/sas/dr17"
EXPECTED_COLUMNS = {"flux", "loglam", "ivar", "and_mask", "or_mask"}
ARCHIVE_FIELDS = [
    "plate", "mjd", "fiberID", "run2d", "z", "zWarning", "class", "subclass",
    "instrument", "survey", "programname", "snMedian", "ra", "dec", "specObjID",
]
GREEN_SCALE_PROVENANCE = "NO_RESCALE_EXACT_SDSS_PAIR_NO_SOURCE_OUTLIER_TRIGGER"


@dataclass(frozen=True)
class Endpoint:
    object_id: str
    role: str
    plate: int
    mjd: int
    fiber: int
    source_family: str
    expected_redshift: float


ENDPOINTS = (
    Endpoint("SDSSJ015957.64+003310.4", "bright", 403, 51871, 549, "LaMassa2015", 0.312),
    Endpoint("SDSSJ015957.64+003310.4", "faint", 3609, 55201, 524, "LaMassa2015", 0.312),
    Endpoint("SDSSJ012648.08-083948.0", "bright", 661, 52163, 604, "Ruan2016", 0.198),
    Endpoint("SDSSJ012648.08-083948.0", "faint", 2878, 54465, 377, "Ruan2016", 0.198),
    Endpoint("SDSSJ101152.98+544206.4", "bright", 945, 52652, 22, "Runnoe2016", 0.246),
    Endpoint("SDSSJ101152.98+544206.4", "faint", 8181, 57073, 827, "Runnoe2016", 0.246),
    Endpoint("SDSSJ000236.25-002724.8", "bright", 387, 51791, 110, "Potts2021", 0.291),
    Endpoint("SDSSJ000236.25-002724.8", "faint", 669, 52559, 306, "Potts2021", 0.291),
    Endpoint("SDSSJ135855.83+493414.2", "bright", 1670, 54553, 73, "Potts2021", 0.116),
    Endpoint("SDSSJ135855.83+493414.2", "faint", 1670, 53438, 61, "Potts2021", 0.116),
    Endpoint("SDSSJ021359.79+004226.81", "bright", 405, 51816, 458, "Green2022", 0.182),
    Endpoint("SDSSJ021359.79+004226.81", "faint", 9383, 58097, 829, "Green2022", 0.182),
)


def coordinates(object_id: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"SDSSJ(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)",
        object_id,
    )
    if match is None:
        raise ValueError(f"cannot parse IAU coordinates: {object_id}")
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra = 15.0 * (float(hh) + float(mm) / 60.0 + float(ss) / 3600.0)
    dec = float(dd) + float(dm) / 60.0 + float(ds) / 3600.0
    return ra, dec if sign == "+" else -dec


def angular_offset_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    mean_dec = np.deg2rad(0.5 * (dec1 + dec2))
    dx = (ra1 - ra2) * np.cos(mean_dec)
    dy = dec1 - dec2
    return float(3600.0 * np.hypot(dx, dy))


def archive_record(endpoint: Endpoint) -> pd.Series:
    table = SDSS.query_specobj(
        plate=endpoint.plate,
        mjd=endpoint.mjd,
        fiberID=endpoint.fiber,
        fields=ARCHIVE_FIELDS,
    )
    if table is None or len(table) != 1:
        raise RuntimeError(f"exact archive lookup failed for {endpoint}")
    return table.to_pandas().iloc[0]


def sas_url(endpoint: Endpoint, run2d: str) -> str:
    stem = f"spec-{endpoint.plate:04d}-{endpoint.mjd:05d}-{endpoint.fiber:04d}.fits"
    if str(run2d) == "26":
        return f"{SAS_BASE}/sdss/spectro/redux/26/spectra/lite/{endpoint.plate:04d}/{stem}"
    return f"{SAS_BASE}/eboss/spectro/redux/{run2d}/spectra/lite/{endpoint.plate:04d}/{stem}"


def fetch(url: str, destination: Path) -> tuple[int, str, str]:
    if destination.exists():
        payload = destination.read_bytes()
        return len(payload), hashlib.sha256(payload).hexdigest(), "PRESENT_CHECKSUM_REUSED"
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return len(response.content), hashlib.sha256(response.content).hexdigest(), "DOWNLOADED_EXACT"


def _load_qc_module():
    path = ROOT / "00_scripts/06_spectrum_quality_control.py"
    spec = importlib.util.spec_from_file_location("p3sf_native_qc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen native-QC module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_spectrum(path: Path, redshift: float, spectrum_id: str) -> Spectrum:
    with fits.open(path) as hdul:
        data = hdul[1].data
        wave = 10.0 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        ivar = np.asarray(data["ivar"], dtype=float)
    error = np.full(ivar.shape, np.nan)
    positive = ivar > 0
    error[positive] = 1.0 / np.sqrt(ivar[positive])
    return Spectrum(
        wave, flux, error, redshift, mask=(~positive).astype(int),
        meta={"spectrum_id": spectrum_id},
    )


def zero_flux_green_feasibility(epoch: GreenEpochSpectrum) -> GreenEpochSpectrum:
    """Retain only preprocessing/coverage/variance state, hiding line outcome."""
    return GreenEpochSpectrum(
        wavelength_rest=epoch.wavelength_rest,
        line_flux=np.zeros_like(epoch.line_flux),
        variance=epoch.variance,
        variance_provenance=epoch.variance_provenance,
        preprocessing_valid=epoch.preprocessing_valid,
        invalid_reason=epoch.invalid_reason,
        scale_factor=epoch.scale_factor,
        scale_provenance=epoch.scale_provenance,
    )


def main() -> int:
    raw_dir = ROOT / "03_spectra/raw_sdss/d091_expansion"
    out_dir = ROOT / "05_analysis/q1_design"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    endpoint_cache: dict[tuple[str, str], tuple[pd.Series, Spectrum, Path]] = {}

    for endpoint in ENDPOINTS:
        record = archive_record(endpoint)
        exact_metadata = (
            int(record.plate), int(record.mjd), int(record.fiberID)
        ) == (endpoint.plate, endpoint.mjd, endpoint.fiber)
        run2d = str(record.run2d)
        url = sas_url(endpoint, run2d)
        filename = (
            f"{endpoint.object_id}__{endpoint.role}__"
            f"spec-{endpoint.plate:04d}-{endpoint.mjd:05d}-{endpoint.fiber:04d}.fits"
        )
        destination = raw_dir / filename
        n_bytes, digest, status = fetch(url, destination)
        spectrum_id = f"{endpoint.plate}-{endpoint.mjd}-{endpoint.fiber}"
        expected_ra, expected_dec = coordinates(endpoint.object_id)

        with fits.open(destination) as hdul:
            n_hdus = len(hdul)
            primary = hdul[0].header
            names = set(hdul[1].columns.names)
            structure_ok = len(hdul) >= 3 and names >= EXPECTED_COLUMNS
            header_exact = (
                int(primary.get("PLATEID", -1)),
                int(primary.get("MJD", -1)),
                int(primary.get("FIBERID", -1)),
            ) == (endpoint.plate, endpoint.mjd, endpoint.fiber)
            coadd_rows = len(hdul[1].data)
            specobj = hdul[2].data[0]
            fits_z = float(specobj["Z"])
            fits_zwarning = int(specobj["ZWARNING"])
            fits_plate = int(specobj["PLATE"])
            fits_mjd = int(specobj["MJD"])
            fits_fiber = int(specobj["FIBERID"])
            specobj_exact = (fits_plate, fits_mjd, fits_fiber) == (
                endpoint.plate, endpoint.mjd, endpoint.fiber
            )
            and_mask_nonzero = int(np.count_nonzero(hdul[1].data["and_mask"]))
            or_mask_nonzero = int(np.count_nonzero(hdul[1].data["or_mask"]))

        coordinate_offset = angular_offset_arcsec(
            expected_ra, expected_dec, float(record.ra), float(record.dec)
        )
        coordinate_ok = coordinate_offset <= 1.0
        redshift_ok = abs(float(record.z) - endpoint.expected_redshift) <= 0.005
        redshift_product_ok = abs(fits_z - float(record.z)) <= 1.0e-5
        identity_pass = bool(
            exact_metadata and header_exact and specobj_exact and coordinate_ok
            and redshift_ok and redshift_product_ok and structure_ok
        )
        path_relative = str(destination.relative_to(ROOT))
        manifest_rows.append({
            "object": endpoint.object_id,
            "role": endpoint.role,
            "plate": endpoint.plate,
            "mjd": endpoint.mjd,
            "fiber": endpoint.fiber,
            "spectrum_id": spectrum_id,
            "survey_generation": f"{record.survey}/{record.instrument}/run2d={run2d}",
            "release": RELEASE,
            "source_product": "SDSS_SPECLITE_FITS",
            "source_url": url,
            "local_filename": path_relative,
            "byte_size": n_bytes,
            "sha256": digest,
            "acquisition_status": status if identity_pass else "ACQUIRED_IDENTITY_FAIL",
            "processing_version": PROCESSING_VERSION,
        })
        identity_rows.append({
            "object": endpoint.object_id,
            "role": endpoint.role,
            "expected_spectrum_id": spectrum_id,
            "archive_spectrum_id": f"{int(record.plate)}-{int(record.mjd)}-{int(record.fiberID)}",
            "specobj_id": str(record.specObjID),
            "archive_ra": float(record.ra),
            "archive_dec": float(record.dec),
            "iau_ra": expected_ra,
            "iau_dec": expected_dec,
            "coordinate_offset_arcsec": coordinate_offset,
            "coordinate_consistent": coordinate_ok,
            "archive_redshift": float(record.z),
            "product_redshift": fits_z,
            "expected_event_redshift": endpoint.expected_redshift,
            "redshift_consistent": redshift_ok and redshift_product_ok,
            "header_identifier_exact": header_exact,
            "specobj_identifier_exact": specobj_exact,
            "archive_identifier_exact": exact_metadata,
            "expected_product_structure": structure_ok,
            "n_hdus": n_hdus,
            "coadd_rows": coadd_rows,
            "and_mask_nonzero_pixels": and_mask_nonzero,
            "or_mask_nonzero_pixels": or_mask_nonzero,
            "zwarning_product": fits_zwarning,
            "identity_pass": identity_pass,
            "identity_failure_reason": "" if identity_pass else "IDENTITY_OR_STRUCTURE_CONFLICT",
        })
        endpoint_cache[(endpoint.object_id, endpoint.role)] = (
            record,
            read_spectrum(destination, float(record.z), spectrum_id),
            destination,
        )
        print(f"acquired/verified {endpoint.object_id} {endpoint.role} {spectrum_id}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    identity = pd.DataFrame(identity_rows)
    manifest["unique_file"] = ~manifest.local_filename.duplicated(keep=False)
    manifest["unique_checksum"] = ~manifest.sha256.duplicated(keep=False)
    identity["pair_redshift_delta"] = identity.groupby("object").archive_redshift.transform(
        lambda values: float(values.max() - values.min())
    )
    identity["pair_redshift_consistent"] = identity.pair_redshift_delta <= 0.001
    identity["identity_pass"] &= identity.pair_redshift_consistent
    manifest.to_csv(ROOT / "03_spectra/raw_sdss/manifest_d091_expansion.csv", index=False)
    identity.to_csv(out_dir / "endpoint_identity_validation_d091.csv", index=False)

    if not manifest.unique_file.all() or not manifest.unique_checksum.all():
        raise RuntimeError("D-091 requires twelve unique files and checksums")
    if not identity.identity_pass.all():
        raise RuntimeError("D-091 endpoint identity gate failed")

    qc_module = _load_qc_module()
    qc_rows: list[dict[str, object]] = []
    yang_rows: list[dict[str, object]] = []
    green_epochs: dict[tuple[str, str], GreenEpochSpectrum] = {}
    vendor = ROOT / "06_fitting/pyqsofit/vendor/PyQSOFit"
    for endpoint in ENDPOINTS:
        record, spectrum, destination = endpoint_cache[(endpoint.object_id, endpoint.role)]
        row = pd.Series({
            "object_id": endpoint.object_id,
            "spectrum_id": f"{endpoint.plate}-{endpoint.mjd}-{endpoint.fiber}",
            "mjd": endpoint.mjd,
            "redshift": float(record.z),
            "zwarning": int(record.zWarning),
            "path": str(destination.relative_to(ROOT)),
        })
        assessed = qc_module.assess(spectrum, row, "SDSS")
        assessed.update({
            "role": endpoint.role,
            "plate": endpoint.plate,
            "fiber": endpoint.fiber,
            "valid_pixel_fraction": float(spectrum.good.mean()),
            "degradation_source_eligible": bool(assessed["qc_pass"]),
        })
        qc_rows.append(assessed)

        _, yang = fit_yang_hbeta_audit_pyqsofit(
            spectrum.wavelength,
            spectrum.flux,
            spectrum.error,
            redshift=spectrum.redshift,
            path=str(vendor),
        )
        yang_rows.append({
            "object": endpoint.object_id,
            "role": endpoint.role,
            "spectrum_id": row.spectrum_id,
            **asdict(yang),
        })
        ra, dec = coordinates(endpoint.object_id)
        green_epochs[(endpoint.object_id, endpoint.role)] = fit_green_epoch_line_spectrum_pyqsofit(
            spectrum.wavelength,
            spectrum.flux,
            spectrum.error,
            redshift=spectrum.redshift,
            path=vendor,
            variance_provenance=VarianceProvenance.NATIVE_SDSS_SUPPORTED,
            ra=ra,
            dec=dec,
            scale_factor=1.0,
            scale_provenance=GREEN_SCALE_PROVENANCE,
        )
        print(f"native measurement audit {endpoint.object_id} {endpoint.role}", flush=True)

    qc = pd.DataFrame(qc_rows)
    yang = pd.DataFrame(yang_rows)
    green_rows: list[dict[str, object]] = []
    validity_rows: list[dict[str, object]] = []
    for object_id, group in qc.groupby("object_id", sort=False):
        bright_green = green_epochs[(object_id, "bright")]
        faint_green = green_epochs[(object_id, "faint")]
        feasibility = measure_green_pixel_nsigma(
            zero_flux_green_feasibility(bright_green),
            zero_flux_green_feasibility(faint_green),
        )
        for role, epoch in (("bright", bright_green), ("faint", faint_green)):
            green_rows.append({
                "object": object_id,
                "role": role,
                "preprocessing_valid": epoch.preprocessing_valid,
                "invalid_reason": epoch.invalid_reason,
                "variance_provenance": VarianceProvenance.NATIVE_SDSS_SUPPORTED.value,
                "n_output_pixels": len(epoch.wavelength_rest),
                "pair_zero_flux_feasibility_available": feasibility.measurement_available,
                "pair_zero_flux_feasibility_reason": feasibility.invalid_reason,
            })
        object_identity = bool(identity.loc[identity.object == object_id, "identity_pass"].all())
        object_qc = bool(group.qc_pass.all())
        object_green = bool(
            bright_green.preprocessing_valid
            and faint_green.preprocessing_valid
            and feasibility.measurement_available
        )
        endpoint_valid = object_identity and object_qc and object_green
        validity_rows.append({
            "transition_id": object_id,
            "reference_evidence_status": "GOLD_REFERENCE_EVIDENCE_PASS",
            "identity_pair_valid": object_identity,
            "both_endpoints_native_qc_valid": object_qc,
            "green_preprocessing_variance_feasible": object_green,
            "yang_bright_native_fit_valid": bool(
                yang[(yang.object == object_id) & (yang.role == "bright")].fit_valid.item()
            ),
            "yang_faint_native_fit_valid": bool(
                yang[(yang.object == object_id) & (yang.role == "faint")].fit_valid.item()
            ),
            "q1_native_endpoint_valid": endpoint_valid,
            "q1_eligible": endpoint_valid,
            "technical_exclusion_reason": "" if endpoint_valid else "IDENTITY_QC_OR_GREEN_DOMAIN_FAIL",
        })

    green = pd.DataFrame(green_rows)
    validity = pd.DataFrame(validity_rows)
    qc.to_csv(out_dir / "native_qc_d091.csv", index=False)
    yang.to_csv(out_dir / "yang_native_fit_audit_d091.csv", index=False)
    green.to_csv(out_dir / "green_native_feasibility_d091.csv", index=False)
    validity.to_csv(out_dir / "added_gold_validity_d091.csv", index=False)
    print(f"endpoint-valid added GOLD: {int(validity.q1_eligible.sum())}/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
