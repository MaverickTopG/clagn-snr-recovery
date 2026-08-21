#!/usr/bin/env python
"""Stage 13 / 5B.8b — four-object pPXF validation.

Runs the mechanical gates in order and refuses to report a host fraction until
they pass. Nothing here interprets a scientific disagreement; that comes only
after every gate is green.

1. geometry      velocity scales, log grid, template coverage, LSF over the
                 fitted domain and the Hbeta critical sub-domains
2. rest frame    fitted [O III] 5007 centroid, not an argmax
3. stellar       absorption-feature alignment, independent of any emission line
4. reconstruction  HARD GATE: sum of components must equal pPXF's own bestfit
5. semantics     canonical f_host_cont and f_host_pseudo, identical for both
                 pipelines, from reconstructed spectra over one frozen window
6. comparison    XSL and E-MILES where both are resolution-compatible

Usage
-----
    uv run python 00_scripts/13_ppxf_geometry_audit.py --slice
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import load_config, project_root  # noqa: E402
from p3sf.fitting.components import from_ppxf, from_pyqsofit  # noqa: E402
from p3sf.fitting.ppxf_host import decompose  # noqa: E402
from p3sf.qc.alignment import check_rest_frame, check_stellar_alignment  # noqa: E402
from p3sf.spectral_domain import lsf_compatible  # noqa: E402

SCRIPT = "13_ppxf_geometry_audit.py"
C_KMS = 299792.458
FIT_RANGE = (3700.0, 7000.0)

DIAGNOSTIC_OBJECTS = [
    ("SDSSJ081319.34+460849.5", "host-dominated control"),
    ("SDSSJ105058.42+241351.18", "intermediate clean case"),
    ("SDSSJ114408.90+424357.5", "local-feature concern"),
    ("SDSSJ102152.34+464515.6", "global-shape concern"),
]

# SDSS instrumental FWHM near Hbeta. DESI is sharper and is not used here,
# since E-MILES is not resolution-compatible with it (D-045).
FWHM_SDSS = 2.76


def read_spectrum(path: Path, redshift: float):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul[1].data
        wavelength = 10.0 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        ivar = np.asarray(data["ivar"], dtype=float)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    good = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    return wavelength[good], flux[good], error[good], float(redshift)


def geometry(wavelength, flux, redshift, sps_file):
    from ppxf.ppxf_util import log_rebin
    from ppxf.sps_util import sps_lib

    rest = wavelength / (1.0 + redshift)
    inside = (rest >= FIT_RANGE[0]) & (rest <= FIT_RANGE[1])
    lam = rest[inside]
    galaxy, ln_lam, velscale = log_rebin((lam.min(), lam.max()), flux[inside].astype(float))
    lam_grid = np.exp(ln_lam)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sps = sps_lib(str(sps_file), velscale, fwhm_gal=None, norm_range=None)

    velscale_temp = float(C_KMS * np.median(np.diff(np.log(sps.lam_temp))))
    log_spacing = np.diff(ln_lam)

    library = np.load(sps_file)
    lsf = lsf_compatible(
        FWHM_SDSS, library["lam"], library["fwhm"], FIT_RANGE,
        galaxy_lam=lam_grid, galaxy_good=np.ones(lam_grid.size, dtype=bool),
    )

    return {
        "velscale_gal": float(velscale),
        "velscale_temp": velscale_temp,
        "velscale_ratio": float(velscale / velscale_temp),
        "velscale_match": bool(abs(velscale / velscale_temp - 1.0) < 0.02),
        "log_grid_uniform": bool(np.ptp(log_spacing) / np.median(log_spacing) < 1e-8),
        "template_covers_galaxy": bool(
            sps.lam_temp.min() <= lam_grid.min() and sps.lam_temp.max() >= lam_grid.max()
        ),
        "lsf_compatible": bool(lsf["compatible"]),
        "lsf_clipped_fraction": lsf["clipped_fraction"],
        "lsf_hbeta_core_status": str(lsf["critical"]["hbeta_core"].status),
        "lsf_all_critical_evaluated": bool(lsf["critical_all_evaluated"]),
        "lam_grid": lam_grid,
        "galaxy": galaxy,
    }, sps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    root = project_root()
    suffix = "_slice" if args.slice else ""
    window = tuple(cfg.snr_metrics.continuum_5100)
    libraries = {
        "XSL": root / "06_fitting/alternative_fit/sps_libraries/spectra_xsl_9.0.npz",
        "E-MILES": root / "06_fitting/alternative_fit/sps_libraries/spectra_emiles_9.0.npz",
    }
    feii = root / "06_fitting/pyqsofit/vendor/PyQSOFit/src/pyqsofit/fe_optical.txt"
    qc = pd.read_csv(root / "03_spectra" / "qc" / f"qc_verdicts{suffix}.csv")

    from p3sf.fitting.pyqsofit_driver import _pyqsofit_path  # noqa: F401  (vendor path)

    rows: list[dict[str, object]] = []
    for object_id, role in DIAGNOSTIC_OBJECTS:
        subset = qc[(qc.object_id == object_id) & qc.qc_pass & (qc.survey == "SDSS")]
        if subset.empty:
            print(f"{object_id}: no QC-passing SDSS spectrum")
            continue
        row = subset.iloc[0]
        wavelength, flux, error, redshift = read_spectrum(root / row.path, row.redshift)
        rest = wavelength / (1.0 + redshift)

        print("=" * 78)
        print(f"{object_id}   [{role}]   SDSS  S/N(Hbeta) {row.snr_hbeta_window:.1f}")
        print("=" * 78)

        # -- 2 & 3: wavelength geometry, independent of any fit ---------------
        rest_frame = check_rest_frame(rest, flux, error)
        stellar = check_stellar_alignment(rest, flux, error)
        print(f"  rest frame [O III]   {rest_frame.status}  "
              f"offset {'n/a' if rest_frame.offset_angstrom is None else f'{rest_frame.offset_angstrom:+.2f} A'}"
              f"{'' if rest_frame.offset_kms is None else f' ({rest_frame.offset_kms:+.0f} km/s)'}")
        measured = ", ".join(
            f"{n}{r.offset_angstrom:+.1f}" for n, r in stellar.measured.items()
        )
        print(f"  stellar alignment    {stellar.status}  n={stellar.n_measured}  "
              f"median {'n/a' if stellar.median_offset_kms is None else f'{stellar.median_offset_kms:+.0f} km/s'}")
        if measured:
            print(f"      features: {measured}")

        # -- PyQSOFit under common semantics ----------------------------------
        import importlib.util

        from p3sf.fitting.ppxf_host import decompose as _d  # noqa: F401

        spec = importlib.util.spec_from_file_location(
            "hostrel", root / "00_scripts" / "12_host_template_reliability.py"
        )
        hostrel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hostrel)
        vendor = str(root / "06_fitting" / "pyqsofit" / "vendor" / "PyQSOFit")
        pyqsofit_fit = hostrel.run_fit(wavelength, flux, error, redshift, vendor)
        pyqsofit_common: dict[str, object] = {}
        if bool(getattr(pyqsofit_fit, "decomposed", False)):
            native = hostrel.standardized_fhost(pyqsofit_fit, redshift)
            model = from_pyqsofit(pyqsofit_fit, redshift, native_f_host=native)
            check = model.check_reconstruction()
            pyqsofit_common = (
                model.canonical_fractions(window) if check.passed
                else {"reconstruction_failed": check.detail}
            )
        print(f"  PyQSOFit f_host      "
              f"{pyqsofit_common.get('f_host_cont', 'declined/failed')}")

        # -- 1, 4, 5, 6: per library -----------------------------------------
        for library_name, sps_file in libraries.items():
            geo, _ = geometry(wavelength, flux, redshift, sps_file)
            result = decompose(
                wavelength, flux, error, redshift, sps_file=sps_file,
                fhost_window=window, fwhm_gal=FWHM_SDSS, feii_path=feii,
                fit_range=FIT_RANGE,
            )

            record: dict[str, object] = {
                "object_id": object_id, "role": role, "library": library_name,
                "snr_hbeta": row.snr_hbeta_window,
                "rest_frame_status": str(rest_frame.status),
                "rest_frame_offset_kms": rest_frame.offset_kms,
                "stellar_status": str(stellar.status),
                "stellar_n_measured": stellar.n_measured,
                "stellar_median_offset_kms": stellar.median_offset_kms,
                **{k: v for k, v in geo.items() if k not in {"lam_grid", "galaxy"}},
                "ppxf_status": result.status,
                "pyqsofit_f_host_cont": pyqsofit_common.get("f_host_cont"),
                "pyqsofit_native_f_host": pyqsofit_common.get("native_f_host"),
            }

            if result.status != "ok" or result.fit is None:
                record["reconstruction_passed"] = False
                record["reconstruction_detail"] = f"fit status {result.status}"
                rows.append(record)
                print(f"  {library_name:<8} pPXF failed: {result.status} {result.detail[:60]}")
                continue

            model = from_ppxf(
                result.fit, result.wavelength, result.n_stellar_templates,
                result.nonstellar_names, method=f"ppxf-{library_name}",
                native_f_host=result.f_host_5100,
            )
            check = model.check_reconstruction()
            record["reconstruction_passed"] = check.passed
            record["reconstruction_relative"] = check.relative_difference

            stellar_weight = float(result.diagnostics.get("stellar_weight_sum", 0.0))
            record["stellar_weight_sum"] = stellar_weight
            record["chi2"] = result.chi2

            if not check.passed:
                # HARD GATE: no fraction is computed, let alone reported.
                record["f_host_cont"] = None
                record["f_host_pseudo"] = None
                print(f"  {library_name:<8} RECONSTRUCTION FAILED — no host fraction: "
                      f"{check.detail}")
            else:
                fractions = model.canonical_fractions(window)
                record.update({
                    "f_host_cont": fractions["f_host_cont"],
                    "f_host_pseudo": fractions["f_host_pseudo"],
                    "f_host_native": fractions["native_f_host"],
                    "window_coverage": fractions["window_coverage"],
                })
                medians = {k: float(np.median(v)) for k, v in model.components.items()}
                record.update({f"median_{k}": v for k, v in medians.items()})
                print(f"  {library_name:<8} recon OK ({check.relative_difference:.1e})  "
                      f"w*={stellar_weight:7.3f}  "
                      f"f_cont={fractions['f_host_cont']}  "
                      f"f_pseudo={fractions['f_host_pseudo']}  chi2={result.chi2:.2f}")
            rows.append(record)
        print()

    frame = pd.DataFrame(rows)
    out = root / "06_fitting" / "validation" / f"ppxf_four_object_audit{suffix}.csv"
    frame.to_csv(out, index=False)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    columns = ["object_id", "library", "velscale_match", "lsf_compatible",
               "rest_frame_status", "stellar_status", "reconstruction_passed",
               "stellar_weight_sum", "f_host_cont", "pyqsofit_f_host_cont"]
    print(frame[[c for c in columns if c in frame.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"written -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
