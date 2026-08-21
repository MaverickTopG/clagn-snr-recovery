#!/usr/bin/env python
"""Stage 17 / Gate C0-D — acquire the 12 frozen clusters and measure S/N.

Execution only. Nothing here is designed or chosen: clusters, window, S/N
definition, strata and lifecycle are all frozen upstream (D-058 … D-066).

Restartable and idempotent. The manifest is the state, written durably after
every cluster, so an interruption resumes from it rather than from the
directory.

**No `k_sigma`, no IVAR calibration, no renormalisation, no correction of any
kind.** Native IVAR exactly as supplied.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402
from p3sf.spectral_domain import MIN_PIXELS  # noqa: E402

# cumulative/, not pernight/: pernight publishes only 716 of 6101 DR1 tiles, so
# 11 of the 12 frozen clusters 404 there. cumulative carries the same uncoadded
# spectra-* product for every tile. "thru{night}" accumulates exposures through
# that night; the frozen clusters are coadd_numnight = 1, and the per-row NIGHT
# check in measure_cluster rejects any file whose exposures span other nights.
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/tiles/cumulative"
W = (5000.0, 5200.0)                       # FROZEN (D-059)
SN_BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9)]   # FROZEN (D-056)
SN_LABELS = ["00-10", "10-20", "20-30", "30-50", "50+"]
MIN_COVERAGE_FRACTION = 0.5                # of W, on valid unmasked pixels


def url_for(tileid: int, night: int, petal: int) -> str:
    return f"{BASE}/{tileid}/{night}/spectra-{petal}-{tileid}-thru{night}.fits.gz"


def sn_label(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    for (lo, hi), lab in zip(SN_BINS, SN_LABELS, strict=True):
        if lo <= value < hi:
            return lab
    return "NA"


def load_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"targetid": str})
    return pd.DataFrame()


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)                      # atomic


def download(url: str, dest: Path) -> tuple[str, str]:
    """Return (status, detail). Idempotent: existing complete file is kept."""
    import requests

    if dest.exists() and dest.stat().st_size > 1_000_000:
        return "DOWNLOAD_COMPLETE", "already present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=3600) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
        tmp.replace(dest)
        return "DOWNLOAD_COMPLETE", ""
    except Exception as error:             # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return "DOWNLOAD_FAILED", f"{type(error).__name__}: {error}"


def measure_cluster(path: Path, cluster: pd.Series) -> tuple[str, str, pd.DataFrame]:
    """Validate the file, then measure W coverage and S/N per target-exposure."""
    from astropy.io import fits

    try:
        with fits.open(path, memmap=False) as hdul:
            names = [h.name for h in hdul]
            required = ["FIBERMAP", "B_WAVELENGTH", "B_FLUX", "B_IVAR", "B_MASK"]
            missing = [n for n in required if n not in names]
            if missing:
                return "INTEGRITY_FAILED", f"missing HDUs {missing}", pd.DataFrame()

            fibermap = hdul["FIBERMAP"].data
            flux = np.asarray(hdul["B_FLUX"].data, dtype=float)
            ivar = np.asarray(hdul["B_IVAR"].data, dtype=float)
            mask = np.asarray(hdul["B_MASK"].data)
            lam = np.asarray(hdul["B_WAVELENGTH"].data, dtype=float)

            if not (len(fibermap) == flux.shape[0] == ivar.shape[0] == mask.shape[0]):
                return "INTEGRITY_FAILED", "FIBERMAP/spectral row mismatch", pd.DataFrame()
            for col in ("TARGETID", "NIGHT", "EXPID"):
                if col not in fibermap.columns.names:
                    return "INTEGRITY_FAILED", f"no {col} provenance", pd.DataFrame()

            nights = np.unique(np.asarray(fibermap["NIGHT"]))
            if int(cluster.night) not in {int(n) for n in nights}:
                return "INTEGRITY_FAILED", f"night mismatch {nights[:3]}", pd.DataFrame()

            targetid = np.asarray(fibermap["TARGETID"])
            night = np.asarray(fibermap["NIGHT"])
            expid = np.asarray(fibermap["EXPID"])
            mjd = (np.asarray(fibermap["MJD"], dtype=float)
                   if "MJD" in fibermap.columns.names else np.full(len(fibermap), np.nan))

            in_w = (lam >= W[0]) & (lam <= W[1])
            n_w = int(in_w.sum())
            if n_w == 0:
                return "INTEGRITY_PASS", "no W pixels in product", pd.DataFrame()

            rows = []
            for i in range(len(fibermap)):
                good = in_w & (ivar[i] > 0) & (mask[i] == 0) & np.isfinite(flux[i])
                n_good = int(good.sum())
                if n_good == 0:
                    status, snr, span = "NO_W_COVERAGE", None, None
                else:
                    span = float(lam[good].max() - lam[good].min())
                    coverage = span / (W[1] - W[0])
                    if n_good < MIN_PIXELS:
                        status, snr = "INSUFFICIENT_W_COVERAGE", None
                    elif coverage < MIN_COVERAGE_FRACTION:
                        status, snr = "W_MASKED", None
                    else:
                        status = "W_COVERAGE_PASS"
                        snr = float(np.median(np.abs(flux[i][good]) * np.sqrt(ivar[i][good])))
                rows.append({
                    "targetid": str(targetid[i]), "night": int(night[i]),
                    "expid": int(expid[i]), "mjd": float(mjd[i]),
                    "tileid": int(cluster.tileid), "petal": int(cluster.petal),
                    "n_w_pixels": n_w, "n_valid_pixels": n_good,
                    "wavelength_span": span,
                    "window_coverage": (span / (W[1] - W[0])) if span is not None else None,
                    "coverage_status": status, "snr_pix_w": snr,
                })
            return "INTEGRITY_PASS", "", pd.DataFrame(rows)
    except Exception as error:             # noqa: BLE001
        return "INTEGRITY_FAILED", f"{type(error).__name__}: {error}", pd.DataFrame()


def main() -> int:
    root = project_root()
    final = root / "02_catalogs" / "final"
    raw = root / "03_spectra" / "raw_desi" / "c0_clusters"
    selected = pd.read_csv(final / "c0_selected_clusters.csv")

    manifest_path = final / "c0_acquisition_manifest.csv"
    measure_path = final / "c0_exposure_measurements.csv"
    manifest = load_manifest(manifest_path)
    done = set(manifest.cluster_id) if "cluster_id" in manifest else set()

    records = manifest.to_dict("records") if not manifest.empty else []
    measurements = (
        pd.read_csv(measure_path, dtype={"targetid": str}) if measure_path.exists()
        else pd.DataFrame()
    )

    print(f"{len(selected)} clusters selected; {len(done)} already in manifest\n", flush=True)

    for cluster in selected.itertuples():
        cid = f"{cluster.tileid}-{cluster.night}-{cluster.petal}"
        if cid in done:
            print(f"{cid}  skip (manifest)", flush=True)
            continue

        url = url_for(cluster.tileid, cluster.night, cluster.petal)
        dest = raw / f"spectra-{cluster.petal}-{cluster.tileid}-{cluster.night}.fits.gz"
        entry = {
            "cluster_id": cid, "tileid": cluster.tileid, "night": cluster.night,
            "petal": cluster.petal, "url": url, "local_path": str(dest.relative_to(root)),
            "n_targets_expected": cluster.n_targets,
            "tsnr2_low": cluster.n_low, "tsnr2_mid": cluster.n_mid,
            "tsnr2_high": cluster.n_high,
            "lifecycle": "EXPOSURES_RESOLVED", "failure_reason": "",
            "n_bytes": None, "sha256": None,
        }

        status, detail = download(url, dest)
        entry["lifecycle"] = status
        entry["failure_reason"] = detail
        if status == "DOWNLOAD_COMPLETE":
            entry["n_bytes"] = dest.stat().st_size
            entry["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            integ, idetail, frame = measure_cluster(dest, cluster)
            entry["lifecycle"] = integ
            entry["failure_reason"] = idetail
            if integ == "INTEGRITY_PASS" and not frame.empty:
                measurements = pd.concat([measurements, frame], ignore_index=True)
                save(measurements, measure_path)
                n_pass = int((frame.coverage_status == "W_COVERAGE_PASS").sum())
                entry["n_rows"] = len(frame)
                entry["n_w_pass"] = n_pass
                entry["lifecycle"] = "C0_ELIGIBLE" if n_pass else "W_COVERAGE_FAILED"

        records.append(entry)
        save(pd.DataFrame(records), manifest_path)
        print(f"{cid}  {entry['lifecycle']}  "
              f"{(entry['n_bytes'] or 0)/1e6:.0f} MB  "
              f"rows={entry.get('n_rows','-')} wpass={entry.get('n_w_pass','-')}"
              f"  {entry['failure_reason'][:60]}", flush=True)

    print("\nacquisition loop complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
