# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/emfit.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : c1121919233406d637e2baa2ea1bda4604420711590c56fab8cff820483dca9b
# =====================================================================
"""DESI emission-line VAC as a broad-Hβ outcome-availability proxy.

`desi_dr1.emfit` (FastSpecFit) provides DESI-epoch broad-Hβ fluxes. This is a
fast *availability* look — how many cohort sources have a detectable broad Hβ at
the DESI epoch and its significance distribution — **not** the primary outcome:
the confirmatory SDSS→DESI change requires our own uniform fit (avoiding the
cross-instrument systematic of mixing DESI's pipeline with SDSS's). Public Data
Lab TAP; no MyDB.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from p3sf.access.cohort_config import DATALAB_TAP


def emfit_query(*, hbeta_zmax: float = 0.8, modulus: int | None = None) -> str:
    """Broadened cohort LEFT-joined to the emfit broad-Hβ columns (public TAP)."""
    subsample = f" AND MOD(x.id1, {modulus})=0" if modulus is not None else ""
    return (
        "SELECT s.specobjid AS source_id, d.targetid, "
        "e.hb_b_flux AS broad_hbeta_flux, e.hb_b_flux_err AS broad_hbeta_flux_err\n"
        "FROM sdss_dr17.x1p5__specobj__desi_dr1__zpix AS x\n"
        "JOIN sdss_dr17.specobjall AS s ON s.specobjid = x.id1\n"
        "JOIN desi_dr1.zpix AS d ON x.id2 = d.id\n"
        "LEFT JOIN desi_dr1.emfit AS e ON e.targetid = d.targetid\n"
        "WHERE s.zwarning=0 AND s.scienceprimary=1 AND s.z>=0 "
        f"AND s.z<{hbeta_zmax} AND s.class IN ('GALAXY','QSO') AND d.zwarn=0{subsample}"
    )


def emfit_proxy(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-source DESI-epoch broad-Hβ availability and 3σ detection.

    A target may carry several emfit rows (multiple coadds); results are
    aggregated to one row per source: available/detected if *any* row is, and
    the maximum flux/significance.
    """
    frame = raw.copy()
    frame["source_id"] = frame["source_id"].astype(str)
    flux = pd.to_numeric(frame["broad_hbeta_flux"], errors="coerce")
    error = pd.to_numeric(frame["broad_hbeta_flux_err"], errors="coerce")
    frame["_flux"] = flux
    frame["_available"] = flux.notna() & error.notna() & (error > 0)
    frame["_significance"] = np.where(frame["_available"], flux / error, np.nan)
    frame["_detected"] = frame["_available"] & (flux > 3 * error)
    aggregated = frame.groupby("source_id", sort=True).agg(
        broad_hbeta_flux=("_flux", "max"),
        broad_hbeta_significance=("_significance", "max"),
        broad_hbeta_detected=("_detected", "any"),
        emfit_available=("_available", "any"),
    )
    return aggregated.reset_index()


def emfit_summary(proxy: pd.DataFrame) -> dict[str, int]:
    return {
        "n_total": int(len(proxy)),
        "emfit_available": int(proxy["emfit_available"].sum()),
        "broad_hbeta_detected": int(proxy["broad_hbeta_detected"].sum()),
    }


def fetch_emfit(query: str, *, endpoint: str = DATALAB_TAP, timeout: float = 590.0) -> pd.DataFrame:
    """Execute the emfit availability query (network I/O)."""
    response = requests.post(
        endpoint,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=timeout,
    )
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))
