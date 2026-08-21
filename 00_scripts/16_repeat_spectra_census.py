#!/usr/bin/env python
"""Stage 16 / Gate C0 — repeat-spectrum census.

**Census only. No uncertainty correction is computed or implemented here.**

Gate C calibrates the pixel-noise model from data where the astrophysical signal
should not change. This stage establishes *what repeat data exist*, so the
complexity of `k_sigma` is chosen from evidence rather than from the three
variables the design happened to name (D-054).

Two quantities are counted **separately** throughout, per D-055:

* ``n_targets``   — unique objects. This is ``N_effective``.
* ``n_pairs``     — usable repeat pairs. A target with 6 exposures yields 15
                    pairs but is still **one** independent calibrator.

Baseline regimes are also counted separately, not merely tagged, because
within-sequence repeats probe photon-noise scaling while cross-night repeats
additionally absorb night-to-night calibration variation. Reporting one number
across both would conflate error sources.

Calibration population priority (D-054): stars first, then stable galaxies, then
short-baseline quasars. AGN are **not** primary — for an AGN pair the difference
carries intrinsic variability that masquerades as underestimated IVAR and biases
`k_sigma` upward.

Usage
-----
    uv run python 00_scripts/16_repeat_spectra_census.py
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

SCRIPT = "16_repeat_spectra_census.py"

#: S/N bins spanning the regime where the DESI DR1 variance bug turns on
#: (negligible at low S/N, significant at 20-30, up to a factor 5 near 100).
SN_BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 10_000)]


SDSS_CENSUS = """
WITH rep AS (
  SELECT bestobjid, class,
         COUNT(*)               AS n_spectra,
         COUNT(DISTINCT mjd)    AS n_nights,
         MAX(mjd) - MIN(mjd)    AS baseline_days,
         MIN(snmedian)          AS sn_min,
         MAX(snmedian)          AS sn_max
  FROM sdss_dr17.specobjall
  WHERE bestobjid > 0 AND zwarning = 0 AND snmedian > 0
  GROUP BY bestobjid, class
  HAVING COUNT(*) > 1
)
SELECT class,
       CASE WHEN sn_min < 10 THEN '00-10'
            WHEN sn_min < 20 THEN '10-20'
            WHEN sn_min < 30 THEN '20-30'
            WHEN sn_min < 50 THEN '30-50'
            ELSE '50+' END                                   AS sn_bin,
       CASE WHEN baseline_days = 0 THEN 'same_night'
            WHEN baseline_days <= 30 THEN 'short_baseline'
            ELSE 'long_baseline' END                         AS regime,
       COUNT(*)                                              AS n_targets,
       SUM(n_spectra)                                        AS n_spectra,
       SUM(n_spectra * (n_spectra - 1) / 2)                  AS n_pairs,
       SUM(CASE WHEN n_nights > 1 THEN 1 ELSE 0 END)         AS n_targets_multinight,
       ROUND(AVG(n_nights)::numeric, 2)                      AS mean_nights
FROM rep
GROUP BY class, sn_bin, regime
ORDER BY class, sn_bin, regime
"""

# DESI: repeats visible in Data Lab are targets with more than one zpix coadd
# (distinct survey/program). Per-exposure spectra live in cframe products in the
# archive, not in Data Lab, so coadd_numexp/coadd_numnight are reported as the
# *within-coadd* exposure structure rather than as independent repeat spectra.
DESI_CENSUS = """
WITH rep AS (
  SELECT CAST(targetid AS VARCHAR)  AS targetid,
         MAX(spectype)              AS spectype,
         COUNT(*)                   AS n_coadds,
         SUM(coadd_numexp)          AS total_exposures,
         SUM(coadd_numnight)        AS total_nights,
         MAX(max_mjd) - MIN(min_mjd) AS baseline_days,
         MIN(tsnr2_lrg_z)           AS tsnr_min
  FROM desi_dr1.zpix
  WHERE zwarn = 0 AND coadd_fiberstatus = 0
  GROUP BY targetid
  HAVING COUNT(*) > 1
)
SELECT spectype,
       CASE WHEN baseline_days = 0 THEN 'same_night'
            WHEN baseline_days <= 30 THEN 'short_baseline'
            ELSE 'long_baseline' END              AS regime,
       COUNT(*)                                   AS n_targets,
       SUM(n_coadds)                              AS n_coadds,
       SUM(n_coadds * (n_coadds - 1) / 2)         AS n_pairs,
       SUM(total_exposures)                       AS total_exposures,
       ROUND(AVG(total_nights)::numeric, 2)       AS mean_nights
FROM rep
GROUP BY spectype, regime
ORDER BY spectype, regime
"""


def run_query(sql: str, label: str) -> pd.DataFrame:
    from dl import queryClient as qc

    print(f"  querying {label} ...", flush=True)
    text = qc.query(sql=sql, fmt="csv", timeout=1800)
    return pd.read_csv(io.StringIO(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdss-only", action="store_true")
    args = parser.parse_args()

    root = project_root()
    out_dir = root / "02_catalogs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("GATE C0 — REPEAT-SPECTRUM CENSUS")
    print("=" * 78)
    print("Census only. No uncertainty correction is computed here.")
    print()

    sdss = run_query(SDSS_CENSUS, "SDSS DR17 specObjAll")
    sdss.to_csv(out_dir / "c0_repeat_census_sdss.csv", index=False)

    print()
    print("SDSS DR17 — unique targets with >1 good spectrum")
    print("-" * 78)
    pivot = sdss.pivot_table(index=["class", "regime"], columns="sn_bin",
                             values="n_targets", aggfunc="sum", fill_value=0)
    print(pivot.to_string())
    print()
    print("SDSS totals by class (targets vs pairs kept separate — D-055):")
    totals = sdss.groupby("class")[["n_targets", "n_spectra", "n_pairs",
                                    "n_targets_multinight"]].sum()
    print(totals.to_string())

    desi = pd.DataFrame()
    if not args.sdss_only:
        print()
        try:
            desi = run_query(DESI_CENSUS, "DESI DR1 zpix")
            desi.to_csv(out_dir / "c0_repeat_census_desi.csv", index=False)
            print()
            print("DESI DR1 — targets with >1 coadd (distinct survey/program)")
            print("-" * 78)
            print(desi.to_string(index=False))
        except Exception as error:  # noqa: BLE001
            print(f"  DESI query failed: {type(error).__name__}: {str(error)[:200]}")

    print()
    print("=" * 78)
    print("NOTES BEARING ON THE C0-A / C0-B / C0-C DECISION")
    print("=" * 78)
    print("* SDSS S/N bin uses the WORSE spectrum of each pair (sn_min): a pair can")
    print("  only calibrate the regime both members reach.")
    print("* DESI per-exposure spectra (cframe) are NOT in Data Lab. The DESI rows")
    print("  here count multi-coadd repeats only; coadd_numexp/coadd_numnight")
    print("  describe within-coadd structure and are not independent repeats.")
    print("  Acquiring cframe products is a separate step if C0-A is wanted for DESI.")
    print("* Star counts are the ones that matter first (D-054): AGN pairs carry")
    print("  intrinsic variability that biases k_sigma upward.")
    print("* n_pairs never substitutes for n_targets. N_effective = n_targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
