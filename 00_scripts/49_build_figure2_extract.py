#!/usr/bin/env python
"""Stage 49 -- compact public inputs for Figure 2.

Figure 2 shows one transition to make the intervention concrete. It enters no
statistic, but it is drawn from production records, and those records are not
redistributable as they stand: the per-spectrum fit archive is 707 MB and records
absolute paths, and the left panel reads the two archival endpoint spectra.

This stage writes exactly what the figure reads, and nothing more:

    figure2_selection_d094.csv       native faint-epoch S/N for the 27 paired
                                     faint-only transitions, so the outcome-blind
                                     selection rule is re-executed, not asserted
    figure2_example_arrays_d094.csv  the five plotted series for the selected
                                     transition, over rest-frame 4600-5200 A
    figure2_example_meta_d094.csv    its redshift and native faint-epoch S/N

The archival spectra are SDSS public data-release products (SDSS-V DR19 and SDSS
legacy), which SDSS places in the public domain with an acknowledgment request.

Values are written at 17 significant digits and read back with round-trip
parsing; the run aborts unless every series, and the selection, reproduce
bit-for-bit. Nothing frozen is modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "05_analysis" / "q1_production" / "d094" / "raw"
D095 = ROOT / "05_analysis" / "q1_production" / "d095" / "tables"
FULL = RAW / "spectrum_fit_results_d094.parquet"
SELECTION = RAW / "figure2_selection_d094.csv"
ARRAYS = RAW / "figure2_example_arrays_d094.csv"
META = RAW / "figure2_example_meta_d094.csv"
EXPECTED = "J075934.95+322143.3"
LO, HI = 4600.0, 5200.0


def _observed(path: str, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Identical to the figure generator: frozen positive-IVAR rule, rest frame."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = {n.lower(): n for n in data.names}
        wave = np.power(10.0, np.asarray(data[names["loglam"]], dtype=float))
        flux = np.asarray(data[names["flux"]], dtype=float)
        ivar = np.asarray(data[names["ivar"]], dtype=float)
    good = np.isfinite(ivar) & (ivar > 0) & np.isfinite(flux) & np.isfinite(wave)
    return wave[good] / (1.0 + z), flux[good]


def _window(w: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = (w >= LO) & (w <= HI)
    return w[keep], f[keep]


def main() -> int:
    if not FULL.exists():
        print(f"ABORT: {FULL.name} is required to build the Figure 2 extract and is absent.")
        return 1
    paired = pd.read_csv(D095 / "green_common_support_snr_paired_transitions_d095.csv")
    faint_only = sorted(paired[paired.arm == "faint_only"].transition_id.unique())
    table = pq.read_table(FULL).to_pandas()
    faint = table[(table.task_kind == "faint_shared") & (table.transition_id.isin(faint_only))]
    native = faint.groupby("transition_id").original_snr.first()
    assert len(native) == 27

    offset = (native - float(native.median())).abs()
    example = sorted(offset[offset == offset.min()].index)[0]
    assert example == EXPECTED, example

    rows = table[table.transition_id == example]
    bright = rows[rows.task_kind == "native_bright"].iloc[0]
    d10 = rows[(rows.task_kind == "faint_shared") & (rows.target_snr == 10.0) & (rows.realization == 0.0)].iloc[0]
    d5 = rows[(rows.task_kind == "faint_shared") & (rows.target_snr == 5.0) & (rows.realization == 0.0)].iloc[0]
    z = float(bright.redshift)

    series = {
        "observed_bright": _window(*_observed(str(bright.local_path), z)),
        "observed_faint": _window(*_observed(str(d10.local_path), z)),
        "line_bright": _window(np.asarray(bright.green_wavelength_rest, float), np.asarray(bright.green_line_flux, float)),
        "line_faint_snr10": _window(np.asarray(d10.green_wavelength_rest, float), np.asarray(d10.green_line_flux, float)),
        "line_faint_snr5": _window(np.asarray(d5.green_wavelength_rest, float), np.asarray(d5.green_line_flux, float)),
    }
    long = pd.concat([pd.DataFrame({"series": k, "wavelength_rest": w, "flux": f}) for k, (w, f) in series.items()],
                     ignore_index=True)
    long.to_csv(ARRAYS, index=False, float_format="%.17g")
    native.rename("original_snr").reset_index().to_csv(SELECTION, index=False, float_format="%.17g")
    pd.DataFrame([{"transition_id": example, "redshift": z, "faint_native_snr": float(d10.original_snr)}]).to_csv(
        META, index=False, float_format="%.17g")

    # Round-trip gate: the extract must reproduce every plotted value exactly.
    back = pd.read_csv(ARRAYS, float_precision="round_trip")
    for key, (w, f) in series.items():
        part = back[back.series == key]
        assert np.array_equal(part.wavelength_rest.to_numpy(), w) and np.array_equal(part.flux.to_numpy(), f), key
    sel = pd.read_csv(SELECTION, float_precision="round_trip").set_index("transition_id").original_snr
    assert sel.equals(native.rename("original_snr"))
    meta = pd.read_csv(META, float_precision="round_trip").iloc[0]
    assert meta.redshift == z and meta.faint_native_snr == float(d10.original_snr)

    print(f"verified: selection re-executes to {example}; all {len(series)} series round-trip bit-for-bit")
    for path in (SELECTION, ARRAYS, META):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:6.1f} KB  sha256 {digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
