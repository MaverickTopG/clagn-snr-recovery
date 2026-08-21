# =====================================================================
# VENDORED CODE — do not sync with upstream; modify freely here.
#   source repo : CLAGN-WISE-ApJ (author's earlier project)
#   source file : src/clagn_apj/spectra_direct.py
#   upstream rev: c3bd1d4ece9f64ccf7ab592880ed7da7832e6f31
#   copied      : 2026-08-14
#   sha256(src) : 0f88d5457fe51bb2c4b2c64f10f40e6d198c11b519b352a3a1cea197f52a16be
# =====================================================================
"""SPARCL-free spectra provider: SDSS SAS + DESI mirror direct FITS.

SPARCL (NOIRLab's unified spectra service) is convenient but a single flaky
point of failure. This module retrieves the same two spectra straight from the
survey file servers instead:

* SDSS/eBOSS -- the ``lite`` coadd FITS from the SDSS Science Archive Server
  (``data.sdss.org``), addressed by plate/mjd/fiber/run2d.
* DESI DR1 -- the per-target row of the healpix ``coadd`` FITS from the DESI DR1
  mirror (``webdav-hdfs.pic.es``; the ``data.desi.lbl.gov/public/dr1`` mount is
  currently 404), read lazily over HTTP range requests so we transfer ~MB per
  target instead of the full ~700 MB coadd.

Identifiers come from public Data Lab TAP (which stays up when SPARCL is down).
Each reader returns a :class:`Spectrum` ready for the PyQSOFit fitter, so the
outcome pipeline is identical to the SPARCL path.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests

SDSS_SAS = "https://data.sdss.org/sas/dr17"
DESI_MIRROR = "https://webdav-hdfs.pic.es/data/public/DESI/DR1"
DESI_REDUX = "iron"  # DR1 reduction


@dataclass(frozen=True)
class Spectrum:
    """A single-epoch spectrum in observed frame, ready for PyQSOFit.

    ``mask`` and ``meta`` are Paper 3 additions to the vendored contract. The
    mask must travel with the spectrum through every counterfactual, because
    the whole point of using empirical noise is to preserve real skyline and
    masking structure rather than replacing it with white noise.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    redshift: float
    mask: np.ndarray | None = None
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def good(self) -> np.ndarray:
        """Finite, positive-error, unmasked pixels."""
        ok = np.isfinite(self.wavelength) & np.isfinite(self.flux)
        ok &= np.isfinite(self.error) & (self.error > 0)
        if self.mask is not None:
            ok &= np.asarray(self.mask) == 0
        return ok

    def rest_wavelength(self) -> np.ndarray:
        return self.wavelength / (1.0 + self.redshift)


def identifier_query(source_ids: list[str]) -> str:
    """ADQL to resolve SDSS (plate/mjd/fiber/run2d) + DESI (targetid/healpix/...)
    identifiers for a batch of cohort ``specobjid`` source ids."""
    inlist = ",".join(str(s) for s in source_ids)
    return (
        "SELECT s.specobjid AS source_id, s.plate, s.mjd, s.fiberid, s.run2d, s.z AS sdss_z, "
        "d.targetid, d.healpix, d.survey, d.program, d.z AS desi_z\n"
        "FROM sdss_dr17.x1p5__specobj__desi_dr1__zpix AS x\n"
        "JOIN sdss_dr17.specobjall AS s ON s.specobjid = x.id1\n"
        "JOIN desi_dr1.zpix AS d ON x.id2 = d.id\n"
        f"WHERE s.specobjid IN ({inlist})"
    )


def sdss_sas_url(plate: int, mjd: int, fiberid: int, run2d: str) -> str:
    """SAS ``lite`` spectrum URL. eBOSS reductions (run2d ``vX_...``) live under
    the ``eboss`` tree; legacy SDSS-I/II reductions under the ``sdss`` tree."""
    tree = "eboss" if str(run2d).startswith("v") else "sdss"
    return (
        f"{SDSS_SAS}/{tree}/spectro/redux/{run2d}/spectra/lite/"
        f"{int(plate):04d}/spec-{int(plate):04d}-{int(mjd)}-{int(fiberid):04d}.fits"
    )


def desi_coadd_url(survey: str, program: str, healpix: int) -> str:
    """DESI DR1 healpix ``coadd`` FITS URL on the mirror."""
    hp = int(healpix)
    return (
        f"{DESI_MIRROR}/spectro/redux/{DESI_REDUX}/healpix/"
        f"{survey}/{program}/{hp // 100}/{hp}/coadd-{survey}-{program}-{hp}.fits"
    )


def _http_size(url: str, timeout: float = 60.0) -> int:
    """True byte size via a range probe (the mirror's HEAD reports 0)."""
    r = requests.get(url, headers={"Range": "bytes=0-1"}, timeout=timeout)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[-1])


def _open_coadd(url: str, timeout: float) -> Any:
    """Open a healpix coadd for lazy section reads.

    Remote URLs go through fsspec with a readahead cache so only the sections we
    touch cross the network. A plain filesystem path opens directly -- the HTTP size
    probe would fail on one, and locally cached coadds are worth supporting for the
    same reason bulk retrieval exists.
    """
    if not str(url).startswith(("http://", "https://")):
        return str(url)
    import fsspec  # type: ignore[import-untyped]

    fs = fsspec.filesystem("http")
    return fs.open(url, block_size=1 << 20, size=_http_size(url, timeout=timeout),
                   cache_type="readahead")


def read_sdss_spectrum(url: str, redshift: float, *, timeout: float = 120.0) -> Spectrum:
    """Download and parse an SDSS SAS lite spectrum (observed frame)."""
    from astropy.io import fits  # type: ignore[import-untyped]

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with fits.open(io.BytesIO(resp.content)) as hdul:
        d = hdul[1].data
        wave = np.asarray(10.0 ** d["loglam"], float)
        flux = np.asarray(d["flux"], float)
        ivar = np.asarray(d["ivar"], float)
    error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.nan)
    return Spectrum(wave, flux, error, float(redshift))


def read_desi_spectrum(
    url: str, targetid: int, redshift: float, *, timeout: float = 300.0
) -> Spectrum:
    """Lazily read one target's B+R+Z spectrum from a DESI healpix coadd via
    HTTP range requests (no full-file download)."""
    return read_desi_spectra([targetid], url, {int(targetid): redshift},
                             timeout=timeout)[int(targetid)]


def read_desi_spectra(
    targetids: Sequence[int],
    url: str,
    redshifts: Mapping[int, float],
    *,
    timeout: float = 300.0,
) -> dict[int, Spectrum]:
    """Read MANY targets from one healpix coadd on a single file open.

    A coadd holds every target in its healpix, so retrieving them one at a time
    reopens the same file once per target and pays the open cost every time. Measured
    on the survey mirror, that cost dominates: draining a coadd runs at roughly a third
    of the per-spectrum wall time of one-at-a-time reads, with the gain growing as more
    targets share the file (`scripts/retrieval_pilot.py`).

    Sorting the *request order* by healpix does not achieve this on its own -- the file
    has to stay open across the targets, which is what this does and what a sorted
    sequence of single reads does not.

    Targets absent from the coadd are omitted from the returned mapping rather than
    raising, since one missing target should not lose the rest of the file's work; the
    single-target :func:`read_desi_spectrum` still raises ``KeyError`` for its one id.

    **The speed is not free, and the caller must handle that.** Holding one handle open
    across many section reads is measurably more fragile than independent requests: the
    pilot saw a 1.25% mid-stream failure rate (``ClientPayloadError``) against 0.00% for
    one-at-a-time reads over the same mirror. A partial read is therefore a normal
    outcome, not an error -- this returns whatever it retrieved before the stream broke,
    and the caller is expected to re-request the missing ids individually. Silently
    accepting the short mapping would trade a 3x speedup for a 1% hole in the sample.
    """
    from astropy.io import fits  # type: ignore[import-untyped]

    handle = _open_coadd(url, timeout)
    out: dict[int, Spectrum] = {}
    try:
        with fits.open(handle) as hdul:
            fmap = hdul["FIBERMAP"].data
            ids = np.asarray(fmap["TARGETID"])
            # the three wavelength grids are shared by every row: read them once
            waves = [np.asarray(hdul[f"{band}_WAVELENGTH"].data, float)
                     for band in ("B", "R", "Z")]
            wave = np.concatenate(waves)
            order = np.argsort(wave)
            wave_sorted = wave[order]
            for tid in targetids:
                matches = np.where(ids == int(tid))[0]
                if len(matches) == 0:
                    continue
                row = int(matches[0])
                flux = np.concatenate([
                    np.asarray(hdul[f"{band}_FLUX"].section[row], float)
                    for band in ("B", "R", "Z")])
                ivar = np.concatenate([
                    np.asarray(hdul[f"{band}_IVAR"].section[row], float)
                    for band in ("B", "R", "Z")])
                flux, ivar = flux[order], ivar[order]
                error = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)),
                                 np.nan)
                out[int(tid)] = Spectrum(wave_sorted, flux, error,
                                         float(redshifts[int(tid)]))
    except Exception:
        # a broken stream loses the rest of this file, not what was already read.
        # Re-raise only when nothing survived and a single id was asked for, so the
        # single-target contract is unchanged.
        if not out and len(targetids) == 1:
            raise
    if len(targetids) == 1 and not out:
        raise KeyError(f"targetid {targetids[0]} not in {url}")
    return out
