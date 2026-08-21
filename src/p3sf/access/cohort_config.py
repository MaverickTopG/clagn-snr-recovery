"""Survey-access constants and cohort-query configuration.

Adapted from ``clagn_apj.config.CohortConfig`` and ``clagn_apj.broadened``
(upstream rev c3bd1d4). Kept separate from :mod:`p3sf.config` because these are
*access* parameters — which tables to hit, which endpoint — not analysis
parameters that could change a result. Analysis constants belong in
``00_admin/frozen_config.yaml``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

DATALAB_TAP = "https://datalab.noirlab.edu/tap/sync"
SDSS_SAS_BASE = "https://data.sdss.org/sas"
DESI_DR1_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"


class CohortConfig(BaseModel):
    """Parameters for the SDSS x DESI parent cohort query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    release: str = "dr1"
    expected_source_count: int = Field(default=749_693, ge=1)
    parent_match_radius_arcsec: float = Field(default=1.5, gt=0)
    desi_wavelength_bounds: tuple[float, float] = (3600.0, 9824.0)
    datalab_profile: str = "default"
    sdss_table: str = "sdss_dr17.specobjall"
    sdss_desi_xmatch_table: str = "sdss_dr17.x1p5__specobj__desi_dr1__zpix"
    desi_table: str = "desi_dr1.zpix"
    emfit_table: str = "desi_dr1.emfit"
    download_chunk_size: int = Field(default=25_000, ge=100)
    url_validation_sample: int = Field(default=100, ge=1)
    minimum_index_eligible: int = Field(default=1000, ge=1)


__all__ = ["CohortConfig", "DATALAB_TAP", "SDSS_SAS_BASE", "DESI_DR1_BASE"]
