# Bounded historical endpoint recovery audit after D-086

**Frozen decision preserved:** `NO_GO_FULL_Q2_REFERENCE_SUPPORT_INSUFFICIENT`  
**Scope:** metadata/source discovery only for the six D-086 `OUT_OF_DOMAIN`
transitions. No spectra were downloaded, no fits or classifiers were run, and no D-084,
D-085, D-086, reference-tier, or pair-eligibility record was modified.

## Result

Only the J123 historical faint endpoint has an exact public source identity:

- WHT/ISIS, 2016-05-31 (MJD 57539);
- blue-arm runs `2364301` and `2364305`, red-arm runs `2364302` and `2364304`;
- two 1800 s science exposures in each arm, matching the published ISIS setup;
- public in the ING/CASU archive under the ING one-year proprietary-period policy.

This is an archive recovery lead, not an eligibility promotion. Acquisition would still
have to retrieve all four raw science frames and relevant same-night biases, flats, arcs,
and standards; record archive checksums; verify target coordinates, times, instrument/slit/
grating/dichroic headers; reproduce the paired-arm reduction and telluric/flux calibration;
confirm useful H-beta and continuum coverage; and establish a defensible flux/aperture basis.

The MMT Blue Channel endpoints for J074, J141, J153, and J210 have exact published epochs
but no immutable exposure IDs. The observatory's public policy explicitly says it does not
maintain a Blue Channel data archive. J074 and J141 therefore have later public DESI records
only; J153 has no admissible later record; and J210's DESI positional match has an identity-
breaking redshift conflict. The Magellan LDSS3-C endpoint for J161 likewise has an exact
published epoch and adequate wavelength setup but no public source record was located;
its D-084 DESI/DESI pair is later and is not the event.

Frozen recovery-status counts are:

- `EXACT_ENDPOINT_RECOVERABLE`: 1 (J123)
- `EXACT_ENDPOINT_NOT_FOUND`: 2 (J153, J210)
- `LATER_SUBSTITUTE_ONLY`: 3 (J074, J141, J161)
- `SOURCE_AMBIGUOUS`: 0

## Reference-support accounting

- Current strict GOLD: **4**.
- Current `PAIR_CONSISTENT_ELIGIBLE` GOLD: **1**.
- GOLD objects in this audit whose exact endpoints could become evaluable if recovered: **0**.
- SILVER/BORDERLINE objects with an exact endpoint publicly recoverable from this audit:
  **1** (J123, SILVER).

Recovering J123 cannot enlarge the strict GOLD set because D-086 tiers are frozen, and it
does not establish pair-consistent eligibility without the independent flux/aperture audit
listed above. Thus this bounded audit produces no meaningful strict-reference expansion.
The full physical Q2 no-go should be frozen permanently for the current reference pool;
controlled-host contamination should remain secondary diagnostic science. The ten existing
3-inch/2-inch aperture-mismatched transitions remain `SPECTRUM_DIAGNOSTIC_ONLY`, with no
rescaling or correction.

## Provenance

- MacLeod et al. 2019 primary article: `doi:10.3847/1538-4357/ab05e2`.
- Machine-readable publication epochs/facilities: VizieR `J/ApJ/874/8`.
- Exact J123 source records: ING public observing-log query for target `J123359`, year 2016,
  WHT, returning the four run numbers above.
- Archive/public-access policy: ING Data Archives.
- MMT retention policy: MMTO “Before Your Run,” Data Policies.
- Magellan archive check: Carnegie Astronomer and Observer Resources and LDSS3/COSMOS public
  instrument/software pages; no public science-product query or source record was found.
- Bright SDSS source identifiers were confirmed through a metadata-only SDSS DR18
  `SpecObjAll` query at the published positions and MJDs.

The row-complete audit is in
`historical_endpoint_recovery_audit_post_d086.csv`.
