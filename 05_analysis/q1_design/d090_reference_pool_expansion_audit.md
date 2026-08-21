# D-090 — one-time prospective Q1 reference-pool expansion audit

## Decision

`EXPAND_Q1_REFERENCE_POOL`

No candidate spectrum was downloaded, degraded, fit, or classified. D-085/D-089 outputs
were not inspected. The audit used primary-paper morphology, continuum and difference-
spectrum descriptions plus public archive metadata. Multiple publications of one physical
transition were collapsed to the discovery event; later spectra were not substituted for a
missing discovery endpoint.

## Frozen rules applied

- D-086 tiers are unchanged. GOLD requires convincing broad-line appearance/disappearance,
  strong endpoint morphology, coordinated continuum behavior, and trustworthy primary
  provenance, with multiline or difference-spectrum corroboration where available.
- D-087 Q1 requires the exact historical bright and faint event endpoints, reproducible
  public retrieval, H-beta coverage and a defensible native-quality path. Q2 aperture/host
  comparability is not imposed.
- D-088 `[5,10]` remains unchanged during this audit, but its support must be re-derived if
  the pool expands.
- D-089 Yang/Green contracts and common `M=50` remain frozen. One transition is one
  independent unit.

## Expansion result

| Quantity | Count |
|---|---:|
| Current strict GOLD | 4 |
| Additional strict GOLD with exact public endpoints | **6** |
| Prospective strict GOLD total | **10** |
| Additional SILVER with exact public endpoints | **19** |
| Current strict-GOLD discovery/source families | 4 |
| Prospective strict-GOLD discovery/source families | **6** |

Current GOLD families are Potts–Villforth, MacLeod 2016, Green 2022 and Ruan 2016.
LaMassa 2015 and Runnoe 2016 are added as independent discovery families. New objects also
increase representation within the existing Ruan, Potts and Green families.

Across every tier in the current 17-object pool there are seven provenance families
(Green, MacLeod 2016, MacLeod 2019, Potts–Villforth, Ruan, Runco and Graham); the strict-GOLD
expansion adds LaMassa and Runnoe, giving nine represented families overall.

## Additional strict-GOLD acquisition candidates

All six have complete H-beta coverage from their reported redshifts and SDSS-family
instrument ranges. `Q1 eligible` is prospective pending acquisition checksum, native QC and
endpoint-identity validation; it is not a claim that unseen local files have passed QC.

| Object | Physical evidence, independent of tested classifiers | Exact bright | Exact faint | Instruments | Public | Proposed status |
|---|---|---|---|---|---|---|
| SDSS J015957.64+003310.4 | Type 1→1.9; Hβ disappears, Hα and continuum fade; optical/X-ray/photometric coherence | 403-51871-0549 | 3609-55201-0524 | SDSS/BOSS | yes | GOLD; acquire |
| SDSS J012648.08−083948.0 | Hβ and Hα completely disappear; Hγ in difference spectrum; continuum dims | 661-52163-0604 | 2878-54465-0377 | SDSS/BOSS | yes | GOLD; acquire |
| SDSS J101152.98+544206.4 | Hβ disappears; Hα drops 55-fold; continuum drops >9.8-fold | 945-52652-0022 | 8181-57073-0827 | SDSS/eBOSS | yes | GOLD; acquire |
| SDSS J000236.25−002724.8 | Hβ and continuum disappear and later return; Hα weakens; difference-spectrum support | 387-51791-0110 | 669-52559-0306 | SDSS/SDSS | yes | GOLD; acquire first off event |
| SDSS J135855.83+493414.2 | Type 2→1; strong Hβ/Hα and blue-continuum emergence; independently described as extreme/clear | 1670-54553-0073 | 1670-53438-0061 | SDSS/SDSS | yes | GOLD; acquire |
| SDSS J021359.79+004226.81 | All broad Balmer lines weaken and become narrow; strong steady continuum dimming | 405-51816-0458 | 9383-58097-0829 | SDSS/eBOSS | yes | GOLD; acquire |

Their twelve SDSS lite products are expected to total approximately 2.4–3.3 MB. The exact
per-record sizes and checksums must be recorded during acquisition.

## Additional exact-public SILVER sensitivity candidates

These nineteen transitions meet the endpoint/public/coverage metadata screen but do not
meet GOLD without weakening D-086. Reasons include intermediate broad-line states,
persistent Hα or Hβ, low-significance published morphology, complex/double-peaked profiles,
or weaker continuum/multiline corroboration.

| Source family | Objects |
|---|---|
| MacLeod 2016 | J002311.06+003517.5, J022556.07+003026.7, J100220.17+450927.3, J214613.31+000930.8, J225240.37+010958.7, J233317.38−002303.4 |
| Potts–Villforth 2021 | J082323.89+422048.3, J172322.31+550413.8 |
| Yang 2018 SDSS-only | J110423.21+634305.3, J115039.32+363258.4, J153355.99+011029.7 |
| Green 2022 | J020514.77−045639.74, J021259.59−003029.43, J024508.67+003710.68, J024932.01+002248.35, J113706.93+481943.68, J135415.54+515925.77, J163620.38+475838.36, J231625.39−002225.50 |

They remain optional sensitivity candidates and are not added to strict primary truth.

## Non-counting records

The full classifier-blind record audit contains 51 deduplicated candidate transitions in
`04_reference_sample/q1_expansion_candidates_d090.csv`. Fourteen have proposed GOLD-level
morphology, but only the six above have exact, reproducibly identified public event
endpoints. Important fail-closed cases are:

- J105513 and J143455 have GOLD-level published morphology but their MacLeod-2019 discovery
  endpoints are non-public follow-up spectra. Later Green/eBOSS spectra are substitutes and
  do not count.
- J132457 has exact public identifiers, but the faint BOSS product has a documented
  extraction failure redward of H-beta.
- Yang LAMOST events are not countable until exact `planid/spid/fiberid` records are
  recovered; coordinates, instrument and MJD alone are insufficient immutable provenance.
- Xinglong/Palomar/MMT discovery endpoints without a reproducible public source product do
  not count.
- Green turn-ons with broad lines in both epochs and marginal/profile-only cases remain
  BORDERLINE, consistent with D-086 treatment of variability without clear appearance or
  disappearance.
- Guo DESI catalogue labels are excluded from reference adjudication because the Guo
  staged classifier is itself under test. Graham changing-state rows are not inherited as
  CL truth. López-Navas and other targeted-follow-up families did not yield a bounded public
  exact-endpoint set.

The family-level screen is recorded in
`04_reference_sample/q1_expansion_source_family_screen_d090.csv`.

## Bounded acquisition and validation plan

1. Acquire only the twelve exact SDSS-family products for the six additional GOLD events.
2. Record canonical archive URL, plate–MJD–fiber, SDSS `specObjID`, byte size and SHA-256;
   confirm each file matches the published discovery endpoint, not a later state.
3. Run native ingestion/QC only: wavelength coverage, masks, inverse variance, redshift,
   historical bright/faint orientation and D-089 measurement-domain compatibility. Do not
   degrade spectra or run recovery classifiers during acquisition validation.
4. If any endpoint fails, retain the transition at its evidence tier but exclude it from
   Q1 eligibility; do not replace it with a later spectrum or promote SILVER.
5. Lock the validated expanded GOLD set. Then prospectively re-derive D-088 support for the
   S/N grid on the expanded population. `[5,10]` remains unchanged until that audit; it is
   neither automatically retained nor changed.
6. Only after the expanded eligibility and support matrix is frozen may full Q1 execution
   be reconsidered. D-089 Yang/Green semantics and common `M=50` remain fixed unless a real
   instrument-domain incompatibility is documented.

## Recommendation

`EXPAND_Q1_REFERENCE_POOL`

Six exact-public strict-GOLD candidates can increase the primary independent-transition
denominator from 4 to a prospective 10 and add two discovery-source families. The already
frozen full-Q1 matrix must therefore remain unexecuted pending bounded acquisition,
validation, and a prospective re-derivation of S/N support.

## Primary sources and archive metadata

- LaMassa et al. 2015, [arXiv:1412.2136](https://arxiv.org/abs/1412.2136)
- MacLeod et al. 2016, [arXiv:1509.08393](https://arxiv.org/abs/1509.08393)
- Ruan et al. 2016, [arXiv:1509.03634](https://arxiv.org/abs/1509.03634)
- Runnoe et al. 2016, [arXiv:1509.03640](https://arxiv.org/abs/1509.03640)
- Yang et al. 2018, [arXiv:1711.08122](https://arxiv.org/abs/1711.08122)
- Potts & Villforth 2021, [arXiv:2104.14225](https://arxiv.org/abs/2104.14225)
- Green et al. 2022, [primary catalogue and table metadata](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/933/180?format=html)
- SDSS public spectral metadata were resolved through the
  [DR18 SkyServer](https://skyserver.sdss.org/dr18/).
