# D-092 — final prospective Q1 reference-universe completeness audit

## Decision

`ADDITIONAL_GOLD_RECOVERABLE`

The validated D-091 reference set remains exactly 10 GOLD and 14 GOLD+SILVER. No
candidate endpoint was downloaded, degraded, fit, or run through Yang/Green. The records
below are prospective acquisition candidates only; they have not been promoted into the
validated reference set.

## Objective universe and stopping rule

The frozen literature cutoff is **2026-08-15 23:59 UTC**: the end of the UTC day before
this audit and before any full-Q1 execution. The source universe includes primary
peer-reviewed, accepted, or full arXiv papers available by that cutoff that report a
systematic spectroscopic broad-line CLAGN family (at least three confirmed events or a
machine-readable catalogue). Earlier papers needed to identify duplicate physical events
are also enumerated. Individual-case hunting is outside this bounded universe.

Relevance is restricted to the frozen optical H-beta Q1 domain. Reference evidence is
classifier-blind and requires real broad-line appearance/disappearance, strong endpoint
morphology, continuum support and multi-line/difference-spectrum support where available.
Prospective GOLD additionally requires exact historical bright and faint endpoints,
immutable identifiers, reproducible public retrieval and H-beta coverage. A later spectrum
never replaces a missing event endpoint. No target sample size was used. The stopping rule
was `FROZEN_RELEVANT_SOURCE_UNIVERSE_EXHAUSTED`.

## Family accounting

The machine-readable census is
`04_reference_sample/q1_completeness_source_family_screen_d092.csv`.

| Quantity | Exact audit count |
|---|---:|
| Families considered | 31 |
| Families already screened (D-090 or earlier) | 13 |
| Newly screened relevant families | 9 |
| Duplicative families | 5 |
| Out-of-Q1-domain families | 4 |
| Out-of-domain published records | 757 |
| Unique newly screened events reaching the strict-GOLD morphology prescreen | 64 |
| Duplicate prospective-GOLD events across families | 2 |
| Prospective-GOLD events failing exact endpoint recovery | 8 |
| Additional strict GOLD with exact public endpoints | **54** |
| Additional exact-public SILVER sensitivity candidates | **74** |
| Validated current GOLD | **10** |
| Prospective GOLD after acquisition and validation | **64** |

The 64-event strict-morphology accounting is closed: 54 unique acquisition-ready events,
two Yang/Zeltyn duplicate physical events, and eight events whose exact endpoint pair is
unrecoverable, source-ambiguous, or not a frozen-pipeline public 1D product. The 74 SILVER
count comprises remaining exact-public H-beta-domain SDSS-V catalogue pairs that do not
meet the unchanged strict-GOLD evidence bar. They are not part of the acquisition list.

Every `OUT_OF_Q1_DOMAIN` row has an explicit reason in the census: changing-look
LINER/transient physics outside the frozen broad-Hbeta reference domain; circular Guo
labels from a classifier family under test; high-redshift high-ionization transitions with
no optical H-beta coverage; or persistent-Hbeta samples that do not satisfy the frozen
appearance/disappearance event definition.

## Exact acquisition-only result

The exact 54-pair acquisition manifest is
`04_reference_sample/q1_completeness_gold_acquisition_d092.csv`. It contains 37
SDSS/LAMOST events from Dong et al. 2025, 11 SDSS/SDSS-V events from Zeltyn et al. 2024,
and six non-duplicate SDSS/LAMOST-or-DESI turn-ons from Yang et al. 2025. Every row records
both historical roles and MJDs, immutable survey identifiers, retrieval family, H-beta
coverage, evidence basis, and expected acquisition size. Status is uniformly
`ACQUIRE_VALIDATE_ONLY`.

The source papers independently establish the relevant family-level evidence: Zeltyn et
al. report 116 visually identified CLAGNs with broad-line appearance/disappearance and
publish endpoint figure sets; Yang et al. report 82 spectroscopically confirmed turn-ons;
Dong et al. report 51 CLAGNs identified by spectral fitting followed by detailed visual
inspection, 41 primarily through H-beta. The acquisition subset is deliberately narrower
than each published catalogue and was selected using the frozen evidence and endpoint
rules, not predicted low-S/N behavior.

## Consequence

`REFERENCE_EXPANSION_SEARCH_EXHAUSTED` applies to the bounded publication universe, but
`FREEZE_FINAL_Q1_REFERENCE_POOL` does not: 54 meaningful prospective strict-GOLD events
are reproducibly identified. Full Q1 remains held. The next permitted operation is bounded
acquisition and native endpoint validation for the exact manifest only. After the validated
population is locked, S/N support must be re-derived prospectively; `[5,10]` is neither
changed nor automatically reconfirmed here. D-089 contracts, `M=50`, object-level
aggregation, Gate C, and every Q2 decision remain frozen.

## Primary sources and public archives

- Zeltyn et al. 2024, [SDSS-V first-year CLAGN catalogue](https://arxiv.org/abs/2401.01933)
- Yang et al. 2025, [turn-on CLQ catalogue](https://arxiv.org/abs/2408.16183)
- Dong et al. 2025, [SDSS/LAMOST CLAGN catalogue](https://arxiv.org/abs/2408.07335)
- Hon et al. 2020, [MaNGA CLAGN search](https://doi.org/10.1093/mnras/staa1939)
- Chen et al. 2026, [SDSS/LAMOST/DESI catalogue](https://arxiv.org/abs/2605.24429)
- [SDSS DR19 data access](https://www.sdss.org/dr19/data_access/)
- [LAMOST DR11 public archive](https://www.lamost.org/dr11/)

