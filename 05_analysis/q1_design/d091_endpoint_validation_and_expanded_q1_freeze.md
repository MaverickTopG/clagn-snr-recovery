# D-091 — exact endpoint validation and expanded Q1 freeze

## Scope and gate result

Only the twelve D-090 plate--MJD--fiber products for the six prospective GOLD
events were acquired. No alternative epoch was searched for or substituted; none of the
nineteen additional SILVER candidates was acquired. No spectrum was degraded and no Q1
classifier-recovery outcome was generated or inspected.

`GO_EXPANDED_FULL_Q1`

All six pairs pass identity, frozen native QC, native SDSS variance provenance and Green
preprocessing feasibility. The final reference counts are therefore:

| quantity | count |
|---|---:|
| previous strict GOLD | 4 |
| prospective added GOLD | 6 |
| endpoint-valid added GOLD | 6 |
| final Q1-eligible GOLD | 10 |
| final GOLD+SILVER sensitivity set | 14 |
| strict-GOLD discovery/source families before / after | 4 / 6 |

Reference evidence and technical applicability remain separate. Every added object retains
`GOLD_REFERENCE_EVIDENCE_PASS`; poor or boundary-contacting native Yang fits are recorded as
criterion-level fail-closed states and do not demote the reference tier.

## Exact acquisition and identity

The 12 DR17 lite FITS products total 2,214,720 bytes: nine legacy SDSS/run2d=26 and three
BOSS-family/run2d=v5_13_2 products. All local paths and SHA-256 digests are unique. Archive,
primary-header and SPECOBJ plate--MJD--fiber fields agree exactly; coordinates are within
0.18 arcsec of the IAU names; archive and FITS redshifts agree; within-pair redshift deltas
are all below 0.001; and every file has the expected COADD structure.

| object | physical bright endpoint | physical faint endpoint | result |
|---|---|---|---|
| J015957 | 403-51871-0549 | 3609-55201-0524 | exact pair identity pass |
| J012648 | 661-52163-0604 | 2878-54465-0377 | exact pair identity pass |
| J101152 | 945-52652-0022 | 8181-57073-0827 | exact pair identity pass |
| J000236 | 387-51791-0110 | 669-52559-0306 | exact pair identity pass |
| J135855 | 1670-54553-0073 | 1670-53438-0061 | exact pair identity pass |
| J021359 | 405-51816-0458 | 9383-58097-0829 | exact pair identity pass |

The complete source URLs, filenames, byte sizes and checksums are in
`03_spectra/raw_sdss/manifest_d091_expansion.csv`; the field-level identity checks are in
`endpoint_identity_validation_d091.csv`.

## Frozen native QC and measurement feasibility

The existing `06_spectrum_quality_control.py` reader and `assess` function were imported
unchanged. The binding S/N is the median per-pixel rest-frame 4700--5100-Angstrom value.
Every endpoint has full formal H-beta-window coverage, passes the frozen maximum masked
fraction and S/N >= 3 rules, and has bit-aware ZWARNING=0/PASS. Green feasibility was tested
after source-faithful preprocessing using native SDSS inverse variance; line flux was zeroed
before the pair feasibility operation so no Green outcome statistic was inspected.

| object | role | rest range (A) | valid pixels | H-beta masked | S/N H-beta | S/N 5100 | ZWARNING | native QC | Yang native fit | Green / degradation |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| J015957 | bright | 2906--7023 | 0.998 | 0.014 | 14.42 | 12.71 | 0 PASS | pass | invalid: strict local coverage | pass / eligible |
| J015957 | faint | 2727--7884 | 0.995 | 0.000 | 12.27 | 14.21 | 0 PASS | pass | invalid: width > 20,000 km/s | pass / eligible |
| J012648 | bright | 3190--7687 | 1.000 | 0.000 | 15.12 | 19.22 | 0 PASS | pass | invalid: active bound | pass / eligible |
| J012648 | faint | 3185--7693 | 0.974 | 0.000 | 18.34 | 22.73 | 0 PASS | pass | valid | pass / eligible |
| J101152 | bright | 3057--7385 | 1.000 | 0.000 | 23.64 | 23.58 | 0 PASS | pass | valid | pass / eligible |
| J101152 | faint | 2899--8325 | 0.998 | 0.000 | 14.11 | 15.29 | 0 PASS | pass | invalid: active bound | pass / eligible |
| J000236 | bright | 2940--7125 | 1.000 | 0.000 | 13.67 | 13.26 | 0 PASS | pass | valid | pass / eligible |
| J000236 | faint | 2961--7115 | 0.983 | 0.000 | 12.39 | 12.61 | 0 PASS | pass | invalid: active bound | pass / eligible |
| J135855 | bright | 3413--8243 | 1.000 | 0.000 | 14.31 | 12.29 | 0 PASS | pass | invalid: active bound | pass / eligible |
| J135855 | faint | 3404--8246 | 1.000 | 0.000 | 9.60 | 8.07 | 0 PASS | pass | invalid: active bound | pass / eligible |
| J021359 | bright | 3201--7776 | 1.000 | 0.000 | 29.13 | 28.87 | 0 PASS | pass | valid | pass / eligible |
| J021359 | faint | 3036--8765 | 0.999 | 0.000 | 19.94 | 17.18 | 0 PASS | pass | valid | pass / eligible |

Five of twelve new native Yang fits are valid. This is not an endpoint exclusion: D-089
already specifies that Yang fit/coverage failures become unclassifiable, never non-CL.
J015's bright strict 2-A local-fit coverage failure is static, so its Yang rows are
unclassifiable in both arms. J012 and J135 have invalid native bright fits, so their
`faint_only` Yang rows are fixed unclassifiable; their `matched` bright spectra are refit and
remain applicable with per-realization fail-closed handling. All fourteen transitions are
Green-applicable with native SDSS variance. MacLeod 2019 remains unclassifiable without the
prospectively unavailable visual evidence. MacLeod 2016, Guo quick search and
Potts--Villforth remain search/non-denominator protocols. Exact transition x arm x criterion
states are frozen in `expanded_q1_classifier_applicability_d091.csv`.

## Expanded native-S/N support and final grid

| target S/N | GOLD faint-only | GOLD matched | GOLD+SILVER faint-only | GOLD+SILVER matched | disposition |
|---:|---:|---:|---:|---:|---|
| 5 | 10/10 | 10/10 | 14/14 | 14/14 | final retained |
| 10 | 8/10 | 8/10 | 12/14 | 12/14 | final retained |
| 15 | 3/10 | 3/10 | 5/14 | 5/14 | underpowered; excluded |
| 20 | 1/10 | 1/10 | 2/14 | 2/14 | underpowered; excluded |
| 30 | 0/10 | 0/10 | 1/14 | 0/14 | underpowered; excluded |
| 40 | 0/10 | 0/10 | 0/14 | 0/14 | unreachable; removed |

The expanded support independently reconfirms the final grid `[5,10]`. S/N 5 retains the
complete primary and sensitivity sets; S/N 10 retains 80% of GOLD and 86% of GOLD+SILVER.
The apparent new reach at 15 is only 3/10 GOLD and 5/14 total, so D-088's underpowered-rung
logic is unchanged; a higher rung is not promoted merely because three added objects reach
it. No spectrum is upgraded. `faint_only` leaves the physical bright endpoint unchanged;
`matched` degrades both endpoints with the frozen object-specific common random numbers.

## Expanded unexecuted matrix and inference units

| quantity | frozen value |
|---|---:|
| PRIMARY_Q1 GOLD transitions | 10 |
| GOLD+SILVER sensitivity transitions | 14 |
| S/N grid | [5,10] |
| primary GOLD conditions | 36 |
| SILVER sensitivity-increment conditions | 16 |
| faint-only / matched conditions | 26 / 26 |
| total transition-arm-rung conditions | 52 |
| common M | 50 |
| condition-realization records | 2,600 |
| unique reusable spectrum-fit inputs, including native bright | 2,614 |
| Yang fits / Green preprocessing fits | 2,614 / 2,614 |
| unique classifier-specific fits | 5,228 |
| final-classifier evaluations (Yang, Green, MacLeod 2019) | 7,800 |

The matrix is `proposed_expanded_final_q1_matrix_d091_unexecuted.csv`; every row has
`execution_authorized=False` and `spectra_generated=False`. Transition/object remains the
independent scientific unit. M=50 realizations, arms, rungs and criteria are repeated
measurements. D-089's seed namespace `p3sf:q1:full:v1`, common faint draw across arms,
Yang/Green contracts and equal-transition aggregation remain unchanged. The existing
stability experiment was not rerun because the new products are supported SDSS-family
spectra and exposed no new measurement/variance domain.

## Source-family robustness freeze

The final GOLD composition is Potts 2021 (3), Ruan 2016 (2), Green 2022 (2), MacLeod 2016
(1), LaMassa 2015 (1), and Runnoe 2016 (1). Ten leave-one-transition-out summaries and six
leave-one-source-family-out summaries are frozen but unexecuted in
`expanded_q1_robustness_freeze_d091.csv`. LOSFO retains 7--9 GOLD transitions depending on
the omitted family.

No added transition was technically rejected. This recommendation authorizes a later
expanded full-Q1 execution under the frozen matrix; D-091 itself generated no degraded
spectrum and executed no recovery classifier.
