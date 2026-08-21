# D-093 — final expanded Q1 acquisition, instrument validation, and bias freeze

## Decision

`GO_FINAL_FULL_Q1`

This is an execution-readiness decision only. No degraded spectrum, realization-level fit,
or Q1 recovery/classifier outcome was generated or inspected.

## Bounded acquisition and identity

The input was exactly the 54-pair D-092 manifest. All 108 frozen endpoint products were
acquired, totaling 20,165,760 bytes, with immutable identifiers, local filenames, byte
sizes, SHA-256 checksums, source URLs, physical roles, families, methods, and epochs in
`d093/endpoint_acquisition_manifest_d093.csv`. No alternate epoch or candidate was queried.

Fifty pairs pass the exact identity gate. Four fail closed:

| Transition | Failure | Consequence |
|---|---|---|
| J151143.41+210104.0 | SDSS/LAMOST product redshifts differ by 0.001510, exceeding the unchanged 0.001 pair tolerance | technically ineligible; GOLD evidence preserved |
| J1311+0705 | frozen DESI epoch is 59295, while the exact TARGETID product exposure is MJD 59296.418 | technically ineligible; no substitute |
| J020649.48−041452.7 | D-092 declares MJD 57742 but freezes PMF 7238–56660–791 | technically ineligible; immutable identifier was acquired but contradiction fails closed |
| J221026.83−001721.1 | SDSS/SDSS-V product redshifts differ by 0.001979, exceeding the unchanged tolerance | technically ineligible; GOLD evidence preserved |

The six Yang-family SDSS identifiers omitted the redundant `run2d`; their frozen PMF and
`specObjID` resolve uniquely to legacy `run2d=26`, and both values were validated inside
each downloaded product. This metadata resolution did not change an endpoint.

## Native QC

The unchanged D-091 rest-H-beta rules were applied independently to all acquired products:
106/108 endpoints pass. Two exact LAMOST endpoints fail the frozen minimum native
H-beta-window S/N:

| Transition | Role | Native Hβ S/N | Technical result |
|---|---|---:|---|
| J000253.52+210109.9 | faint | 2.862548 | `LOW_CONTINUUM_SNR`; Q1-ineligible |
| J131001.32+143705.2 | faint | 1.959443 | `LOW_CONTINUUM_SNR`; Q1-ineligible |

No threshold was changed. Bit masks and warnings are reported, while SDSS/LAMOST positive-
IVAR pixel use follows D-091 exactly. The full endpoint audit is in
`d093/native_qc_d093.csv`.

`GOLD_REFERENCE_EVIDENCE` remains separate from `Q1_NATIVE_ENDPOINT_VALID`: all six
technical exclusions retain strict-GOLD reference evidence but do not enter Q1 execution.

## Instrument-domain gate

| Family | Wavelength/sampling | Uncertainty and masks | Resolution/degradation | Yang | Green |
|---|---|---|---|---|---|
| SDSS legacy | vacuum log-lambda | native IVAR; AND/OR masks reported | WDISP retained; noise only | supported | supported |
| LAMOST DR11 | `VACUUM=T`, log-linear | native IVAR; AND/OR/FIB masks reported | native low-resolution LSF; noise only | supported | supported |
| SDSS-V DR19 | vacuum BOSS log-lambda | native IVAR; AND/OR masks reported | WDISP retained; noise only | supported | supported |
| DESI EDR/Fuji | vacuum linear grid, official camera coadd | native IVAR and DESI MASK | resolution matrix preserved; noise only | supported | supported |

For every family, rest conversion is `lambda/(1+z)`, Green uses flux-conserving overlap-
weighted rest-frame 2-A bins with propagated `sum(w^2 variance)`, and the Q1 operator changes
noise without altering the native grid or LSF. No empirical variance correction was added.
The exact gate is `d093/instrument_domain_validation_d093.csv`; the new supported variance
provenance enums are enforced in code and tests.

Native per-transition measurement feasibility is separate from family compatibility.
Among final GOLD, Yang is prospectively applicable for 49/58 transitions and Green for
54/58. The others are `UNCLASSIFIABLE_INSTRUMENT_DOMAIN`, never non-CL. Yang's ratio
`<0.3`, valid-nondetection-only zero rule, 1200–20000 km/s domain, and fail-closed fit
semantics are unchanged. Green remains the exact 2-A/16-pixel/4750-subtraction/4750–4940
maximum with inclusive `>=3`.

## Final independent sample

| Quantity | Count |
|---|---:|
| Validated pre-D-092 GOLD | 10 |
| Prospective new GOLD | 54 |
| Frozen pairs acquired | 54/54 |
| Identity-valid exact pairs | 50 |
| Endpoint/native-QC-valid new GOLD | **48** |
| Final Q1-eligible GOLD | **58** |
| Final GOLD+SILVER sensitivity set | **62** |

Final GOLD composition is Dong 2025 SDSS/LAMOST 34, Zeltyn 2024 SDSS-V 9, Yang 2025
turn-on 5, Potts 3, Ruan 2, Green 2, LaMassa 1, Runnoe 1, and MacLeod 2016 1. It contains
43 turn-off and 15 turn-on transitions. Ordered bright/faint instrument pairs are 32
SDSS/LAMOST, 10 SDSS/SDSS, six LAMOST/SDSS, five SDSS/SDSS-V, four SDSS-V/SDSS, and one
DESI/SDSS.

## Discovery-method circularity and family imbalance

The classifier-blind family matrix is frozen in
`d093/discovery_classifier_independence_d093.csv`. It classifies every family × Yang/Green
combination as `DISCOVERY_INDEPENDENT`, `PARTIALLY_METHOD_OVERLAPPING`, or
`DIRECTLY_CRITERION_SELECTED`. Green-2022-family transitions are directly selected for the
Green final criterion and are excluded only from the prespecified discovery-independent
Green sensitivity summary. They remain in the main reference truth and transition-equal
primary estimand. Related but non-identical Yang/Dong/Zeltyn methods are retained as
`PARTIALLY_METHOD_OVERLAPPING`, not falsely treated as direct scalar selection.

The primary remains `mean_i(p_i)`, one equal weight per transition. Prespecified robustness
summaries are leave-one-source-family-out and
`mean_family(mean_transition_within_family(p_i))`, giving every family equal weight.
Neither replaces the primary. Exact frozen rows are in
`d093/source_family_robustness_d093.csv`.

## Re-derived S/N support and final grid

| S/N | All GOLD faint-only | All GOLD matched | Yang-applicable faint/matched | Green-applicable faint/matched |
|---:|---:|---:|---:|---:|
| 5 | 48/58 | 47/58 | 41/49 / 40/49 | 44/54 / 43/54 |
| 10 | 29/58 | 27/58 | 25/49 / 24/49 | 27/54 / 25/54 |
| 15 | 14/58 | 12/58 | 13/49 / 12/49 | 14/54 / 12/54 |
| 20 | 7/58 | 7/58 | 7/49 / 7/49 | 7/54 / 7/54 |
| 30 | 1/58 | 1/58 | 1/49 / 1/49 | 1/54 / 1/54 |
| 40 | 0/58 | 0/58 | 0/49 / 0/49 | 0/54 / 0/54 |

`[5,10]` remains final. Although the larger N supplies some cells at 15, those cells are a
minority (24% faint-only, 21% matched), are strongly source/instrument selected, and would
break comparability with the prospectively frozen D-088/D-091 design. No spectrum is
upgraded. All family-stratified counts are in `d093/final_snr_support_d093.csv`.

## Final unexecuted matrix

| Quantity | Count |
|---|---:|
| GOLD primary conditions | 151 |
| SILVER sensitivity-increment conditions | 16 |
| Total conditions | **167** |
| M | 50 |
| Condition-realizations | **8,350** |
| Reusable spectrum-fit inputs | **8,412** |
| Yang fit inputs / condition evaluations | 8,412 / 8,350 |
| Green preprocessing inputs / condition evaluations | 8,412 / 8,350 |
| Unclassifiable-by-design condition-classifier cells | 205 |

The 205 cells are 167 MacLeod-2019 visual-final cells, 26 Yang cells, and 12 Green cells.
MacLeod remains non-denominator without prospective visual evidence. The exact matrix and
applicability rows are `d093/final_q1_matrix_d093_unexecuted.csv` and
`d093/final_classifier_applicability_d093.csv`.

`M=50`, seed namespace `p3sf:q1:full:v1`, common-random-number structure, and transition-
level aggregation remain frozen. No instrument changed the stochastic operator, so no new
realization pilot was opened.

## Final freeze

The literature universe remains exhausted under D-092, and no further reference search is
permitted. D-093 freezes PRIMARY_Q1 at 58 GOLD and SENSITIVITY_Q1 at 62 GOLD+SILVER, grid
`[5,10]`, and `M=50`. There is no remaining instrument-level blocker:

`GO_FINAL_FULL_Q1`
