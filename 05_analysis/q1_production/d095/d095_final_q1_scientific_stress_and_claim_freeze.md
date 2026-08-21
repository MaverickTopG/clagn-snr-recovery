# D-095 — final Q1 scientific stress test and manuscript-claim freeze

Date: 2026-08-16  
Frozen input: D-094 immutable production products  
Decision: **Q1_SCIENCE_PARTIAL**

No D-094 result was modified, rerun, retuned or replaced. D-095 did not import or invoke
the production execution path. It read the frozen D-094 condition, classifier, fit-audit,
transition and checksum products only.

## 1. Independent accounting reconstruction

D-095 independently reconstructs:

- 167 unique frozen conditions;
- 8,350 unique condition-realizations;
- 25,050 terminal classifier outcomes, exactly three per condition-realization;
- 501 transition/condition/classifier summaries, each with M=50.

Every row obeys

`N_CL + N_nonCL + N_unclassifiable = N_eligible = 50`

and

`N_CL + N_nonCL = N_classifiable`.

Reconstructed transition counts, `p_i`, `u_i`, and primary/GOLD+SILVER aggregates match
the frozen D-094 tables exactly (absolute tolerance `1e-15`). All six immutable raw files
match their frozen byte sizes and SHA-256 hashes. Independent accounting status is PASS.

## 2. Green S/N effect on paired common support

Only GOLD transitions with classifiable Green `p_i` at both rungs in the same arm enter
the paired contrast.

| Arm | Paired N | Mean `p_i(10)-p_i(5)` | 95% paired CI | Median | IQR | Range | Positive / zero / negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| faint-only | 27 | +0.290 | 0.172–0.416 | +0.120 | 0.000–0.630 | −0.020–+0.880 | 18 / 8 / 1 |
| matched | 25 | +0.259 | 0.144–0.383 | +0.180 | 0.000–0.440 | −0.060–+0.960 | 18 / 6 / 1 |

The interval is the percentile 95% CI for the paired-transition mean from 5,000
object-level resamples. It does not resample realizations as independent AGN.

The full supported-rung comparison and paired common-support comparison are:

| Arm | Full R(5) | Full R(10) | Unmatched difference | Common-support R(5) | Common-support R(10) | Paired difference |
|---|---:|---:|---:|---:|---:|---:|
| faint-only | 0.329 | 0.573 | +0.245 | 0.284 | 0.573 | +0.290 |
| matched | 0.132 | 0.374 | +0.242 | 0.115 | 0.374 | +0.259 |

The paired effect is at least as large as the unmatched-rung difference. Therefore the
central Green conclusion is not produced by the lower-support S/N=10 population being a
different or easier set of transitions. The headline claim is based on the paired values.

No “substantial change” cutoff was prospectively frozen, so D-095 reports the complete
continuous distribution and sign counts without inventing a dichotomizing threshold.

## 3. Paired Green arm effect

Only transitions classifiable in both arms at the same rung enter this contrast. Positive
values mean faint-only recovery exceeds matched recovery.

| S/N | Paired N | Mean `p_i(faint-only)-p_i(matched)` | 95% paired CI | Median | IQR | Range | Positive / zero / negative |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 43 | +0.197 | 0.113–0.292 | +0.020 | 0.000–0.330 | −0.180–+0.980 | 23 / 15 / 5 |
| 10 | 25 | +0.245 | 0.141–0.362 | +0.120 | 0.000–0.440 | −0.040–+0.860 | 17 / 5 / 3 |

Thus degrading the bright epoch as well as the faint epoch reduces Green recovery in the
paired reference experiment at both rungs, while retaining visible transition-level
heterogeneity.

## 4. Yang failure-mode decomposition

Across all Q1 conditions Yang has 5,756 unclassifiable rows. Of these, 1,300 are frozen
inapplicable-by-design and 4,456 are applicable rows that fail closed. Mutually exclusive
priority categories for the 4,456 applicable failures are:

| Frozen failure category | N | Fraction of applicable failures |
|---|---:|---:|
| degraded fit-boundary failure | 3,387 | 76.0% |
| native boundary failure | 550 | 12.3% |
| local fit invalidity | 498 | 11.2% |
| measurement unavailable | 21 | 0.5% |
| nondetection-semantics failure | 0 | 0.0% |
| other | 0 | 0.0% |

Boundary failures therefore dominate: native plus degraded boundary categories account
for 3,937/4,456 = 88.4% of applicable Yang failures. Nonexclusive diagnostics find 3,635
degraded boundary incidences, 550 native boundary incidences, 1,250 local-invalidity
incidences and 32 measurement-unavailable incidences, allowing overlaps.

Within GOLD specifically, the priority counts are 3,031 degraded boundary, 550 native
boundary, 475 local invalidity, 19 measurement unavailable and 1,300 inapplicable by
design. The source-permitted valid-nondetection-as-zero convention was used in 18
classifiable Yang rows; it caused no unclassifiable failure. No validity rule is changed.

## 5. Yang/Green both-classifiable contingency

The following counts use GOLD realization cells only. The four central columns include
only cells where both classifiers are genuinely classifiable.

| Arm | S/N | Both classifiable | Green CL / Yang CL | Green CL / Yang non-CL | Green non-CL / Yang CL | Green non-CL / Yang non-CL |
|---|---:|---:|---:|---:|---:|---:|
| faint-only | 5 | 613 | 90 | 113 | 160 | 250 |
| faint-only | 10 | 372 | 71 | 174 | 44 | 83 |
| matched | 5 | 695 | 36 | 24 | 224 | 411 |
| matched | 10 | 442 | 37 | 103 | 81 | 221 |

Disagreement counts are therefore 273, 218, 248 and 184, respectively. Asymmetric
unclassifiability is kept separate:

| Arm | S/N | Green classifiable / Yang unclassifiable | Green unclassifiable / Yang classifiable | Both unclassifiable |
|---|---:|---:|---:|---:|
| faint-only | 5 | 1,587 | 30 | 170 |
| faint-only | 10 | 978 | 0 | 100 |
| matched | 5 | 1,455 | 23 | 177 |
| matched | 10 | 808 | 0 | 100 |

Unclassifiability is not interpreted as classifier disagreement or as a physical
difference between CLAGN.

## 6. Prespecified robustness on the paired Green effect

| Analysis | faint-only mean `p_i(10)-p_i(5)` | matched mean `p_i(10)-p_i(5)` |
|---|---:|---:|
| Primary GOLD | +0.290 | +0.259 |
| GOLD+SILVER | +0.294 | +0.268 |
| Discovery-independent Green | +0.298 | +0.240 |
| Equal-source-family weighted | +0.180 | +0.303 |
| LOTO range | +0.267 to +0.302 | +0.230 to +0.273 |
| LOSFO range | +0.215 to +0.337 | +0.239 to +0.291 |

Magnitude variation is material even though every prespecified sign is positive.
Equal-family weighting attenuates the faint-only effect from +0.290 to +0.180 (−0.110,
38%) while increasing the matched effect to +0.303. Removing Dong2025 produces the
largest faint-only attenuation, from +0.290 to +0.215 (−0.075, 26%); removing Zeltyn2024
gives +0.248 (−0.042, 14%). No single matched-arm family removal attenuates by more than
0.020: the lowest is +0.239 after removing LaMassa2015, followed by +0.240 after removing
Green2022 and +0.242 after removing Dong2025.

The correct robustness statement is therefore that the Green S/N direction persists,
while its magnitude depends meaningfully on family weighting—especially faint-only.

## 7. Mask-contract incident disclosure

The root cause, 6,305 affected inputs, 2,107 unaffected inputs, pre-interpretation
detection, provenance-only rerun rule, exact seed verification and evidence against an
outcome-conditioned rerun are frozen in
`mask_contract_incident_methods_disclosure_d095.md`. D-095 independently reconstructs
the 6,305/2,107 partition and every one of the 8,350 degraded-task seeds.

## 8. Frozen claims and overall status

The complete claim tiers are frozen in `allowed_scientific_claims_d095.md`.

The exact overall decision is:

    Q1_SCIENCE_PARTIAL

Reasons:

1. Independent accounting and reproducibility checks pass.
2. Green's paired/common-support S/N and arm effects have positive confidence intervals
   and survive every prespecified sensitivity in direction.
3. Effect magnitude is heterogeneous and source-family-weight sensitive.
4. Yang does not show the same dependence and is dominated by frozen boundary failures.
5. Yang/Green disagreement is large even when both are classifiable.
6. MacLeod remains non-denominator, and no population-rate or demographic-explanation
   interpretation is justified.

No manuscript prose is changed by D-095.
