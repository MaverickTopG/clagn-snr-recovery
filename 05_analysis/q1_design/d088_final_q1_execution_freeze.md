# D-088 — final Q1 grid and execution-readiness freeze

**Reference/eligibility basis:** D-086/D-087 unchanged.  
**Final Q1 grid:** `[5, 10]`, binding metric = rest-frame H-beta window `[4700,5100]` A.  
**Full-Q1 execution:** `NO_GO_FULL_Q1_SOURCE_MEASUREMENT_AND_FIT_VALIDITY_UNRESOLVED`.  
**Full degraded spectra generated:** none. A prospectively bounded two-object engineering
stability pilot ran in memory and retained only scalar/fit-status CSV outputs.

## Native support on the exact D-087 endpoints

Support means the native H-beta-window S/N is at least the target: the faint epoch for
`faint_only`, and both epochs for `matched`. A spectrum is never upgraded. The continuum
5100-A S/N is retained as a reported diagnostic but is not substituted for the frozen
binding metric.

| S/N | GOLD faint-only | GOLD matched | GOLD+SILVER faint-only | GOLD+SILVER matched | disposition |
|---:|---:|---:|---:|---:|---|
| 5 | 4 | 4 | 8 | 8 | final |
| 10 | 3 | 3 | 7 | 7 | final |
| 15 | 1 | 1 | 3 | 3 | underpowered; excluded |
| 20 | 1 | 1 | 2 | 2 | underpowered; excluded |
| 30 | 0 | 0 | 1 | 0 | underpowered; excluded |
| 40 | 0 | 0 | 0 | 0 | unreachable; removed |

The final `[5,10]` grid is a prospective support decision, not an outcome-selected grid.
At S/N 5 it retains all 4 PRIMARY_Q1 GOLD and all 8 GOLD+SILVER transitions. At S/N 10 it
retains 3/4 GOLD and 7/8 GOLD+SILVER; J233 is absent because degrading cannot increase its
native S/N. Rungs 15 and above cannot support a defensible primary transition-level
contrast. Exact endpoint values and per-arm flags are in `native_snr_support_d088.csv`.

## Mandatory validity chain

Every transition/arm/rung/realization/criterion follows this monotone chain:

`eligible -> degradation_succeeded -> spectral_QC_passed -> fit_completed -> fit_valid ->`
`required_measurement_available -> variance_calibration_supported -> classifiable`.

The first failed stage is retained explicitly. No later stage can restore eligibility.
Failure at any required stage ends as `unclassifiable`, never `non-CL`. The code-level
state machine is `p3sf.design.q1.Q1Validity`.

- Degradation validity requires a supported native rung, deterministic seed provenance,
  finite propagated errors, the requested H-beta-window S/N within the frozen numerical
  tolerance, and exact faint-draw identity across arms. `faint_only` also requires the
  bright arrays to be bit-for-bit unchanged.
- Spectral QC requires usable masks/errors and rest-frame coverage for every measurement
  consumed by the criterion.
- `fit_completed` only says the fitter returned. `fit_valid` additionally requires finite,
  physically admissible required outputs, no required parameter pinned to a bound, and the
  frozen local H-beta/[O III] residual/quality audit. A PyQSOFit `status=ok` alone is not this
  audit.
- Deterministic PyQSOFit point estimates are retained as point estimates. Formal fit errors
  are not fabricated. A criterion requiring an unavailable uncertainty remains
  unclassifiable.
- A source-defined nondetection may map to a protocol value only where the primary source
  defines that operation and the nondetection measurement itself is valid. Fit failure,
  invalid fit, missing coverage, or missing uncertainty never becomes a nondetection.
- Variance-consuming criteria require supported calibrated variance for the exact record
  and measurement domain. J082 retains the T2-B restriction; its exact Q1 pair is SDSS-only,
  so this does not license a later DESI substitute.

## Criterion-specific realization strategy

- **Yang 2024 final:** refit the degraded spectrum per realization and use faithful broad
  H-beta point fluxes plus a valid protocol nondetection state. No formal fitted-flux error
  is required. The native bright fit is deterministic and reusable in `faint_only`; the
  matched bright epoch is refit per realization. Final labels remain unavailable until the
  fit-boundary/local-validity audit is implemented.
- **Green 2022 final:** recompute its source-protocol continuum-subtracted pixel statistic
  and propagated variance per realization. Required operations include the common spectral
  rescaling basis, about-2-A rebinning, 32-A median smoothing, and the maximum over
  4750–4940 A relative to 4750 A. A fitted line-flux/error surrogate is prohibited. The
  source measurement pipeline and its variance validation are not implemented, so current
  final labels are unclassifiable.
- **MacLeod 2019 final:** requires prospective visual H-beta appearance/disappearance plus
  source-protocol `N_sigma(Hbeta)>3`. D-086 chose no post-outcome visual adjudication;
  therefore these cells are unclassifiable when the visual evidence is unavailable.
- MacLeod 2016, Guo final, and Potts–Villforth remain non-denominator protocols. Guo quick
  screen remains search-stage only. None may be promoted to a final classifier.

## Bounded realization-stability pilot

The subset and settings were fixed before inspecting pilot outcomes:

- GOLD/high-native-S/N J082 and SILVER/lower-native-S/N J012;
- both final candidate rungs 5 and 10;
- `faint_only` and `matched` arms;
- pilot `M=20`;
- base seed `314159`, namespace `p3sf:q1:d088:stability:v1`;
- identical faint stream across arms at fixed transition/rung/realization, a distinct
  matched-bright stream, and independent object-specific streams.

This produced 160 arm-level rows and 162 actual diagnostic fits (two reusable native-bright
fits plus 160 realization-dependent faint/matched-bright fits). All engineering fits
returned `status=ok`, and achieved S/N was numerically stable. These are not final-valid fit
claims. All 160 Yang-final and all 160 Green-final rows were unclassifiable under the frozen
validity rules.

The diagnostic-only Yang shortcut was stable for J012 (20/20 CL in every condition) and for
J082 at S/N 10 (20/20 non-CL in both arms). J082 at S/N 5 changed between CL and non-CL:
15/20 versus 5/20 in each arm. Its prefix CL fraction changed from 1.0 at M=5, to 0.9 at
M=10, to 0.75 at M=20. This is engineering evidence that M=20 is not automatically adequate;
it is not a population or classifier result.

Because no pilot row could enter either executable final-classifier denominator, a final M
cannot be chosen from final-label stability. `mc_realizations: 50` remains a legacy planning
placeholder and is explicitly not D-088 authorization. **Final M = NOT FROZEN.** The exact
blockers are the missing fit-boundary/local-validity audit for Yang and the missing
source-faithful Green measurement/variance pipeline. After those are closed, the same
prospective subset and candidate-M convergence rule must be rerun under a new namespace;
the pilot namespace must never be reused for full execution.

## Frozen seeds and aggregation

The seed base remains `314159`. The proposed full-run namespace is
`p3sf:q1:full:v1`, but it is inactive while M is unfrozen. Seeds hash namespace,
transition ID, exact spectrum ID, rung, realization, and stream role. The faint key omits
arm so the same draw is reused across arms; matched bright uses its own role-specific key.

For transition `i`, condition `(s,a)`, and final criterion `k`:

`p_i(s,a,k) = n_CL,i / n_classifiable,i`, undefined when the transition has no classifiable
realization, and `u_i(s,a,k) = n_unclassifiable,i / M`.

The empirical recovery summary is the equal-transition mean of defined `p_i`; the
unclassifiable summary is the equal-transition mean of `u_i`. Realizations are never pooled
as independent AGN. Also report the full nested counts `N_eligible`, `N_fit_completed`,
`N_fit_valid`, `N_classifiable`, `N_CL`, `N_nonCL`, and `N_unclassifiable`, with
`R=N_CL/N_classifiable` and `U=N_unclassifiable/N_eligible` at the transition level specified
in D-087. Primary and sensitivity results stay separate.

Prespecified robustness outputs are arm contrast, rung contrast, criterion contrast,
transition-level response tables, per-transition `p_i/u_i`, classifiable-realization counts,
and leave-one-transition-out summaries for the four-GOLD PRIMARY_Q1 set. No population
probability claim is licensed by four GOLD transitions.

## Final unexecuted matrix and workload

The matrix has 14 PRIMARY_Q1 GOLD transition-arm-rung conditions and 16 SILVER sensitivity
increment conditions, 30 total. With three source-verified final-classifier rows this is 90
criterion-evaluation cells per realization; MacLeod 2019 is retained but expected to be
unclassifiable without visual evidence.

For a future fixed M, full GOLD+SILVER execution would contain `30M` condition-realization
records and `90M` classifier evaluations. Reusing the common faint spectrum across arms and
the deterministic native bright fit gives `30M + 8` unique spectrum fits (primary GOLD only:
`14M + 4`). No numeric final fit count is scientifically frozen because M is unresolved.

## Recommendation

`NO_GO_FULL_Q1_SOURCE_MEASUREMENT_AND_FIT_VALIDITY_UNRESOLVED`

The grid, validity chain, seed convention, aggregation, denominators, and unexecuted matrix
are frozen. Full Q1 must not run until the Yang fit-validity audit and source-faithful Green
measurement/variance layer are implemented and validated, followed by a fresh bounded
final-label stability check that can freeze M. D-084 through D-087, all reference tiers,
eligibility sets, classifiers, Gate C, and the permanent physical-Q2 no-go remain unchanged.
