# D-089 — Q1 source-measurement validity and stability closeout

## Scope and preserved decisions

D-088 remains binding: final S/N rungs `[5,10]`, `faint_only` primary arm, `matched`
secondary arm, the exact D-087 GOLD/SILVER eligibility sets, the 30-condition matrix,
object-specific common random numbers and equal-transition aggregation. D-084 through D-088,
Gate C, host work and the permanent physical-Q2 no-go are unchanged. No full-Q1 degraded
spectrum was generated.

## Yang H-beta measurement contract

For each epoch the production audit exposes exactly:

`fit_completed`, `optimizer_success`, `parameter_at_bound`, `local_hbeta_coverage`,
`local_mask_fraction`, `continuum_valid`, `broad_component_present`,
`broad_component_amplitude`, `broad_component_width`, `broad_flux`,
`residual_rms_local`, `finite_measurement`, `fit_valid`, `invalid_reason`, and
`valid_nondetection`.

Hard validity is prospective and structural:

1. the fit completes and the H-beta optimizer reports success;
2. every 2-Angstrom bin over rest 4640--5100 Angstrom has usable coverage;
3. the continuum optimizer succeeds and its model is finite;
4. no active broad-component scale, width or center is pinned to an allowed boundary;
5. broad flux is finite and nonnegative and any present component has finite positive width;
6. the fitted broad-line FWHM is within Yang's 1200--20,000 km/s source-model domain.

Failure of any condition is `unclassifiable`. Local residual RMS is persisted only as a
diagnostic; no empirical RMS or mask-fraction cutoff was selected after observing labels.
Complete 2-Angstrom coverage is the mask/coverage gate.

Yang final is the strict fitted broad-line ratio
`f(Hbeta,faint)/f(Hbeta,bright) < 0.3`. Only a fit that passes all validity gates and whose
broad component is genuinely absent at its zero-scale model boundary is a valid fitted
nondetection; only then does Yang locally substitute faint flux zero. Fit failure,
component unavailability and an arbitrary numerical zero remain unclassifiable. The zero is
not written to a master measurement table.

The audit contains 416 rows: 16 exact native endpoint fits and 400 pilot fits. Of these,
293 pass and 123 fail closed. Active parameter-bound contact occurs in 120 rows; four rows
are outside the Yang source width range (one overlaps the boundary set). Native invalid fits
are J012 faint, J082 bright, J102 faint and J233 faint. Exact row-level values and reasons
are in `measurement_validity_d089/yang_fit_audit_rows.csv` and the 16-row native subset in
`measurement_validity_d089/yang_native_fit_audit.csv`.

## Green exact pixel statistic

- Frame: rest wavelength.
- Epoch preprocessing: separate Galactic dereddening and host/continuum decomposition;
  power law, optical/UV Fe II and Balmer continuum; no polynomial term. The resulting line
  spectrum is used.
- Scaling: the exact SDSS/SDSS pairs retain scale 1 because no source outlier trigger was
  prespecified. Provenance is
  `NO_RESCALE_EXACT_SDSS_PAIR_NO_SOURCE_OUTLIER_TRIGGER`.
- Sampling: flux-conserving 2-Angstrom bins; variance propagated from the exact overlap
  weights.
- Pixel array: `(f_bright-f_dim)/sqrt(var_bright+var_dim)`; signed, never absolute.
- Smoothing: centered 16-pixel median (32 Angstrom) with explicitly clipped endpoint
  windows.
- Statistic: subtract the smoothed 4750-Angstrom value, then take the maximum signed
  relative value over 4750--4940 Angstrom.
- Final Green comparison: inclusive `N_sigma(Hbeta) >= 3`.
- Masks/provenance: every contributing bin must have full coverage and every contributing
  input variance must be `NATIVE_SDSS_SUPPORTED` or `GATE_C_CORRECTED_SUPPORTED`; otherwise
  the measurement is unavailable and the classifier is unclassifiable.

All D-089 pilot inputs are exact SDSS pairs and use native SDSS inverse-variance semantics.
No DESI correction is applied. J082's unsupported later DESI T2-B variance therefore does
not propagate into its exact SDSS/SDSS Q1 pair.

## Deterministic fixtures

`tests/test_q1_source_measurements.py` freezes Yang arithmetic and the strict 0.3 edge,
valid nondetection semantics, boundary/failure/source-width-domain failure to
unclassifiable, Green analytic per-pixel variance arithmetic and inclusive 3 edge, signed
direction, required-region mask behavior, remote-mask irrelevance, and unsupported-variance
failure. No empirical real-object label is frozen in a unit test.

## Corrected frozen stability pilot

Counts are `(CL, non-CL, unclassifiable)` at common `M=50`; `p50=CL/classifiable` and
`u50=unclassifiable/50`.

| transition | S/N | arm | criterion | counts | p50 | u50 |
|---|---:|---|---|---:|---:|---:|
| J012 | 5 | faint_only | Green | 2,48,0 | 0.040 | 0.00 |
| J012 | 5 | faint_only | Yang | 3,13,34 | 0.188 | 0.68 |
| J012 | 5 | matched | Green | 0,50,0 | 0.000 | 0.00 |
| J012 | 5 | matched | Yang | 3,10,37 | 0.231 | 0.74 |
| J012 | 10 | faint_only | Green | 33,17,0 | 0.660 | 0.00 |
| J012 | 10 | faint_only | Yang | 6,16,28 | 0.273 | 0.56 |
| J012 | 10 | matched | Green | 10,40,0 | 0.200 | 0.00 |
| J012 | 10 | matched | Yang | 4,13,33 | 0.235 | 0.66 |
| J082 | 5 | faint_only | Green | 29,21,0 | 0.580 | 0.00 |
| J082 | 5 | faint_only | Yang | 0,0,50 | undefined | 1.00 |
| J082 | 5 | matched | Green | 0,50,0 | 0.000 | 0.00 |
| J082 | 5 | matched | Yang | 1,32,17 | 0.030 | 0.34 |
| J082 | 10 | faint_only | Green | 50,0,0 | 1.000 | 0.00 |
| J082 | 10 | faint_only | Yang | 0,0,50 | undefined | 1.00 |
| J082 | 10 | matched | Green | 13,37,0 | 0.260 | 0.00 |
| J082 | 10 | matched | Yang | 1,33,16 | 0.029 | 0.32 |

The complete summary also reports prefix recovery at M=5/10/20/40/50 and finite-measurement
mean, standard deviation, median, q10 and q90 in
`measurement_validity_d089/stability_summary.csv`.

## Convergence and final M

M=20 was not sufficient: corrected final-label prefixes changed materially between M=10 and
M=20 in several groups. Before producing realizations 20--49, the final-step acceptance rule
was written to `measurement_validity_d089/convergence_rule_before_extension.md`: every group
must have `|p50-p40|<=0.10` (or both values undefined) and `|u50-u40|<=0.10`.

All 16 groups pass. The maximum recovery change is 0.060 and maximum unclassifiable-fraction
change is 0.045. Exact group checks are in
`measurement_validity_d089/convergence_m40_m50.csv`. A single classifiable
J082/SNR-10/matched Yang draw has ratio 196.38 because its degraded bright fitted flux is
small; the source ratio is unbounded, the value is retained, and the categorical prefix is
stable. It is why robust quantiles are reported beside mean/SD rather than silently clipped.

Freeze a common `M=50`. No object-specific M is allowed.

For transition `i`, rung `s`, arm `a`, and final classifier `k`:

`p_i(s,a,k) = n_CL,i / n_classifiable,i`, undefined when no realization is classifiable;

`u_i(s,a,k) = n_unclassifiable,i / 50`.

Across-transition summaries give each transition equal weight. Realizations remain repeated
measurements, not independent AGN, and are never pooled across objects.

## Gate decision

`GO_FULL_Q1`

Yang validity is explicit and fail-closed; Green's statistic and variance provenance are
source-faithful; all fixed pilot groups pass the prospective M50 rule; aggregation preserves
transition-level independence. This decision authorizes no execution in D-089: the frozen
30-condition full-Q1 matrix remains unexecuted.
