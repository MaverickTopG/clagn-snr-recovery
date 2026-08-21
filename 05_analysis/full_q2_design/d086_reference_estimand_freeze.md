# D-086 — reference truth, pair eligibility, and full-Q2 estimand freeze

**Status:** `NO_GO_FULL_Q2_REFERENCE_SUPPORT_INSUFFICIENT`  
**Execution:** no full-Q2 spectra, fits, or classifier outputs generated.

## 1. Independent Paper-3 reference tier

The 17 development candidates were adjudicated without opening or using D-085 classifier
outcomes. The old benchmark `confidence=gold` field was treated only as source provenance.
Evidence was restricted to raw authoritative transition endpoints where available and
descriptive primary-source spectra/light-curve provenance otherwise.

The prospective tier rule was:

- `GOLD`: an unambiguous broad-line appearance/disappearance plus at least one independent
  corroborator (another broad line, coordinated continuum change, or repeated endpoint
  spectra), without relying on the tested decision statistic;
- `SILVER`: credible transition evidence with one important limitation, such as single-line
  support, retained broad emission, unavailable local follow-up endpoint, or host/line
  complexity;
- `BORDERLINE`: strong variability but no decisive appearance/disappearance, unresolved
  spectral-quality concern, or explicitly marginal source description.

Frozen counts are **4 GOLD, 9 SILVER, 4 BORDERLINE**. The exact primary reference set is:

- SDSSJ082942.66+415436.8
- SDSSJ102152.34+464515.6
- SDSSJ105325.40+302419.34
- SDSSJ233602.98+001728.7

This GOLD set is unexpectedly too small for a persuasive population recovery estimate, and
only one GOLD event survives physical pair-consistent eligibility. The tier rule is not
weakened. Before full Q2, the candidate pool must instead be expanded from independent
discovery routes and exact endpoints acquired and adjudicated under the same frozen rule.
SILVER is a prespecified sensitivity set only. BORDERLINE is never ground truth.

## 2. Independent unit and resampling

One `object_id`/real transition is one scientific unit. Host shapes, rungs, criteria,
spectra, fits, and any future realizations are repeated measurements on that unit. Every
count in `p3sf.stats.q2` deduplicates by object within `(f,H,k)`. Any future bootstrap or
randomization resamples transition IDs and carries all rows belonging to the sampled ID;
pixels, spectra, host shapes, cells, and fitted realizations are never resampled as AGN.

## 3. Pair-consistent eligibility

The literal shared-amplitude estimand requires exact transition endpoints with a defensible
common flux/aperture basis. “Both are SDSS” is insufficient: SDSS Legacy used 3 arcsec fibers
whereas BOSS/eBOSS used 2 arcsec fibers.

Frozen accounting across all 17 transitions:

- `PAIR_CONSISTENT_ELIGIBLE`: **1** — J082, exact SDSS-Legacy 3″/3″ endpoints;
- `SPECTRUM_DIAGNOSTIC_ONLY`: **10** — exact endpoints exist but use 3″/2″ or 2″/3″
  apertures, so constant physical galaxy flux is not defensible;
- `OUT_OF_DOMAIN`: **6** — an exact event endpoint is unavailable locally. Later DESI
  observations cannot replace the historical event endpoints.

Among GOLD, J102, J105325, and J233 are diagnostic-only because of 3″/2″ aperture mismatch.
J161's same-aperture DESI pair from D-084 is a later repeated condition, not the published
53501/57596 transition, and is therefore out of domain for reference-event full Q2.

## 4. Frozen empirical estimands

For criterion `k`, rung `f`, and explicit host shape `H`, define

$$R(f,H,k)=\frac{N_{\rm CL}(f,H,k)}{N_{\rm classifiable}(f,H,k)}$$

and

$$U(f,H,k)=\frac{N_{\rm unclassifiable}(f,H,k)}{N_{\rm eligible}(f,H,k)}.$$

`R` is the **empirical recovery fraction over the locked reference transitions**. It is not
a survey selection function or an independent probability observation per cell. `R` is
undefined when `N_classifiable=0`; it is never filled with zero. Report, without omission:

`N_eligible`, `N_fit_completed`, `N_fit_valid`, `N_classifiable`, `N_CL`, `N_nonCL`, and
`N_unclassifiable`, with

$$N_{\rm CL}+N_{\rm nonCL}=N_{\rm classifiable},\qquad
N_{\rm classifiable}+N_{\rm unclassifiable}=N_{\rm eligible}.$$

Fit failures, invalid fits, unavailable protocol measurements, and criterion-support gaps
are terminal subclasses of `N_unclassifiable`; none disappear by conditioning on fit
survivors.

## 5. Frozen scientific contrasts

- **Host amplitude:** for each explicit `H`, report
  $R(f,H,k)-R(0,k)$ at `f=0.1,0.5,0.85`. The single no-op has no host-shape level.
- **Host shape at fixed amplitude:** report all three values and their range
  $\max_H R(f,H,k)-\min_H R(f,H,k)$, plus named pairwise contrasts where defined.
  Never average or weight H1/H2/H3.
- **Transition × host:** list each transition's label trajectory and whether it changes
  relative to its no-op; only pairs classifiable in both compared cells receive a flip/no-flip
  statement. Unclassifiable transitions remain visible.
- **Criterion × host:** report the vector of criterion-specific changes under the same cell;
  do not collapse criteria or convert search-stage results into labels.
- **Shape dependence by rung:** compute the H1/H2/H3 range separately at 0.1, 0.5, and 0.85
  to test whether D-084's shape dependence is confined to the extreme rung.

No confidence interval or stronger population interpretation is authorized at the current
GOLD support. A later object-level interval requires an independently expanded reference
pool and transition-cluster resampling.

## 6. Conditions and no-op invariance

For each eligible pair the matrix remains exactly one no-op plus
`[0.1,0.5,0.85] × [XSL_H1,XSL_H2,XSL_H3]` = **10 conditions**. With the presently eligible
GOLD set, the unexecuted matrix therefore contains **10 cells**, all for J082. There are no
new rungs, templates, weights, shape averages, or independently normalized epochs.

Before any nonzero cell, each pair's no-op must hash-identically reproduce authoritative
wavelength, flux, corrected IVAR/uncertainty, mask, resolution, and frozen measurement
inputs/classifications. `NOOP_IDENTITY_FAIL` is an engineering stop; it produces no science
label and no recovery estimate.

## 7. Prospective visual-final decision

**Option B is frozen.** No injected-cell human adjudication will be introduced for the
current full-Q2 design.

- MacLeod 2019 final status is unclassifiable when the required blinded visual-transition
  flag is unavailable.
- Guo final catalogue status remains a non-denominator confirmation protocol; its numeric
  quick screen stays a search stage.
- Potts–Villforth remains a non-denominator search stage; its final visual type-transition
  rejection is not reconstructed.

Changing to Option A would require a new prospective design gate, independent adjudicators,
fixed displays/order/masking, and reliability rules **before any injected spectrum is
viewed**. It cannot be added in response to observed flips.

## 8. J082 and GO/NO-GO

J082 remains unclassifiable for every criterion that requires its unsupported calibrated
T2-B Hβ variance. Its corrected point estimate and calibrated continuum quantities remain
available only under their existing criterion-specific support policy. No unsupported
quantity becomes non-CL.

The final recommendation is:

> **NO-GO for full-Q2 execution.**

The physical GOLD pair-consistent denominator is one transition, and that transition has a
binding uncertainty-calibration limitation for the variance-based source-verified criteria.
This cannot establish the preregistered transition, criterion, or host-shape interactions.
Expand and independently adjudicate the candidate pool, acquire exact same-basis endpoints,
and close remaining Gates B and D before seeking a new execution authorization. Do not
change the tier or pair-eligibility rules.
