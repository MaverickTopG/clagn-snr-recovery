# Preregistered analysis plan

**Project:** The spectroscopic selection function of changing-look AGN — dependence on
signal-to-noise, host dilution, and classification criterion.

**Registered:** 2026-08-14
**Config at registration:** `00_admin/frozen_config.yaml`, config_version 1
**Repository at registration:** `e674430bbd4c23ddb599ec46fdd95fdcd54a3135`

This document is written **before** any outcome is inspected. It is prospective. If it is
ever revised, the revision is recorded in `00_admin/decisions_log.md` with a date, a
reason, and a commit hash, and the original text is preserved in git history rather than
overwritten in spirit.

---

## 1. The question

If the same genuine astrophysical transition were observed under different realistic
observing conditions, what is the probability that astronomers would classify it as a
changing-look AGN?

$$P(C = 1 \mid T,\ S/N,\ f_{\rm host},\ R,\ z,\ \mathrm{survey},\ D)$$

- $T$ — the underlying real transition
- $C$ — receives a CLAGN classification
- $f_{\rm host}$ — host-galaxy continuum fraction at 5100 Å
- $R$ — spectral resolution
- $D$ — the adopted CLAGN definition

### Notation used throughout the manuscript

$$T_i = \text{underlying reference event for object } i$$
$$C_{ijk} = \text{CL classification for object } i \text{ under condition } j \text{ using definition } k$$

This project studies the map $T_i \rightarrow C_{ijk}$. It does not study
$P(T = 1)$.

### Thesis

Changing-look classification is partly an **observational measurement** rather than a
purely intrinsic label: classification probability depends systematically on spectral
quality, host dilution, and the mathematical criterion used to define a transition.

---

## 2. Primary questions

These four are the paper. They are fixed. No result may be promoted into this list later.

| ID | Question |
|----|----------|
| **Q1** | How does CL classification probability change with spectral S/N? |
| **Q2** | How does classification probability change with host-galaxy fraction? |
| **Q3** | Do commonly used CLAGN criteria assign the same label to the same underlying transition? |
| **Q4** | After controlling for S/N and host fraction, is there residual sensitivity to the survey/instrument pairing? |

## 3. Secondary questions

Exploratory. **Non-promotable**: none of these may become a headline claim in this paper,
regardless of how large or clean the resulting effect is.

1. Turn-on versus turn-off differences
2. Hβ versus Hα
3. Spectral resolution
4. Fe II strength
5. Continuum-change amplitude
6. Line width
7. Redshift
8. Calibration perturbations
9. Whether published demographic differences could plausibly be amplified by
   observational selection
10. Fit-failure probability as a function of observing condition

---

## 4. What this project will not claim

Binding. If an analysis appears to support one of these, the correct response is to
report the analysis and explicitly decline the claim.

1. **The intrinsic CLAGN occurrence rate.** We estimate $P(C \mid T, \ldots)$, never
   $P(T = 1 \mid \text{all AGN})$.
2. **True cosmic CLAGN demographics.**
3. **Causal dependence on Eddington ratio.**
4. **A complete SDSS/DESI targeting correction.** We model
   $P(\text{classified} \mid \text{observed})$, not
   $P(\text{targeted}) \cdot P(\text{observed}) \cdot P(\text{classified} \mid \text{observed})$.
5. **Physical BLR formation or destruction** inferred from line detectability alone.
6. **That [O III] is invariant across different fibre apertures.** It is a diagnostic,
   never a commandment. See §9.
7. **That simulated spectra perfectly reproduce nature.**
8. **That thousands of noise realizations constitute thousands of independent AGN.**
   The independent scientific unit is the object. See §7.

### Language rule

The estimand is called a **conditional classification efficiency** or a **spectroscopic
recovery probability**. The phrase *survey selection function* is not used for it, because

$$P(\text{in catalogue}) = P(\text{targeted}) \cdot P(\text{observed}) \cdot P(\text{classified} \mid \text{observed})$$

and only the last factor is modelled. `14_referee_audit/prior_rejection_lessons.md`
carries the inherited language blacklist and applies here too.

---

## 5. Design

Counterfactual spectroscopy on real transitions.

1. Build a reference library of genuine transitions (Gold / Silver / Borderline).
2. Fit each epoch with a full decomposition:
   $F_\lambda = F_{\rm host} + F_{\rm AGN} + F_{\rm FeII} + F_{\rm narrow} + F_{\rm broad} + \epsilon$.
3. Generate counterfactual observations of the *same* event at controlled
   $(S/N,\ f_{\rm host},\ R,\ \text{calibration})$.
4. **Re-fit every counterfactual from scratch.** The pipeline sees only
   $(F_{\rm sim}(\lambda), \sigma(\lambda))$. A manipulated fit is never evaluated.
5. Apply published CLAGN criteria verbatim.
6. Measure $\hat{P}(C = 1)$ per condition, per criterion.

Grids, realization counts, and seeds are in `00_admin/frozen_config.yaml`.

### Q1 has three arms, because a pair has two S/N values (D-010)

A spectrum pair does not have one signal-to-noise ratio. It has `SNR_bright` and
`SNR_faint`, and the faint-state S/N is partly coupled to the physical transition itself:
in a turn-off the continuum falls along with the broad line, so the largest-amplitude
events have the poorest faint-state data. Collapsing the pair to one number would hide
the asymmetry this question exists to measure.

| Arm | Design | Role |
|---|---|---|
| **faint_only** | degrade the faint epoch; bright held at native quality | **primary** |
| **matched** | both epochs degraded to the same target | secondary — the literature convention |
| **joint** | 2-D `SNR_bright × SNR_faint` | exploratory, only where support permits |

The primary arm asks the question the paper is actually about — when does a real
transition stop being classifiable as the crucial faint spectrum worsens — and does not
discard a good bright spectrum merely because its partner is poor.

The binding metric is the Hβ-window S/N, since that governs broad-line detectability. The
5100 Å continuum S/N is recorded alongside and never used alone.

### Degradation is one-directional

S/N counterfactuals only ever *degrade*:
$\sigma_{\rm added}^2 = \sigma_{\rm target}^2 - \sigma_{\rm original}^2$, which requires
$\sigma_{\rm target} \geq \sigma_{\rm original}$. We never claim to upgrade an observed
low-S/N spectrum.

### Host injection uses the object's own host

Host dilution is imposed by rescaling the object's **own recovered** host template against
its own AGN component. A foreign galaxy spectrum is never added.

### Aperture is not simulated

Host-fraction manipulation approximates the dominant continuum-dilution consequence of
aperture variation but does not constitute a complete spatial aperture simulation. Residual
aperture effects are handled empirically through the same-instrument comparisons (Q4).

---

## 6. Ground truth, and why it is not circular

Gold status is adjudicated **without reference to any criterion under test**. An
adjudicator sees: high-S/N spectra, the full decomposition, posterior broad-line flux
distributions, difference spectra, continuum behaviour, and multiple lines where available.
An adjudicator does **not** see the output of the $N_\sigma$, flux-ratio, type-change, or
difference-spectrum classifiers.

Defining Gold by $N_\sigma > 3$ and then testing whether $N_\sigma > 3$ is a good criterion
would be circular. It is prohibited.

A subset is independently adjudicated by two people and inter-rater agreement is reported.

**Borderline cases are retained**, not discarded. They are informative about why criteria
disagree, but they never define ground truth.

---

## 7. Statistical commitments

### The independent unit is the object

With $N$ objects and $M$ Monte Carlo realizations each, the sample size is $N$, not
$N \times M$. All population confidence intervals come from a bootstrap that resamples
**objects** with replacement and carries all of a selected object's simulations with it.
Never bootstrap pixels, spectra, or realizations. Enforced by
`bootstrap.cluster: object_id` in the frozen config.

### Two uncertainties, reported separately

- **Simulation uncertainty** — how classification fluctuates between noise realizations of
  the same AGN.
- **Population uncertainty** — how results vary across different AGN.

These are never pooled into a single error bar.

### Primary contrasts

Exactly three, declared now:

| ID | Contrast |
|----|----------|
| **PC1** | Sensitivity at faint-epoch S/N = 10 versus 20, primary `faint_only` arm |
| **PC2** | Sensitivity at $f_{\rm host}$ = 0.3 versus 0.7 |
| **PC3** | Sensitivity difference between major classification criteria |

No multiplicity adjustment is applied to these three. Any secondary formal test uses Holm
adjustment. Reporting emphasises effect sizes and confidence intervals over p-values.

### Functional form

No linearity is assumed. Threshold behaviour is expected. Empirical binned completeness
surfaces are primary; restricted cubic splines / GAMs are supportive.

### Operating points, not AUC

Results are reported at interpretable operating points ("at S/N = 10 and
$f_{\rm host}$ = 0.7, criterion A recovers X% while criterion B recovers Y%"). ROC AUC is
not emphasised, because it averages over regimes astronomers do not observe in.

### Precision

Positive predictive value is reported only against an explicitly stated assumed prevalence,
because PPV depends on prevalence. Sensitivity, false-positive rate, specificity, and
balanced accuracy are reported unconditionally.

### Power

Computed once the Gold sample size is known, from the **number of objects**. No arbitrary
$N$ is manufactured. We report the minimum resolvable difference in classification
probability given $N_{\rm Gold}$.

---

## 8. Outcome is three-class, not binary

$$C \in \{\text{CL},\ \text{non-CL},\ \text{unclassifiable}\}$$

At S/N = 3 the pipeline cannot determine Hβ. That is **not** "not a CLAGN". Collapsing
`unclassifiable` into `non-CL` would manufacture a completeness deficit that is really a
measurement failure.

Consequently, fit failures are data. We report
$P(\text{fit failure} \mid S/N, f_{\rm host})$ alongside every completeness surface.
Missingness is not assumed random: if low-S/N and host-dominated spectra fail
disproportionately, then $P(\text{missing} \mid S/N, f_{\rm host}) \neq \text{const}$, and
analysing only successful fits would be survivorship bias.

---

## 9. Calibration and [O III]

[O III] is used three ways: as a calibration **warning flag**, as a **within-instrument**
consistency check, and in **sensitivity analysis**. Every primary result is repeated under
three calibration treatments — none, [O III] rescaling, pseudo-photometric scaling — and
the comparison is reported as a matrix. If conclusions do not survive all three, that
sensitivity *is* the result.

SDSS-I/II used 3″ fibres, BOSS 2″, DESI 1.5″. Therefore
$F_{\rm host}^{\rm SDSS} \neq F_{\rm host}^{\rm DESI}$ and possibly
$F_{\rm [OIII]}^{\rm SDSS} \neq F_{\rm [OIII]}^{\rm DESI}$ even for an unchanged galaxy. A
single global aperture constant applied to all objects is specifically prohibited: prior
work in this group applied a fixed 2.83× factor that exceeded its own detection threshold,
while the true mismatch for the one object checked in detail was ≈15%.

---

## 10. Uncertainties are validated, not trusted

DESI DR1 documents that its inverse-variance underestimates uncertainty increasingly above
S/N ≈ 20–30. Since $N_\sigma \propto 1/\sigma$, a wrong $\sigma$ is a wrong classification
probability.

For every spectrum we compute normalized residuals in line-free windows,
$r_\lambda = (F_\lambda - M_\lambda)/\sigma_\lambda$, and inspect $\mathrm{std}(r)$, which
should be ≈1. Where it is not, uncertainties are renormalised. This is done **separately**
for SDSS Legacy, BOSS/eBOSS, and DESI, and stratified by S/N. The measured factors are
written back into `noise_model.measured_factors`; none is assumed in advance.

Non-detections are never stored as zero flux. Every dim state carries a fitted flux, an
uncertainty, a detection significance, and an upper limit.

---

## 11. Classification criteria

Implemented **verbatim at their published thresholds** first. Threshold sweeps are labelled
robustness, never primary. Full transcription with citations in
`01_literature/selection_definitions.md`.

| Family | Form |
|---|---|
| **A** Significance | $N_\sigma = \|F_b - F_d\| / \sqrt{\sigma_b^2 + \sigma_d^2}$ |
| **B** Flux ratio | $F_{\rm bright} / F_{\rm faint}$ |
| **C** Appearance / disappearance / type change | transition in broad-line detectability or spectral type |
| **D** Difference spectrum | broad-line detection in the difference of two epochs |

Continuous statistics are retained alongside the binary labels — $\Delta \log F_{H\beta}$,
$F_b/F_d$, $S_\Delta = (F_b - F_d)/\sqrt{\sigma_b^2 + \sigma_d^2}$, $\Delta$EW,
$\Delta L_{5100}$ — because binary labels change abruptly at a threshold while the
astrophysics changes smoothly.

Turn-on and turn-off are analysed **separately**, not pooled. A line appearing above a
detection threshold is not statistically equivalent to one disappearing below it.

---

## 12. Controls

Completeness alone is insufficient; the false-positive surface
$P(\text{false CL} \mid S/N, f_{\rm host})$ is equally important.

- **Control A — synthetic null.** One latent spectrum, two independent noise realizations.
  No astrophysical change. Any CL label is a false positive by construction.
- **Control B — short-baseline repeats.** Same object, intervals too short for the modelled
  transition. Contains real instrumental and calibration variation.
- **Control C — empirically stable AGN.** Repeat spectra, stable photometry, no broad-line
  change; matched to Gold in redshift, continuum S/N, host fraction, and luminosity where
  feasible.

---

## 13. Locked validation and replication

Strata are assigned by SHA-256 of `object_id` (`p3sf.infra.splits`), 60/20/20 by **object**.

- **Development (buckets 0–5).** All debugging, window choices, and numerical fixes.
- **Validation (buckets 6–7).** Opened only when the pipeline is frozen. No threshold tuning
  after opening. Legitimate bugs may be fixed, documented in `decisions_log.md`, after which
  the pipeline is refrozen.
- **Replication (buckets 8–9).** Not inspected until the validation analysis is complete.
  Success is agreement in **sign and approximate magnitude** — overlapping intervals,
  similar curves — not identical p-values.

---

## 14. Stopping gates

Finishing a script is not a reason to proceed.

| Gate | Condition |
|---|---|
| **A** Reference sample | Gold labels are defensible and non-circular |
| **B** Fitting | Measurements agree acceptably with DESI EmFit; residuals behave |
| **C** Errors | DESI variance behaviour verified and corrected |
| **D** Simulator | Degraded spectra reproduce properties of genuine low-quality observations |
| **E** Controls | False-positive behaviour quantified |
| **F** Validation | No methodological tuning after this point |
| **G** Replication | Primary effect reproduces in sign and shape |
| **H** Manuscript | Every strong claim corresponds directly to an experiment |

---

## 15. Exclusions

Every excluded object gets a row in `00_admin/exclusions_log.csv` with a reason code drawn
from the frozen vocabulary in that file's header. The manuscript reports "N objects were
excluded for predefined reason X". The phrase "we visually removed several weird spectra"
is not acceptable and its absence is enforced by a test.

A CONSORT-style sample-flow diagram accompanies the reference-sample section.

---

## 16. A null result is publishable

If $P(C)$ barely changes with S/N, that is informative, because the contemporary literature
strongly suspects such dependence. If all definitions agree, that is useful. If host
dilution hardly matters, that is interesting.

The only genuinely bad outcome is simulations that are not realistic enough to support
inference — which is why simulator validation (Gate D) carries as much weight as the
headline experiment.

We are not optimising for a sensational conclusion. "Criterion A is more complete but has a
higher false-positive rate" is a better result than "criterion A is more complete", because
it gives the field a tradeoff rather than a ranking.
