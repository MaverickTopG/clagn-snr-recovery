# Published CLAGN definitions, transcribed

Primary sources used for D-085: MacLeod et al. 2016
([arXiv:1509.08393](https://arxiv.org/abs/1509.08393)); MacLeod et al. 2019
([arXiv:1810.00087](https://arxiv.org/abs/1810.00087)); Green et al. 2022
([arXiv:2201.09123](https://arxiv.org/abs/2201.09123)); Potts & Villforth 2021
([arXiv:2104.14225](https://arxiv.org/abs/2104.14225)); Guo et al. DESI II
([arXiv:2408.00402](https://arxiv.org/abs/2408.00402)).

> **STATUS: SOURCE-VERIFICATION GATE D-085.** MacLeod (2016), MacLeod (2019),
> Green (2022), Guo DESI II, and Potts & Villforth (2021) were checked against
> their primary papers. Rows still marked `TO_VERIFY` remain locked.
>
> Every threshold, window, and quantity in this file must be read out of the paper itself
> and the `verified_against_pdf` column in `literature_matrix.csv` flipped to `yes` before
> the corresponding classifier is written. Preregistration §11 requires criteria to be
> implemented **verbatim at their published thresholds**; a plausible-looking guess would
> silently become a strawman, which is exactly the failure this paper exists to expose.
>
> Gate: `19_apply_nsigma_definition.py` and its siblings refuse to run against an
> unverified row.
>
> **Owner: the author, first-hand.** This transcription is deliberately not automated
> and not delegated. If a referee asks whether a criterion was implemented as published,
> the strongest answer is that the threshold was read out of the paper by the person
> making the claim. Do not fill these tables from secondary sources or from memory.

For each criterion we must pin down five things, because papers differ on all five and the
differences are the subject of Q3:

1. **The quantity** — line flux, equivalent width, or line-to-continuum ratio?
2. **The line** — Hβ, Hα, Mg II, He II?
3. **The continuum window** — where is the continuum measured, and rest or observed frame?
4. **The threshold** — exact numeric value.
5. **The auxiliary requirements** — S/N floors, visual inspection, photometric corroboration,
   host subtraction, minimum baseline. These are part of the definition even when the paper
   presents them as sample cuts.

---

## Source-verified MacLeod / Green Hβ pixel-deviation statistic

This is **not** integrated fitted broad-line flux divided by its fit uncertainty. For each
rest-frame spectral element,

$$N_\sigma(\lambda) = \frac{f_{\rm bright}(\lambda)-f_{\rm dim}(\lambda)}
{\sqrt{\sigma_{\rm bright}^2(\lambda)+\sigma_{\rm dim}^2(\lambda)}}.$$

The source workflow rescales epochs ([O III], with documented host-based fallbacks),
subtracts continuum components, rebins flux and variance to about 2 Å/pixel rest frame,
and applies a 32 Å running median (Green: normally 16 pixels). The reported Hβ statistic
is the maximum smoothed deviation over 4750–4940 Å relative to the smoothed value at
4750 Å.

| Source | Final decision | Candidate-stage items that are **not** final cuts |
|---|---|---|
| MacLeod et al. 2016 | No scalar final classifier; final ten were visually identified BEL changes | $|\Delta g|>1$ mag photometric search |
| MacLeod et al. 2019 | visual broad-Hβ appearance/disappearance **and** $N_\sigma(H\beta)>3$ | $\Delta g>1$, $\Delta r>0.5$ and light-curve vetting |
| Green et al. 2022 | bona-fide catalogue status at $N_\sigma(H\beta)\geq3$ (Table 2 note) | qualitative/incomplete visual candidate discovery and heterogeneous TDSS targeting |

MacLeod excluded spectra with median S/N ≲2 before decomposition. Green states no universal
spectral-S/N floor. A missing corrected variance array, failed preprocessing/rescaling, or
missing required MacLeod visual adjudication is `unclassifiable`, never `non-CL`.
Broad-Hβ fit nondetection is not itself a zero: this statistic consumes continuum-subtracted
pixel flux and variance. J082 remains unclassifiable for this family wherever its DESI
Hβ-region uncertainty correction is unsupported.

---

## Source-verified Yang broad-line flux ratio

Yang et al., *Galaxies Lighting Up* ([arXiv:2408.16183](https://arxiv.org/abs/2408.16183)),
defines a CL transition when at least one fitted broad line obeys

$$\frac{F_{\rm faint}}{F_{\rm bright}} < 0.3.$$

The strict inequality is the final rule. The paper calls the exact 0.3 choice somewhat
arbitrary but uses it for the catalogue; this project therefore does not retune it.

The source fits continuum/host and emission-line models, restricts broad-line FWHM to
1200–20,000 km/s, and uses BIC model selection to decide whether a broad component exists.
It explicitly states that the ratio rule remains valid when no broad line is detectable in
the faint state, in which case its catalogue uses `f_line,faint=0`. This is a
criterion-local fitted-nondetection convention. It does not turn fit failure, an unavailable
component, or an arbitrary numerical zero into a nondetection.

The Paper-3 production audit therefore requires optimizer completion, full local H-beta
coverage, a valid continuum solution, finite required quantities, and no active broad
component pinned to a fit boundary. A valid fitted nondetection may become zero inside the
Yang classifier only. Every other missing or invalid measurement is `unclassifiable`.

**Reported disagreement to verify.** Recent turn-on work is said to find that only 38 of 75
Hβ CLQs selected by flux ratio also satisfy $N_\sigma > 3$. If true this is the single
strongest existing motivation for Q3 and belongs in the introduction. **Verify the numbers
against the paper before citing them.**

---

## Family C — appearance / disappearance / spectral type change

**Form.** A transition in broad-line *detectability* or in assigned spectral type
(e.g. Type 1 ↔ Type 1.9/2).

| Item | Value | Source | Verified |
|---|---|---|---|
| Detection threshold defining "present" | TO_VERIFY | TO_VERIFY | **no** |
| Type assignment rule | TO_VERIFY | TO_VERIFY | **no** |
| Is visual inspection part of the definition | TO_VERIFY | TO_VERIFY | **no** |

**Why this one is the most S/N-sensitive.** "Disappearance" is a statement about a detection
limit, not about a flux. Improving S/N can make a weak residual broad Hβ detectable and
thereby *remove* the CL label from a genuine transition. If the counterintuitive effect this
paper is chasing exists anywhere, it is here — and this family must therefore be evaluated
against explicit upper limits (`BroadLineMeasurement.upper_limit`), never against a flux
coerced to zero.

---

## Guo DESI II — staged protocol, not one classifier

Step 1 defines, for each BEL pixel, the same variance-normalized bright-minus-dim deviation
and defines total-flux fractional change

$$R=(F_{\rm bright}-F_{\rm dim})/F_{\rm dim}.$$

The quick screen requires **the same one** of Mg II, Hβ, or Hα to satisfy both strict cuts
`Max(N_sigma) > 3` and `R > 1.5`. Notice that this R is not `F_bright/F_dim`; its cut is
equivalent to the latter ratio being greater than 2.5 only when both fluxes are finite and
the dim flux is positive. The paper supplies neither BEL integration windows nor a
nondetection/zero-denominator convention, so those cases are `unclassifiable`; zero or an
upper limit is not substituted.

Step 2 repeats `R > 1.5` on decomposed BEL flux after old-stellar-host subtraction. Step 3
normally requires relative [O III] flux difference below 20% and synthetic-to-observed
pseudo-photometry agreement below 0.5 mag. The published visual-inspection branches retain
some weak/low-S/N [O III] or low-redshift host-contaminated cases. Therefore
`GUO_QUICK_SCREEN` is a `SEARCH_STAGE`, and a quick-screen pass is never a final Guo CL label.
Missing any required branch makes strict final-protocol reproduction unclassifiable.

## Potts & Villforth — verified search protocol, no invented final classifier

The source uses Galactic-extinction-corrected, same-instrument SDSS Legacy repeat spectra,
Gaussian smoothing, the absolute epoch difference, and a third-degree polynomial over the
available difference-spectrum range. Its frozen measurements are:

- initial mean variability over rest 4100–5500 Å greater than
  `3e-18 erg s^-1 cm^-2 A^-1` (about 0.16 mag at magnitude 21);
- mean polynomial-continuum slope over rest 3500–3700 Å;
- highest continuum-subtracted monochromatic flux in rest 4856–4866 Å (Hβ) and
  6558–6568 Å (Hα), in `1e-17 erg s^-1 cm^-2`;
- Table 1 `(continuum, Hα, Hβ)` strict `>` thresholds: strong
  `(-2e-2, 3.5, 5.25)`, intermediate `(-1.8e-3, 1.8, 2.7)`, weak
  `(-6e-4, 0.8, 1.2)`.

This is not executable as an exact final classifier. The Gaussian kernel width is unreported;
the prose's candidate-combination wording conflicts with the Table 1 note; thresholds were
optimized on 184 changed-pipeline-class objects; and visual inspection reduced 941 candidates
to six CLQs by rejecting spectra without an AGN type transition. Code therefore freezes the
measurements and thresholds, returns a non-denominator `SEARCH_STAGE` result, and refuses to
invent the missing choices.

---

## Cross-cutting notes

**Thresholds are primary, sweeps are robustness.** Implement the published value exactly.
"2.8 seems close enough" is prohibited. Sweeps are labelled robustness in every figure and
table.

**Auxiliary cuts are part of the definition.** A paper that requires S/N > 10 before applying
$N_\sigma > 3$ has a *different* criterion from one that does not, and the difference is
precisely what Q1 measures. Record these in the `snr_cut` column, not as a footnote.

**Every final criterion returns three states.** `CL`, `non-CL`, `unclassifiable` — never a
silent negative. Search-stage outcomes are separately typed `pass`, `fail`, or
`unclassifiable` and have `counts_toward_classifier_denominator=False` by construction.

**Denominators are pair-level and nested.** At every Q2 cell and criterion report distinct
eligible pairs, fit-success pairs, protocol-measurement-complete pairs, and classifiable pairs.
The classifiable fraction is divided by all eligible pairs. Search stages are reported in
their own protocol accounting and are excluded from the final classifier denominator.
