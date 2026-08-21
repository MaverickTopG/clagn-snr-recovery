# Protocol deviations

The prospective plan of 2026-08-14 is preserved unchanged in
`original_preregistered_analysis_2026-08-14.md`, and the configuration written alongside it in
`original_prospective_config_2026-08-14.yaml`. The analysis the paper reports differs from that
plan in the ways listed here.

Timing is taken from the project's dated decision records. Those records place every design
change below at 2026-08-15 or 2026-08-16, and place the first generation of Q1 recovery outcomes
after all of them. Where a claim about ordering cannot be supported by a record, it is not made.

---

## A. Physical host-fraction experiment — no-go

**Planned.** A host-fraction recovery experiment, injecting object-specific host templates
recovered by spectral decomposition and measuring recovery against a grid of host fractions
(0.1 to 0.85), with 0.3 versus 0.7 as a primary contrast.

**Final.** Not performed. Recorded as a permanent no-go for this reference pool.

**Reason.** Host decompositions proved model-sensitive or non-identifiable: near-identical
global fit quality yielded substantially different host fractions, and for several objects
stellar information present under one model disappeared under a reasonable alternative. Corrected
refits produced five model-sensitive cases, three with no primary host template, and none with a
robust fixed object-specific template. Separately, the historical pairs mix apertures — SDSS
legacy 3-arcsec against BOSS/eBOSS 2-arcsec — so a constant absolute galaxy flux across a pair is
not physically justified, and no aperture rescaling was permitted.

**Timing.** Decided in the records dated 2026-08-16, before Q1 recovery outcomes existed.

**Consequence.** The paper reports no population `P(C | f_host)` and no host-completeness curve.
What survives is a controlled diagnostic in which a stellar continuum of known shape is *added*
at a known share. That quantity is an added-contamination fraction, not a physical total host
fraction, and the paper treats it as a secondary diagnostic only.

## B. Reference sample expansion

**Planned.** A smaller reference set assembled from the source families known at the time.

**Final.** 58 GOLD transitions and 62 GOLD+SILVER.

**Reason.** The initial strict set was too small for persuasive aggregate inference. A bounded
literature audit enumerated 31 source families, exhausted the relevant accessible universe, and
identified additional pairs with exact public endpoints. Those were acquired and validated
against unchanged identity and quality criteria; some qualified on evidence but failed technical
eligibility and were not admitted.

**Timing.** The expansion, its acquisition, and its validation are recorded on 2026-08-16, ahead
of the record of full Q1 execution. The literature search cutoff is recorded as
2026-08-15 23:59 UTC.

**Consequence.** The set is a census of published transitions, not a probability sample. One
family contributes 34 of 58, which is why family-composition sensitivities are reported and why
the paper states that the direction of the effect is stable while its magnitude is not.

## C. Final S/N grid

**Planned.** More rungs, including a 10-versus-20 primary contrast.

**Final.** Rungs 5 and 10 only.

**Reason.** Native support. Degradation may not raise a spectrum's S/N, so a transition can only
occupy a rung its own spectra already reach. Support at 15, 20 and 30 was too sparse to carry
inference, and 40 was unreachable.

**Timing.** The grid was fixed from support counts before execution, not chosen after seeing
effect sizes at each rung.

**Consequence.** The paper cannot describe the shape of recovery outside 5 to 10, or separate a
smooth curve from a threshold-like response.

## D. Monte Carlo realization count

**Final.** M = 50, common to every transition.

**Reason and timing.** A two-transition pilot found M = 20 insufficient. Before the draws beyond
40 were generated, an acceptance rule was written down requiring every group to satisfy
`|p50 - p40| <= 0.10` and `|u50 - u40| <= 0.10`. The remaining draws were then produced and all
16 groups met it.

**Consequence.** This bounds Monte Carlo error for the groups examined. It is not a convergence
proof for all 58 transitions, and the paper does not describe it as one.

## E. Independent second adjudicator — not performed

**Planned.** The prospective plan states that a subset would be independently adjudicated by two
people, with inter-rater agreement reported.

**Final.** This was not done. The reference tiers were assigned by a single adjudicator, the
author, without an independent second reading.

No inter-rater agreement statistic exists, and none is reported anywhere in this repository or
the paper. The manuscript states the limitation in its own text. The machine-readable manifest
gives the source publication, tier, evidence summary and exact endpoint identifiers for all 62
transitions, so an external reader can recheck any assignment against the cited literature.

## F. Instrument and survey pairing

**Planned.** Instrument or survey pairing as a primary question with its own inferential result.

**Final.** Descriptive only. Instrument-pair splits are reported as composition and
compatibility checks.

**Reason.** The strata are small, between three and eight transitions, and cross-instrument
coverage is heterogeneous. They are not powered for instrument-specific effect estimates.

**Consequence.** The paper reports that the paired effect is positive in every populated
instrument-pair stratum, and states explicitly that this shows the direction is not confined to
one combination rather than establishing identical calibration across products.

## G. Operational protocol implementations

Neither protocol is a reproduction of a published pipeline end to end.

**Green.** The final statistic and threshold follow the published definition, including its
rebinning, smoothing length, reference bin and edge rule. The preceding spectral decomposition is
this project's PyQSOFit implementation; the original authors used different fitting software.

**Yang.** The published Hβ flux-ratio threshold and FWHM domain are applied to this project's
PyQSOFit measurements. Yang et al. used QGfit, with skewed-Voigt profiles and a model-selection
step. **This project does not reproduce the QGfit pipeline**, and results for this protocol
describe the composite evaluated here.

## H. Conditioning on the realized spectrum

The degradation operator adds noise to the archival spectrum as observed. Each input is itself
one noisy realization of its source, and the experiment holds that realization fixed while adding
further noise around it. It does not estimate a noiseless latent spectrum and draw fresh
independent observations from it.

Unconditionally the target variance is recovered; conditional on the observed spectrum, the
spread among realizations comes only from the added component. The recovery fractions therefore
describe added noise on the spectra that were actually taken.

## I. Configuration file status

`frozen_config.yaml` is a working copy of the prospective configuration and is what the code
loads at runtime, because the package's validators and its directory discovery were built against
that schema. It still contains grids the paper does not report, including the host-fraction grid
from section A, and it carries `frozen: false`.

It is retained for that mechanical reason only. The configuration describing the published
analysis is `final_paper3_analysis_config.yaml`, loaded by
`p3sf.config.load_final_paper_config`. The generic `load_config` returns the prospective plan and
is aliased as `load_prospective_config` so the distinction is visible at the call site.
