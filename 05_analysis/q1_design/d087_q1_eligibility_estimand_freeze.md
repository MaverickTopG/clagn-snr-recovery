# D-087 — Q1 eligibility, estimands, and provisional execution matrix

**Q2 decision:** `FULL_PHYSICAL_Q2_PERMANENT_NO_GO_CURRENT_REFERENCE_POOL`  
**Q1 execution:** `NO_GO_Q1_FINAL_GRID_AND_MEASUREMENT_VALIDITY_NOT_CLOSED`  
**Execution performed:** none; no degraded spectra, fits, or classifier outcomes.

## Q1-specific eligibility

Q1 asks whether the same historical transition retains its classification after an S/N
intervention. It does not assert a constant absolute host contribution and therefore does
not apply Q2's 3-inch/2-inch aperture comparability exclusion. It does require the exact
historical bright and faint endpoints. Later DESI spectra are not substitutes.

The locked sets are:

- `PRIMARY_Q1`: **4 GOLD** — J082, J102, J105325, J233.
- `SENSITIVITY_Q1`: **8 GOLD+SILVER** — primary plus J012, J081, J105058, J222.
- `BORDERLINE`: disagreement/diagnostic-only, never ground truth.

Six transitions lack a local exact historical faint endpoint: J074, J123, J141, J153,
J161, and J210. J123's four exact WHT/ISIS source records remain a future optional SILVER
resource, but they are not acquired or reduced. J090 is additionally held from any reference
set by its BORDERLINE tier and pending native H-beta Gate-B adjudication.

Four GOLD transitions are unexpectedly small for a persuasive full-population Q1 recovery
claim. The tiers are unchanged; execution does not begin to compensate for the small N.

## Frozen intervention and grid status

- Primary arm: `faint_only`; degrade only the exact historical faint/line-poor endpoint.
- Secondary arm: `matched`; degrade both exact endpoints to the same target S/N.
- In `faint_only`, the bright endpoint is bit-for-bit unchanged.
- Common random numbers are object/pair/spectrum-specific. The same faint draw is reused
  across arms at fixed object, rung, and realization; the matched bright draw has a distinct
  stream.
- The grid `[5,10,15,20]` is **development-only provisional**, not a final scientific grid.
  D-027 requires support to be re-derived on the full reference population before results
  are reported. D-030 allows a prospective change only for materially different support, a
  measurement-validity boundary, or a published source requirement.

Using the lower of each endpoint's native continuum-5100 and H-beta-window S/N, and never
upgrading a spectrum, the provisional matrix contains **18 primary GOLD object-arm-rung
conditions** and **22 incremental SILVER conditions**, or **40 total**. Crossing the 40
conditions with the three source-verified final-classifier rows produces 120 prospective
evaluation cells, but MacLeod 2019 is expected to remain unclassifiable without visual
adjudication. No realization count is invented; D-019's criterion-by-criterion MC strategy
remains unresolved.

## Classifier roles

- Yang 2024: final quantitative classifier; denominator-eligible when its faithful protocol
  measurement and fit are valid.
- Green 2022: final quantitative classifier; denominator-eligible when the source-protocol
  pixel statistic and calibrated variance exist.
- MacLeod 2019: source-verified final classifier, but injected/degraded cells are
  unclassifiable without the frozen prospective visual flag.
- MacLeod 2016: visual search/identification protocol; non-denominator.
- Guo quick screen: search-stage only. Guo final is a staged/visual confirmation protocol,
  not reconstructed from the quick screen and non-denominator here.
- Potts-Villforth: search-stage protocol; non-denominator.

J082's unsupported T2-B calibrated-variance limitation remains binding for every condition
that consumes that variance. Its exact historical Q1 pair is SDSS-only, so Gate C is not
invoked by that pair; this does not authorize substituting the later DESI record or using an
unsupported T2-B-dependent quantity.

## Frozen estimands and denominators

For source-verified final classifier `k`, S/N rung `s`, and arm `a`:

$$R(s,a,k)=\frac{N_{\rm CL}(s,a,k)}{N_{\rm classifiable}(s,a,k)}$$

and

$$U(s,a,k)=\frac{N_{\rm unclassifiable}(s,a,k)}{N_{\rm eligible}(s,a,k)}.$$

`R` is the empirical recovery fraction over the locked transition set and is undefined when
`N_classifiable=0`. Always report:

`N_eligible`, `N_fit_completed`, `N_fit_valid`, `N_classifiable`, `N_CL`, `N_nonCL`, and
`N_unclassifiable`, subject to

$$N_{\rm CL}+N_{\rm nonCL}=N_{\rm classifiable},\qquad
N_{\rm classifiable}+N_{\rm unclassifiable}=N_{\rm eligible}.$$

Fit failures, invalid fits, missing source measurements, absent visual evidence, and
unsupported calibrated variance remain in `N_unclassifiable`. Search/confirmation protocols
never enter these final-classifier denominators. One real transition is one independent
unit; noise realizations, arms, rungs, spectra, and classifiers are repeated measurements.

## Recommendation

**NO-GO for full Q1 execution.** The current matrix is a complete design over the provisional
development grid, not an authorized final matrix. Before execution, re-derive grid support
on an independently adequate locked reference population, prospectively decide whether the
grid remains unchanged, close applicable measurement-validity requirements, and freeze the
criterion-specific realization strategy. Do not weaken tiers or add later endpoints.
