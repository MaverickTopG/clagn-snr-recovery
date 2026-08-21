# Final contradiction audit

Every search term from the release brief was run across the tracked tree with `git grep -il`.
Matches are classified as:

**A** historical, allowed only inside clearly labeled preserved material
**B** final-paper statement, must match the manuscript
**C** obsolete active configuration or code comment, fixed
**D** legitimate scientific limitation or context, retained

| Term | Where | Class | Disposition |
|---|---|---|---|
| `10 versus 20` | `frozen_config.yaml` | A | Prospective plan. File now carries a status header naming it as such and pointing to the final config. Pristine copy preserved separately. |
| `frozen: false` | `frozen_config.yaml` | A | Same. The flag was never flipped because the locked-validation stage it belonged to was not part of the final design. |
| `14 GOLD`, `10 GOLD` | `frozen_config.yaml` | A | Superseded counts from before the reference expansion. Final counts are in `final_paper3_analysis_config.yaml`. |
| `host_fraction` | `frozen_config.yaml` | A | The abandoned host-fraction grid. Deviation A in `PROTOCOL_DEVIATIONS.md`. |
| `host_fraction` | `src/p3sf/counterfactual/host.py`, `criteria/metrics.py`, `fitting/*`, `spectral_domain.py`, tests | D | Working code for the controlled stellar-contamination diagnostic, which the paper does report as a secondary result. Retained. |
| `two people`, `independently adjudicated`, `inter-rater` | preserved preregistration only | A | Promised prospectively, not performed. Stated plainly as deviation E. No active file claims it was done. |
| `Q2` | `05_analysis/full_q2_design/` table names | A | Design tables for the experiment that was a no-go. Retained as evidence that it was designed and abandoned, not silently dropped. |
| `QGfit` | `01_literature/literature_matrix.csv` | D | Correctly describes Yang et al.'s own method in the literature matrix. Nothing claims this project reproduces it; `PROTOCOL_DEVIATIONS.md` section G states it does not. |
| `survey selection function` | `src/p3sf/__init__.py` | D→C | The sentence is a correct disclaimer, but it pointed at `preregistered_analysis.md`, now a stub. Repointed to `PROTOCOL_DEVIATIONS.md`. |
| `decisions_log.md` | 6 active files | C | Referred to a private file. Replaced with `PROTOCOL_DEVIATIONS.md` or removed where it was internal commentary. Zero active references remain. |
| `14_referee_audit` | `tests/test_environment.py` | A | Inside the development-tree layout test, which is skipped in this repository and exists to describe the private tree. Not a claim to a reader. |
| `0.3` | ~110 files | D | Overwhelmingly version strings and numeric data. The scientifically meaningful use is the Yang flux-ratio threshold, which is correct and current. |
| `scratchpad`, `prior_rejection_lessons`, `/Users/`, `/home/` | — | — | No matches. |

## Result

No active configuration, README, documentation or result code presents a superseded plan as
current. The prospective plan survives in two clearly labeled preserved files and in one working
config that now states its own status in its header.
