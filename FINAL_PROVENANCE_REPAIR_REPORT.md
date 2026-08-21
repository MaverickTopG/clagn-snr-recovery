# Final provenance repair report

Repository: `MaverickTopG/clagn-snr-recovery`. Nothing pushed, nothing committed, no tag beyond
the local safety tag, existing five commit SHAs untouched.

## 1. Original HEAD

`59778cb75b44ce6b8c50f581fab72132e981ac73` — unchanged. Safety tag
`before-provenance-repair-2026-08-20` created locally and not pushed.

## 2–3. Preserved originals, SHA-256 verified byte-identical

| Preserved as | SHA-256 |
|---|---|
| `00_admin/original_prospective_config_2026-08-14.yaml` | `1c01b7ef7e9f70d93573572b614047e0bc7a33ebc2390a5d67d86b253ec3262e` |
| `00_admin/original_preregistered_analysis_2026-08-14.md` | `d40fb6dbb838d130eb074100d067696cd30bbb2a034eddefe2eeca35ecd7a9b0` |

Both match their sources exactly. The 381-line preregistration is preserved whole.

## 4–5. Final configuration

`00_admin/final_paper3_analysis_config.yaml`, `final_config_version: 1`,
`status: final_paper_analysis`. Every value transcribed from production artifacts: 58/62
reference, 43/15 turn-off/turn-on, rungs [5, 10], M=50, seed namespace `p3sf:q1:full:v1`,
167/8350/8412/16700 production, transition as independent unit, 5000 bootstrap resamples, Green
threshold 3.0 inclusive with `significance_interpretation: false`, Yang 0.3 with
`original_qgfit_pipeline_reproduced: false`, physical host experiment `no_go`, stellar-shape
diagnostic `secondary_diagnostic_only`.

## 6–9. Protocol deviations

`00_admin/PROTOCOL_DEVIATIONS.md` documents nine, each with plan, outcome, reason, timing and
consequence. Timing comes from dated decision records, which place every design change at
2026-08-15/16 and the first Q1 outcomes after all of them.

- **Q2 no-go (A).** Host decompositions model-sensitive or non-identifiable; aperture mismatch
  between 3-arcsec and 2-arcsec fibers. No population `P(C | f_host)`. Only an added-contamination
  diagnostic survives, and it is not a physical host fraction.
- **Independent second adjudicator (E).** The preregistration promised two-person adjudication of
  a subset with inter-rater agreement reported. **It was not performed.** Stated plainly, with no
  excuse. No inter-rater statistic exists anywhere in the repository, and none is claimed.
- **Instrument Q4 downgrade (F).** From primary question to descriptive composition check; strata
  hold three to eight transitions and are not powered for instrument-specific estimates.
- Also: reference expansion (B), final S/N grid (C), M=50 (D), protocol implementations (G),
  realized-spectrum conditioning (H), and the configuration-file status (I).

## 10. README reproduction

Replaced the vague instruction with literal paths determined by reading the scripts. The frozen
tables ship with the repository, so the four summary scripts run with no download. The two things
not shipped are named with exact destinations: `spectrum_fit_results_d094.parquet` at
`05_analysis/q1_production/d094/raw/`, and survey spectra under
`03_spectra/raw_q1_d093/{sdss_legacy,sdssv_dr19,lamost_dr11,desi_edr}/`, retrieved using the
identifiers and SHA-256 values in `endpoint_bindings_d094.csv`. An `## Analysis provenance`
section of four sentences was added.

## 11. Duplicate configuration — resolved

`00_admin/` and `config/` held three byte-identical files. `00_admin/` is authoritative:
`project_root()` locates the repository by `00_admin/frozen_config.yaml`. The `config/` copies of
`frozen_config.yaml`, `requirements-lock.txt` and `exclusion_reason_codes.yaml` were removed.

## 12. Licensing

Root `LICENSE` is now plain MIT, copyright 2026 Ayansh Singh, so GitHub will detect it correctly.
`DATA_LICENSE.md` places the author-generated derived tables under CC BY 4.0 and lists the actual
covered directories, explicitly excluding survey data, third-party software and published
literature. README states all three tiers.

## 13. Third-party FITS — removed

`config/pyqsofit_qsopar.fits` is **byte-identical** (`37d181e0…`) to `qsopar.fits` in the pinned
PyQSOFit clone, which is **GPL-3.0**. It was a verbatim copy of a GPL file sitting in a repository
now licensed MIT, with no GPL notice.

Nothing read it: `_pyqsofit_path()` returns `os.path.dirname(pyqsofit.__file__)`, so PyQSOFit
locates its own parameter file inside its own package. The only reference was a test assertion
added during the previous audit.

Removed, and `config/` with it. Verified afterwards that PyQSOFit still resolves its parameter
file from the cloned package. The test now asserts that **no** FITS file is tracked at all.

## 14. `pyproject.toml`

`version = "1.0.0"`. Dependency comments made self-contained: references to a private decision
log and to the author's earlier project removed, the scientific reason for each pin retained. No
dependency version altered.

## 15. Verification

| | Exit | Result |
|---|---:|---|
| In place, direct | **0** | 391 passed, 3 skipped, ruff PASS, mypy PASS (50 files), environment PASS |
| Fresh tree, README instructions followed literally, direct | **0** | 391 passed, 3 skipped, all four checks PASS |

## 16. Scientific anchors — 18/18 unchanged

58 GOLD, 62 GOLD+SILVER, 43 turn-off, 15 turn-on, 167 conditions, 8350 condition-realizations,
8412 fit inputs, 16700 automated classifications, Green faint-only N=27 mean 0.2896, matched N=25
mean 0.2592, arm 0.1967 and 0.2448, equal-family 0.1801, Yang 4456 invalid, 3937 boundary-related,
88.4%.

## 17. Contradiction audit

`FINAL_CONTRADICTION_AUDIT.md`. No active configuration, README, documentation or result code
presents a superseded plan as current. Zero active references to private files remain.

## 18. Unresolved issues

**`frozen_config.yaml` remains the file the code loads.** Repointing the default was assessed and
rejected: the `Config` model requires `host_fraction_grid`, `primary_contrasts` and `slice_grid`,
and `project_root()` discovers the repository by that filename. Making the final config the
default would force it to declare the very host-fraction experiment that was abandoned. Per the
brief's own instruction, the loaders were separated instead — `load_final_paper_config()` for the
paper, `load_config()` aliased as `load_prospective_config()` for the plan — and the file now
states its status in its own header. This is plumbing; no algorithm changed.

**Thirteen internal design memos were removed** from `05_analysis/` in the previous session and
that deletion is carried in this change set. They were D-number freeze documents matching the
"internal decision logs" exclusion. The evidence they contained is summarized in
`PROTOCOL_DEVIATIONS.md`; the underlying data tables all remain.

**The software DOI is absent by design.** `CITATION.cff` carries the dataset DOI
`10.5281/zenodo.22022007` and no software DOI, which cannot exist until the release is archived.
