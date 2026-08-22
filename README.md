# Spectroscopic Recovery of Changing-Look AGN

Code for the analysis associated with:

> Singh, A. "Signal-to-Noise and Classification Protocols Shape the Spectroscopic Recovery of
> Changing-Look AGN."

This is a curated publication release; its five commits organize the released files and do not
reconstruct the original development history.

## Overview

The experiment holds a physical changing-look transition fixed and asks whether it would still
be recovered from noisier spectra. Exact historical endpoint spectra for 58 published
transitions are degraded to target median native-pixel signal-to-noise ratios of 5 and 10 in
the rest-frame 4700--5100 A window, with 50 noise realizations per condition, and each
realization is reclassified under two operational protocols: a pixel statistic following Green
et al. (2022), and an Hbeta flux-ratio rule following Yang et al. (2025) applied to PyQSOFit
measurements.

Recovery is aggregated with the transition, not the realization, as the independent unit.

## Data

Reproducibility dataset, including the reference manifest, endpoint identifiers, condition
matrix, stored classification outcomes and transition-level estimands:

https://doi.org/10.5281/zenodo.22022007

Survey spectra are not redistributed. They are retrieved from SDSS, SDSS-V, LAMOST and DESI
using the immutable identifiers in `endpoint_bindings.csv` in that dataset.

## Installation

Python 3.12.

```
uv sync --all-extras
uv run python 00_scripts/00_make_environment.py
```

The first command installs the package and its development tools; the second clones the pinned
PyQSOFit revision, which is not redistributed here. Both are required before the tests run.

## Reproduction

The frozen result tables are in this repository, so the analyses that summarize them run
without downloading anything:

```
uv run python 00_scripts/35_analyze_full_q1.py
uv run python 00_scripts/36_d095_q1_scientific_stress_test.py
uv run python 00_scripts/37_d099_transition_level_disagreement.py
uv run python 00_scripts/40_d100_referee_stratifications.py
```

These read `05_analysis/q1_production/` and write to `05_analysis/derived/`. They finish in
minutes and reproduce every number the paper reports.

Two things are deliberately not shipped, and each is needed only for a fuller rerun.

**Per-spectrum fit results** (`spectrum_fit_results_d094.parquet`, 675 MB) back the Green
decomposition audit and the raw-product checksum test. Both are skipped when it is absent.
The file is too large to ship here and is **not** part of the Zenodo dataset; it is produced by
rerunning the production campaign (`00_scripts/34_execute_full_q1.py`). If you have it, place it at:

```
05_analysis/q1_production/d094/raw/spectrum_fit_results_d094.parquet
```

The one analysis that needs those per-pixel arrays, the Green decomposition variance sensitivity
(`00_scripts/42_green_decomposition_variance_sensitivity.py`), does not require it. A compact
extract is shipped instead:

```
05_analysis/q1_production/d094/raw/green_pixel_window_d094.parquet   13 MB
```

It carries the rest-frame 4730-4962 A pixels the Green statistic can reach, for the 6994 epochs
that analysis uses. `00_scripts/43_build_green_window_input.py` builds it from the full product and
aborts unless every epoch rebins bitwise-identically from the extract; Stage 42 produces
byte-identical output from either input.

**Survey spectra** are needed only to rerun the 8412-fit production campaign
(`00_scripts/34_execute_full_q1.py`, many hours). They are not redistributed. Retrieve each from
its originating archive using the identifiers in
`05_analysis/q1_production/d094/raw/endpoint_bindings_d094.csv`, which gives the survey, the
spectrum identifier and the expected SHA-256 for all 116 endpoints, then place them under:

```
03_spectra/raw_q1_d093/{sdss_legacy,sdssv_dr19,lamost_dr11,desi_edr}/
```

Everything the Zenodo dataset contains is already present in this repository; nothing needs to be
retrieved from it to reproduce the reported numbers.

## Tests

```
./verify.sh --fast
```

Runs the test suite, `ruff`, `mypy`, and an environment check.

## Repository structure

| Path | Contents |
|---|---|
| `src/p3sf/` | Analysis package: spectral access, degradation operator, fitting drivers, classification protocols, statistics |
| `00_scripts/` | Numbered pipeline stages, run in order |
| `00_admin/` | Analysis configuration and provenance records |
| `01_literature/` | Published thresholds transcribed with citations |
| `04_reference_sample/` | Reference-sample screening tables |
| `05_analysis/` | Frozen design, production and result tables |
| `tests/` | Test suite |

## Analysis provenance

The prospective plan written before the experiment ran is preserved unchanged in
`00_admin/original_preregistered_analysis_2026-08-14.md`, alongside its configuration. The
analysis actually reported in the paper is described by
`00_admin/final_paper3_analysis_config.yaml`. Where the two differ — the abandoned host-fraction
experiment, the expanded reference sample, the final S/N grid, and an independent second
adjudication that was planned but not performed — the differences and their timing are set out in
`00_admin/PROTOCOL_DEVIATIONS.md`. This release corresponds to the final paper analysis.

## Citation

Please cite both the paper and the dataset:

```
@misc{singh2026dataset,
  author = {Singh, Ayansh},
  title  = {Reproducibility Dataset for: Signal-to-Noise and Classification Protocols
            Shape the Spectroscopic Recovery of Changing-Look AGN},
  year   = {2026},
  doi    = {10.5281/zenodo.22022007}
}
```

## License

- **Code:** MIT, see `LICENSE`.
- **Author-generated derived tables:** CC BY 4.0, see `DATA_LICENSE.md`.
- **Survey products and third-party software:** original terms apply. No survey spectrum is
  redistributed here, and PyQSOFit is GPL-3.0 and cloned from upstream rather than included.
