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

Download the dataset above and place its tables where the scripts expect them, then:

```
uv run python 00_scripts/34_execute_full_q1.py
uv run python 00_scripts/35_analyze_full_q1.py
uv run python 00_scripts/36_d095_q1_scientific_stress_test.py
uv run python 00_scripts/37_d099_transition_level_disagreement.py
```

The production step reruns 8412 spectral fits and takes many hours. The later scripts read
stored outputs and finish in minutes.

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
| `tests/` | Test suite |
| `config/` | PyQSOFit line-parameter file |

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

MIT for the code in this repository; see `LICENSE`. Survey data products remain subject to the
terms of the originating archives, and third-party dependencies remain under their own
licenses.
