# Data license

`LICENSE` (MIT) covers the source code in this repository. The derived data products listed
below are author-generated results of this study and are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Covered by CC BY 4.0

| Path | Contents |
|---|---|
| `04_reference_sample/` | Reference-sample screening and candidate tables |
| `05_analysis/q1_design/` | Design, eligibility, support and measurement-validity tables |
| `05_analysis/q1_production/` | Condition matrix, classification outcomes, transition-level estimands, aggregate result tables |
| `05_analysis/full_q2_design/` | Tables from the host-fraction design that was not executed |
| `05_analysis/derived/` | Summaries recomputed from the stored outcomes |
| `00_admin/exclusions_log.csv` | Exclusion records |

## Not covered

**Survey data.** No spectrum from SDSS, SDSS-V, LAMOST or DESI is redistributed here. The
identifiers, redshifts, coordinates and checksums in `endpoint_bindings_d094.csv` and the
reference tables describe those products so a reader can retrieve them. The products themselves
remain subject to the terms and acknowledgment requirements of the originating collaborations.

**Third-party software.** PyQSOFit is GPL-3.0 and is not redistributed here; it is cloned from
upstream at a pinned revision by `00_scripts/00_make_environment.py`. All other dependencies
remain under their own licenses.

**Published literature.** `01_literature/literature_matrix.csv` and `selection_definitions.md`
record thresholds and definitions transcribed from published papers, with citations. The
underlying publications remain the property of their authors and publishers.
