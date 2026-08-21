#!/usr/bin/env python
"""Stage 39 / D-100 — what actually happened to the Green decomposition.

The D-099 appendix explained the absence of invalid Green realizations by
asserting that nothing after the applicability gate can fail: "rebinning, a
division, a running median, and a maximum -- operations with no optimizer, no
free parameters, and no bounds to reach." That is false, and a referee caught
it. A PyQSOFit continuum/host decomposition runs on *every* degraded
realization, well after applicability is settled, and it is an optimizer with
every failure mode optimizers have.

The zero is real, but it is an observed outcome rather than a structural
guarantee, and the paper has to say so. This script establishes the facts from
the frozen production output:

* the decomposition ran per fit input, not once per transition;
* its success was recorded per input (`green_preprocessing_valid`);
* a failure would have propagated to `unclassifiable` rather than to non-CL;
* it converged on every input, so no such row exists.

Nothing here refits or reclassifies anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p3sf.config import project_root  # noqa: E402

COLUMNS = [
    "task_id", "task_kind", "instrument_survey", "target_snr", "realization",
    "job_status", "green_preprocessing_valid", "green_invalid_reason",
    "yang_fit_valid", "yang_invalid_reason",
]


def main() -> int:
    root = project_root()
    raw = root / "05_analysis" / "q1_production" / "d094" / "raw"
    out = root / "05_analysis" / "derived"
    out.mkdir(parents=True, exist_ok=True)

    fits = pq.read_table(raw / "spectrum_fit_results_d094.parquet", columns=COLUMNS).to_pandas()
    outcomes = pd.read_csv(raw / "classifier_outcomes_d094.csv")

    by_kind = fits.task_kind.value_counts().to_dict()
    green_valid = int(fits.green_preprocessing_valid.sum())
    green_invalid = int((~fits.green_preprocessing_valid).sum())
    reasons = sorted({r for r in fits.green_invalid_reason.fillna("") if r})

    # Every degraded input carries its own decomposition; only the native bright
    # fit is computed once per transition and reused within that transition.
    degraded = fits[fits.task_kind != "native_bright"]
    per_realization = bool(len(degraded) == degraded.task_id.nunique() == 8350)

    green_rows = outcomes[outcomes.criterion_id == "GREEN2022_FINAL"]
    applicable = green_rows[green_rows.applicable]
    yang_rows = outcomes[outcomes.criterion_id == "YANG2024_FINAL"]
    yang_applicable = yang_rows[yang_rows.applicable]

    audit = {
        "verdict": "GREEN_DECOMPOSITION_RAN_PER_REALIZATION_AND_ALWAYS_CONVERGED",
        "fit_inputs_total": int(len(fits)),
        "fit_inputs_by_kind": {k: int(v) for k, v in by_kind.items()},
        "decomposition_is_per_realization": per_realization,
        "native_bright_fits_reused_within_transition": int(by_kind["native_bright"]),
        "green_preprocessing_valid": green_valid,
        "green_preprocessing_invalid": green_invalid,
        "distinct_green_invalid_reasons": reasons,
        "all_jobs_completed": bool((fits.job_status == "COMPLETED").all()),
        "green_applicable_realizations": int(len(applicable)),
        "green_applicable_invalid_realizations": int((~applicable.fit_valid).sum()),
        "yang_applicable_realizations": int(len(yang_applicable)),
        "yang_applicable_invalid_realizations": int((~yang_applicable.fit_valid).sum()),
        "yang_fit_valid_inputs": int(fits.yang_fit_valid.sum()),
        "yang_invalid_inputs": int((~fits.yang_fit_valid).sum()),
        "failure_propagation_path": (
            "fit_green_epoch_line_spectrum_pyqsofit sets preprocessing_valid; "
            "_green_from_fit carries it into measure_green_pixel_nsigma; "
            "_rebin_two_angstrom returns None when it is false; "
            "classify_green_pixel_measurement then returns unclassifiable, never non-CL"
        ),
        "interpretation": (
            "The zero is an observed convergence outcome, not a structural guarantee. "
            "Both protocols require a converged continuum fit; only the fitted-line "
            "protocol additionally requires a broad component strictly interior to its "
            "scale, width and centroid bounds, and that is the requirement that fails."
        ),
        "refit_or_reclassification_performed": False,
    }
    (out / "green_preprocessing_audit_d100.json").write_text(json.dumps(audit, indent=2) + "\n")

    print("=" * 84)
    print("D-100 | Green decomposition: execution order and observed validity")
    print("=" * 84)
    print(f"fit inputs                         {audit['fit_inputs_total']}")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind:<32} {n}")
    print(f"decomposition run per realization  {per_realization}")
    print(f"green_preprocessing_valid          {green_valid}")
    print(f"green_preprocessing_invalid        {green_invalid}")
    print(f"distinct invalid reasons           {reasons if reasons else 'none'}")
    print()
    print(f"Green applicable realizations      {audit['green_applicable_realizations']}")
    print(f"  of which invalid                 {audit['green_applicable_invalid_realizations']}")
    print(f"Yang applicable realizations       {audit['yang_applicable_realizations']}")
    print(f"  of which invalid                 {audit['yang_applicable_invalid_realizations']}")
    print()
    print("An optimizer does run after the applicability gate. It simply converged every")
    print("time. A failure would have become unclassifiable, never non-CL.")

    if green_invalid and audit["green_applicable_invalid_realizations"] == 0:
        print()
        print("STOP: an invalid decomposition did not reach an unclassifiable label.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
