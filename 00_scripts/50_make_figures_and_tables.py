#!/usr/bin/env python3
"""Build the ApJ submission manuscript assets from frozen D-094/D-095 products.

Every number and every plotted array is read from a stored result file. Nothing
here fits, degrades, classifies or resamples a spectrum.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _find_root() -> Path:
    """The project root is marked by the frozen configuration in both repositories."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "00_admin" / "frozen_config.yaml").exists():
            return parent
    raise SystemExit("project root not found: 00_admin/frozen_config.yaml is missing")


ROOT = _find_root()
SUB = ROOT / "05_analysis/manuscript/submission_apj"
# The public code release carries no manuscript. There the generator reads the
# stage 37/40 products from 05_analysis/derived, draws Figure 2 from the verified
# stage 49 extract, and writes its outputs beside them rather than into a
# manuscript directory that does not exist.
PUBLIC = not (SUB / "manuscript.tex").exists()
OUT = ROOT / "05_analysis/manuscript_assets" if PUBLIC else SUB
FIG = OUT / "figures"
TAB = OUT / "tables"
DERIVED = ROOT / "05_analysis/derived" if PUBLIC else TAB
MANIFEST = ROOT / "05_analysis/derived/reference_transition_manifest.csv" if PUBLIC else SUB / "reference_transition_manifest.csv"
D094RAW = ROOT / "05_analysis/q1_production/d094/raw"
D094 = ROOT / "05_analysis/q1_production/d094/tables"
D095 = ROOT / "05_analysis/q1_production/d095/tables"
D093 = ROOT / "05_analysis/q1_design/d093"
HOST = ROOT / "05_analysis/host_shape_pilot/execution"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLACK = "#111111"
DARK = "#4d4d4d"
MID = "#858585"
LIGHT = "#c7c7c7"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# Figure 1: reference flow and design. The two branches that make up the final
# GOLD set are drawn separately, because a single chain reading 54 -> 48 -> 58
# hides where the other ten transitions come from.
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 14.7)
ax.set_ylim(0, 7.6)
ax.axis("off")
boxes = [
    (0.10, 5.55, 2.62, 1.30, "31 literature\nsource families", BLACK),
    (2.82, 5.55, 2.62, 1.30, "64 high-confidence\nmorphology\ncandidates", DARK),
    (5.54, 5.55, 2.62, 1.30, "54 exact public\nendpoint pairs", DARK),
    (8.26, 5.55, 2.62, 1.30, "48 newly\nvalidated GOLD\n(identity + QC)", DARK),
    (8.26, 3.60, 2.62, 1.05, "10 previously\nvalidated GOLD", DARK),
    (11.85, 4.40, 2.65, 1.85, "$48+10=58$\nhigh-confidence\ntransitions\nanalysis-eligible", BLACK),
    (11.85, 1.05, 2.65, 1.55, "Native support\nby arm\nand rung", BLACK),
    (7.35, 1.15, 3.55, 1.35, "Faint-only: 48 / 29\nMatched: 47 / 27\n(S/N 5 / S/N 10)", DARK),
    (4.55, 1.15, 2.10, 1.35, "$M=50$ draws\nper condition", MID),
    (0.15, 1.15, 3.65, 1.35, "Green-statistic\nprotocol\nYang-ratio/PyQSOFit\nprotocol", BLACK),
]
for x, y, w, h, label, color in boxes:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.6))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.9)
for x1, x2 in [(2.72, 2.82), (5.44, 5.54), (8.16, 8.26)]:
    ax.annotate("", (x2, 6.20), (x1, 6.20), arrowprops=dict(arrowstyle="->", lw=1.1))
# The two independently validated branches merge into the final GOLD set.
ax.annotate("", (11.85, 5.70), (10.88, 6.20), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.annotate("", (11.85, 4.95), (10.88, 4.13), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(11.45, 5.34, "$+$", ha="center", va="center", fontsize=13, weight="bold")
ax.text(3.85, 3.95,
        "Reference-evidence tier was assigned before any\n"
        "recovery outcome. Analysis eligibility is a separate\n"
        "technical gate on the endpoint spectra: published\n"
        "evidence can be convincing while the archival pair\n"
        "still fails identity or native quality.",
        ha="center", va="center", fontsize=6.9, style="italic", color=DARK)
ax.annotate("", (13.17, 2.60), (13.17, 4.40), arrowprops=dict(arrowstyle="->", lw=1.1))
for x1, x2 in [(11.85, 10.90), (7.35, 6.65), (4.55, 3.80)]:
    ax.annotate("", (x2, 1.825), (x1, 1.825), arrowprops=dict(arrowstyle="->", lw=1.1))
ax.text(7.35, 7.30, "Outcome-blind reference construction and supported-noise experiment",
        ha="center", weight="bold", fontsize=9.5)
ax.text(7.35, 0.45, "One transition per physical AGN; realizations are repeated measurements",
        ha="center", style="italic", fontsize=8.4)
save(fig, "fig1_experimental_design")


# Figure 2: one representative transition, plotted entirely from stored production
# arrays. The object is fixed by a rule that cannot see a recovery outcome: among
# the 27 paired common-support faint-only transitions, take the one whose native
# faint H-beta-window S/N is nearest that set's median, breaking ties by the
# lexicographically smallest identifier. With N=27 the median is attained exactly
# by J075934.95+322143.3, and no tie-break is needed.
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from astropy.io import fits  # noqa: E402

_paired = pd.read_csv(D095 / "green_common_support_snr_paired_transitions_d095.csv")
_fo = sorted(_paired[_paired.arm == "faint_only"].transition_id.unique())
_FIT_ARCHIVE = D094RAW / "spectrum_fit_results_d094.parquet"
if _FIT_ARCHIVE.exists():
    _fits_tbl = pq.read_table(_FIT_ARCHIVE).to_pandas()
    _faint = _fits_tbl[(_fits_tbl.task_kind == "faint_shared") & (_fits_tbl.transition_id.isin(_fo))]
    _native = _faint.groupby("transition_id").original_snr.first()
else:
    # Stage 49 extract: the same native S/N values, so the rule is re-executed.
    _native = pd.read_csv(D094RAW / "figure2_selection_d094.csv", float_precision="round_trip") \
        .set_index("transition_id").original_snr
    _native = _native[_native.index.isin(_fo)]
assert len(_native) == 27
_median_snr = float(_native.median())
_offset = (_native - _median_snr).abs()
EXAMPLE = sorted(_offset[_offset == _offset.min()].index)[0]
assert EXAMPLE == "J075934.95+322143.3", EXAMPLE

if _FIT_ARCHIVE.exists():
    _rows = _fits_tbl[_fits_tbl.transition_id == EXAMPLE]
    _bright = _rows[_rows.task_kind == "native_bright"].iloc[0]
    _d10 = _rows[(_rows.task_kind == "faint_shared") & (_rows.target_snr == 10.0) & (_rows.realization == 0.0)].iloc[0]
    _d5 = _rows[(_rows.task_kind == "faint_shared") & (_rows.target_snr == 5.0) & (_rows.realization == 0.0)].iloc[0]
else:
    from types import SimpleNamespace

    _arr = pd.read_csv(D094RAW / "figure2_example_arrays_d094.csv", float_precision="round_trip")
    _meta = pd.read_csv(D094RAW / "figure2_example_meta_d094.csv", float_precision="round_trip").iloc[0]
    assert _meta.transition_id == EXAMPLE

    def _series(name: str) -> tuple[np.ndarray, np.ndarray]:
        part = _arr[_arr.series == name]
        return part.wavelength_rest.to_numpy(), part.flux.to_numpy()

    def _line_row(name: str, **extra: object) -> SimpleNamespace:
        w, f = _series(name)
        return SimpleNamespace(green_wavelength_rest=w, green_line_flux=f, **extra)

    _bright = _line_row("line_bright", redshift=float(_meta.redshift), local_path="extract:observed_bright")
    _d10 = _line_row("line_faint_snr10", original_snr=float(_meta.faint_native_snr), local_path="extract:observed_faint")
    _d5 = _line_row("line_faint_snr5")
_z = float(_bright.redshift)


def _observed(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Archival flux on the rest-frame grid, using the frozen positive-IVAR rule."""
    if path.startswith("extract:"):
        return _series(path.split(":", 1)[1])  # already rest-frame and windowed
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = {n.lower(): n for n in data.names}
        wave = np.power(10.0, np.asarray(data[names["loglam"]], dtype=float))
        flux = np.asarray(data[names["flux"]], dtype=float)
        ivar = np.asarray(data[names["ivar"]], dtype=float)
    good = np.isfinite(ivar) & (ivar > 0) & np.isfinite(flux) & np.isfinite(wave)
    return wave[good] / (1.0 + _z), flux[good]


def _headroom(ax, arrays, pad_top=0.42, pad_bot=0.10):
    """Robust limits: one deep noise spike should not set the scale."""
    lo = min(float(np.percentile(a, 0.5)) for a in arrays)
    hi = max(float(np.percentile(a, 99.5)) for a in arrays)
    span = hi - lo
    ax.set_ylim(lo - pad_bot * span, hi + pad_top * span)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
wb, fb = _observed(str(_bright.local_path))
wf, ff = _observed(str(_d10.local_path))
ax = axes[0]
shown = []
for w, f, color, label in [
    (wf, ff, MID, "Faint epoch, native"),
    (wb, fb, BLACK, "Bright epoch, native"),
]:
    keep = (w >= 4600) & (w <= 5200)
    ax.plot(w[keep], f[keep], color=color, lw=0.8, label=label)
    shown.append(f[keep])
ax.axvspan(4750, 4940, color=LIGHT, alpha=0.40, zorder=0, lw=0)
_headroom(ax, shown)
ax.set_xlim(4600, 5200)
ax.set_xlabel(r"Rest wavelength ($\AA$)")
ax.set_ylabel(r"$f_\lambda$ (archival units)")
ax.set_title("Archival endpoints", fontsize=8.8)
ax.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.4, borderpad=0.1)

ax = axes[1]
shown = []
for row, color, lw, label in [
    (_d5, LIGHT, 0.8, "Faint, degraded to S/N 5"),
    (_d10, MID, 0.8, "Faint, degraded to S/N 10"),
    (_bright, BLACK, 0.9, "Bright, native"),
]:
    w = np.asarray(row.green_wavelength_rest, dtype=float)
    f = np.asarray(row.green_line_flux, dtype=float)
    keep = (w >= 4600) & (w <= 5200)
    ax.plot(w[keep], f[keep], color=color, lw=lw, label=label)
    shown.append(f[keep])
ax.axvspan(4750, 4940, color=LIGHT, alpha=0.40, zorder=0, lw=0)
_headroom(ax, shown, pad_top=0.55)
ax.set_xlim(4600, 5200)
ax.set_xlabel(r"Rest wavelength ($\AA$)")
ax.set_ylabel("Host- and continuum-subtracted flux")
ax.set_title("As the pixel statistic receives them", fontsize=8.8)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper left",
          fontsize=6.8, handlelength=1.4, borderpad=0.1)
fig.suptitle(
    f"{EXAMPLE}  ($z={_z:.3f}$; native faint-epoch H"
    r"$\beta$-window S/N " + f"{float(_d10.original_snr):.1f})",
    y=1.02, fontsize=9.2,
)
save(fig, "fig2_representative_spectra")


# Figure 3: Green paired S/N effects. The D-095 delta column is ignored because a
# legacy CSV serializer embedded a pandas Series string; differences are exactly
# reconstructed from the frozen p_i columns.
rows = read_csv(D095 / "green_common_support_snr_paired_transitions_d095.csv")
summary = {r["arm"]: r for r in read_csv(D095 / "green_common_support_snr_summary_d095.csv")}
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.25),
                         gridspec_kw={"width_ratios": [1.0, 1.0, 1.14], "wspace": 0.38})
for ax, arm, color, marker, title in zip(
    axes[:2], ["faint_only", "matched"], [BLACK, DARK], ["o", "s"], ["Faint-only", "Matched"], strict=True
):
    sub = [r for r in rows if r["arm"] == arm]
    for r in sub:
        y = [float(r["p_i_5"]), float(r["p_i_10"])]
        ax.plot([5, 10], y, color=color, alpha=0.23, lw=0.9)
        ax.scatter([5, 10], y, color=color, marker=marker, alpha=0.45, s=10, zorder=2)
    s = summary[arm]
    m5 = float(s["common_support_R_snr5"])
    m10 = float(s["common_support_R_snr10"])
    ax.plot([5, 10], [m5, m10], color="black", lw=2.4, marker="o", ms=5, zorder=5)
    delta = float(s["delta_p_mean_10_minus_5"])
    lo, hi = float(s["paired_mean_ci_low"]), float(s["paired_mean_ci_high"])
    ax.text(0.04, 0.965, f"N={len(sub)}\nmean $\\Delta p$={delta:.3f}\n95% bootstrap interval\n{lo:.3f}\u2013{hi:.3f}",
            transform=ax.transAxes, va="top", fontsize=6.6)
    ax.set_title(title)
    ax.set_xticks([5, 10])
    ax.set_xlabel("Target median native-pixel S/N")
    ax.set_xlim(4.3, 10.7)
    ax.set_ylim(-0.04, 1.04)
axes[0].set_ylabel("Transition recovery $p_i$")
axes[1].set_yticklabels([])

# Third panel: the distribution of the per-transition change itself. The two
# left panels show that recovery rises on average; they do not show that the
# rise is concentrated. Sorting the differences makes the gap visible -- a group
# sitting at zero, a group spread high, and almost nothing between.
for arm, color, marker, label in [("faint_only", BLACK, "o", "Faint-only"),
                                  ("matched", DARK, "s", "Matched")]:
    deltas = np.sort(np.array(
        [float(r["p_i_10"]) - float(r["p_i_5"]) for r in rows if r["arm"] == arm]))
    fraction = (np.arange(len(deltas)) + 0.5) / len(deltas)
    axes[2].step(deltas, fraction, where="post", color=color, lw=1.4, label=label)
    axes[2].scatter(deltas, fraction, color=color, marker=marker, s=9, zorder=3)
axes[2].axvline(0.0, color=MID, lw=0.8, ls=":", zorder=0)
axes[2].set_xlabel(r"Per-transition change $\Delta p_i$")
axes[2].set_ylabel("Cumulative fraction", labelpad=2)
axes[2].set_title("Distribution of the change")
axes[2].set_xlim(-0.12, 1.02)
axes[2].set_ylim(0, 1.02)
axes[2].legend(frameon=False, fontsize=6.8, loc="lower right", handlelength=1.4)

fig.suptitle("Green recovery on paired common-support transitions", y=1.01)
save(fig, "fig3_green_paired_snr")


# Figure 4: Green paired arm effects.
arm_rows = read_csv(D095 / "green_common_support_arm_paired_transitions_d095.csv")
arm_summary = {int(r["snr"]): r for r in read_csv(D095 / "green_common_support_arm_summary_d095.csv")}
fig, ax = plt.subplots(figsize=(5.6, 3.35))
rng = np.random.default_rng(1968)
for x, snr, color, marker in [(0, 5, BLACK, "o"), (1, 10, DARK, "s")]:
    vals = np.array([
        float(r["p_i_faint_only"]) - float(r["p_i_matched"])
        for r in arm_rows if int(r["snr"]) == snr
    ])
    jitter = rng.uniform(-0.11, 0.11, len(vals))
    ax.scatter(x + jitter, vals, s=18, marker=marker, alpha=0.48, color=color, edgecolor="none")
    s = arm_summary[snr]
    mean = float(s["delta_p_mean_faint_only_minus_matched"])
    lo, hi = float(s["paired_mean_ci_low"]), float(s["paired_mean_ci_high"])
    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="black",
                capsize=4, lw=1.8, ms=6, zorder=5)
    ax.text(x, 1.03, f"N={len(vals)}", ha="center", va="bottom")
ax.axhline(0, color="0.35", lw=0.9, ls="--")
ax.set_xticks([0, 1], ["S/N 5", "S/N 10"])
ax.set_ylabel("$p_i$(faint-only) - $p_i$(matched)")
ax.set_ylim(-0.24, 1.12)
ax.set_title("Paired Green arm contrast")
save(fig, "fig4_green_arm_effect")


# Figure 5: protocol applicability, measurement validity, and disagreement.
agg = read_csv(D094 / "aggregate_estimands_bootstrap_d094.csv")
yang = [r for r in agg if r["selection_scope"] == "PRIMARY_GOLD" and r["criterion_id"] == "YANG2024_FINAL"]
cont = read_csv(D095 / "yang_green_both_classifiable_contingency_d095.csv")
fail_rows = read_csv(D095 / "yang_failure_decomposition_summary_d095.csv")
fail_counts = Counter()
for r in fail_rows:
    if r["failure_category"] != "inapplicable_by_design":
        fail_counts[r["failure_category"]] += int(r["N"])
assert sum(fail_counts.values()) == 4456
assert fail_counts["degraded_fit_boundary_failure"] == 3387
assert fail_counts["native_boundary_failure"] == 550
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.7),
                         gridspec_kw={"width_ratios": [1.45, 0.95, 1.10], "wspace": 0.42})
labels = ["F5", "F10", "M5", "M10"]
# Left panel: the three states an applicable realization can end in, for both
# protocols on a common denominator. Conditioning on a classifiable measurement,
# as the recovery fraction does, hides the third state entirely -- and that state
# is where most of the fitted-line protocol's realizations end up.
_st = pd.read_csv(D094 / "three_state_outcomes_d094.csv")
_st_by = {(r.protocol, r.arm, int(r.snr)): r for r in _st.itertuples()}
_order = [("Faint-only", 5), ("Faint-only", 10), ("Matched", 5), ("Matched", 10)]
_pos, _cl, _non, _inv, _tick = [], [], [], [], []
for gi, proto in enumerate(["Green-statistic", "Yang-ratio"]):
    for bi, (arm, snr) in enumerate(_order):
        row = _st_by[(proto, arm, snr)]
        _pos.append(gi * 5.4 + bi * 1.06)
        _cl.append(row.p_cl)
        _non.append(row.p_noncl)
        _inv.append(row.p_invalid)
        _tick.append(labels[bi])
_cl, _non, _inv = np.array(_cl), np.array(_non), np.array(_inv)
axes[0].bar(_pos, _cl, color=LIGHT, edgecolor=BLACK, label="CL recovered")
axes[0].bar(_pos, _non, bottom=_cl, color=DARK, edgecolor=BLACK, label="Valid non-CL")
axes[0].bar(_pos, _inv, bottom=_cl + _non, color="white", edgecolor=BLACK,
            hatch="////", label="No valid measurement")
axes[0].set_xticks(_pos, _tick, fontsize=7)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Fraction of applicable realizations")
axes[0].set_title("Three-state outcome")
axes[0].text(1.59, -0.115, "Green-statistic", ha="center", fontsize=7.4,
             transform=axes[0].get_xaxis_transform())
axes[0].text(6.99, -0.115, "Yang-ratio", ha="center", fontsize=7.4,
             transform=axes[0].get_xaxis_transform())
axes[0].legend(frameon=False, fontsize=6.6, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), handlelength=1.2,
               columnspacing=1.0, handletextpad=0.4)
cats = ["Degraded\nboundary", "Native\nboundary", "Local fit\ninvalid", "Unavailable"]
vals = [fail_counts[k] for k in ["degraded_fit_boundary_failure","native_boundary_failure","local_fit_invalidity","measurement_unavailable"]]
axes[1].barh(np.arange(4), vals, color=[BLACK, DARK, MID, LIGHT], edgecolor=BLACK)
axes[1].set_yticks(np.arange(4), cats)
axes[1].invert_yaxis()
axes[1].set_xlabel("Applicable unclassifiable rows")
axes[1].set_title("Applicable invalid rows")
# Right panel: the transition-level disagreement, not the pooled realization
# fraction. The transition is the independent unit, so it is the one that
# belongs in a main figure; marker area shows how many both-classifiable draws
# each d_i rests on, which ranges from 1 to 49.
per_tr = read_csv(DERIVED / "green_yang_transition_disagreement_d099.csv")
summary_rows = {
    (r["arm"], int(r["snr"])): r
    for r in read_csv(DERIVED / "green_yang_equal_transition_summary_d099.csv")
    if r["selection_scope"] == "PRIMARY_GOLD"
}
rng2 = np.random.default_rng(20260818)
for x, (arm, snr) in enumerate([("faint_only",5),("faint_only",10),("matched",5),("matched",10)]):
    pts = [r for r in per_tr if r["arm"] == arm and int(r["snr"]) == snr
           and r["reference_tier"] == "GOLD"]
    y = np.array([float(r["disagreement_i"]) for r in pts])
    n = np.array([int(r["N_both_classifiable"]) for r in pts])
    axes[2].scatter(x + rng2.uniform(-0.16, 0.16, len(y)), y, s=3 + 1.5 * np.sqrt(n),
                    facecolor=MID, edgecolor=BLACK, linewidth=0.3, alpha=0.65, zorder=2)
    s_row = summary_rows[(arm, snr)]
    mean = float(s_row["mean_disagreement_equal_transition"])
    lo, hi = float(s_row["ci_low"]), float(s_row["ci_high"])
    axes[2].errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=BLACK,
                     capsize=3, lw=1.5, ms=4.5, zorder=5)
    axes[2].text(x, 1.04, f"{len(y)}", ha="center", va="bottom", fontsize=6.5)
axes[2].set_xticks(range(4), labels)
axes[2].set_ylim(-0.05, 1.12)
axes[2].set_ylabel("Transition disagreement $d_i$")
axes[2].set_title("Both classifiable")
fig.suptitle("Conditional recovery can conceal measurement invalidity", y=1.03)
fig.subplots_adjust(wspace=0.62)
save(fig, "fig5_classifier_failure_modes")


# Figure 5: prespecified robustness of paired Green S/N effect.
rob = read_csv(D095 / "green_paired_robustness_d095.csv")
names = ["Primary GOLD", "GOLD+SILVER", "Direct-selection excluded", "Equal-family", "LOTO range", "LOSFO range"]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), sharey=True)
for ax, arm, color, marker, title in zip(axes, ["faint_only", "matched"], [BLACK, DARK], ["o", "s"], ["Faint-only", "Matched"], strict=True):
    sub = [r for r in rob if r["arm"] == arm]
    direct = {}
    ranges = {}
    for analysis in ["PRIMARY_GOLD", "GOLD_PLUS_SILVER", "DISCOVERY_INDEPENDENT_GREEN", "EQUAL_SOURCE_FAMILY_WEIGHTED"]:
        direct[analysis] = float(next(r for r in sub if r["analysis"] == analysis)["mean_delta_p_10_minus_5"])
    for analysis in ["LEAVE_ONE_TRANSITION_OUT", "LEAVE_ONE_SOURCE_FAMILY_OUT"]:
        v = [float(r["mean_delta_p_10_minus_5"]) for r in sub if r["analysis"] == analysis]
        ranges[analysis] = (min(v), max(v))
    vals = [direct["PRIMARY_GOLD"], direct["GOLD_PLUS_SILVER"], direct["DISCOVERY_INDEPENDENT_GREEN"], direct["EQUAL_SOURCE_FAMILY_WEIGHTED"]]
    y = np.arange(6)
    ax.scatter(vals, y[:4], color=color, marker=marker, s=35, zorder=3)
    ax.scatter([vals[3]], [y[3]], facecolor="white", edgecolor=BLACK, marker="D", s=58, zorder=4)
    for yi, key in [(4, "LEAVE_ONE_TRANSITION_OUT"), (5, "LEAVE_ONE_SOURCE_FAMILY_OUT")]:
        lo, hi = ranges[key]
        ax.plot([lo, hi], [yi, yi], color=color, lw=4, solid_capstyle="round")
        ax.scatter([(lo+hi)/2], [yi], color="white", edgecolor=color, marker=marker, s=28, zorder=3)
    ax.axvline(0, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(0.14, 0.35)
    ax.set_xlabel("Mean paired $p_i(10)-p_i(5)$")
    ax.set_title(title)
axes[0].set_yticks(np.arange(6), names)
axes[0].invert_yaxis()
fig.suptitle("Prespecified robustness of the Green S/N direction", y=1.02)
save(fig, "fig6_green_robustness")


# Appendix host figure: only the frozen 0.85 shape-dependent cells.
host = read_csv(HOST / "pilot_classification_outcomes.csv")
host = [r for r in host if r["criterion"] == "yang_2024_flux_ratio" and r["template_id"] != "NONE" and float(r["requested_f_added_star"]) == 0.85 and r["pilot_arm_id"] in {"DIAG_J161_BRIGHT", "DIAG_J161_FAINT", "PAIR_J161_FAINT_ANCHORED"}]
arm_order = ["DIAG_J161_BRIGHT", "DIAG_J161_FAINT", "PAIR_J161_FAINT_ANCHORED"]
shape_order = ["XSL_H1", "XSL_H2", "XSL_H3"]
fig, ax = plt.subplots(figsize=(6.2, 3.45))
for j, arm in enumerate(arm_order):
    for k, shape in enumerate(shape_order):
        r = next(x for x in host if x["pilot_arm_id"] == arm and x["template_id"] == shape)
        x = j + (k - 1) * 0.2
        stat = float(r["statistic"])
        ax.scatter(x, stat, s=48, marker=["o","s","^"][k], label=shape if j == 0 else None)
ax.axhline(0.3, color="black", ls="--", lw=1, label="Yang-ratio threshold")
ax.set_ylim(-0.1, 4.8)
ax.set_xticks(range(3), ["Bright-only\ndiagnostic", "Faint-only\ndiagnostic", "Pair-consistent"])
ax.set_ylabel("Faint/bright broad-H$\\beta$ flux ratio")
ax.set_title("Controlled stellar-shape diagnostic at $f_{\\mathrm{star,add}}=0.85$")
ax.legend(frameon=False, ncol=2)
save(fig, "figa1_host_shape_diagnostic")


# Table 1: frozen sample composition.
gold = read_csv(D093 / "final_gold_sample_d093.csv")
sens = read_csv(D093 / "final_sensitivity_sample_d093.csv")
gc = Counter(r["source_family"] for r in gold)
sc = Counter(r["source_family"] for r in sens)
families = sorted(sc)
labels_family = {
    "Dong2025_SDSS_LAMOST": "Dong et al. (2025)",
    "Zeltyn2024_SDSSV": "Zeltyn et al. (2024)",
    "Yang2025_turn_on": "Yang et al. (2025)",
    "Potts2021": r"Potts \& Villforth (2021)",
    "Ruan2016": "Ruan et al. (2016)",
    "Green2022": "Green et al. (2022)",
    "LaMassa2015": "LaMassa et al. (2015)",
    "Runnoe2016": "Runnoe et al. (2016)",
    "MacLeod2016": "MacLeod et al. (2016)",
    "Yang2018": "Yang et al. (2018)",
}
lines = [r"\begin{deluxetable}{lrr}", r"\tablecaption{Outcome-blind reference sample composition\label{tab:sample}}", r"\tablehead{\colhead{Source family} & \colhead{GOLD} & \colhead{GOLD+SILVER}}", r"\startdata"]
for _fam in families:
    pass
r"""
    lines.append(f"{labels_family[fam]} & {gc[fam]} & {sc[fam]} \\\\")
lines += [r"\hline", f"Total & {len(gold)} & {len(sens)} \\\", r"\enddata", r"\tablecomments{A transition is counted once even if it appears in more than one publication. GOLD is the primary classifier-blind reference set; SILVER enters only the prespecified sensitivity analysis.}", r"\end{deluxetable}"]
"""
lines = [r"\begin{deluxetable}{lrr}", r"\tablecaption{Outcome-blind reference sample composition\label{tab:sample}}", r"\tablehead{\colhead{Source family} & \colhead{GOLD} & \colhead{GOLD+SILVER}}", r"\startdata"]
for fam in families:
    lines.append(f"{labels_family[fam]} & {gc[fam]} & {sc[fam]} " + r"\\")
lines.extend([r"\hline", f"Total & {len(gold)} & {len(sens)} " + r"\\", r"\enddata", r"\tablecomments{A transition is counted once even if it appears in more than one publication. GOLD is the primary outcome-blind reference set; SILVER enters only the prespecified sensitivity analysis.}", r"\end{deluxetable}"])
(TAB / "table1_sample_composition.tex").write_text("\n".join(lines) + "\n")


# Table 3: transition-leading primary estimates and separate realization accounting.
green = [r for r in agg if r["selection_scope"] == "PRIMARY_GOLD" and r["criterion_id"] == "GREEN2022_FINAL"]
by = {(r["criterion_id"], r["arm"], int(r["snr"])): r for r in agg if r["selection_scope"] == "PRIMARY_GOLD"}
_states = pd.read_csv(D094 / "three_state_outcomes_d094.csv")
_yields = pd.read_csv(D094 / "operational_yield_d094.csv")
_state_by = {(r.protocol, r.arm, int(r.snr)): r for r in _states.itertuples()}
_yield_by = {(r.protocol, r.arm, int(r.snr)): r for r in _yields.itertuples()}
lines = [r"\begin{deluxetable*}{llrrrrrrrrcc}", r"\tablecaption{Primary transition outcomes, realization accounting, and operational yield\label{tab:recovery}}", r"\tablehead{\colhead{Protocol} & \colhead{Arm} & \colhead{S/N} & \colhead{$N_{\rm tr,elig}$} & \colhead{$N_{\rm tr,app}$} & \colhead{$N_{\rm tr,class}$} & \colhead{$n_{\rm real,app}$} & \colhead{$n_{\rm CL}$} & \colhead{$n_{\rm nonCL}$} & \colhead{$n_{\rm real,invalid}$} & \colhead{$R$} & \colhead{$Y$}}", r"\startdata"]
for crit, cname in [("GREEN2022_FINAL", "Green-statistic"), ("YANG2024_FINAL", "Yang-ratio")]:
    for arm, aname in [("faint_only", "Faint-only"), ("matched", "Matched")]:
        for snr in (5, 10):
            r = by[(crit, arm, snr)]
            st = _state_by[(cname, aname, snr)]
            yl = _yield_by[(cname, aname, snr)]
            n_app = 50 * int(r['N_classifier_applicable'])
            n_invalid = n_app - int(r['N_classifiable'])
            assert st.n_applicable == n_app and st.n_invalid == n_invalid
            lines.append(f"{cname} & {aname} & {snr} & {r['N_transition_eligible']} & {r['N_classifier_applicable']} & {r['N_transition_classifiable']} & {n_app} & {st.n_cl} & {st.n_noncl} & {n_invalid} & {float(r['R_equal_transition']):.3f} & {yl.Y_operational:.3f} \\\\")
lines += [r"\enddata", r"\tablecomments{$R$ is the equal-transition mean of $p_i=n_{\rm CL}/n_{\rm classifiable}$, the recovery fraction \emph{given} a valid measurement. $Y$ is the equal-transition mean of $Y_i=n_{\rm CL}/n_{\rm applicable}$, the operational recovery yield, which keeps invalid measurements in the denominator. The two coincide for the Green-statistic protocol because no applicable realization was invalid, and differ by a factor of three to four for the Yang-ratio protocol. $Y$ is a yield for this reference set under this intervention, not a completeness. For each row $n_{\rm real,app}=n_{\rm CL}+n_{\rm nonCL}+n_{\rm real,invalid}$; realizations for which the protocol was inapplicable are excluded here and reported in Appendix Table~\ref{tab:applicability}.}", r"\end{deluxetable*}"]
(TAB / "table3_primary_recovery.tex").write_text("\n".join(lines) + "\n")


# Appendix: outcome geometry with protocol applicability held fixed.
_ca = pd.read_csv(D094 / "common_applicability_d094.csv")
lines = [r"\begin{deluxetable*}{llrrrrrrrr}", r"\tablecaption{Outcome geometry on transitions applicable under both protocols\label{tab:commonapp}}", r"\tablehead{\colhead{Arm} & \colhead{S/N} & \colhead{Protocol} & \colhead{$N_{\rm tr}$} & \colhead{$n_{\rm real,app}$} & \colhead{$n_{\rm CL}$} & \colhead{$n_{\rm nonCL}$} & \colhead{$n_{\rm invalid}$} & \colhead{$R$} & \colhead{$Y$}}", r"\startdata"]
for r in _ca.itertuples():
    lines.append(f"{r.arm} & {int(r.snr)} & {r.protocol} & {int(r.N_tr_common)} & {int(r.n_applicable)} & {int(r.n_cl)} & {int(r.n_noncl)} & {int(r.n_invalid)} & {r.R_conditional:.3f} & {r.Y_operational:.3f} " + r"\\")
lines += [r"\enddata", r"\tablecomments{Both protocols are restricted to the transitions applicable under both, so within each arm and rung the two rows share an identical transition set and an identical number of applicable realizations. Between one and seven transitions are dropped per protocol and condition. The Green-statistic protocol has no invalid realization in any condition, so its $R$ and $Y$ coincide; the Yang-ratio/PyQSOFit invalid fraction is 0.632--0.677 and its $R$ exceeds its $Y$ by a factor of 2.97--3.84. Holding applicability fixed therefore does not change the difference in outcome geometry reported in Section~\ref{sec:threestate}. The comparison concerns these two operational implementations, not the published methods whose decision rules they follow.}", r"\end{deluxetable*}"]
(TAB / "tablea13_common_applicability.tex").write_text("\n".join(lines) + "\n")


# Appendix: fit validity before and after degradation.
_vb = pd.read_csv(D094 / "fit_validity_baseline_d094.csv")
lines = [r"\begin{deluxetable}{llrrr}", r"\tablecaption{Fit validity on undegraded and degraded spectra\label{tab:validbase}}", r"\tablehead{\colhead{Spectra} & \colhead{Target S/N} & \colhead{$N_{\rm fits}$} & \colhead{Yang valid} & \colhead{Green valid}}", r"\startdata"]
for r in _vb.itertuples():
    lines.append(f"{r.spectra} & {r.target_snr} & {int(r.n_fits)} & {r.yang_valid_fraction:.3f} & {r.green_valid_fraction:.3f} " + r"\\")
lines += [r"\enddata", r"\tablecomments{Fractions of spectral fits that satisfied each protocol's validity requirements. The three groups have different memberships and different denominators---the degraded sets contain only spectra whose native quality supports the target---so the rows are not a controlled series and the columns should not be differenced. They establish one point: the Yang-ratio/PyQSOFit fit is already invalid for 19 of the 62 undegraded native bright-epoch spectra, so its invalidity is present before any noise is added rather than produced by degradation. The Green-statistic preprocessing was valid for every fit input.}", r"\end{deluxetable}"]
(TAB / "tablea14_fit_validity_baseline.tex").write_text("\n".join(lines) + "\n")


# Appendix: how well determined the quoted medians are.
_mu = pd.read_csv(D094 / "paired_median_uncertainty_d094.csv")
lines = [r"\begin{deluxetable}{lrcc}", r"\tablecaption{Resampling spread of the paired difference: mean against median\label{tab:medianspread}}", r"\tablehead{\colhead{Arm} & \colhead{$N_{\rm tr}$} & \colhead{Mean (95\% interval)} & \colhead{Median (95\% interval)}}", r"\startdata"]
for r in _mu.itertuples():
    lines.append(f"{r.arm} & {int(r.N_tr)} & ${r.mean:+.3f}$ ({r.mean_ci_low:.3f}--{r.mean_ci_high:.3f}) & ${r.median:+.3f}$ ({r.median_ci_low:.3f}--{r.median_ci_high:.3f}) " + r"\\")
lines += [r"\enddata", r"\tablecomments{Both intervals are 95\% percentile intervals from 5000 transition-bootstrap resamples using the same seed construction as Table~\ref{tab:paired}. They describe resampling variation within this reference census and are not confidence intervals for a parent population. The median is much the less well determined of the two summaries: its interval is roughly twice the width of the mean's and, in the faint-only arm, reaches zero. The medians quoted in Section~\ref{sec:results} should therefore be read as sample medians indicating the shape of a heterogeneous distribution, not as precisely estimated quantities.}", r"\end{deluxetable}"]
(TAB / "tablea15_median_uncertainty.tex").write_text("\n".join(lines) + "\n")


# Appendix: the prespecified S/N metric against line-free continuum windows.
_cs = pd.read_csv(D094 / "continuum_snr_summary_d094.csv")
_labels = {"blue_4435_4700": r"4435--4700~\AA", "red_5100_5535": r"5100--5535~\AA"}
lines = [r"\begin{deluxetable*}{lrrrrrrr}", r"\tablecaption{The prespecified H$\beta$-window S/N against line-free continuum windows\label{tab:contsnr}}", r"\tablehead{\colhead{Continuum window} & \colhead{$N_{\rm ep}$} & \colhead{Pearson $r$} & \colhead{Spearman $\rho$} & \colhead{Median ratio} & \colhead{Support S/N 5} & \colhead{Disagree 5} & \colhead{Disagree 10}}", r"\startdata"]
for r in _cs.itertuples():
    lines.append(f"{_labels[r.continuum_window]} & {int(r.n_endpoints)} & {r.pearson_r:.3f} & {r.spearman_rho:.3f} & {r.median_ratio_hbeta_over_continuum:.3f} & {int(r.support_hbeta_snr5)}/{int(r.support_continuum_snr5)} & {int(r.support_disagree_snr5)} & {int(r.support_disagree_snr10)} " + r"\\")
lines += [r"\enddata", r"\tablecomments{Median per-pixel S/N recomputed on the 114 native endpoint spectra over two conventional line-free continuum windows either side of the H$\beta$ complex, both outside the 4700--5100~\AA\ analysis window. The recomputed H$\beta$-window value reproduces the frozen metric exactly, so the comparison is between metrics rather than between implementations. ``Median ratio'' is the H$\beta$-window value divided by the continuum value. ``Support'' counts endpoints reaching S/N 5 under the H$\beta$ metric and under the continuum metric; ``Disagree'' counts endpoints whose support status differs between the two at that rung. The two measures track each other closely overall, but membership is not invariant, and the disagreement is larger at S/N 10 than at S/N 5. The experiment was not rerun under a continuum metric and its results remain conditional on the prespecified H$\beta$-window definition.}", r"\end{deluxetable*}"]
(TAB / "tablea16_continuum_snr.tex").write_text("\n".join(lines) + "\n")


# Table 4: paired primary effects and prespecified sensitivity summaries.
lines = [r"\begin{deluxetable*}{llrrrrrr}", r"\tablecaption{Paired Green contrasts and robustness\label{tab:paired}}", r"\tablehead{\colhead{Contrast} & \colhead{Arm/rung} & \colhead{$N$} & \colhead{Mean} & \colhead{95\% interval} & \colhead{Median} & \colhead{$N_+/N_0/N_-$} & \colhead{Sensitivity range}}", r"\startdata"]
for arm, label in [("faint_only", "Faint-only"), ("matched", "Matched")]:
    s = summary[arm]
    los = [float(r["mean_delta_p_10_minus_5"]) for r in rob if r["arm"] == arm and r["analysis"] == "LEAVE_ONE_SOURCE_FAMILY_OUT"]
    lines.append(f"$p_i(10)-p_i(5)$ & {label} & {s['paired_transition_count']} & {float(s['delta_p_mean_10_minus_5']):+.3f} & {float(s['paired_mean_ci_low']):.3f}--{float(s['paired_mean_ci_high']):.3f} & {float(s['delta_p_median_10_minus_5']):+.3f} & {s['N_positive']}/{s['N_zero']}/{s['N_negative']} & LOSFO {min(los):.3f}--{max(los):.3f} \\\\")
r"""
for snr in (5, 10):
    s = arm_summary[snr]
    lines.append(f"$p_i({{\rm faint}})-p_i({{\rm matched}})$ & S/N {snr} & {s['paired_transition_count']} & {float(s['delta_p_mean_faint_only_minus_matched']):+.3f} & {float(s['paired_mean_ci_low']):.3f}--{float(s['paired_mean_ci_high']):.3f} & {float(s['delta_p_median_faint_only_minus_matched']):+.3f} & {s['N_positive']}/{s['N_zero']}/{s['N_negative']} & \nodata \\\\")
"""
for snr in (5, 10):
    s = arm_summary[snr]
    lines.append(f"$p_i({{\\rm faint}})-p_i({{\\rm matched}})$ & S/N {snr} & {s['paired_transition_count']} & {float(s['delta_p_mean_faint_only_minus_matched']):+.3f} & {float(s['paired_mean_ci_low']):.3f}--{float(s['paired_mean_ci_high']):.3f} & {float(s['delta_p_median_faint_only_minus_matched']):+.3f} & {s['N_positive']}/{s['N_zero']}/{s['N_negative']} & \\nodata " + r"\\")
lines += [r"\enddata", r"\tablecomments{Intervals are 95\% percentile intervals from 5000 transition-bootstrap resamples, in which the transition is the resampled unit and common-support membership is held fixed. They summarize the sensitivity of the equal-transition statistic to resampling the observed transitions; because this census is not a probability sample, they are not confidence intervals for a parent population. LOSFO gives the leave-one-source-family-out range for the S/N contrast. The complete robustness set also includes GOLD+SILVER, direct-criterion-selection-excluded, equal-family, and leave-one-transition-out analyses. Means and medians differ substantially: the response across transitions is heterogeneous.}", r"\end{deluxetable*}"]
(TAB / "table4_paired_effects.tex").write_text("\n".join(lines) + "\n")


# Appendix tables.
failure_values = [fail_counts[k] for k in ["degraded_fit_boundary_failure","native_boundary_failure","local_fit_invalidity","measurement_unavailable"]]
assert failure_values == [3387, 550, 498, 21]
failure_total = sum(failure_values)
assert failure_total == 4456
gold_only = Counter()
for r in read_csv(D095 / "yang_failure_decomposition_summary_d095.csv"):
    if r["reference_tier"] == "GOLD" and r["failure_category"] != "inapplicable_by_design":
        gold_only[r["failure_category"]] += int(r["N"])
assert [gold_only[k] for k in ["degraded_fit_boundary_failure","native_boundary_failure","local_fit_invalidity","measurement_unavailable"]] == [3031, 550, 475, 19]
assert sum(gold_only.values()) == 4075
lines = [r"\begin{deluxetable}{lrr}", r"\tablecaption{Yang applicable unclassifiable failure modes\label{tab:yangfail}}", r"\tablehead{\colhead{Prespecified failure category} & \colhead{$N$} & \colhead{Fraction}}", r"\startdata"]
for label, n in zip(["Degraded fit-boundary", "Native boundary", "Local fit invalidity", "Measurement unavailable"], failure_values, strict=True):
    lines.append(f"{label} & {n} & {n/failure_total:.3f} \\\\")
lines += [r"\enddata", r"\tablecomments{Counts cover the full production set of 62 transitions; the 58-transition primary set gives 3031, 550, 475, and 19, totalling 4075. Categories are mutually exclusive under the priority ordering fixed before production. Boundary-related categories total 3937/4456 (88.4\%), or 3581/4075 (87.9\%) on the primary set alone. Valid nondetections were assigned zero line flux where the source protocol permitted; none of them produced an invalid measurement.}", r"\end{deluxetable}"]
(TAB / "tablea1_yang_failures.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Green--Yang contingency and asymmetric unclassifiability\label{tab:contingency}}", r"\scriptsize", r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r"Arm & S/N & Both & C/C & C/N & N/C & N/N & C/U & U/C & U/U \\", r"\midrule"]
for r in cont:
    lines.append(f"{r['arm'].replace('_','-')} & {r['snr']} & {r['N_both_classifiable']} & {r['Green_CL__Yang_CL']} & {r['Green_CL__Yang_nonCL']} & {r['Green_nonCL__Yang_CL']} & {r['Green_nonCL__Yang_nonCL']} & {r['N_Green_classifiable_Yang_unclassifiable']} & {r['N_Green_unclassifiable_Yang_classifiable']} & {r['N_both_unclassifiable']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\smallskip", r"\parbox{0.93\textwidth}{\scriptsize \textit{Note.} Each pair is Green/Yang. C, N, and U denote CL, non-CL, and unclassifiable. The central four cells include only realizations for which both protocols are classifiable. Unclassifiability is neither disagreement nor non-CL.}", r"\end{table*}"]
# Split the wide accounting into two one-column appendix tables so the ApJ
# two-column draft does not create a nearly empty float page.
lines = [r"\begin{deluxetable}{llrrrrr}", r"\tabletypesize{\scriptsize}", r"\tablecaption{Green--Yang both-classifiable contingency\label{tab:contingency}}", r"\tablehead{\colhead{Arm} & \colhead{S/N} & \colhead{Both} & \colhead{C/C} & \colhead{C/N} & \colhead{N/C} & \colhead{N/N}}", r"\startdata"]
for r in cont:
    lines.append(f"{r['arm'].replace('_','-')} & {r['snr']} & {r['N_both_classifiable']} & {r['Green_CL__Yang_CL']} & {r['Green_CL__Yang_nonCL']} & {r['Green_nonCL__Yang_CL']} & {r['Green_nonCL__Yang_nonCL']} " + r"\\")
lines += [r"\enddata", r"\tablecomments{Each pair is Green/Yang. C and N denote CL and non-CL. Only realization cells classifiable under both protocols enter this table.}", r"\end{deluxetable}", "", r"\begin{deluxetable}{llrrr}", r"\tabletypesize{\scriptsize}", r"\tablecaption{Asymmetric protocol unclassifiability\label{tab:asymunclass}}", r"\tablehead{\colhead{Arm} & \colhead{S/N} & \colhead{Class./Unclass.} & \colhead{Unclass./Class.} & \colhead{Unclass./Unclass.}}", r"\startdata"]
for r in cont:
    lines.append(f"{r['arm'].replace('_','-')} & {r['snr']} & {r['N_Green_classifiable_Yang_unclassifiable']} & {r['N_Green_unclassifiable_Yang_classifiable']} & {r['N_both_unclassifiable']} " + r"\\")
lines += [r"\enddata", r"\tablecomments{Each pair is Green/Yang. Class. and Unclass. denote classifiable and unclassifiable. Unclassifiability is neither disagreement nor non-CL.}", r"\end{deluxetable}"]
(TAB / "tablea2_contingency.tex").write_text("\n".join(lines) + "\n")


# Machine-readable reference manifest assembled only from prespecified reference
# and immutable endpoint tables. This is documentation, not a new adjudication.
bindings = read_csv(ROOT / "05_analysis/q1_production/d094/raw/endpoint_bindings_d094.csv")
binding_by = {(r["transition_id"], r["role"]): r for r in bindings}
evidence = {}
for path, id_col, evidence_col in [
    (ROOT / "04_reference_sample/q1_completeness_gold_acquisition_d092.csv", "object", "evidence_basis"),
    (ROOT / "04_reference_sample/q1_expansion_candidates_d090.csv", "object", "evidence_summary"),
    (ROOT / "04_reference_sample/reference_tiers_d086.csv", "object_id", "adjudication_basis"),
]:
    for row in read_csv(path):
        evidence.setdefault(row[id_col], row[evidence_col])
publication = {
    "Dong2025_SDSS_LAMOST": "Dong et al. (2025)", "Zeltyn2024_SDSSV": "Zeltyn et al. (2024)",
    "Yang2025_turn_on": "Yang et al. (2025)", "Potts2021": "Potts & Villforth (2021)",
    "Ruan2016": "Ruan et al. (2016)", "Green2022": "Green et al. (2022)",
    "LaMassa2015": "LaMassa et al. (2015)", "Runnoe2016": "Runnoe et al. (2016)",
    "MacLeod2016": "MacLeod et al. (2016)", "Yang2018": "Yang et al. (2018)",
}
generic_evidence = {
    "Dong2025_SDSS_LAMOST": "Broad-Hbeta appearance/disappearance, endpoint morphology, visual and photometric confirmation",
    "Zeltyn2024_SDSSV": "Visual broad-line appearance/disappearance with published endpoint spectra",
    "Yang2025_turn_on": "Turn-on broad-line emergence with optical/MIR variability and spectroscopic confirmation",
    "Green2022": "Broad-Hbeta change, continuum evolution, and multi-epoch spectral evidence",
    "Potts2021": "Difference-spectrum and multi-line transition evidence",
}
manifest_rows = []
for row in sens:
    tid = row["transition_id"]
    bright, faint = binding_by[(tid, "bright")], binding_by[(tid, "faint")]
    manifest_rows.append({
        "transition_id": tid,
        "source_publication": publication[row["source_family"]],
        "source_family": row["source_family"],
        "transition_direction": row["event"],
        "reference_evidence": evidence.get(tid, generic_evidence.get(row["source_family"], "Published broad-line transition morphology and continuum evidence under the prespecified adjudication rule")),
        "reference_tier": row["reference_tier"],
        "q1_technical_eligibility": "ELIGIBLE",
        "exclusion_reason": "",
        "instrument_pair": row["instrument_pair"],
        "redshift": bright["redshift"],
        "bright_endpoint_identifier": bright["spectrum_id"],
        "faint_endpoint_identifier": faint["spectrum_id"],
        "bright_variance_provenance": bright["variance_provenance"],
        "faint_variance_provenance": faint["variance_provenance"],
    })
manifest_path = MANIFEST
with manifest_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0]))
    writer.writeheader()
    writer.writerows(manifest_rows)
assert len(manifest_rows) == 62
assert len({r["transition_id"] for r in manifest_rows}) == 62
assert sum(r["reference_tier"] == "GOLD" for r in manifest_rows) == 58


# Appendix sample-flow table.
lines = [r"\begin{deluxetable}{lr}", r"\tablecaption{Outcome-blind reference flow\label{tab:flow}}", r"\tablehead{\colhead{Stage} & \colhead{$N$}}", r"\startdata",
         r"Literature source families considered & 31 \\", r"High-confidence morphology candidates & 64 \\",
         r"Exact public endpoint pairs identified & 54 \\", r"New pairs passing identity gate & 50 \\",
         r"New pairs passing identity and native QC & 48 \\", r"Previously validated GOLD pairs & 10 \\",
         r"\hspace{1em}\textit{sum of the two validated branches} & \textbf{58} \\", r"Final GOLD+SILVER sensitivity transitions & 62 \\",
         r"\enddata", r"\tablecomments{The 54 exact-public pairs arose from the final prospective expansion audit. Four failed endpoint identity and two additional pairs failed native S/N quality criteria; the strict evidence tier was not weakened.}", r"\end{deluxetable}"]
(TAB / "tablea3_reference_flow.tex").write_text("\n".join(lines) + "\n")


# Appendix paired-subset composition, reconstructed by joining immutable IDs.
gold_by = {r["transition_id"]: r for r in gold}
paired_by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ("faint_only", "matched")}
lines = [r"\begin{deluxetable*}{lrrrrrrrr}", r"\tablecaption{Composition of the Green common-support subsets\label{tab:pairedcomposition}}",
         r"\tablehead{\colhead{Arm} & \colhead{$N_{\rm tr}$} & \colhead{On/Off} & \colhead{Dong} & \colhead{Zeltyn} & \colhead{Yang25} & \colhead{Other} & \colhead{SDSS/LAMOST} & \colhead{$z_{\rm med}$}}", r"\startdata"]
for arm, label in [("faint_only", "Faint-only"), ("matched", "Matched")]:
    ids = [r["transition_id"] for r in paired_by_arm[arm]]
    sub = [gold_by[x] for x in ids]
    fam = Counter(x["source_family"] for x in sub)
    event = Counter(x["event"] for x in sub)
    ipair = Counter(x["instrument_pair"] for x in sub)
    zmed = float(np.median([float(binding_by[(x, "bright")]["redshift"]) for x in ids]))
    other = len(sub) - fam["Dong2025_SDSS_LAMOST"] - fam["Zeltyn2024_SDSSV"] - fam["Yang2025_turn_on"]
    values = [
        label,
        len(sub),
        f"{event['turn_on']}/{event['turn_off']}",
        fam["Dong2025_SDSS_LAMOST"],
        fam["Zeltyn2024_SDSSV"],
        fam["Yang2025_turn_on"],
        other,
        ipair["SDSS_LEGACY/LAMOST_DR11"],
        f"{zmed:.3f}",
    ]
    lines.append(" & ".join(map(str, values)) + r" \\")
lines += [r"\enddata", r"\tablecomments{On/Off gives turn-on/turn-off counts. SDSS/LAMOST denotes the ordered SDSS-legacy bright/LAMOST faint pair; complete instrument-pair membership is supplied in the machine-readable transition manifest. The faint-only and matched redshift ranges are 0.060--0.623 and 0.060--0.348, respectively.}", r"\end{deluxetable*}"]
(TAB / "tablea4_paired_composition.tex").write_text("\n".join(lines) + "\n")


# Appendix instrument-domain documentation; entries reproduce D-093 validation.
lines = [r"\begin{deluxetable*}{p{0.10\textwidth}p{0.13\textwidth}p{0.13\textwidth}p{0.14\textwidth}p{0.13\textwidth}p{0.13\textwidth}p{0.12\textwidth}}",
         r"\tablecaption{Instrument-domain compatibility\label{tab:instrument}}",
         r"\tablehead{\colhead{Product} & \colhead{Wavelength / sampling} & \colhead{Flux / variance} & \colhead{Masks used here} & \colhead{Resolution treatment} & \colhead{Green} & \colhead{Yang-ratio fit}}", r"\startdata",
         r"SDSS legacy (DR17 archive) & Vacuum log-$\lambda$ & $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$; native IVAR & Finite flux and positive IVAR; survey flags documented but not additionally rejected & Native WDISP/grid retained; noise only & Supported variance and coverage & Supported \\",
         r"SDSS-V DR19 & Vacuum BOSS log-$\lambda$ & Same SDSS flux unit; native IVAR & Finite flux and positive IVAR & Native WDISP/grid retained; noise only & Supported variance and coverage & Supported \\",
         r"LAMOST DR11 v2.0 & Vacuum log-linear & Native low-resolution flux and IVAR & Finite flux and positive IVAR; AND/OR/FIB flags documented & Native low-resolution LSF/grid retained; noise only & Supported variance and coverage & Supported \\",
         r"DESI EDR (Fuji) & Vacuum linear camera coadd & $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$; native IVAR & Finite flux, positive IVAR, and supported DESI mask & Resolution matrix and grid retained; noise only & Supported variance and coverage & Supported \\",
         r"\enddata", r"\tablecomments{All degradation occurs on the native observed-frame grid before protocol-specific rest-frame processing. Instrument compatibility was fixed before recovery outcomes.}", r"\end{deluxetable*}"]
(TAB / "tablea5_instrument_domain.tex").write_text("\n".join(lines) + "\n")


# Appendix prospective Monte Carlo convergence table.
conv = read_csv(ROOT / "05_analysis/q1_design/measurement_validity_d089/convergence_m40_m50.csv")
max_dp = max(abs(float(r["abs_delta_recovery_M50_M40"])) for r in conv if r["abs_delta_recovery_M50_M40"])
max_du = max(abs(float(r["abs_delta_unclassifiable_M50_M40"])) for r in conv)
assert max_dp <= 0.10 and max_du <= 0.10
lines = [
    r"\begin{deluxetable}{lcc}",
    r"\tablecaption{Prospective choice of 50 noise realizations\label{tab:mcconv}}",
    r"\tablehead{\colhead{Check} & \colhead{Rule} & \colhead{Observed maximum}}",
    r"\startdata",
    f"Recovery prefix change & $|p_{{50}}-p_{{40}}|\\leq0.10$ & {max_dp:.3f} " + r"\\",
    f"Unclassifiable prefix change & $|u_{{50}}-u_{{40}}|\\leq0.10$ & {max_du:.3f} " + r"\\",
    r"\enddata",
    r"\tablecomments{The rule was recorded after $M=20$ proved insufficient and before the remaining draws were produced; all 16 pilot transition--arm--rung--protocol groups then satisfied it. The final bootstrap intervals condition on the resulting $M=50$ Monte Carlo approximation.}",
    r"\end{deluxetable}",
]
(TAB / "tablea6_mc_convergence.tex").write_text("\n".join(lines) + "\n")

# Table 5: equal-transition Green-Yang disagreement, derived by
# 00_scripts/37_d099_transition_level_disagreement.py from the stored
# classifications. Realizations of one transition are repeated measurements on
# one AGN, so the transition is the unit here and the pooled realization
# fraction is shown only for comparison.
dis = read_csv(DERIVED / "green_yang_equal_transition_summary_d099.csv")
dis = [r for r in dis if r["selection_scope"] == "PRIMARY_GOLD"]
assert len(dis) == 4
lines = [
    r"\begin{deluxetable*}{llrccccc}",
    r"\tablecaption{Green--Yang disagreement with the transition as the unit of analysis\label{tab:disagree}}",
    r"\tablehead{\colhead{Arm} & \colhead{S/N} & \colhead{$N_{\rm tr}$} & \colhead{Mean $d_i$} & "
    r"\colhead{95\% interval} & \colhead{Median $d_i$} & \colhead{$n_{\rm both\ class}$ median (range)} & "
    r"\colhead{Pooled cells}}",
    r"\startdata",
]
for r in sorted(dis, key=lambda r: (r["arm"], int(r["snr"]))):
    lines.append(
        f"{r['arm'].replace('_','-')} & {int(r['snr'])} & {r['N_transitions_defined']} & "
        f"{float(r['mean_disagreement_equal_transition']):.3f} & "
        f"{float(r['ci_low']):.3f}--{float(r['ci_high']):.3f} & "
        f"{float(r['median_disagreement']):.3f} & "
        f"{int(float(r['denominator_median']))} ({r['denominator_min']}--{r['denominator_max']}) & "
        f"{float(r['pooled_realization_disagreement']):.3f} " + r"\\"
    )
lines += [
    r"\enddata",
    r"\tablecomments{$d_i$ is the fraction of a transition's realizations, among those classifiable "
    r"under both protocols, in which the two protocols disagree. Every transition with at least one "
    r"such realization is included; no minimum denominator was specified in advance and none is "
    r"imposed, so the denominators are very unequal and their distribution is given. Intervals are "
    r"95\% percentile intervals from 5000 transition-bootstrap resamples, describing variation "
    r"across this reference census rather than a parent population. The final column is the "
    r"pooled realization-cell fraction of Table~\ref{tab:contingency}, shown to confirm that the "
    r"conclusion does not depend on the choice of unit.}",
    r"\end{deluxetable*}",
]
(TAB / "table5_transition_disagreement.tex").write_text("\n".join(lines) + "\n")


# Appendix Table A7: why a protocol is or is not applicable to a condition.
# Reasons come from the applicability matrix fixed before execution, never from
# an outcome.
app = read_csv(D093 / "final_classifier_applicability_d093.csv")
label = {
    "APPLICABLE_SUPPORTED_VARIANCE": ("Green-statistic", "Applicable", "Both endpoints carry validated native variance and full required coverage"),
    "APPLICABLE_NATIVE_SDSS_VARIANCE_SUPPORTED": ("Green-statistic", "Applicable", "Exact SDSS/SDSS pair with native SDSS variance semantics"),
    "APPLICABLE_FAIL_CLOSED_PER_REALIZATION": ("Yang-ratio", "Applicable", "Native bright fit valid and 4640--5100 \\AA\\ covered; each realization must still pass validity"),
    "UNCLASSIFIABLE_INSTRUMENT_DOMAIN": (None, "Not applicable", "Endpoint instrument domain unsupported for this protocol"),
    "UNCLASSIFIABLE_NATIVE_BRIGHT_FIT_INVALID": (None, "Not applicable", "Native bright-epoch fit itself invalid, so no ratio denominator exists"),
    "UNCLASSIFIABLE_STATIC_NATIVE_BRIGHT_HBETA_COVERAGE": (None, "Not applicable", "Native bright epoch does not cover the required H$\\beta$ region"),
    "UNCLASSIFIABLE_NO_PROSPECTIVE_VISUAL_EVIDENCE": ("MacLeod", "Not applicable", "Requires visual-final evidence unavailable for simulated spectra"),
}
counts: dict[tuple[str, str, str], int] = {}
for row in app:
    crit = {"GREEN2022_FINAL": "Green-statistic", "YANG2024_FINAL": "Yang-ratio",
            "MACLEOD2019_FINAL": "MacLeod"}[row["criterion"]]
    _, state, reason = label[row["applicability"]]
    counts[(crit, state, reason)] = counts.get((crit, state, reason), 0) + 1
lines = [
    r"\begin{deluxetable*}{lllr}",
    r"\tablecaption{Why a protocol is applicable to a transition--arm--rung condition\label{tab:applicability}}",
    r"\tablehead{\colhead{Protocol} & \colhead{State} & \colhead{Reason fixed before execution} & \colhead{Conditions}}",
    r"\startdata",
]
order = ["Green-statistic", "Yang-ratio", "MacLeod"]
for crit in order:
    for (c, state, reason), n in sorted(counts.items(), key=lambda kv: (kv[0][1], -kv[1])):
        if c == crit:
            lines.append(f"{crit} & {state} & {reason} & {n} " + r"\\")
assert sum(counts.values()) == 501
lines += [
    r"\enddata",
    r"\tablecomments{One row of the underlying matrix is one transition--arm--rung condition for one "
    r"protocol; there are 167 conditions and three protocols. Applicability was assigned before any "
    r"realization was drawn, and is distinct from a realization that is applicable but yields an "
    r"invalid measurement. No applicable Green-statistic realization produced an invalid measurement; "
    r"4456 of 7050 applicable Yang-ratio realizations did (Table~\ref{tab:yangfail}).}",
    r"\end{deluxetable*}",
]
(TAB / "tablea7_applicability_reasons.tex").write_text("\n".join(lines) + "\n")


# Appendix Table A12: the source-family screen behind the completeness claim.
# Read from the frozen D-092 audit; no family is re-screened here.
fam = read_csv(ROOT / "04_reference_sample/q1_completeness_source_family_screen_d092.csv")
assert len(fam) == 31, f"expected 31 source families, found {len(fam)}"
DISPOSITION = {
    "ALREADY_COMPLETE": "Screened earlier; already represented",
    "SCREENED_D092_ADDITIONAL_GOLD": "Newly screened; contributed transitions",
    "SCREENED_D092_NO_EXACT_ADDITION": "Newly screened; no exact public pair",
    "DEDUPLICATED": "Duplicate of another family",
    "EXCLUDED_DOMAIN": r"Outside the H$\beta$ domain",
}
ORDER = ["SCREENED_D092_ADDITIONAL_GOLD", "ALREADY_COMPLETE",
         "SCREENED_D092_NO_EXACT_ADDITION", "DEDUPLICATED", "EXCLUDED_DOMAIN"]
assert set(r["final_audit_state"] for r in fam) <= set(DISPOSITION)
lines = [
    r"\begin{deluxetable}{llc}",
    r"\tabletypesize{\scriptsize}",
    r"\tablecaption{Source families enumerated in the literature screen\label{tab:families}}",
    r"\tablehead{\colhead{Source family} & \colhead{Disposition} & \colhead{Transitions}}",
    r"\startdata",
]
for state in ORDER:
    rows = [r for r in fam if r["final_audit_state"] == state]
    if not rows:
        continue
    lines.append(r"\cutinhead{" + DISPOSITION[state] + f" ({len(rows)})" + "}")
    for r in sorted(rows, key=lambda r: r["primary_publication"]):
        n = r["additional_exact_gold_unique"]
        n = int(n) if str(n).strip() not in ("", "nan") else 0
        lines.append(f"{r['primary_publication']} & {DISPOSITION[state]} & "
                     f"{n if n else chr(92) + 'nodata'} " + r"\\")
lines += [
    r"\enddata",
    r"\tablecomments{All 31 families enumerated before the literature cutoff of 2026 August 15, "
    r"23:59 UTC. ``Transitions'' counts strict high-confidence transitions the family contributed "
    r"to the final reference set at that stage; a dash means none were added, either because the "
    r"family was already represented or because it yielded no exact public endpoint pair. "
    r"Dispositions were assigned before any recovery outcome existed.}",
    r"\end{deluxetable}",
]
(TAB / "tablea12_source_families.tex").write_text("\n".join(lines) + "\n")


print(f"wrote figures to {FIG}")
print(f"wrote tables to {TAB}")
