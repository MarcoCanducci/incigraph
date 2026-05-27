#!/usr/bin/env python3
"""
build_l3.py
===========
Build Figure 2 (the length-three convergence figure) from the InciGraph
parquet deposit.

Two panels:
  A. nine curated length-3 deprivation contrasts, grouped by third-condition
     endpoint (COPD, drug/alcohol, heart failure, depression)
  B. third-condition frequency in the top-60 deprivation-ranked length-3
     candidate set (shows that the convergence pattern in A is a property
     of the ranked candidate pool, not the curated rows)

All inputs come from the parquet via the `incigraph` Python API.

USAGE
-----
  python figures/build_l3.py \\
      --parquet-dir ./incigraph_data \\
      --out-dir ./figures

OUTPUTS
-------
  <out-dir>/l3_figure.pdf
  <out-dir>/l3_figure.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import incigraph as ig
from incigraph.ci import irr_ci


# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COL_COPD = "#185FA5"   # respiratory
COL_DA   = "#C04828"   # substance use
COL_HF   = "#0F6E56"   # cardiovascular
COL_DEP  = "#7F4FA0"   # mental health
COL_OTHER = "#888780"
COL_TEXT = "#1F1F1E"
COL_MUTE = "#666660"

ENDPOINT_META = {
    "COPD":  dict(color=COL_COPD,  label="COPD"),
    "DA":    dict(color=COL_DA,    label="Drug/alcohol misuse"),
    "HF":    dict(color=COL_HF,    label="Heart failure"),
    "DEP":   dict(color=COL_DEP,   label="Depression"),
    "OTHER": dict(color=COL_OTHER, label="Other endpoints"),
}

# --------------------------------------------------------------------------
# Panel A: nine curated length-3 deprivation contrasts.
# Each spec describes the parquet query that recovers the IRR; we re-verify
# every row against the parquet on every build, so the figure stays in sync
# with the deposit.
# --------------------------------------------------------------------------
PANEL_A_SPECS = [
    # --- COPD endpoint ---
    dict(endpoint="COPD",
         label="Hypertension → Asthma → COPD",
         sublabel="IMD 5 vs IMD 1 — within White",
         sequence=[3, 23, 24], stratification="ETHNICITY+IMD",
         comparison={"ethnicity": "WHITE", "imd": 5.0},
         reference={"ethnicity": "WHITE", "imd": 1.0}),
    dict(endpoint="COPD",
         label="Osteoarthritis → Asthma → COPD",
         sublabel="IMD 5 vs IMD 1 — within White",
         sequence=[30, 23, 24], stratification="ETHNICITY+IMD",
         comparison={"ethnicity": "WHITE", "imd": 5.0},
         reference={"ethnicity": "WHITE", "imd": 1.0}),
    dict(endpoint="COPD",
         label="Depression → Anxiety → COPD",
         sublabel="IMD 5 vs IMD 1 — within Female",
         sequence=[10, 11, 24], stratification="IMD+SEX",
         comparison={"sex": "F", "imd": 5.0},
         reference={"sex": "F", "imd": 1.0}),

    # --- Drug/alcohol endpoint ---
    dict(endpoint="DA",
         label="Hypertension → Type 2 diabetes → Drug/alcohol misuse",
         sublabel="IMD 5 vs IMD 1 — unconditional",
         sequence=[3, 8, 60], stratification="IMD",
         comparison={"imd": 5.0}, reference={"imd": 1.0}),
    dict(endpoint="DA",
         label="Osteoarthritis → Hypertension → Drug/alcohol misuse",
         sublabel="IMD 4+5 vs IMD 1+2 — within White",
         sequence=[30, 3, 60], stratification="ETHNICITY+IMD",
         pooled_imd=True, conditioning={"ethnicity": "WHITE"}),

    # --- Heart failure endpoint ---
    dict(endpoint="HF",
         label="Valvular disease → AF → Heart failure",
         sublabel="IMD 4+5 vs IMD 1+2 — unconditional",
         sequence=[5, 2, 1], stratification="IMD",
         pooled_imd=True, conditioning={}),
    dict(endpoint="HF",
         label="Osteoarthritis → AF → Heart failure",
         sublabel="IMD 4+5 vs IMD 1+2 — within White",
         sequence=[30, 2, 1], stratification="ETHNICITY+IMD",
         pooled_imd=True, conditioning={"ethnicity": "WHITE"}),

    # --- Depression endpoint ---
    dict(endpoint="DEP",
         label="Cancer → Anxiety → Depression",
         sublabel="IMD 5 vs IMD 1 — within Male",
         sequence=[22, 11, 10], stratification="IMD+SEX",
         comparison={"sex": "M", "imd": 5.0},
         reference={"sex": "M", "imd": 1.0}),
    dict(endpoint="DEP",
         label="Epilepsy → Anxiety → Depression",
         sublabel="IMD 5 vs IMD 1 — unconditional",
         sequence=[21, 11, 10], stratification="IMD",
         comparison={"imd": 5.0}, reference={"imd": 1.0}),
]


def compute_pooled_imd(spec) -> dict:
    """IRR for IMD 4+5 vs IMD 1+2 (sum across two quintiles per side,
    then apply irr_ci on the summed counts). Optional conditioning is
    applied first."""
    df = ig.get_sequence(spec["sequence"], stratification=spec["stratification"])
    df = df[~df["imd_missing"]]
    for col, val in spec.get("conditioning", {}).items():
        df = df[df[col].astype(str).str.upper() == str(val).upper()]
    c = df[df["imd"].isin([4.0, 5.0])][["numerator", "denominator"]].sum()
    r = df[df["imd"].isin([1.0, 2.0])][["numerator", "denominator"]].sum()
    return irr_ci(c["numerator"], c["denominator"],
                  r["numerator"], r["denominator"])


def compute_panel_a() -> pd.DataFrame:
    """Re-verify each curated Panel A row against the parquet.

    Raises a clean FileNotFoundError if the L3 parquet is missing, since
    every Panel A row is length 3.
    """
    rows = []
    try:
        for spec in PANEL_A_SPECS:
            if spec.get("pooled_imd"):
                r = compute_pooled_imd(spec)
            else:
                df = ig.get_sequence(spec["sequence"],
                                     stratification=spec["stratification"])
                r = ig.compute_irr(df, comparison=spec["comparison"],
                                   reference=spec["reference"])
            rows.append(dict(
                endpoint=spec["endpoint"],
                label=spec["label"], sublabel=spec["sublabel"],
                irr=r["irr"], lo=r["lower_ci"], hi=r["upper_ci"],
                n_comp=int(r["n_comp"]), n_ref=int(r["n_ref"]),
            ))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Cannot build the L3 figure — required parquet file missing. "
            f"({e}) "
            "This figure requires length-3 estimates; ensure "
            "incigraph_L3.parquet is present in your --parquet-dir."
        ) from e
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Panel B: re-run the length-3 deprivation candidate ranking inline and
# count third-condition frequencies in the top-60.
# --------------------------------------------------------------------------
MIN_EVENTS = 30


def compute_panel_b() -> pd.Series:
    """Identify the top-60 length-3 deprivation candidates and tabulate
    their third-condition (endpoint) frequencies."""
    try:
        L3 = ig.load_estimates(3)
    except FileNotFoundError as e:
        # No L3 data -> empty bar chart with a note. Acceptable degradation.
        return pd.Series({k: 0 for k in
                          ("DA", "COPD", "DEP", "HF", "OTHER")})

    candidates = []
    imd_strats = [s for s in ig.available_stratifications() if "IMD" in s]
    other_axis = {"AGE_CATG": "age_catg", "ETHNICITY": "ethnicity",
                  "SEX": "sex"}
    for strat in imd_strats:
        block = L3[L3["stratification_key"].astype(str) == strat]
        if block.empty:
            continue
        other_cols = [other_axis[a] for a in strat.split("+") if a != "IMD"]
        block = block[~block["imd_missing"]]
        for c in other_cols:
            block = block[block[c].notna()]
        gcols = ["sequence", "target_disease_idx",
                 "target_disease_short"] + other_cols
        for keys, sub in block.groupby(gcols, observed=True):
            n_c = sub[sub["imd"] == 5.0][["numerator", "denominator"]].sum()
            n_r = sub[sub["imd"] == 1.0][["numerator", "denominator"]].sum()
            if min(n_c["numerator"], n_r["numerator"]) < MIN_EVENTS:
                continue
            r = irr_ci(n_c["numerator"], n_c["denominator"],
                       n_r["numerator"], n_r["denominator"])
            if not np.isfinite(r["irr"]):
                continue
            rank_score = abs(r["log_irr"]) - 1.96 * r["se_log_irr"]
            candidates.append(dict(
                third_name=keys[2], rank_score=rank_score,
            ))
    if not candidates:
        return pd.Series({k: 0 for k in
                          ("DA", "COPD", "DEP", "HF", "OTHER")})
    cand = pd.DataFrame(candidates).sort_values("rank_score", ascending=False)
    top60 = cand.head(60)

    def categorise(name):
        return ({"COPD": "COPD", "DRUG_ALCOHOL": "DA",
                 "HF": "HF", "DEPRESSION": "DEP"}.get(str(name), "OTHER"))

    top60["endpoint"] = top60["third_name"].apply(categorise)
    return top60["endpoint"].value_counts().reindex(
        ["DA", "COPD", "DEP", "HF", "OTHER"], fill_value=0)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(panel_A: pd.DataFrame, panel_B_counts: pd.Series,
           out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": "#999992", "axes.linewidth": 0.6})

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.4, 1.0], wspace=0.35,
                           left=0.11, right=0.97, top=0.86, bottom=0.18)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # === Panel A ===
    n = len(panel_A)
    axA.set_xscale("log"); axA.set_xlim(0.9, 8); axA.set_ylim(-0.6, n - 0.4)

    groups = {}
    for i, row in panel_A.iterrows():
        groups.setdefault(row["endpoint"], []).append(i)
    for ep, idxs in groups.items():
        axA.axhspan(min(idxs) - 0.5, max(idxs) + 0.5,
                    color=ENDPOINT_META[ep]["color"], alpha=0.07, zorder=0)

    axA.axvline(1.0, color="#999992", lw=1.0, ls="--", zorder=1)
    axA.text(1.0, -0.5, "IRR = 1", fontsize=7.6, color=COL_MUTE,
             ha="center", va="top")

    for i, row in panel_A.iterrows():
        y = n - 1 - i
        c = ENDPOINT_META[row["endpoint"]]["color"]
        axA.plot([row["lo"], row["hi"]], [y, y], color=c, lw=2.2,
                 solid_capstyle="round", zorder=3)
        axA.plot([row["irr"]], [y], "o", color=c, ms=8.5, mec="white",
                 mew=1.2, zorder=4)
        axA.text(1.05, y + 0.32, row["label"], fontsize=9.2,
                 fontweight="bold", ha="left", va="center")
        axA.text(1.05, y + 0.08, row["sublabel"], fontsize=7.7,
                 style="italic", color=COL_MUTE, ha="left", va="center")
        axA.text(7.7, y + 0.20,
                 f"IRR = {row['irr']:.2f}  [{row['lo']:.2f}–{row['hi']:.2f}]",
                 fontsize=8.7, fontweight="bold", ha="right", va="center")
        axA.text(7.7, y - 0.22,
                 f"N = {row['n_comp']:,} vs {row['n_ref']:,} events",
                 fontsize=7.6, color=COL_MUTE, ha="right", va="center")

    # endpoint brackets outside the left edge (axes-coords)
    for ep, idxs in groups.items():
        y_lo, y_hi = axA.get_ylim()
        ymin = (n - 1 - max(idxs) - 0.4 - y_lo) / (y_hi - y_lo)
        ymax = (n - 1 - min(idxs) + 0.4 - y_lo) / (y_hi - y_lo)
        c = ENDPOINT_META[ep]["color"]
        axA.plot([-0.025, -0.025], [ymin, ymax], color=c, lw=4,
                 solid_capstyle="butt", clip_on=False,
                 transform=axA.transAxes)
        axA.text(-0.037, (ymin + ymax) / 2,
                 f"→ {ENDPOINT_META[ep]['label']}",
                 fontsize=9, fontweight="bold", color=c,
                 ha="right", va="center", rotation=90, clip_on=False,
                 transform=axA.transAxes)

    axA.set_xticks([1, 2, 3, 4, 5, 6, 8])
    axA.set_xticklabels(["1", "2", "3", "4", "5", "6", "8"], fontsize=8)
    axA.set_xlabel(
        "Crude incidence rate ratio (log scale)\n"
        "most-deprived vs least-deprived, contrast pair given per row",
        fontsize=9.5, labelpad=4)
    axA.set_yticks([])
    for sp in ("left", "top", "right"):
        axA.spines[sp].set_visible(False)
    axA.tick_params(axis="y", length=0)
    axA.grid(axis="x", alpha=0.25, lw=0.5, zorder=0)
    axA.text(0.0, 1.085, "A", transform=axA.transAxes,
             fontsize=18, fontweight="bold", color=COL_TEXT,
             va="top", ha="left")
    axA.text(0.04, 1.085,
             "Selected length-three deprivation contrasts, grouped by endpoint",
             transform=axA.transAxes, fontsize=12.5, fontweight="bold",
             color=COL_TEXT, va="top", ha="left")
    axA.text(0.04, 1.030,
             "Different early diagnoses converge on a small set of "
             "deprivation-patterned third conditions.",
             transform=axA.transAxes, fontsize=8.2, style="italic",
             color=COL_MUTE, va="top", ha="left")

    # === Panel B ===
    bar_order = ["DA", "COPD", "DEP", "HF", "OTHER"]
    total = int(panel_B_counts.sum())
    y_pos = np.arange(len(bar_order))[::-1]
    max_pct = max(int(panel_B_counts.get(k, 0)) / total * 100
                  for k in bar_order) if total else 0
    for y, k in zip(y_pos, bar_order):
        c = int(panel_B_counts.get(k, 0))
        pct = c / total * 100 if total else 0
        axB.barh(y, pct, color=ENDPOINT_META[k]["color"], height=0.7,
                 edgecolor="white", linewidth=0.8)
        axB.text(-2.5, y, ENDPOINT_META[k]["label"], fontsize=9,
                 color=COL_TEXT, ha="right", va="center")
        if total:
            axB.text(pct + 1.5, y, f"{c}/{total} ({pct:.0f}%)",
                     fontsize=8.4, color=COL_TEXT, va="center")
    axB.set_xlim(0, (max_pct or 50) * 1.35)
    axB.set_ylim(-0.6, len(bar_order) - 0.2)
    axB.set_yticks([])
    axB.set_xlabel("% of top-60 length-3 deprivation candidates",
                   fontsize=9.5, labelpad=4)
    for sp in ("left", "top", "right"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(axis="y", length=0)
    axB.grid(axis="x", alpha=0.25, lw=0.5, zorder=0)
    axB.text(0.0, 1.085, "B", transform=axB.transAxes,
             fontsize=18, fontweight="bold", color=COL_TEXT,
             va="top", ha="left")
    axB.text(0.07, 1.085,
             "Third-condition concentration\nin top-60 candidates",
             transform=axB.transAxes, fontsize=12.5, fontweight="bold",
             color=COL_TEXT, va="top", ha="left", linespacing=1.15)
    if total:
        in_4 = sum(int(panel_B_counts.get(k, 0))
                   for k in ("DA", "COPD", "DEP", "HF"))
        axB.text(0.0, 0.98,
                 f"The four endpoint groups in Panel A account for "
                 f"{in_4 / total * 100:.0f}% of the ranked candidate set.",
                 transform=axB.transAxes, fontsize=8.2, style="italic",
                 color=COL_MUTE, va="top", ha="left")

    # === figure title ===
    fig.suptitle(
        "Length-three deprivation contrasts converge on respiratory, "
        "substance-use, cardiovascular and mental-health endpoints",
        fontsize=13.0, fontweight="bold", color=COL_TEXT, y=0.965)

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.parquet_dir is not None:
        ig.set_data_dir(args.parquet_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.stderr.write("[info] computing Panel A (nine curated L3 contrasts)...\n")
    A = compute_panel_a()
    sys.stderr.write(f"       {len(A)} rows; "
                     f"IRR range {A['irr'].min():.2f}–{A['irr'].max():.2f}\n")

    sys.stderr.write("[info] computing Panel B (top-60 third-condition concentration)...\n")
    B = compute_panel_b()
    total = int(B.sum())
    sys.stderr.write(f"       top-60 = {total} candidates; "
                     f"endpoint counts: {dict(B)}\n")

    sys.stderr.write("[info] rendering...\n")
    render(A, B, args.out_dir / "l3_figure")
    sys.stderr.write(
        f"[info] wrote {args.out_dir / 'l3_figure.png'} "
        f"and {args.out_dir / 'l3_figure.pdf'}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
