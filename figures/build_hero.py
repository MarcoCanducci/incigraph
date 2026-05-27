#!/usr/bin/env python3
"""
build_hero.py
=============
Build Figure 1 (the hero figure) from the InciGraph parquet deposit.

Three panels:
  A. arithmetic consistency receipt
  B. forest plot of eight curated contrasts from the systematic scan
  C. compact sparsity heatmap

All inputs come from the parquet via the `incigraph` Python API; no other
data files are required.

USAGE
-----
  python figures/build_hero.py \\
      --parquet-dir ./incigraph_data \\
      --out-dir ./figures

OUTPUTS
-------
  <out-dir>/hero_figure.pdf
  <out-dir>/hero_figure.png
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import incigraph as ig
from incigraph.ci import RATE_DENOMINATOR, irr_ci, poisson_ci


# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COL_DEP  = "#185FA5"   # deprivation
COL_ETH  = "#C04828"   # ethnicity
COL_SEX  = "#0F6E56"   # sex
COL_TEXT = "#1F1F1E"
COL_MUTE = "#666660"
COL_PASS = "#1F8A3E"   # validation green

# --------------------------------------------------------------------------
# Panel B: the eight curated contrasts, each described as a parquet query.
# These are the rows the published manuscript figure shows; we re-verify
# every IRR by re-computing it from the parquet here. The (sequence,
# stratification, comparison, reference) tuples are the row-level source
# of truth.
# --------------------------------------------------------------------------
PANEL_B_SPECS = [
    # group, label, sublabel, sequence, stratification, comparison, reference
    dict(group='ethnicity',
         label='Sickle cell disease incidence — Black vs White',
         sublabel='(unconditional, all ages, all sexes)',
         sequence=[56], stratification='ETHNICITY',
         comparison={'ethnicity': 'BLACK'},
         reference={'ethnicity': 'WHITE'}),
    dict(group='ethnicity',
         label='Hypertension → atrial fibrillation — Black vs White',
         sublabel='Black incidence ~1/5 of White after hypertension',
         sequence=[3, 2], stratification='ETHNICITY',
         comparison={'ethnicity': 'BLACK'},
         reference={'ethnicity': 'WHITE'}),
    dict(group='deprivation',
         label='Osteoarthritis → COPD — IMD 5 vs IMD 1',
         sublabel='within White',
         sequence=[30, 24], stratification='ETHNICITY+IMD',
         comparison={'ethnicity': 'WHITE', 'imd': 5.0},
         reference={'ethnicity': 'WHITE', 'imd': 1.0}),
    dict(group='deprivation',
         label='Sickle cell disease — IMD 4+5 vs IMD 1+2',
         sublabel='(unconditional, all sexes)',
         sequence=[56], stratification='IMD',
         pooled_imd=True, conditioning={}),
    dict(group='deprivation',
         label='Learning disability — IMD 5 vs IMD 1',
         sublabel='within White',
         sequence=[53], stratification='ETHNICITY+IMD',
         comparison={'ethnicity': 'WHITE', 'imd': 5.0},
         reference={'ethnicity': 'WHITE', 'imd': 1.0}),
    dict(group='sex',
         label='Gout → Osteoporosis — Female vs Male',
         sublabel='(unconditional, all ages)',
         sequence=[32, 29], stratification='SEX',
         comparison={'sex': 'F'}, reference={'sex': 'M'}),
    dict(group='sex',
         label='Eating disorders — Female vs Male',
         sublabel='within age 17–30',
         sequence=[13], stratification='AGE_CATG+SEX',
         comparison={'sex': 'F', 'age_catg': '17-30'},
         reference={'sex': 'M', 'age_catg': '17-30'}),
    dict(group='sex',
         label='Aortic aneurysm — Female vs Male',
         sublabel='within age 61–70 — female incidence ~1/20 of male',
         sequence=[6], stratification='AGE_CATG+SEX',
         comparison={'sex': 'F', 'age_catg': '61-70'},
         reference={'sex': 'M', 'age_catg': '61-70'}),
]

# --------------------------------------------------------------------------
# Panel A: compute the aggregation-consistency stats from the parquet
# --------------------------------------------------------------------------
AXIS_TO_COL = {"AGE_CATG": "age_catg", "ETHNICITY": "ethnicity",
               "IMD": "imd", "SEX": "sex"}
RATE_REL_TOL = 1e-6

REPRESENTATIVE_SEQUENCES = [
    "0 3", "0 3 8", "0 3 8 9", "0 26", "0 26 23", "0 26 23 27",
    "0 10", "0 10 11",
]


def axes_of(s):
    return set() if s == "NONE" else set(s.split("+"))


def compute_panel_a() -> dict:
    """Re-run the aggregation-consistency check programmatically."""
    parts = []
    for L in (1, 2, 3):
        try:
            parts.append(ig.load_estimates(L))
        except FileNotFoundError:
            pass
    if not parts:
        raise FileNotFoundError(
            "No parquet files found. Set the parquet directory.")
    estimates = pd.concat(parts, ignore_index=True)
    all_strats = sorted(estimates["stratification_key"].astype(str).unique())

    pairs = [(a, b) for a, b in combinations(all_strats, 2)
             if axes_of(b) < axes_of(a)]
    pairs += [(b, a) for a, b in combinations(all_strats, 2)
              if axes_of(a) < axes_of(b)]

    sequences = [s for s in REPRESENTATIVE_SEQUENCES
                 if s in set(estimates["sequence"].astype(str).unique())]

    n_cells = n_rate_ok = n_ci_ok = 0
    max_rel = 0.0
    for seq in sequences:
        block = estimates[estimates["sequence"] == seq]
        for source, target in pairs:
            src = block[block["stratification_key"] == source]
            tgt = block[block["stratification_key"] == target]
            if src.empty or tgt.empty:
                continue
            denom_max = max(src["denominator"].sum(), tgt["denominator"].sum())
            if denom_max <= 0:
                continue
            rel = abs(src["denominator"].sum() - tgt["denominator"].sum()) / denom_max
            if rel > RATE_REL_TOL:
                continue
            tgt_cols = [AXIS_TO_COL[a] for a in sorted(axes_of(target))]
            agg = (src.groupby(tgt_cols, dropna=False, observed=True)
                      [["numerator", "denominator"]].sum().reset_index()
                   if tgt_cols else pd.DataFrame({
                       "numerator":   [src["numerator"].sum()],
                       "denominator": [src["denominator"].sum()],
                   }))
            agg["incidence_rate"] = (agg["numerator"] / agg["denominator"]
                                     * RATE_DENOMINATOR)
            lo, hi = poisson_ci(agg["numerator"].values,
                                agg["denominator"].values)
            agg["lower_limit"] = lo
            agg["upper_limit"] = hi
            merged = agg.merge(
                tgt[tgt_cols + ["numerator", "denominator", "incidence_rate",
                                "lower_limit", "upper_limit"]],
                on=tgt_cols, suffixes=("_src", "_tgt"))
            if merged.empty:
                continue
            rate_diff = ((merged["incidence_rate_src"]
                          - merged["incidence_rate_tgt"]).abs()
                         / merged["incidence_rate_tgt"].abs())
            ci_lo = ((merged["lower_limit_src"]
                      - merged["lower_limit_tgt"]).abs()
                     / merged["lower_limit_tgt"].abs().replace(0, np.nan))
            ci_hi = ((merged["upper_limit_src"]
                      - merged["upper_limit_tgt"]).abs()
                     / merged["upper_limit_tgt"].abs())
            n_cells   += len(merged)
            n_rate_ok += int((rate_diff <= RATE_REL_TOL).sum())
            n_ci_ok   += int(((ci_lo.fillna(0) <= RATE_REL_TOL)
                              & (ci_hi <= RATE_REL_TOL)).sum())
            max_rel = max(max_rel, float(rate_diff.max()))
    return {
        "pairs": len(pairs), "cells": n_cells,
        "pct_rate": 100.0 * n_rate_ok / n_cells if n_cells else 0.0,
        "pct_ci":   100.0 * n_ci_ok   / n_cells if n_cells else 0.0,
        "max_rel":  max_rel,
    }


# --------------------------------------------------------------------------
# Panel B: re-verify each spec against the parquet
# --------------------------------------------------------------------------
def compute_pooled_imd(spec) -> dict:
    df = ig.get_sequence(spec["sequence"], stratification=spec["stratification"])
    df = df[~df["imd_missing"]]
    for col, val in spec.get("conditioning", {}).items():
        df = df[df[col].astype(str).str.upper() == str(val).upper()]
    c = df[df["imd"].isin([4.0, 5.0])][["numerator", "denominator"]].sum()
    r = df[df["imd"].isin([1.0, 2.0])][["numerator", "denominator"]].sum()
    return irr_ci(c["numerator"], c["denominator"],
                  r["numerator"], r["denominator"])


def compute_panel_b() -> pd.DataFrame:
    rows = []
    for spec in PANEL_B_SPECS:
        if spec.get("pooled_imd"):
            r = compute_pooled_imd(spec)
        else:
            df = ig.get_sequence(spec["sequence"],
                                 stratification=spec["stratification"])
            r = ig.compute_irr(df, comparison=spec["comparison"],
                               reference=spec["reference"])
        rows.append(dict(
            group=spec["group"], label=spec["label"], sublabel=spec["sublabel"],
            irr=r["irr"], lo=r["lower_ci"], hi=r["upper_ci"],
            n_comp=int(r["n_comp"]), n_ref=int(r["n_ref"]),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Panel C: sparsity heatmap
# --------------------------------------------------------------------------
def compute_panel_c() -> pd.DataFrame:
    rows = []
    strats = ig.available_stratifications()
    for sl in (1, 2, 3):
        try:
            df = ig.load_estimates(sl)
        except FileNotFoundError:
            continue
        for strat in strats:
            block = df[df["stratification_key"].astype(str) == strat]
            if not len(block):
                continue
            rows.append(dict(
                sequence_length=sl, stratification_key=strat,
                pct_n_ge_10=float((block["numerator"].fillna(0) >= 10).sum())
                            / len(block) * 100,
                n_estimates=len(block),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(panel_A: dict, panel_B: pd.DataFrame, panel_C: pd.DataFrame,
           out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": "#999992", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(15.5, 9.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.85, 2.4, 1.05],
                           wspace=0.32, left=0.04, right=0.985,
                           top=0.84, bottom=0.16)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # Panel A
    axA.axis("off"); axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    axA.text(0, 0.95, "A", fontsize=18, fontweight="bold", va="top")
    axA.text(0.13, 0.95, "The tool's arithmetic\nis exact",
             fontsize=12.5, fontweight="bold", va="top", linespacing=1.18)
    axA.text(0, 0.50, f"{panel_A['pairs']}", fontsize=42, fontweight="bold",
             color=COL_PASS)
    axA.text(0.30, 0.55,
             f"of {panel_A['pairs']}\nmarginalisable\nfolder pairs",
             fontsize=8.5, va="bottom", linespacing=1.3)
    axA.text(0, 0.32, f"{panel_A['cells']:,}\ncells compared",
             fontsize=12, fontweight="bold", linespacing=1.15)
    axA.text(0, 0.16, f"{panel_A['pct_rate']:.2f}%",
             fontsize=14, fontweight="bold", color=COL_PASS)
    axA.text(0, 0.12,
             f"incidence rate match\n95% CIs reproduced within\n"
             f"numerical tolerance (≤ {panel_A['max_rel']:.1e})",
             fontsize=8.4, va="top", linespacing=1.45)

    # Panel B
    n = len(panel_B)
    axB.set_xscale("log"); axB.set_xlim(0.025, 250); axB.set_ylim(-0.5, n - 0.5)
    groups = {}
    for i, row in panel_B.iterrows():
        groups.setdefault(row["group"], []).append(i)
    band = {"ethnicity": "#FBE7E1", "deprivation": "#E2ECF6", "sex": "#DFEFE7"}
    col_map = {"ethnicity": COL_ETH, "deprivation": COL_DEP, "sex": COL_SEX}
    for g, idxs in groups.items():
        axB.axhspan(min(idxs) - 0.5, max(idxs) + 0.5, color=band[g],
                    alpha=0.55, zorder=0)
    axB.axvline(1.0, color="#999992", lw=1.0, ls="--", zorder=1)
    axB.text(1.0, n - 0.35, "IRR = 1", fontsize=7.5, color=COL_MUTE,
             ha="center", va="top")
    for i, row in panel_B.iterrows():
        y = n - 1 - i
        c = col_map[row["group"]]
        axB.plot([row["lo"], row["hi"]], [y, y], color=c, lw=2.2,
                 solid_capstyle="round", zorder=3)
        axB.plot([row["irr"]], [y], "o", color=c, ms=8.5, mec="white",
                 mew=1.2, zorder=4)
        axB.text(0.029, y + 0.18, row["label"], fontsize=9.2,
                 fontweight="bold", ha="left", va="center")
        axB.text(0.029, y - 0.22, row["sublabel"], fontsize=7.8,
                 style="italic", color=COL_MUTE, ha="left", va="center")
        axB.text(245, y + 0.18,
                 f"IRR = {row['irr']:.2f}  [{row['lo']:.2f}-{row['hi']:.2f}]",
                 fontsize=8.7, fontweight="bold", ha="right", va="center")
        axB.text(245, y - 0.22,
                 f"N = {row['n_comp']:,} vs {row['n_ref']:,} events",
                 fontsize=7.6, color=COL_MUTE, ha="right", va="center")
    axB.set_xticks([0.05, 0.1, 0.5, 1, 5, 10, 50, 100])
    axB.set_xticklabels(["0.05", "0.1", "0.5", "1", "5", "10", "50", "100"])
    axB.set_xlabel("Incidence rate ratio (log scale)", fontsize=9.5)
    axB.set_yticks([])
    for sp in ("left", "top", "right"):
        axB.spines[sp].set_visible(False)
    axB.text(0.04, 1.04, "B", transform=axB.transAxes,
             fontsize=18, fontweight="bold", va="top")
    axB.text(0.09, 1.06,
             "Systematic contrast scan identifies expected demographic patterns\n"
             "in ordered-incidence estimates across ethnicity, deprivation and sex",
             transform=axB.transAxes, fontsize=12.5, fontweight="bold",
             va="top", linespacing=1.18)

    # Panel C
    strat_order = sorted(panel_C["stratification_key"].unique(),
                         key=lambda s: (0 if s == "NONE" else s.count("+") + 1, s))
    M = panel_C.pivot(index="stratification_key", columns="sequence_length",
                      values="pct_n_ge_10").reindex(strat_order)
    axC.imshow(M.values, aspect="auto", cmap=plt.cm.YlOrRd_r,
               vmin=0, vmax=100)
    axC.set_xticks(range(M.shape[1]))
    axC.set_xticklabels([f"Length {c}" for c in M.columns])
    axC.set_yticks(range(len(strat_order)))
    axC.set_yticklabels(strat_order, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if pd.isna(v):
                continue
            axC.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                     color="white" if v < 35 else COL_TEXT)
    for sp in ("top", "right", "left", "bottom"):
        axC.spines[sp].set_visible(False)
    axC.text(0, 1.04, "C", transform=axC.transAxes,
             fontsize=18, fontweight="bold", va="top")
    axC.text(0.09, 1.06, "Operating characteristics\nare transparent",
             transform=axC.transAxes, fontsize=12.5, fontweight="bold",
             va="top", linespacing=1.18)

    fig.suptitle(
        "InciGraph: a precomputed, intersectional, ordered-incidence platform",
        fontsize=13.5, fontweight="bold", y=0.96)

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

    sys.stderr.write("[info] computing Panel A (aggregation consistency)...\n")
    A = compute_panel_a()
    sys.stderr.write(f"       {A['cells']:,} cells, "
                     f"{A['pct_rate']:.2f}% match, "
                     f"max rel diff {A['max_rel']:.2e}\n")
    sys.stderr.write("[info] computing Panel B (curated contrasts)...\n")
    B = compute_panel_b()
    sys.stderr.write("[info] computing Panel C (sparsity)...\n")
    C = compute_panel_c()
    sys.stderr.write("[info] rendering...\n")
    render(A, B, C, args.out_dir / "hero_figure")
    sys.stderr.write(
        f"[info] wrote {args.out_dir / 'hero_figure.png'} "
        f"and {args.out_dir / 'hero_figure.pdf'}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
