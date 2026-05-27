#!/usr/bin/env python3
"""
plot_incigraph.py
=================
Produce the InciGraph cardiometabolic validation figure: a two-panel plot,
one line/bar per ethnicity, x-axis = IMD deprivation quintile.

  * Panel A : T2D after hypertension          (CSV from --panel-a)
  * Panel B : CKD after hypertension + T2D    (CSV from --panel-b)

Two modes, selected with --mode:
  * rate  (default) : incidence rate per 100,000 person-years, slope plot
                      with 95% CI bands. The headline validation figure.
  * count           : raw numerators (event counts) per stratum, grouped
                      bar chart. The companion figure showing how many
                      events sit behind each rate -- i.e. which strata are
                      large enough to trust. Best kept as a supplementary
                      exhibit alongside the rate figure.

Each input CSV is the tidy long-format output of collect_incigraph.py:
  sequence_label, target_index, target_name,
  ethnicity, imd, numerator, denominator,
  incidence_rate, lower_limit, upper_limit

USAGE
-----
  # headline rate figure
  python plot_incigraph.py --panel-a panelA_long.csv \
                           --panel-b panelB_long.csv \
                           --age-label "Ages 51-60" \
                           --out cardiometabolic_validation.png

  # companion numerator figure (same CSVs, just a different --mode)
  python plot_incigraph.py --panel-a panelA_long.csv \
                           --panel-b panelB_long.csv \
                           --age-label "Ages 51-60" \
                           --mode count \
                           --out cardiometabolic_numerators.png

  # single-panel works in either mode -- pass only --panel-a:
  python plot_incigraph.py --panel-a panelA_long.csv --out one_panel.png

NOTES
-----
  * X-axis is IMD 1 (least deprived) -> 5 (most deprived).
  * rate mode: y-axis is incidence rate per 100,000 person-years. Each panel
    is capped at 1.25x its OWN highest incidence rate, computed from the
    rates only -- the confidence intervals do not affect the cap. Wide CI
    bands on sparse strata may therefore clip at the top edge; this is
    intended, the rate lines and markers always stay visible. The cap is
    per-panel by definition, so --free-y has no effect in rate mode.
    95% CI is drawn as a shaded band per ethnicity. Strata with numerator
    < --min-count are drawn with an open marker and noted in the caption.
  * count mode: y-axis is the raw event count (numerator); shared scale
    across panels by default, --free-y scales each panel independently.
    Bars below --min-count are hatched and noted in the caption.
  * Mixed and Ethnicity-missing are excluded from the figure by default
    (Mixed strata are typically too sparse for stable intersectional rate
    estimates; missing is an ascertainment artefact, not a real demographic
    group). Pass --include-all-ethnicities to plot every group present.
  * Output format is taken from the --out extension (.png, .pdf, .svg).
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed, color-blind-safe palette per ethnicity, plus a stable z-order.
# Names are upper-cased to match collect_incigraph.py output.
ETHNICITY_STYLE = {
    "WHITE":       dict(color="#185FA5", marker="o"),
    "SOUTH_ASIAN": dict(color="#C04828", marker="s"),
    "BLACK":       dict(color="#0F6E56", marker="^"),
    "MIXED_RACE":  dict(color="#7F77DD", marker="D"),
    "OTHERS":      dict(color="#BA7517", marker="v"),
    "MISSING":     dict(color="#888780", marker="P"),
}
# Order ethnicities are drawn / listed in the legend.
ETHNICITY_ORDER = ["WHITE", "SOUTH_ASIAN", "BLACK", "MIXED_RACE",
                   "OTHERS", "MISSING"]

# Ethnicities excluded from the figure by default. MISSING is not a real
# demographic group (it is an ascertainment artefact, not interpretable as a
# population), and MIXED_RACE strata are typically too sparse for stable
# intersectional rate estimates. Both can be restored with
# --include-all-ethnicities.
EXCLUDED_ETHNICITIES = {"MIXED_RACE", "MISSING"}

PRETTY_ETHNICITY = {
    "WHITE": "White",
    "SOUTH_ASIAN": "South Asian",
    "BLACK": "Black",
    "MIXED_RACE": "Mixed",
    "OTHERS": "Other",
    "MISSING": "Ethnicity missing",
}

IMD_TICKS = [1, 2, 3, 4, 5]
IMD_TICKLABELS = ["1\n(least\ndeprived)", "2", "3", "4", "5\n(most\ndeprived)"]


def load_panel(csv_path):
    """Load one tidy CSV, validate the columns, return (df, title_string)."""
    df = pd.read_csv(csv_path)
    required = {"ethnicity", "imd", "incidence_rate",
                "lower_limit", "upper_limit", "numerator"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path}: missing required columns {sorted(missing)}. "
            "Was this produced by collect_incigraph.py?"
        )
    df["ethnicity"] = df["ethnicity"].astype(str).str.strip().str.upper()
    df["imd"] = pd.to_numeric(df["imd"], errors="coerce")
    for c in ("incidence_rate", "lower_limit", "upper_limit", "numerator"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["imd"].notna()].copy()
    df["imd"] = df["imd"].astype(int)

    # title: prefer the sequence_label column if present
    if "sequence_label" in df.columns and df["sequence_label"].notna().any():
        title = str(df["sequence_label"].dropna().iloc[0])
    else:
        title = csv_path
    return df, title


def _ethnicity_draw_list(df, exclude=frozenset()):
    """Ethnicities present, in the canonical order, minus any excluded ones,
    plus any unstyled extras."""
    present = [e for e in ETHNICITY_ORDER
               if e in set(df["ethnicity"]) and e not in exclude]
    extras = sorted(set(df["ethnicity"]) - set(ETHNICITY_ORDER) - exclude)
    for e in extras:
        sys.stderr.write(f"[warn] no preset style for ethnicity {e!r}; "
                         "drawing in grey.\n")
    return present + extras


def draw_panel_rate(ax, df, title, min_count, ymax=None, exclude=frozenset()):
    """Rate mode: IR vs IMD, one line per ethnicity, 95% CI bands."""
    for eth in _ethnicity_draw_list(df, exclude):
        style = ETHNICITY_STYLE.get(eth, dict(color="#888780", marker="o"))
        sub = df[df["ethnicity"] == eth].sort_values("imd")
        if sub.empty:
            continue

        x = sub["imd"].to_numpy()
        y = sub["incidence_rate"].to_numpy()
        lo = sub["lower_limit"].to_numpy()
        hi = sub["upper_limit"].to_numpy()
        n = sub["numerator"].fillna(0).to_numpy()

        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.12,
                        linewidth=0)
        ax.plot(x, y, color=style["color"], linewidth=1.6, zorder=3,
                label=PRETTY_ETHNICITY.get(eth, eth.title()))
        stable = n >= min_count
        ax.scatter(x[stable], y[stable], color=style["color"],
                   marker=style["marker"], s=34, zorder=4,
                   edgecolors="white", linewidths=0.6)
        if (~stable).any():
            ax.scatter(x[~stable], y[~stable], facecolors="white",
                       edgecolors=style["color"], marker=style["marker"],
                       s=34, zorder=4, linewidths=1.2)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("IMD deprivation quintile", fontsize=10)
    ax.set_xticks(IMD_TICKS)
    ax.set_xticklabels(IMD_TICKLABELS, fontsize=8)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, ymax) if ymax is not None else ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#000000", alpha=0.06, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_panel_count(ax, df, title, min_count, ymax=None, exclude=frozenset()):
    """Count mode: raw numerators per stratum, grouped bar chart.

    Bars are grouped by IMD quintile; within each group, one bar per
    ethnicity. Bars with numerator < min_count are hatched so sparse strata
    are visible at a glance."""
    draw_list = _ethnicity_draw_list(df, exclude)
    n_eth = len(draw_list)
    if n_eth == 0:
        return

    group_width = 0.8
    bar_width = group_width / n_eth

    for j, eth in enumerate(draw_list):
        style = ETHNICITY_STYLE.get(eth, dict(color="#888780", marker="o"))
        sub = (df[df["ethnicity"] == eth]
               .set_index("imd").reindex(IMD_TICKS))
        counts = sub["numerator"].fillna(0).to_numpy()
        # offset each ethnicity's bar within the IMD group, centred on the tick
        offsets = (np.array(IMD_TICKS, dtype=float)
                   - group_width / 2 + bar_width * (j + 0.5))
        bars = ax.bar(offsets, counts, width=bar_width * 0.92,
                      color=style["color"], linewidth=0,
                      label=PRETTY_ETHNICITY.get(eth, eth.title()))
        # hatch the sparse bars
        for b, c in zip(bars, counts):
            if c < min_count:
                b.set_hatch("////")
                b.set_edgecolor("white")
                b.set_linewidth(0.4)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("IMD deprivation quintile", fontsize=10)
    ax.set_xticks(IMD_TICKS)
    ax.set_xticklabels(IMD_TICKLABELS, fontsize=8)
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(0, ymax) if ymax is not None else ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#000000", alpha=0.06, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    ap = argparse.ArgumentParser(
        description="Plot the InciGraph cardiometabolic validation figure "
                    "(IR vs deprivation, lines = ethnicity).")
    ap.add_argument("--panel-a", required=True,
                    help="Tidy CSV for panel A (e.g. T2D after hypertension).")
    ap.add_argument("--panel-b", default=None,
                    help="Tidy CSV for panel B (e.g. CKD after HTN+T2D). "
                         "Omit for a single-panel figure.")
    ap.add_argument("--out", required=True,
                    help="Output figure path (.png, .pdf or .svg).")
    ap.add_argument("--mode", choices=["rate", "count"], default="rate",
                    help="rate: incidence rate slope plot with 95%% CI "
                         "(default, the headline figure). count: raw "
                         "numerators as grouped bars (the companion figure "
                         "showing event counts behind each rate).")
    ap.add_argument("--age-label", default="Ages 51-60",
                    help="Age band label shown in the figure caption.")
    ap.add_argument("--min-count", type=int, default=10,
                    help="rate mode: numerator below this is drawn with an "
                         "open marker. count mode: bars below this are "
                         "hatched. Either way it is flagged in the caption "
                         "(default: 10).")
    ap.add_argument("--free-y", action="store_true",
                    help="count mode only: let each panel scale its y-axis "
                         "independently (default: shared across panels). In "
                         "rate mode the y-axis is always per-panel, capped at "
                         "1.25x that panel's highest incidence rate, so "
                         "--free-y is redundant and ignored there.")
    ap.add_argument("--include-all-ethnicities", action="store_true",
                    help="Plot every ethnicity present in the data. By "
                         "default Mixed and Ethnicity-missing are excluded "
                         "(Mixed strata are typically too sparse for stable "
                         "rates; missing is not a real demographic group).")
    ap.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = ap.parse_args()

    panels = [load_panel(args.panel_a)]
    if args.panel_b:
        panels.append(load_panel(args.panel_b))

    # ethnicities to drop from the figure (and from the y-axis cap)
    exclude = frozenset() if args.include_all_ethnicities \
        else EXCLUDED_ETHNICITIES
    if exclude:
        sys.stderr.write(
            "[info] excluding from the figure: "
            + ", ".join(sorted(exclude))
            + " (use --include-all-ethnicities to keep them)\n")

    # which draw function and labels the mode uses
    if args.mode == "rate":
        draw_fn = draw_panel_rate
        ylabel = "Incidence rate per 100,000 person-years"
    else:  # count
        draw_fn = draw_panel_count
        ylabel = "Event count (numerator)"

    # ----- per-panel y-axis caps -----
    # rate mode: ALWAYS cap each panel at 1.25x its own highest incidence
    #   rate, independently of the confidence intervals (so wide CI bands
    #   may clip at the top edge -- intended). --free-y has no effect here;
    #   the rate cap is per-panel by definition.
    # count mode: keep the shared-vs-free behaviour (shared upper limit
    #   unless --free-y), since counts have no CI to ignore.
    # In both modes, excluded ethnicities are dropped BEFORE the cap is
    # computed, so a hidden group can never drive the axis.
    def _kept(df):
        return df[~df["ethnicity"].isin(exclude)]

    panel_ymax = []
    if args.mode == "rate":
        for df, _ in panels:
            top_rate = _kept(df)["incidence_rate"].max(skipna=True)
            panel_ymax.append(top_rate * 1.25 if np.isfinite(top_rate)
                              else None)
        if args.free_y:
            sys.stderr.write(
                "[info] rate mode caps each panel at 1.25x its own max "
                "incidence rate; --free-y is redundant here and ignored.\n")
    else:  # count
        if args.free_y:
            panel_ymax = [None] * len(panels)
        else:
            hi = max(_kept(p[0])["numerator"].max(skipna=True)
                     for p in panels)
            shared = hi * 1.08 if np.isfinite(hi) else None
            panel_ymax = [shared] * len(panels)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.8), squeeze=False)
    axes = axes[0]

    panel_letters = ["A", "B", "C", "D"]
    any_sparse = False
    for i, (ax, (df, title)) in enumerate(zip(axes, panels)):
        draw_fn(ax, df, title, args.min_count, ymax=panel_ymax[i],
                exclude=exclude)
        ax.text(-0.06, 1.06, panel_letters[i], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="right")
        # only flag sparsity among ethnicities actually drawn
        if (_kept(df)["numerator"].fillna(0) < args.min_count).any():
            any_sparse = True

    # y-label only on the left panel
    axes[0].set_ylabel(ylabel, fontsize=10)

    # one shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    if args.mode == "rate":
        caption = (f"{args.age_label}. Lines = ethnicity. "
                   "Shaded band = 95% CI. X-axis: IMD 1 (least deprived) to "
                   "5 (most deprived).")
        if any_sparse:
            caption += (f" Open markers: stratum numerator < {args.min_count} "
                        "(wide CI, interpret with caution).")
        suptitle = "InciGraph internal validation: cardiometabolic cascade"
    else:
        caption = (f"{args.age_label}. Bars = ethnicity, grouped by IMD "
                   "quintile. Y-axis: raw event count behind each incidence "
                   "rate. X-axis: IMD 1 (least deprived) to 5 (most deprived).")
        if any_sparse:
            caption += (f" Hatched bars: numerator < {args.min_count} "
                        "(rate estimate unstable, interpret with caution).")
        suptitle = ("InciGraph internal validation: event counts behind the "
                    "cardiometabolic cascade")

    fig.text(0.5, -0.08, caption, ha="center", fontsize=8.5,
             color="#444441", wrap=True)
    fig.suptitle(suptitle, fontsize=12.5, y=1.04)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    sys.stderr.write(f"[info] wrote {args.mode} figure -> {args.out}\n")


if __name__ == "__main__":
    main()
