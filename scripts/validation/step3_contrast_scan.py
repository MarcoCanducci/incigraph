#!/usr/bin/env python3
"""
step3_contrast_scan.py
======================
Systematic scan of crude incidence-rate-ratio contrasts across deprivation,
ethnicity, sex and age for every sufficiently populated ordered disease
sequence in the InciGraph parquet deposit.

This is Step 3 of the validation pipeline. It produces the data behind the
manuscript's "top contrasts" tables (Panel B of the hero figure, the full
length-3 figure, and the supplement's full scan tables).

The scan is HYPOTHESIS-GENERATING. Every IRR is crude (no confounding
adjustment, no competing-risks handling). The BH-FDR step is a SCREENING
SAFEGUARD against chance signals; survival of BH is not evidence of effect.
Rank by the stability-weighted score (lower bound of |log IRR|), not p-value.

INPUTS (from the parquet)
-------------------------
For each (sequence, stratification, target_disease), we pull every demographic
stratum's numerator and denominator. We then form pairwise contrasts as
described below.

CONTRASTS COMPUTED
------------------
For each sequence (length 1, 2, 3) and each stratification where the relevant
demographic axis is present:

* deprivation:   IMD 5 vs IMD 1; IMD 4+5 vs IMD 1+2
                  - unconditional (in IMD-only strata)
                  - within ethnicity (in ETHNICITY+IMD strata)
                  - within sex (in IMD+SEX strata)
                  - within age (in AGE_CATG+IMD strata)
* ethnicity:     SOUTH_ASIAN/WHITE, BLACK/WHITE, MIXED_RACE/WHITE,
                 OTHERS/WHITE (where each is supported)
                  - unconditional / within IMD / within sex / within age
* sex:           F/M    (SEX='I' is its OWN category, not a pooled total)
                  - unconditional / within ethnicity / within IMD / within age
* age (supplementary): 51-60 vs 31-40, 61-70 vs 41-50, 71-80 vs 51-60
                  - unconditional / within other strata

For every contrast we require BOTH groups to meet the event threshold. The
HEADLINE threshold is N >= 30. A sensitivity run at N >= 10 is also produced.

OUTPUTS
-------
  <out-dir>/demographic_contrast_scan_all.csv
       Every contrast attempted (N>=30 path), with raw and BH-adjusted
       p-values and the stability-weighted rank score.
  <out-dir>/demographic_contrast_scan_top_hits.csv
       Top 20 per PRIMARY contrast type (deprivation, ethnicity, sex),
       BH-FDR-surviving, ranked by score. Age contrasts EXCLUDED from
       this file -- they go in the next one.
  <out-dir>/demographic_contrast_scan_age_contrasts.csv
       The supplementary age-only top-hits.
  <out-dir>/demographic_contrast_scan_sensitivity_n10.csv
       Top hits at the N>=10 sensitivity threshold.
  <out-dir>/demographic_contrast_scan.json
       Provenance sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import incigraph as ig
from incigraph.ci import irr_ci
from incigraph.disease_index import DISEASE_NAMES


# ----------------------------------------------------------------------
# Contrast specifications
# ----------------------------------------------------------------------
# Each spec describes one *kind* of contrast we want to compute. The
# `compare` and `reference` entries pick out rows of the parquet by exact
# match on the relevant column; `axis` says which column varies.

DEPRIVATION_CONTRASTS = [
    {"name": "IMD 5 vs IMD 1",
     "axis": "imd", "compare": [5.0], "reference": [1.0]},
    {"name": "IMD 4+5 vs IMD 1+2",
     "axis": "imd", "compare": [4.0, 5.0], "reference": [1.0, 2.0]},
]
ETHNICITY_CONTRASTS = [
    {"name": "South Asian vs White",
     "axis": "ethnicity", "compare": ["SOUTH_ASIAN"], "reference": ["WHITE"]},
    {"name": "Black vs White",
     "axis": "ethnicity", "compare": ["BLACK"], "reference": ["WHITE"]},
    {"name": "Mixed vs White",
     "axis": "ethnicity", "compare": ["MIXED_RACE"], "reference": ["WHITE"]},
    {"name": "Others vs White",
     "axis": "ethnicity", "compare": ["OTHERS"], "reference": ["WHITE"]},
]
SEX_CONTRASTS = [
    # 'I' is excluded from this contrast type because it is a third sex
    # category and not a meaningful F/M comparator. Reported separately.
    {"name": "Female vs Male",
     "axis": "sex", "compare": ["F"], "reference": ["M"]},
]
AGE_CONTRASTS = [
    {"name": "51-60 vs 31-40",
     "axis": "age_catg", "compare": ["51-60"], "reference": ["31-40"]},
    {"name": "61-70 vs 41-50",
     "axis": "age_catg", "compare": ["61-70"], "reference": ["41-50"]},
    {"name": "71-80 vs 51-60",
     "axis": "age_catg", "compare": ["71-80"], "reference": ["51-60"]},
]

CONTRAST_TYPES = {
    "deprivation": DEPRIVATION_CONTRASTS,
    "ethnicity":   ETHNICITY_CONTRASTS,
    "sex":         SEX_CONTRASTS,
    "age":         AGE_CONTRASTS,
}

AXIS_COL = {
    "deprivation": "imd",
    "ethnicity":   "ethnicity",
    "sex":         "sex",
    "age":         "age_catg",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure. Returns BH-adjusted p-values
    the same shape as the input; NaN is propagated."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    n = len(pv)
    ranks = np.arange(1, n + 1)
    adj = pv[order] * n / ranks
    # enforce monotonicity from the largest rank down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    # put back in original order
    result = np.full(n, np.nan)
    result[order] = adj
    out[valid] = result
    return out


def sum_groups(rows: pd.DataFrame, axis: str, values) -> tuple[float, float]:
    """Sum numerator and denominator across `axis` taking values in `values`.

    Returns (sum_n, sum_y). If no matching rows, returns (0, 0).
    """
    sub = rows[rows[axis].isin(values)]
    if sub.empty:
        return 0.0, 0.0
    return float(sub["numerator"].sum()), float(sub["denominator"].sum())


def stratifications_for(contrast_type: str,
                        all_strats: list[str]) -> list[tuple[str, str]]:
    """Return (stratification_key, conditioning_label) pairs to scan for
    this contrast type.

    For contrast_type = 'deprivation' we look at every stratification that
    contains 'IMD' (the contrast axis). The conditioning axes are the
    OTHER columns in the stratification key. e.g.:
      - 'IMD' alone -> unconditional
      - 'IMD+ETHNICITY' -> conditional on each ethnicity
      - 'AGE_CATG+IMD+SEX' -> conditional on every (age, sex) pair
    """
    contrast_axis_col = {
        "deprivation": "IMD",
        "ethnicity":   "ETHNICITY",
        "sex":         "SEX",
        "age":         "AGE_CATG",
    }[contrast_type]
    out = []
    for sk in all_strats:
        if contrast_axis_col not in sk:
            continue
        # the conditioning axes are the other dimensions in the key
        dims = sk.split("+") if sk != "NONE" else []
        cond_axes = [d for d in dims if d != contrast_axis_col]
        out.append((sk, "+".join(cond_axes) if cond_axes else ""))
    return out


def scan_one_sequence(seq: str,
                      seq_length: int,
                      target_idx: int,
                      target_short: str,
                      data_for_seq: pd.DataFrame,
                      min_events: int,
                      all_strats: list[str]) -> list[dict]:
    """Compute every contrast for one sequence. Returns a list of records."""
    records = []
    for ctype, contrasts in CONTRAST_TYPES.items():
        for sk, cond_axes in stratifications_for(ctype, all_strats):
            block = data_for_seq[data_for_seq["stratification_key"] == sk]
            if block.empty:
                continue
            # Conditioning: iterate over every unique combination of the
            # conditioning axes. Empty cond_axes means a single unconditional
            # contrast.
            cond_cols = []
            if cond_axes:
                for ax in cond_axes.split("+"):
                    col = ax.lower() if ax != "AGE_CATG" else "age_catg"
                    cond_cols.append(col)

            if cond_cols:
                # exclude rows where any conditioning value is missing/unknown
                #   ETHNICITY=MISSING / IMD missing -> not a meaningful
                #   conditioning stratum
                cond_block = block.copy()
                if "imd" in cond_cols and "imd_missing" in cond_block.columns:
                    cond_block = cond_block[~cond_block["imd_missing"]]
                # also drop any conditioning axis equal to nan (defensive)
                for col in cond_cols:
                    cond_block = cond_block[cond_block[col].notna()]
                if cond_block.empty:
                    continue
                groups = cond_block.groupby(cond_cols, observed=True)
                group_iter = list(groups)
            else:
                group_iter = [(("",), block)]

            for keys, group in group_iter:
                cond_label = ""
                if cond_cols:
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    cond_label = "; ".join(
                        f"{ax}={val}" for ax, val in zip(cond_axes.split("+"),
                                                          keys)
                    )

                for spec in contrasts:
                    axis = spec["axis"]
                    if axis not in group.columns:
                        continue
                    n_c, y_c = sum_groups(group, axis, spec["compare"])
                    n_r, y_r = sum_groups(group, axis, spec["reference"])
                    if min(n_c, n_r) < min_events:
                        continue
                    result = irr_ci(n_c, y_c, n_r, y_r)
                    if not np.isfinite(result["irr"]):
                        continue
                    records.append({
                        "sequence": seq,
                        "sequence_length": seq_length,
                        "target_condition": target_short,
                        "stratification_key": sk,
                        "contrast_type": ctype,
                        "contrast_name": spec["name"],
                        "conditioning_strata": cond_label,
                        "comparison_group": "+".join(map(str, spec["compare"])),
                        "reference_group":  "+".join(map(str, spec["reference"])),
                        "comparison_numerator": n_c,
                        "reference_numerator":  n_r,
                        "comparison_denominator": y_c,
                        "reference_denominator":  y_r,
                        "comparison_rate": result["comparison_rate"]
                            if "comparison_rate" in result else
                            (n_c / y_c * 1e5 if y_c > 0 else np.nan),
                        "reference_rate":  result["reference_rate"]
                            if "reference_rate" in result else
                            (n_r / y_r * 1e5 if y_r > 0 else np.nan),
                        "incidence_rate_ratio": result["irr"],
                        "log_irr":    result["log_irr"],
                        "se_log_irr": result["se_log_irr"],
                        "lower_95_ci_irr": result["lower_ci"],
                        "upper_95_ci_irr": result["upper_ci"],
                        "p_raw":      result["p_raw"],
                        "event_threshold_used": min_events,
                    })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--event-threshold", type=int, default=30,
                    help="Headline event threshold. Default 30.")
    ap.add_argument("--sensitivity-threshold", type=int, default=10,
                    help="Secondary threshold for sensitivity output. Default 10.")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Top hits per contrast type in the top_hits file. "
                         "Default 20.")
    ap.add_argument("--bh-alpha", type=float, default=0.05,
                    help="BH-FDR alpha threshold. Default 0.05.")
    args = ap.parse_args()

    if args.parquet_dir is not None:
        ig.set_data_dir(args.parquet_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    sys.stderr.write("[info] loading estimates from parquet...\n")
    parts = []
    for L in (1, 2, 3):
        try:
            parts.append(ig.load_estimates(L))
        except FileNotFoundError:
            sys.stderr.write(f"[warn] L{L} parquet missing, skipping\n")
    if not parts:
        sys.stderr.write("[error] no parquet files found\n")
        return 2
    estimates = pd.concat(parts, ignore_index=True)
    sys.stderr.write(f"[info] total estimates: {len(estimates):,}\n")

    all_strats = sorted(estimates["stratification_key"].astype(str).unique().tolist())
    sys.stderr.write(f"[info] stratifications: {all_strats}\n")

    # ---- scan each sequence at the headline threshold ----
    sys.stderr.write(
        f"[info] scanning at headline threshold N >= {args.event_threshold}...\n"
    )
    all_records = []
    grouped = estimates.groupby(["sequence", "sequence_length",
                                 "target_disease_idx",
                                 "target_disease_short"], observed=True)
    n_seqs = len(grouped)
    for i, ((seq, sl, tidx, tshort), block) in enumerate(grouped, 1):
        recs = scan_one_sequence(
            seq=str(seq), seq_length=int(sl),
            target_idx=int(tidx), target_short=str(tshort),
            data_for_seq=block, min_events=args.event_threshold,
            all_strats=all_strats,
        )
        all_records.extend(recs)
        if i % 200 == 0:
            sys.stderr.write(f"       {i}/{n_seqs} sequences ({len(all_records):,} contrasts so far)\n")

    sys.stderr.write(f"[info] total contrasts at headline: {len(all_records):,}\n")
    if not all_records:
        sys.stderr.write("[warn] no contrasts passed the threshold; nothing to write\n")
        return 0

    df = pd.DataFrame(all_records)

    # ---- BH-FDR within contrast_type ----
    df["p_bh_adjusted"] = np.nan
    df["survives_bh_fdr"] = False
    for ct, sub in df.groupby("contrast_type"):
        adj = bh_adjust(sub["p_raw"].values)
        df.loc[sub.index, "p_bh_adjusted"] = adj
        df.loc[sub.index, "survives_bh_fdr"] = adj < args.bh_alpha

    # ---- stability-weighted ranking score ----
    # Lower bound of |log IRR|'s 95% CI on the log scale -- prefers contrasts
    # that are large AND well-supported.
    df["rank_score"] = (
        np.abs(df["log_irr"]) - 1.959963984540054 * df["se_log_irr"]
    )

    # round for cleaner CSV
    rounders = {
        "incidence_rate_ratio": 4, "log_irr": 4, "se_log_irr": 5,
        "lower_95_ci_irr": 4, "upper_95_ci_irr": 4,
        "p_raw": 8, "p_bh_adjusted": 8,
        "rank_score": 5,
        "comparison_rate": 4, "reference_rate": 4,
        "comparison_denominator": 1, "reference_denominator": 1,
    }
    for c, p in rounders.items():
        if c in df.columns:
            df[c] = df[c].round(p)

    # ---- write the full table ----
    all_path = args.out_dir / "demographic_contrast_scan_all.csv"
    df.to_csv(all_path, index=False)
    sys.stderr.write(f"[info] wrote full scan (csv)     -> {all_path}\n")
    # Also write a parquet copy. The full scan is ~14M rows at the real
    # scale; parquet is ~10x smaller on disk and ~50x faster to read back.
    # The curator and any downstream tooling prefer the parquet form.
    all_path_pq = args.out_dir / "demographic_contrast_scan_all.parquet"
    df.to_parquet(all_path_pq, engine="pyarrow", compression="zstd",
                  index=False)
    sys.stderr.write(f"[info] wrote full scan (parquet) -> {all_path_pq}\n")

    # ---- top hits: primary types only, BH-surviving, top-N by rank score ----
    primary_types = ["deprivation", "ethnicity", "sex"]
    top_primary = (
        df[df["contrast_type"].isin(primary_types) & df["survives_bh_fdr"]]
        .sort_values("rank_score", ascending=False)
        .groupby("contrast_type", group_keys=False, observed=True)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    top_path = args.out_dir / "demographic_contrast_scan_top_hits.csv"
    top_primary.to_csv(top_path, index=False)
    sys.stderr.write(
        f"[info] wrote primary top hits -> {top_path} ({len(top_primary)} rows)\n"
    )

    # ---- age contrasts (supplementary) ----
    age_top = (
        df[(df["contrast_type"] == "age") & df["survives_bh_fdr"]]
        .sort_values("rank_score", ascending=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    age_path = args.out_dir / "demographic_contrast_scan_age_contrasts.csv"
    age_top.to_csv(age_path, index=False)
    sys.stderr.write(
        f"[info] wrote age contrasts (supp.) -> {age_path} ({len(age_top)} rows)\n"
    )

    # ---- sensitivity scan at the lower threshold ----
    sys.stderr.write(
        f"[info] scanning at sensitivity threshold N >= {args.sensitivity_threshold}...\n"
    )
    sens_records = []
    for i, ((seq, sl, tidx, tshort), block) in enumerate(grouped, 1):
        recs = scan_one_sequence(
            seq=str(seq), seq_length=int(sl),
            target_idx=int(tidx), target_short=str(tshort),
            data_for_seq=block, min_events=args.sensitivity_threshold,
            all_strats=all_strats,
        )
        sens_records.extend(recs)
    if sens_records:
        sdf = pd.DataFrame(sens_records)
        sdf["p_bh_adjusted"] = np.nan
        sdf["survives_bh_fdr"] = False
        for ct, sub in sdf.groupby("contrast_type"):
            adj = bh_adjust(sub["p_raw"].values)
            sdf.loc[sub.index, "p_bh_adjusted"] = adj
            sdf.loc[sub.index, "survives_bh_fdr"] = adj < args.bh_alpha
        sdf["rank_score"] = (
            np.abs(sdf["log_irr"]) - 1.959963984540054 * sdf["se_log_irr"]
        )
        for c, p in rounders.items():
            if c in sdf.columns:
                sdf[c] = sdf[c].round(p)
        sens_top = (
            sdf[sdf["contrast_type"].isin(primary_types)
                & sdf["survives_bh_fdr"]]
            .sort_values("rank_score", ascending=False)
            .groupby("contrast_type", group_keys=False, observed=True)
            .head(args.top_n)
            .reset_index(drop=True)
        )
        sens_path = args.out_dir / "demographic_contrast_scan_sensitivity_n10.csv"
        sens_top.to_csv(sens_path, index=False)
        sys.stderr.write(
            f"[info] wrote sensitivity top hits -> {sens_path} "
            f"({len(sens_top)} rows)\n"
        )
    else:
        sens_path = None

    # ---- provenance ----
    from incigraph.api import _resolve_dir
    sidecar = {
        "script_name": "step3_contrast_scan.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_dir": str(_resolve_dir(args.parquet_dir)),
        "estimates_total": len(estimates),
        "stratifications_scanned": all_strats,
        "sequences_scanned": n_seqs,
        "total_contrasts_headline": int(len(df)),
        "primary_top_hits": int(len(top_primary)),
        "age_top_hits": int(len(age_top)),
        "sensitivity_top_hits": int(len(sens_top)) if sens_path else 0,
        "event_threshold_primary": args.event_threshold,
        "event_threshold_sensitivity": args.sensitivity_threshold,
        "bh_alpha": args.bh_alpha,
        "bh_scope":
            "within each contrast_type independently "
            "(screening safeguard, not confirmatory inference)",
        "ranking":
            "stability-weighted score = |log IRR| - 1.96 * SE(log IRR); "
            "the lower bound of |log IRR| at 95% CI",
        "framing":
            "Hypothesis-generating crude incidence rate ratios. "
            "NOT adjusted for confounding, competing risks, or "
            "differential coding by demographic group.",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "incigraph": ig.__version__,
        },
    }
    sidecar_path = args.out_dir / "demographic_contrast_scan.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    sys.stderr.write(f"[info] wrote sidecar -> {sidecar_path}\n")
    sys.stderr.write(f"[info] elapsed: {sidecar['elapsed_seconds']}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
