#!/usr/bin/env python3
"""
step1_aggregation_consistency.py
================================
Verify that InciGraph's stored estimates are internally coherent at the
parquet level: when one stratification scheme is a strict marginalisation
of another (e.g. ETHNICITY+IMD is ETHNICITY+IMD+AGE_CATG pooled over age),
summing the numerator and denominator from the more granular subset must
reproduce the less granular subset's values, and recomputing the rate and
95% CI from the cumulative count must match the stored less-granular rate
and CI to within numerical tolerance.

This is Step 1 of the validation pipeline -- the reviewer-armour. The
arithmetic check is what justifies the manuscript's "internal consistency"
claim and the hero figure's Panel A.

Reframing note (parquet vs original xlsx tree)
-----------------------------------------------
On the xlsx tree this check compared two FILE TREES. On the parquet it
compares two SUBSETS of one big table -- the underlying property is the
same, but the lookup is faster and the script is shorter. The xlsx-level
validation was done once and archived in the published Zenodo deposit.
This script checks the *parquet's* internal consistency, which is what
matters for downstream parquet users.

PROCEDURE
---------
For every pair of stratifications (source, target) where target's axes are
a strict subset of source's:

  1) Verify marginalisation: total denominator over the relevant sequence
     should agree to within tolerance between source and target. If it
     doesn't, the two are not strict marginalisations -- record but do
     not compare per-cell.

  2) For each sequence the user includes in the panel, aggregate the
     source rows on the target's axes (summing numerator and denominator),
     recompute IR and 95% CI from the cumulative count, then compare
     against the target's stored values cell by cell.

OUTPUTS
-------
  <out-dir>/aggregation_consistency_detailed.csv  -- per-cell comparison
  <out-dir>/aggregation_consistency_summary.csv   -- per-pair summary
  <out-dir>/aggregation_consistency.json          -- provenance sidecar

GATE
----
If any marginalisable pair has rate_match below 99%, the script EXITS
NON-ZERO so the pipeline halts. This is intentional: Step 2 and Step 3
have no meaning if the data is not internally consistent.

CONVENTIONS PRESERVED
---------------------
- IMD blank means MISSING (a real stratum). The marginalisation sum keeps
  these rows so the totals agree.
- SEX = 'I' is its own demographic category. Kept in the sums.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import incigraph as ig
from incigraph.ci import poisson_ci, RATE_DENOMINATOR


# Numerical tolerances for "match"
RATE_REL_TOL = 1e-6     # relative tolerance for rates and CIs
COUNT_ABS_TOL = 0.5     # numerators are integers; tolerance is half a count

# Representative sequences used to verify marginalisation. Extending this
# panel exercises more of the data but the test is the same per sequence.
REPRESENTATIVE_SEQUENCES = [
    "0 3",          # hypertension
    "0 3 8",        # hypertension -> T2D
    "0 3 8 9",      # hypertension -> T2D -> CKD
    "0 26",         # eczema
    "0 26 23",      # eczema -> asthma
    "0 26 23 27",   # eczema -> asthma -> allergic rhinitis
    "0 10",         # depression
    "0 10 11",      # depression -> anxiety
]

# Stratification axes we recognise. Mapping the parquet column name to the
# key-segment name used in stratification_key strings.
AXIS_TO_COL = {
    "AGE_CATG":  "age_catg",
    "ETHNICITY": "ethnicity",
    "IMD":       "imd",
    "SEX":       "sex",
}


def axes_of(strat: str) -> set[str]:
    """'AGE_CATG+ETHNICITY+IMD' -> {'AGE_CATG','ETHNICITY','IMD'}; 'NONE' -> {}."""
    return set() if strat == "NONE" else set(strat.split("+"))


def is_marginalisation(source: str, target: str) -> bool:
    """True if target's axes are a strict subset of source's axes."""
    s, t = axes_of(source), axes_of(target)
    return t < s


def find_pairs(strats: list[str]) -> list[tuple[str, str]]:
    """All (source, target) pairs where target axes are a strict subset of
    source's. Both stratifications must exist in the data."""
    pairs = []
    for source, target in combinations(strats, 2):
        if is_marginalisation(source, target):
            pairs.append((source, target))
        elif is_marginalisation(target, source):
            pairs.append((target, source))
    return pairs


def aggregate_source(source_block: pd.DataFrame,
                     target_axes: set[str]) -> pd.DataFrame:
    """Sum numerator and denominator across the axes that target marginalises.

    `source_block` is a slice of the parquet for one (sequence, source
    stratification). target_axes is the set of axes target keeps.
    The result has one row per (target stratum) cell with summed counts and
    recomputed rate / CI.
    """
    target_cols = [AXIS_TO_COL[a] for a in sorted(target_axes)]
    # We use dropna=False so that IMD-missing (NaN), SEX='I', and any other
    # blank-but-real categories survive the groupby as their own cells.
    if target_cols:
        agg = (source_block
               .groupby(target_cols, dropna=False, observed=True)
               [["numerator", "denominator"]].sum()
               .reset_index())
    else:
        agg = pd.DataFrame({
            "numerator":   [source_block["numerator"].sum()],
            "denominator": [source_block["denominator"].sum()],
        })

    # Also carry the imd_missing flag for the IMD-missing stratum if IMD is
    # an axis of the target. (When the source has IMD blank, those rows have
    # imd_missing=True and group together cleanly because all share NaN.)
    if "IMD" in target_axes:
        # rebuild imd_missing from imd: NaN imd means imd_missing=True
        agg["imd_missing"] = agg["imd"].isna()

    # recompute rate and CI from the cumulative count
    agg["incidence_rate"] = np.where(
        agg["denominator"] > 0,
        agg["numerator"] / agg["denominator"] * RATE_DENOMINATOR,
        np.nan,
    )
    lo, hi = poisson_ci(agg["numerator"].values, agg["denominator"].values)
    agg["lower_limit"] = lo
    agg["upper_limit"] = hi
    return agg


def compare_pair(seq: str, source: str, target: str,
                 estimates_for_seq: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the consistency check for one (sequence, source, target) triple.

    Returns (detailed_df, summary_dict).
    """
    # Step 1a: total denominators agree? (marginalisation precondition)
    src = estimates_for_seq[estimates_for_seq["stratification_key"] == source]
    tgt = estimates_for_seq[estimates_for_seq["stratification_key"] == target]
    src_total_y = float(src["denominator"].sum())
    tgt_total_y = float(tgt["denominator"].sum())
    if max(src_total_y, tgt_total_y) > 0:
        denom_rel_diff = abs(src_total_y - tgt_total_y) / max(src_total_y,
                                                              tgt_total_y)
    else:
        denom_rel_diff = np.nan
    marginalisation_ok = denom_rel_diff < RATE_REL_TOL

    if not marginalisation_ok:
        # The two stratifications represent different cohorts; we don't
        # do the per-cell comparison.
        return pd.DataFrame(), {
            "sequence": seq,
            "source_stratification": source,
            "target_stratification": target,
            "marginalisation_ok": False,
            "marginalisation_denom_rel_diff": denom_rel_diff,
            "n_cells_compared": 0,
            "percent_numerator_match": np.nan,
            "percent_denominator_match": np.nan,
            "percent_rate_match": np.nan,
            "percent_ci_exact_match": np.nan,
            "max_absolute_numerator_diff": np.nan,
            "max_relative_rate_diff": np.nan,
            "n_failed_rate_match": 0,
        }

    # Step 1b: aggregate source rows on the target axes, then join against
    # target's stored rows, then compare cell by cell.
    tgt_axes = axes_of(target)
    target_cols = [AXIS_TO_COL[a] for a in sorted(tgt_axes)]
    src_agg = aggregate_source(src, tgt_axes)
    tgt_view = tgt.copy()

    # Make sure target has the imd_missing flag matching the source's
    # encoding so the join keys line up.
    if "IMD" in tgt_axes and "imd_missing" not in tgt_view.columns:
        tgt_view["imd_missing"] = tgt_view["imd"].isna()

    join_keys = target_cols.copy()
    if join_keys:
        merged = src_agg.merge(
            tgt_view[join_keys + ["numerator", "denominator", "incidence_rate",
                                  "lower_limit", "upper_limit"]],
            on=join_keys,
            suffixes=("_src", "_tgt"),
            how="inner",
        )
    else:
        # Target is NONE (unstratified): src_agg has exactly one row (the
        # full marginal sum) and tgt has exactly one row (the unstratified
        # estimate). pd.merge with an empty `on` list raises; we line the
        # two single rows up by hand instead.
        if len(src_agg) != 1 or len(tgt_view) != 1:
            sys.stderr.write(
                f"[warn] unexpected row counts comparing {source} -> {target} "
                f"for seq {seq}: src_agg has {len(src_agg)} rows, tgt has "
                f"{len(tgt_view)}; skipping per-cell check\n"
            )
            merged = pd.DataFrame()
        else:
            # Concatenate the two single rows side by side with _src / _tgt
            # suffixes on the columns that exist in both. The result has
            # one row, the same shape as the merge-based path produces.
            shared = ["numerator", "denominator", "incidence_rate",
                      "lower_limit", "upper_limit"]
            src_row = src_agg.iloc[0]
            tgt_row = tgt_view.iloc[0]
            merged_data = {}
            for col in shared:
                merged_data[f"{col}_src"] = [src_row[col]]
                merged_data[f"{col}_tgt"] = [tgt_row[col]]
            merged = pd.DataFrame(merged_data)

    if merged.empty:
        return pd.DataFrame(), {
            "sequence": seq,
            "source_stratification": source,
            "target_stratification": target,
            "marginalisation_ok": True,
            "marginalisation_denom_rel_diff": denom_rel_diff,
            "n_cells_compared": 0,
            "percent_numerator_match": np.nan,
            "percent_denominator_match": np.nan,
            "percent_rate_match": np.nan,
            "percent_ci_exact_match": np.nan,
            "max_absolute_numerator_diff": np.nan,
            "max_relative_rate_diff": np.nan,
            "n_failed_rate_match": 0,
        }

    # diffs.
    # The merge added "_src" / "_tgt" suffixes to columns that exist in
    # both src_agg and tgt_view (numerator, denominator, incidence_rate,
    # lower_limit, upper_limit). We compare on those.
    merged["numerator_diff"] = (
        merged["numerator_src"] - merged["numerator_tgt"]
    ).abs()
    denom_max = merged[["denominator_src", "denominator_tgt"]].max(axis=1)
    merged["denominator_rel_diff"] = np.where(
        denom_max > 0,
        (merged["denominator_src"] - merged["denominator_tgt"]).abs() / denom_max,
        0.0,
    )
    rate_max = merged[["incidence_rate_src", "incidence_rate_tgt"]].abs().max(axis=1)
    merged["rate_rel_diff"] = np.where(
        rate_max > 0,
        (merged["incidence_rate_src"] - merged["incidence_rate_tgt"]).abs() / rate_max,
        0.0,
    )
    lo_max = merged[["lower_limit_src", "lower_limit_tgt"]].abs().max(axis=1)
    hi_max = merged[["upper_limit_src", "upper_limit_tgt"]].abs().max(axis=1)
    merged["lower_rel_diff"] = np.where(
        lo_max > 0,
        (merged["lower_limit_src"] - merged["lower_limit_tgt"]).abs() / lo_max,
        0.0,
    )
    merged["upper_rel_diff"] = np.where(
        hi_max > 0,
        (merged["upper_limit_src"] - merged["upper_limit_tgt"]).abs() / hi_max,
        0.0,
    )

    # match flags
    merged["numerator_match"]   = merged["numerator_diff"] <= COUNT_ABS_TOL
    merged["denominator_match"] = merged["denominator_rel_diff"] <= RATE_REL_TOL
    merged["rate_match"]        = merged["rate_rel_diff"] <= RATE_REL_TOL
    merged["ci_exact_match"] = (
        (merged["lower_rel_diff"] <= RATE_REL_TOL)
        & (merged["upper_rel_diff"] <= RATE_REL_TOL)
    )

    # tidy detailed rows: add metadata columns directly to the merged df
    detailed = merged.copy()
    detailed.insert(0, "sequence", seq)
    detailed.insert(1, "source_stratification", source)
    detailed.insert(2, "target_stratification", target)
    detailed.insert(3, "marginalisation_ok", True)
    detailed.insert(4, "comparison_key",
                    "|".join(target_cols) if target_cols else "(none)")

    # summary
    n_cells = len(merged)
    summary = {
        "sequence": seq,
        "source_stratification": source,
        "target_stratification": target,
        "marginalisation_ok": True,
        "marginalisation_denom_rel_diff": denom_rel_diff,
        "n_cells_compared": n_cells,
        "percent_numerator_match":
            float(merged["numerator_match"].sum()) / n_cells * 100,
        "percent_denominator_match":
            float(merged["denominator_match"].sum()) / n_cells * 100,
        "percent_rate_match":
            float(merged["rate_match"].sum()) / n_cells * 100,
        "percent_ci_exact_match":
            float(merged["ci_exact_match"].sum()) / n_cells * 100,
        "max_absolute_numerator_diff": float(merged["numerator_diff"].max()),
        "max_relative_rate_diff":  float(merged["rate_rel_diff"].max()),
        "n_failed_rate_match":
            int((~merged["rate_match"]).sum()),
    }
    return detailed, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="Sequences to test. Default: a panel of 8 "
                         "representative sequences (see source).")
    ap.add_argument("--gate-threshold", type=float, default=99.0,
                    help="Minimum percent_rate_match required on every "
                         "marginalisable pair. Below this the script "
                         "exits non-zero (halting the pipeline). "
                         "Default 99.")
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
            pass
    if not parts:
        sys.stderr.write("[error] no parquet files found\n")
        return 2
    estimates = pd.concat(parts, ignore_index=True)
    sys.stderr.write(f"[info] total estimates: {len(estimates):,}\n")

    all_strats = sorted(estimates["stratification_key"].astype(str).unique().tolist())
    pairs = find_pairs(all_strats)
    sys.stderr.write(
        f"[info] stratifications: {all_strats}\n"
        f"[info] candidate (source, target) pairs: {len(pairs)}\n"
    )

    sequences = args.sequences or REPRESENTATIVE_SEQUENCES
    # only keep sequences actually in the data
    available = set(estimates["sequence"].astype(str).unique())
    sequences = [s for s in sequences if s in available]
    sys.stderr.write(f"[info] sequences tested: {sequences}\n")

    detailed_rows, summary_rows, gate_failures = [], [], []
    n_attempted = n_succeeded = 0
    for seq in sequences:
        block = estimates[estimates["sequence"] == seq]
        for source, target in pairs:
            n_attempted += 1
            detailed, summary = compare_pair(seq, source, target, block)
            summary_rows.append(summary)
            if len(detailed):
                detailed_rows.append(detailed)
                n_succeeded += 1
                rm = summary["percent_rate_match"]
                if np.isfinite(rm) and rm < args.gate_threshold:
                    gate_failures.append({
                        "sequence": seq,
                        "source": source, "target": target,
                        "percent_rate_match": rm,
                    })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = (pd.concat(detailed_rows, ignore_index=True)
                   if detailed_rows else pd.DataFrame())

    summary_path  = args.out_dir / "aggregation_consistency_summary.csv"
    detailed_path = args.out_dir / "aggregation_consistency_detailed.csv"
    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    sys.stderr.write(f"[info] wrote summary  -> {summary_path}\n")
    sys.stderr.write(f"[info] wrote detailed -> {detailed_path}\n")

    from incigraph.api import _resolve_dir
    sidecar = {
        "script_name": "step1_aggregation_consistency.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_dir": str(_resolve_dir(args.parquet_dir)),
        "stratifications": all_strats,
        "pairs_tested": n_attempted,
        "pairs_marginalisation_ok": n_succeeded,
        "sequences_tested": sequences,
        "tolerances": {"rate_rel_tol": RATE_REL_TOL,
                       "count_abs_tol": COUNT_ABS_TOL},
        "gate_threshold": args.gate_threshold,
        "gate_failures": gate_failures,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "incigraph": ig.__version__,
        },
    }
    sidecar_path = args.out_dir / "aggregation_consistency.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    sys.stderr.write(f"[info] wrote sidecar -> {sidecar_path}\n")
    sys.stderr.write(f"[info] elapsed: {sidecar['elapsed_seconds']}s\n")

    # Gate
    if gate_failures:
        sys.stderr.write(
            f"\n[GATE FAIL] {len(gate_failures)} pair(s) fell below "
            f"{args.gate_threshold}% rate match. Step 2 and Step 3 "
            "should NOT be trusted. Inspect the detailed CSV.\n"
        )
        for gf in gate_failures[:5]:
            sys.stderr.write(
                f"  seq={gf['sequence']} src={gf['source']} "
                f"tgt={gf['target']} rate_match={gf['percent_rate_match']:.2f}%\n"
            )
        return 1
    sys.stderr.write("[gate] all marginalisable pairs passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
