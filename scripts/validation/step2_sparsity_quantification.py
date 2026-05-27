#!/usr/bin/env python3
"""
step2_sparsity_quantification.py
================================
Quantify where InciGraph's estimates are statistically supported, by
counting how many cells meet event-count thresholds across sequence
length and stratification scheme.

This is Step 2 of the validation pipeline. It produces the data behind
the operating-characteristics figure (Panel C of the hero figure) and the
manuscript's transparency statement about where the tool's estimates are
sparse vs supported.

On the xlsx tree this script needed a two-stage manifest + checkpoint
runner to manage the runtime cost of opening tens of thousands of
workbooks. Against the parquet it collapses to one groupby. Same outputs.

OUTPUTS
-------
  <out-dir>/sparsity_by_stratification.csv  -- aggregated summary,
       indexed by (sequence_length, stratification_key)
  <out-dir>/sparsity_heatmap_n_per_cell.csv -- the n_estimates value
       behind each heatmap cell, for the figure caption
  <out-dir>/sparsity_quantification.json    -- provenance sidecar

cell_status categories (mutually exclusive):
  observed_events       numerator > 0 and denominator > 0
  observed_zero         numerator == 0 and denominator > 0
  suppressed_or_missing numerator is null OR denominator <= 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import incigraph as ig


def classify_cells(df: pd.DataFrame) -> pd.Series:
    """Return a categorical column with cell_status for each row."""
    num = df["numerator"]
    den = df["denominator"]
    observed = den.fillna(0) > 0
    has_events = observed & (num.fillna(0) > 0)
    zero_events = observed & (num.fillna(-1) == 0)
    status = pd.Series("suppressed_or_missing", index=df.index, dtype="object")
    status[has_events]  = "observed_events"
    status[zero_events] = "observed_zero"
    return status


def summarise_one_block(df: pd.DataFrame) -> dict:
    """Per (sequence_length, stratification_key) summary."""
    status = classify_cells(df)
    n_total       = len(df)
    n_events      = int((status == "observed_events").sum())
    n_zero        = int((status == "observed_zero").sum())
    n_suppressed  = int((status == "suppressed_or_missing").sum())
    n_observed    = n_events + n_zero
    # numerator/denominator stats over the OBSERVED subset
    observed_mask = status.isin(["observed_events", "observed_zero"])
    n_obs = df.loc[observed_mask, "numerator"].astype(float)
    y_obs = df.loc[observed_mask, "denominator"].astype(float)

    def _quantiles(series, ps=(0.25, 0.5, 0.75)):
        if len(series) == 0:
            return (np.nan, np.nan, np.nan)
        return tuple(series.quantile(p) for p in ps)

    n_q1, n_med, n_q3 = _quantiles(n_obs)
    y_q1, y_med, y_q3 = _quantiles(y_obs)

    # threshold tallies over ALL rows (suppressed = fail)
    def _pct_ge(series, thresh):
        if n_total == 0:
            return np.nan
        return float((series.fillna(0) >= thresh).sum()) / n_total * 100

    # ...and over observed only
    def _pct_ge_observed(series, thresh):
        if n_observed == 0:
            return np.nan
        return float((series.fillna(0) >= thresh).sum()) / n_observed * 100

    nums = df["numerator"]
    return {
        "n_estimates": n_total,
        "n_observed_events": n_events,
        "n_observed_zero": n_zero,
        "n_suppressed_or_missing": n_suppressed,
        "median_numerator": n_med,
        "iqr_numerator": f"{n_q1:.0f} - {n_q3:.0f}" if not np.isnan(n_q1) else "",
        "median_denominator": y_med,
        "iqr_denominator": f"{y_q1:.0f} - {y_q3:.0f}" if not np.isnan(y_q1) else "",
        "percent_zero": (n_zero / n_total * 100) if n_total else np.nan,
        "percent_suppressed_or_missing":
            (n_suppressed / n_total * 100) if n_total else np.nan,
        "percent_n_ge_5":   _pct_ge(nums, 5),
        "percent_n_ge_10":  _pct_ge(nums, 10),
        "percent_n_ge_30":  _pct_ge(nums, 30),
        "percent_n_ge_100": _pct_ge(nums, 100),
        "percent_n_ge_10_among_observed":
            _pct_ge_observed(nums[observed_mask], 10),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", type=Path, default=None,
                    help="Directory with the parquet files. Defaults to "
                         "INCIGRAPH_DATA env var or ./incigraph_data.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory for the output CSVs and JSON sidecar.")
    args = ap.parse_args()

    if args.parquet_dir is not None:
        ig.set_data_dir(args.parquet_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Load all three length files. Memory cost: ~50 M rows total at the full
    # scale, which is acceptable; pyarrow uses categorical dtypes for the
    # text columns so the resident set stays under a few GB.
    sys.stderr.write("[info] loading L1, L2, L3 estimates...\n")
    parts = []
    for L in (1, 2, 3):
        try:
            df = ig.load_estimates(L)
        except FileNotFoundError:
            sys.stderr.write(f"[warn] L{L} parquet missing, skipping\n")
            continue
        sys.stderr.write(f"[info] L{L}: {len(df):,} rows\n")
        parts.append(df)
    if not parts:
        sys.stderr.write("[error] no parquet files found\n")
        return 2

    estimates = pd.concat(parts, ignore_index=True)
    n_total = len(estimates)
    sys.stderr.write(
        f"[info] total estimates: {n_total:,}\n"
    )

    # === group and summarise ===
    sys.stderr.write("[info] computing per-(length, stratification) summaries...\n")
    summary_rows = []
    grouped = estimates.groupby(["sequence_length", "stratification_key"],
                                observed=True)
    for (sl, strat), block in grouped:
        record = {"sequence_length": int(sl), "stratification_key": str(strat)}
        record.update(summarise_one_block(block))
        # demographic_depth: number of axes in the stratification key
        if record["stratification_key"] == "NONE":
            record["demographic_depth"] = 0
        else:
            record["demographic_depth"] = (
                record["stratification_key"].count("+") + 1
            )
        summary_rows.append(record)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["sequence_length", "demographic_depth", "stratification_key"]
    ).reset_index(drop=True)

    summary_path = args.out_dir / "sparsity_by_stratification.csv"
    summary.to_csv(summary_path, index=False)
    sys.stderr.write(f"[info] wrote summary -> {summary_path}\n")

    # === n-per-cell table for the heatmap figure caption ===
    n_table = summary[["sequence_length", "stratification_key", "n_estimates"]]
    n_table_path = args.out_dir / "sparsity_heatmap_n_per_cell.csv"
    n_table.to_csv(n_table_path, index=False)
    sys.stderr.write(f"[info] wrote n-per-cell table -> {n_table_path}\n")

    # === provenance sidecar ===
    # Pull the resolved data dir back out of the api module for the record.
    # We use the internal _resolve_dir helper rather than reading _DATA_DIR
    # directly, so the recorded path reflects the actual lookup precedence
    # (set_data_dir -> env -> default).
    from incigraph.api import _resolve_dir
    resolved_dir = str(_resolve_dir(args.parquet_dir))

    sidecar = {
        "script_name": "step2_sparsity_quantification.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_dir": resolved_dir,
        "estimates_total": n_total,
        "stratifications_seen": sorted(
            estimates["stratification_key"].astype(str).unique().tolist()
        ),
        "outputs": {
            "summary": str(summary_path),
            "n_per_cell": str(n_table_path),
        },
        "thresholds_reported": [5, 10, 30, 100],
        "elapsed_seconds": round(time.time() - t_start, 2),
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "incigraph": ig.__version__,
        },
    }
    sidecar_path = args.out_dir / "sparsity_quantification.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    sys.stderr.write(f"[info] wrote provenance sidecar -> {sidecar_path}\n")
    sys.stderr.write(f"[info] elapsed: {sidecar['elapsed_seconds']}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
