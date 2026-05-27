#!/usr/bin/env python3
"""
curate_ordered_transition_candidates.py
=======================================
Extract clinically interpretable candidate rows for Panel B of the hero
figure (or the standalone length-3 figure) from the demographic contrast
scan output produced by step3_contrast_scan.py.

GOAL
----
For one sequence length (1, 2, or 3) and one contrast type (deprivation,
ethnicity, sex, or age), find crude IRR contrasts that are:
  (a) present in the scan,
  (b) BH-survived at q=0.05 within their contrast type,
  (c) supported by at least --min-events in BOTH groups,
  (d) clinically interpretable.

The script handles (a)-(c) plus the disease-name decoding. You make the
final clinical-recognisability call (d) by reading the output CSVs.

INPUTS
------
The full scan output. By default we read the parquet form if it exists
(<scan-dir>/demographic_contrast_scan_all.parquet); falls back to the CSV
form (<scan-dir>/demographic_contrast_scan_all.csv) for older runs.

OUTPUTS
-------
  <out-dir>/candidates_L<N>_<contrast_type>_top<K>.csv  -- per-type rankings
  <out-dir>/candidates_L<N>_all.csv                     -- combined file
  <out-dir>/concentration_report_L<N>.txt               -- diagnostic

The concentration report flags whether the top-N is dominated by one
condition at any sequence position. Heavy concentration usually means
either a real "everything funnels into X" pattern (informative) or one
well-coded condition driving apparent gradients across upstream pairings
(artefactual); the report tells you which to investigate.

USAGE
-----
  # Length-2 deprivation+ethnicity candidates (default)
  python curate_ordered_transition_candidates.py \\
      --scan-dir ./step3_output --out-dir ./candidates

  # Length-3, the InciGraph signature query
  python curate_ordered_transition_candidates.py \\
      --scan-dir ./step3_output --out-dir ./candidates \\
      --sequence-length 3 --min-events 30 --top-n 60
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

import incigraph as ig
from incigraph.disease_index import (
    DISEASE_NAMES, N_DISEASES, parse_sequence
)


def find_scan_file(scan_dir: Path) -> Path:
    """Prefer the parquet form; fall back to CSV."""
    for name in ("demographic_contrast_scan_all.parquet",
                 "demographic_contrast_scan_all.csv"):
        p = scan_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find demographic_contrast_scan_all.{{parquet,csv}} "
        f"in {scan_dir}. Run step3_contrast_scan.py first."
    )


def read_scan(scan_file: Path) -> pd.DataFrame:
    """Load the scan output (handles parquet or CSV)."""
    if scan_file.suffix == ".parquet":
        return pd.read_parquet(scan_file)
    return pd.read_csv(scan_file)


def decode_with_steps(seq_str, expected_length):
    """Like ig.decode_sequence but also returns the indices and per-step
    short names, so the curator can record both. Returns None on parse
    failure or length mismatch."""
    try:
        idxs = parse_sequence(seq_str)
    except ValueError:
        return None
    if len(idxs) != expected_length:
        return None
    names = tuple(DISEASE_NAMES[i - 1] for i in idxs)
    return tuple(idxs), names, " -> ".join(names)


def concentration_check(df_decoded, contrast_type, expected_length):
    """Are the top-N dominated by a single condition? Returns the text block."""
    if not len(df_decoded):
        return f"{contrast_type}: no rows to check.\n"
    lines = [f"--- Concentration check for {contrast_type} top-{len(df_decoded)} ---"]
    flagged = False
    labels = ("FIRST ", "SECOND", "THIRD ")
    for step in range(expected_length):
        col = f"step_{step + 1}_name"
        counts = df_decoded[col].value_counts()
        top = counts.index[0]
        top_count = int(counts.iloc[0])
        pct = top_count / len(df_decoded) * 100
        lines.append(f"Most common {labels[step]} condition: {top}  "
                     f"({top_count}/{len(df_decoded)} = {pct:.0f}%)")
        if pct >= 40:
            flagged = True
    if flagged:
        lines.append(
            "  WARNING: rank is dominated by one condition. Consider "
            "widening the candidate pool by lowering --min-events or "
            "raising --top-n."
        )
    else:
        lines.append("  OK: top-N spans multiple multimorbidity transitions.")
    return "\n".join(lines) + "\n"


def curate_one_type(scan: pd.DataFrame, contrast_type: str,
                    sequence_length: int, top_n: int,
                    min_events: int) -> pd.DataFrame:
    """Filter, decode, rank top-N. Returns a tidy DataFrame."""
    sub = scan[
        (scan["sequence_length"] == sequence_length)
        & (scan["contrast_type"] == contrast_type)
        & (scan.get("survives_bh_fdr",
                    scan.get("survives_bh_fdr_005", True)))
        & (scan["comparison_numerator"] >= min_events)
        & (scan["reference_numerator"] >= min_events)
    ].copy()
    if not len(sub):
        return pd.DataFrame()

    # decode sequences -- drop rows where decoding fails
    decoded = sub["sequence"].apply(
        lambda s: decode_with_steps(s, sequence_length))
    keep = decoded.notna()
    sub = sub.loc[keep].copy()
    if not len(sub):
        return pd.DataFrame()
    decoded = decoded[keep]
    for step in range(sequence_length):
        sub[f"step_{step + 1}_idx"]  = decoded.apply(lambda t: t[0][step])
        sub[f"step_{step + 1}_name"] = decoded.apply(lambda t: t[1][step])
    sub["sequence_decoded"] = decoded.apply(lambda t: t[2])

    sub = sub.sort_values("rank_score", ascending=False).head(top_n)

    # round display-numerics
    for col, places in [
        ("incidence_rate_ratio", 3), ("lower_95_ci_irr", 3),
        ("upper_95_ci_irr", 3), ("rank_score", 4),
        ("comparison_rate", 2), ("reference_rate", 2),
        ("comparison_denominator", 1), ("reference_denominator", 1),
        ("p_raw", 6), ("p_bh_adjusted", 6),
    ]:
        if col in sub.columns:
            sub[col] = sub[col].round(places)

    return sub


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scan-dir", type=Path, required=True,
                    help="Directory containing the step3_contrast_scan.py "
                         "outputs.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory for the candidate output files.")
    ap.add_argument("--sequence-length", type=int, default=2,
                    choices=[1, 2, 3],
                    help="Length of the ordered transition to curate "
                         "(default: 2).")
    ap.add_argument("--contrast-types", nargs="+",
                    default=["deprivation", "ethnicity"],
                    help="Which contrast types to curate (default: "
                         "deprivation ethnicity). Use 'sex' or 'age' to "
                         "include the secondary ones.")
    ap.add_argument("--min-events", type=int, default=100,
                    help="Minimum numerator in BOTH groups. Default 100 "
                         "for the headline. For length-3 you may want 30.")
    ap.add_argument("--top-n", type=int, default=30,
                    help="Top-N rows per contrast type. Default 30.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    scan_file = find_scan_file(args.scan_dir)
    sys.stderr.write(f"[info] reading scan output: {scan_file}\n")
    scan = read_scan(scan_file)
    sys.stderr.write(f"[info] scan rows: {len(scan):,}\n")

    sys.stderr.write(
        textwrap.dedent(f"""
        [info] prefilter: sequence_length=={args.sequence_length},
               contrast types in {args.contrast_types}, BH-FDR survived,
               both groups >= {args.min_events} events.
        """).strip() + "\n"
    )

    output_columns_base = (
        ["sequence_decoded", "sequence",
         "contrast_type", "conditioning_strata",
         "comparison_group", "reference_group",
         "incidence_rate_ratio", "lower_95_ci_irr", "upper_95_ci_irr",
         "comparison_numerator", "reference_numerator",
         "comparison_rate", "reference_rate",
         "rank_score", "p_bh_adjusted"]
    )
    step_columns = []
    for s in range(args.sequence_length):
        step_columns += [f"step_{s+1}_idx", f"step_{s+1}_name"]

    combined_parts, report_parts = [], []

    for ct in args.contrast_types:
        sub = curate_one_type(
            scan, contrast_type=ct,
            sequence_length=args.sequence_length,
            top_n=args.top_n, min_events=args.min_events,
        )
        cols = [c for c in step_columns + output_columns_base if c in sub.columns]
        out_path = args.out_dir / (
            f"candidates_L{args.sequence_length}_{ct}_top{args.top_n}.csv"
        )
        sub[cols].to_csv(out_path, index=False)
        sys.stderr.write(
            f"[info] wrote {len(sub)} {ct} candidates -> {out_path}\n"
        )

        report_parts.append(
            concentration_check(sub, ct, args.sequence_length))
        if len(sub):
            combined_parts.append(sub[cols])

            sys.stderr.write(f"\n=== {ct.upper()} top {min(10, len(sub))} ===\n")
            preview_cols = [c for c in (
                "sequence_decoded", "conditioning_strata",
                "comparison_group", "reference_group",
                "incidence_rate_ratio",
                "lower_95_ci_irr", "upper_95_ci_irr",
                "comparison_numerator", "reference_numerator",
                "rank_score") if c in sub.columns]
            sys.stderr.write(sub[preview_cols].head(10).to_string(index=False) + "\n")

    if combined_parts:
        combined = pd.concat(combined_parts, ignore_index=True)
        combo_path = args.out_dir / f"candidates_L{args.sequence_length}_all.csv"
        combined.to_csv(combo_path, index=False)
        sys.stderr.write(f"\n[info] wrote combined candidates -> {combo_path}\n")

    report_path = args.out_dir / f"concentration_report_L{args.sequence_length}.txt"
    with open(report_path, "w") as f:
        f.write(f"Concentration check on candidate top-N "
                f"(sequence_length={args.sequence_length})\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(report_parts))
    sys.stderr.write(f"[info] wrote concentration report -> {report_path}\n")

    sys.stderr.write(
        f"\nNext step: open candidates_L{args.sequence_length}_all.csv and "
        "skim the decoded sequences. Send the chosen rows back to be worked "
        "into the figure.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
