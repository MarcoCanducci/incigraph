#!/usr/bin/env python3
"""
collect_incigraph.py
====================
Extract a tidy CSV of estimates for one disease sequence and one
stratification scheme from the InciGraph parquet deposit. The output is
plot-ready (plot_incigraph.py consumes it directly).

This is a thin CLI wrapper around incigraph.get_sequence(). All the heavy
lifting -- sequence parsing, age-band pooling with CI recomputation,
IMD-missing / SEX='I' handling -- happens inside the library.

USAGE
-----
  # Length-2 query: incidence of T2D after hypertension, by ethnicity x IMD,
  # for the 51-60 cohort
  python collect_incigraph.py \\
      --sequence 0 3 8 \\
      --stratification ETHNICITY+IMD+AGE_CATG \\
      --age 51-60 \\
      --out panelA_long.csv

  # Multi-band age aggregation: pool 51-60 and 61-70 into a 51-70 cohort,
  # with numerator/denominator summed and CI recomputed per the supplement
  python collect_incigraph.py \\
      --sequence 0 3 8 9 \\
      --stratification ETHNICITY+IMD+AGE_CATG \\
      --age 51-60 61-70 \\
      --out panelB_5170_long.csv

  # Single-disease, no age stratification:
  python collect_incigraph.py \\
      --sequence 0 3 \\
      --stratification ETHNICITY+IMD \\
      --out htn_long.csv

OUTPUT
------
Long-format CSV, one row per (stratum) cell:
  sequence, sequence_length, target_disease_idx, target_disease_short,
  ethnicity, sex, imd, imd_missing, age_catg,
  numerator, denominator, incidence_rate, lower_limit, upper_limit

The IMD-missing rows (imd_missing=True) and SEX='I' rows are INCLUDED by
default because they are real demographic strata. Pass --drop-missing-imd
and/or --drop-sex-i to filter them out for plotting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import incigraph as ig


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Directory with the parquet files. Defaults to the "
             "INCIGRAPH_DATA env var, or ./incigraph_data.",
    )
    ap.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        required=True,
        help="Sequence as disease indices. Accepts '0 3 8' (with cohort "
             "root) or '3 8'. Length 1, 2, or 3 post-root.",
    )
    ap.add_argument(
        "--stratification",
        required=True,
        help="Canonical stratification key, e.g. 'NONE', 'ETHNICITY+IMD', "
             "'AGE_CATG+ETHNICITY+IMD'. Run --list-stratifications for "
             "the full set.",
    )
    ap.add_argument(
        "--age",
        nargs="+",
        default=None,
        help="One or more age bands. Single band keeps stored CIs; two or "
             "more bands sum the numerator/denominator within each "
             "stratum and recompute IR and 95%% CI per the supplement "
             "(chi-squared for N<10, Byar's for N>=10). Ignored if the "
             "stratification has no age axis.",
    )
    ap.add_argument(
        "--drop-missing-imd",
        action="store_true",
        help="Filter out rows with imd_missing=True. By default these are "
             "kept (they represent a real demographic stratum).",
    )
    ap.add_argument(
        "--drop-sex-i",
        action="store_true",
        help="Filter out rows with sex=='I'. By default these are kept "
             "(they represent a real demographic category, not a pooled "
             "total).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the long-format CSV.",
    )
    ap.add_argument(
        "--list-stratifications",
        action="store_true",
        help="List available stratification keys and exit.",
    )
    args = ap.parse_args()

    if args.parquet_dir is not None:
        ig.set_data_dir(args.parquet_dir)

    if args.list_stratifications:
        for k in ig.available_stratifications():
            print(k)
        return 0

    # Call the API. ValueError is the user-facing error -- bad sequence,
    # missing stratification, unknown age band etc. -- so we let it bubble
    # up with a clean message rather than a stack trace.
    try:
        df = ig.get_sequence(
            sequence=args.sequence,
            stratification=args.stratification,
            age_bands=args.age,
        )
    except (ValueError, KeyError) as e:
        sys.stderr.write(f"[error] {e}\n")
        return 2
    except FileNotFoundError as e:
        sys.stderr.write(
            f"[error] could not find parquet files. {e}\n"
            "        Set --parquet-dir or the INCIGRAPH_DATA env var.\n"
        )
        return 2

    n_initial = len(df)
    if args.drop_missing_imd and "imd_missing" in df.columns:
        df = df[~df["imd_missing"]].copy()
    if args.drop_sex_i and "sex" in df.columns:
        df = df[df["sex"].astype(str).str.upper() != "I"].copy()
    n_final = len(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    sys.stderr.write(
        f"[info] sequence: {ig.decode_sequence(args.sequence)}\n"
        f"[info] stratification: {args.stratification}\n"
    )
    if args.age:
        if len(args.age) == 1:
            sys.stderr.write(f"[info] age band: {args.age[0]}\n")
        else:
            sys.stderr.write(
                f"[info] age bands pooled: {', '.join(args.age)} "
                "(IR and 95% CI recomputed per supplement)\n"
            )
    if n_final != n_initial:
        sys.stderr.write(
            f"[info] dropped {n_initial - n_final} rows "
            f"(missing-IMD or SEX='I' filters)\n"
        )
    sys.stderr.write(
        f"[info] wrote {n_final} rows -> {args.out}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
