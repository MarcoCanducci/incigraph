#!/usr/bin/env python3
"""
verify_install.py
=================
One-stop verification that the incigraph package is installed correctly
and behaves as the manuscript claims.

Runs four checks in sequence; stops at the first failure. Each check is
printed with a clear header and a pass/fail summary at the end.

  Check 1 -- Package importable, version present, public API exposed.
  Check 2 -- Unit tests pass (pytest). Catches arithmetic regressions.
  Check 3 -- API smoke test against the real parquet deposit.
  Check 4 -- End-to-end pipeline: Step 1 aggregation consistency.

The script is non-destructive: it reads from your parquet directory but
does not modify it. Outputs (validation pipeline) go to a temp folder
that is cleaned up unless --keep-output is passed.

USAGE
-----
  python verify_install.py
  python verify_install.py --parquet-dir C:\\path\\to\\incigraph_data
  python verify_install.py --skip pytest      # skip unit tests
  python verify_install.py --skip step1       # skip the end-to-end check
  python verify_install.py --keep-output      # don't delete temp output
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ANSI colour codes that work on most modern terminals (including Windows
# Terminal and PowerShell since 2018). Fall back to no-colour if stdout
# isn't a TTY so log files stay readable.
_TTY = sys.stdout.isatty()
def _c(code: str) -> str:
    return code if _TTY else ""

GREEN  = _c("\033[32m")
RED    = _c("\033[31m")
YELLOW = _c("\033[33m")
BLUE   = _c("\033[34m")
BOLD   = _c("\033[1m")
DIM    = _c("\033[2m")
RESET  = _c("\033[0m")


def banner(text: str) -> None:
    """Print a labelled section header."""
    bar = "=" * 72
    print(f"\n{BOLD}{BLUE}{bar}\n{text}\n{bar}{RESET}", flush=True)


def ok(text: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {text}", flush=True)


def fail(text: str) -> None:
    print(f"{RED}[FAIL]{RESET} {text}", flush=True)


def info(text: str) -> None:
    print(f"{DIM}[info]{RESET} {text}", flush=True)


def skip(text: str) -> None:
    print(f"{YELLOW}[skip]{RESET} {text}", flush=True)


# ----------------------------------------------------------------------
# Check 1: package importable
# ----------------------------------------------------------------------
def check_import() -> bool:
    """Confirm `import incigraph` finds the real package (not an empty
    namespace) and that the public API names are present."""
    banner("Check 1 / 4 -- Package importable")
    try:
        import incigraph as ig
    except ImportError as e:
        fail(f"could not import incigraph: {e}")
        print(f"\n  {YELLOW}Fix:{RESET} run `pip install -e .` from the repo root.")
        return False

    expected = {
        "set_data_dir", "load_estimates", "load_metadata", "get_sequence",
        "compute_irr", "list_sequences", "decode_sequence",
        "available_stratifications", "poisson_ci", "irr_ci",
        "DISEASE_NAMES", "N_DISEASES", "__version__",
    }
    actual = set(dir(ig))
    missing = expected - actual
    if missing:
        fail(f"missing public API names: {sorted(missing)}")
        print(f"\n  {YELLOW}Fix:{RESET} your install resolved to an incomplete "
              "package.\n"
              "  This often means there is a stray empty `incigraph/` directory "
              "shadowing\n"
              "  the real one. Run:\n"
              "    pip uninstall -y incigraph\n"
              "    pip install -e .\n"
              f"  Currently imported from: {getattr(ig, '__file__', '<unknown>')}")
        return False

    info(f"imported from {ig.__file__}")
    info(f"version = {ig.__version__}")
    info(f"public API: {len([n for n in expected if not n.startswith('_')])} "
         "names exposed")
    ok("incigraph importable, all public names present")
    return True


# ----------------------------------------------------------------------
# Check 2: pytest
# ----------------------------------------------------------------------
def check_pytest(repo_root: Path) -> bool:
    """Run the pytest suite. These tests don't need the parquet."""
    banner("Check 2 / 4 -- Unit tests (pytest)")
    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        fail(f"tests directory not found at {tests_dir}")
        print(f"\n  {YELLOW}Fix:{RESET} run this script from the repo root.")
        return False

    info(f"running pytest in {tests_dir}")
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-q",
         "--no-header"],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        fail(f"pytest failed ({proc.returncode}) in {elapsed:.1f}s")
        print("\n  --- pytest stdout ---")
        print(proc.stdout)
        if proc.stderr.strip():
            print("\n  --- pytest stderr ---")
            print(proc.stderr)
        print(f"\n  {YELLOW}Fix:{RESET} a unit test failed. Look at the "
              "traceback above; the\n"
              "  most likely cause is a Python or dependency version mismatch.\n"
              "  Compare your `pip freeze` against the dependencies in "
              "pyproject.toml.")
        return False

    # try to extract the pass count from pytest's summary line
    summary = ""
    for line in proc.stdout.strip().splitlines()[::-1]:
        if "passed" in line or "failed" in line:
            summary = line.strip()
            break
    if summary:
        info(summary)
    info(f"elapsed: {elapsed:.1f}s")
    ok("all unit tests pass")
    return True


# ----------------------------------------------------------------------
# Check 3: API smoke test against the real parquet
# ----------------------------------------------------------------------
def check_api_smoke(parquet_dir: Path) -> bool:
    """Confirm the API can read the parquet and produces sensible results."""
    banner("Check 3 / 4 -- API smoke test against the parquet")
    if not parquet_dir.exists():
        fail(f"parquet directory not found: {parquet_dir}")
        print(f"\n  {YELLOW}Fix:{RESET} either pass --parquet-dir explicitly, "
              "or set\n"
              "  the INCIGRAPH_DATA environment variable, or place the "
              "parquet\n"
              "  files in ./incigraph_data relative to this script.")
        return False

    expected_files = ["incigraph_L1.parquet", "incigraph_L2.parquet",
                      "incigraph_L3.parquet", "incigraph_metadata.parquet"]
    missing_files = [f for f in expected_files
                     if not (parquet_dir / f).exists()]
    if missing_files:
        fail(f"missing parquet files in {parquet_dir}: {missing_files}")
        print(f"\n  {YELLOW}Fix:{RESET} re-download the deposit from Zenodo, or\n"
              "  re-run scripts/incigraph_to_parquet.py against your "
              "xlsx tree.")
        return False

    import incigraph as ig
    ig.set_data_dir(parquet_dir)

    # --- subcheck: stratifications enumerable ---
    try:
        strats = ig.available_stratifications()
    except Exception as e:
        fail(f"available_stratifications() raised: {e}")
        return False
    if not strats:
        fail("no stratifications found in the parquet")
        return False
    info(f"stratifications: {len(strats)} keys -- "
         f"{', '.join(strats[:3])}{'...' if len(strats) > 3 else ''}")

    # --- subcheck: metadata has the right size ---
    try:
        meta = ig.load_metadata()
    except Exception as e:
        fail(f"load_metadata() raised: {e}")
        return False
    n_l1 = int((meta["sequence_length"] == 1).sum())
    if n_l1 != 64:
        fail(f"expected 64 length-1 sequences in metadata; got {n_l1}")
        return False
    info(f"metadata: {len(meta):,} unique sequences "
         f"(L1={n_l1}, L2={int((meta['sequence_length']==2).sum()):,}, "
         f"L3={int((meta['sequence_length']==3).sum()):,})")

    # --- subcheck: a specific query returns a non-empty frame ---
    # We pick a sequence/stratification that should always exist in a
    # complete parquet deposit. If it doesn't, the schema has drifted.
    test_specs = [
        ("HYPERTENSION (length 1)", [3], "ETHNICITY+IMD"),
        ("HYPERTENSION -> T2D (length 2)", [3, 8], "ETHNICITY+IMD"),
    ]
    for label, seq, strat in test_specs:
        try:
            df = ig.get_sequence(seq, stratification=strat)
        except Exception as e:
            fail(f"get_sequence({seq}, {strat!r}) raised: {e}")
            return False
        if len(df) == 0:
            fail(f"get_sequence({seq}, {strat!r}) returned an empty frame")
            return False
        info(f"{label}: {len(df)} rows")

    # --- subcheck: IRR computation produces a finite number ---
    df = ig.get_sequence([3, 8], stratification="ETHNICITY+IMD")
    try:
        r = ig.compute_irr(
            df,
            comparison={"ethnicity": "SOUTH_ASIAN", "imd": 5.0},
            reference={"ethnicity": "WHITE", "imd": 1.0},
        )
    except Exception as e:
        fail(f"compute_irr raised: {e}")
        return False
    if not (r["irr"] > 0):
        fail(f"compute_irr returned non-positive IRR: {r['irr']}")
        return False
    info(f"sample IRR (South Asian/IMD5 vs White/IMD1, HTN -> T2D): "
         f"{r['irr']:.3f}  [{r['lower_ci']:.3f} - {r['upper_ci']:.3f}]")

    ok("API reads the parquet and produces sensible results")
    return True


# ----------------------------------------------------------------------
# Check 4: end-to-end pipeline (Step 1)
# ----------------------------------------------------------------------
def check_step1(repo_root: Path, parquet_dir: Path,
                out_dir: Path) -> bool:
    """Run step1_aggregation_consistency.py and confirm the gate passes."""
    banner("Check 4 / 4 -- End-to-end pipeline (Step 1)")
    script = repo_root / "scripts" / "validation" / "step1_aggregation_consistency.py"
    if not script.exists():
        fail(f"could not find {script}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"running {script.name} ...")
    info(f"  parquet: {parquet_dir}")
    info(f"  out:     {out_dir}")
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script),
         "--parquet-dir", str(parquet_dir),
         "--out-dir", str(out_dir)],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0

    # script writes to stderr by convention; show its log
    log_lines = proc.stderr.strip().splitlines()
    for line in log_lines[-12:]:
        print(f"  {DIM}{line}{RESET}")

    if proc.returncode != 0:
        fail(f"step1 exited with code {proc.returncode} in {elapsed:.1f}s")
        # Distinguish a clean "gate failed" exit from an actual crash.
        # Step 1 prints '[GATE FAIL]' to stderr when the consistency gate
        # fails -- a real-data issue. Anything else with returncode != 0
        # is a script crash (Python traceback, unexpected exception, etc.)
        gate_failed = any("[GATE FAIL]" in line for line in log_lines)
        if gate_failed:
            print(f"\n  {YELLOW}Diagnosis:{RESET} the consistency gate failed.\n"
                  "  This means the parquet contains marginalisable folder pairs\n"
                  "  whose rates disagree beyond numerical tolerance. The\n"
                  "  aggregation_consistency_summary.csv file in your out-dir\n"
                  "  lists which pairs failed.\n"
                  "  This is a data problem, not a script problem.")
        else:
            # Script crash: surface the traceback from stderr so the user
            # can see what blew up.
            print(f"\n  {YELLOW}Diagnosis:{RESET} step1 crashed with an "
                  "exception, not a gate failure.")
            print("  The exception's traceback (from stderr):\n")
            # show the last block of stderr -- typically the Python traceback
            tb_lines = log_lines
            # find where the traceback starts (last 'Traceback' marker)
            for i, line in enumerate(tb_lines):
                if "Traceback" in line:
                    tb_lines = tb_lines[i:]
                    break
            for line in tb_lines[-40:]:
                print(f"    {line}")
        return False

    info(f"elapsed: {elapsed:.1f}s")
    ok("aggregation consistency gate passed")
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", type=Path, default=None,
                    help="Directory with the parquet deposit. Defaults to "
                         "INCIGRAPH_DATA env var or ./incigraph_data.")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["import", "pytest", "smoke", "step1"],
                    help="Skip a check. Can be passed multiple times.")
    ap.add_argument("--keep-output", action="store_true",
                    help="Don't delete the step1 output directory at the end.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    parquet_dir = (
        args.parquet_dir
        or (Path(os.environ["INCIGRAPH_DATA"])
            if "INCIGRAPH_DATA" in os.environ else None)
        or (repo_root / "incigraph_data")
    )

    t_start = time.monotonic()
    out_dir = Path(tempfile.mkdtemp(prefix="incigraph_verify_"))
    results: dict[str, str] = {}

    try:
        # Check 1
        if "import" in args.skip:
            results["import"] = "skipped"
            skip("Check 1 -- import (--skip)")
        else:
            results["import"] = "PASS" if check_import() else "FAIL"
            if results["import"] != "PASS":
                return _summary(results, out_dir, args, t_start)

        # Check 2
        if "pytest" in args.skip:
            results["pytest"] = "skipped"
            skip("Check 2 -- pytest (--skip)")
        else:
            results["pytest"] = "PASS" if check_pytest(repo_root) else "FAIL"
            if results["pytest"] != "PASS":
                return _summary(results, out_dir, args, t_start)

        # Check 3
        if "smoke" in args.skip:
            results["smoke"] = "skipped"
            skip("Check 3 -- API smoke test (--skip)")
        else:
            results["smoke"] = (
                "PASS" if check_api_smoke(parquet_dir) else "FAIL"
            )
            if results["smoke"] != "PASS":
                return _summary(results, out_dir, args, t_start)

        # Check 4
        if "step1" in args.skip:
            results["step1"] = "skipped"
            skip("Check 4 -- step1 (--skip)")
        else:
            results["step1"] = (
                "PASS" if check_step1(repo_root, parquet_dir, out_dir)
                else "FAIL"
            )

        return _summary(results, out_dir, args, t_start)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted by user.{RESET}")
        return 130
    finally:
        # cleanup if --keep-output wasn't passed and the summary didn't already
        # handle it (we may exit via KeyboardInterrupt before the summary runs)
        if not args.keep_output and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


def _summary(results: dict, out_dir: Path, args, t_start: float) -> int:
    banner("Summary")
    for name in ("import", "pytest", "smoke", "step1"):
        if name not in results:
            continue
        status = results[name]
        if status == "PASS":
            colour, marker = GREEN, "PASS"
        elif status == "FAIL":
            colour, marker = RED, "FAIL"
        else:
            colour, marker = YELLOW, "skip"
        print(f"  {colour}[{marker}]{RESET} {name}")

    elapsed = time.monotonic() - t_start
    print(f"\n  total elapsed: {elapsed:.1f}s")
    if args.keep_output:
        print(f"  outputs kept at: {out_dir}")

    all_pass = all(v == "PASS" for v in results.values()
                   if v != "skipped")
    if all_pass and "FAIL" not in results.values():
        print(f"\n  {GREEN}{BOLD}All checks passed.{RESET}\n")
        return 0
    print(f"\n  {RED}{BOLD}One or more checks failed.{RESET}  "
          "See the messages above for the fix.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
