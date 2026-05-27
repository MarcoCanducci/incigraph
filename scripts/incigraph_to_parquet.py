#!/usr/bin/env python3
"""
incigraph_to_parquet.py
=======================
One-shot converter that turns the full InciGraph xlsx tree into a small set
of partitioned parquet files for public distribution.

Why
---
The xlsx tree is ~1.2 GB across thousands of workbooks. Programmatic access
is slow (Excel parsing per sheet) and hostile to anything that isn't pandas.
A flat tidy parquet table is ~200-400 MB total, loads in seconds, and is
trivially queryable from any modern data tool.

Output layout
-------------
  <out-dir>/incigraph_L1.parquet        # length-1 estimates
  <out-dir>/incigraph_L2.parquet        # length-2 estimates
  <out-dir>/incigraph_L3.parquet        # length-3 estimates
  <out-dir>/incigraph_metadata.parquet  # small lookup table: every sequence
                                        # that appears in the data, with its
                                        # human-readable name
  <out-dir>/incigraph_to_parquet.json   # provenance sidecar

Each row in the L1/L2/L3 files = one estimate (one cell of one disease
column in one stratum of one sequence workbook).

Schema
------
  sequence              str            "0 3" or "0 3 8 9" (cohort-root prefix)
  sequence_length       int8           1, 2 or 3
  step_1_idx ... step_3_idx  int16     1-based disease index; null for shorter
                                       sequences
  target_disease_idx    int16          the disease whose incidence is reported
                                       (equals step_{sequence_length}_idx)
  target_disease_short  category       e.g. "HYPERTENSION"  (stripped form)
  target_disease_raw    str            the original "BD_MEDI:..." column header
  stratification_key    category       canonical, e.g. "ETHNICITY+IMD"; "NONE"
                                       for unstratified
  ethnicity             category       nullable when not stratified
  sex                   category       nullable; "I" is its OWN category
                                       (not a pooled total)
  imd                   float32        1..5; null when not stratified OR when
                                       the row is the IMD-missing row (see
                                       imd_missing)
  imd_missing           bool           true ONLY for the blank-IMD row, which
                                       represents IMD-missing patients (a real
                                       stratum, not a pooled total)
  age_catg              category       nullable; preserved verbatim
  numerator             int32          event count; null for suppressed cells
  denominator           float64        person-years
  incidence_rate        float64        per 100,000 PY
  lower_limit           float64        95% CI lower
  upper_limit           float64        95% CI upper
  source_folder         category       provenance: which stratification folder

The metadata file has one row per unique (sequence, sequence_length) and
contains the human-readable trajectory:
  sequence              str            "0 3 8"
  sequence_length       int8           2
  step_1_short ... step_3_short  category   short names; nullable
  sequence_decoded      str            "HYPERTENSION -> T2D"

Usage
-----
  python incigraph_to_parquet.py \
      --root /path/to/InciGraph \
      --out-dir ./parquet_out \
      [--workers 4] [--resume] [--no-verify]

Conventions baked in
--------------------
- IMD blank means MISSING, not pooled.
- SEX = 'I' is its own category, not a sex-pooled total.
- The cohort-root 0 is preserved as the leading element of `sequence`.
- Folder schema fingerprints are computed and checked; halt on mismatch.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------
# Disease index <-> name mapping. 1-based; index N corresponds to the
# disease column whose BD_MEDI header ends in ":N".
# --------------------------------------------------------------------------
DISEASE_NAMES = [
    "HF", "AF", "HYPERTENSION", "PAD", "VALVULAR", "AORTIC_ANEURYSM",
    "T1DM", "T2D", "CKD3-5", "DEPRESSION", "ANXIETY", "BIPOLAR",
    "EATING_DISORDERS", "SCHIZOPHRENIA", "AUTISM",
    "LIVER_ALCOHOL", "NAFLD", "OTHER_LIVER", "DEMENTIA", "PARKINSONS",
    "EPILEPSY", "CANCER", "ASTHMA", "COPD", "OSA",
    "ECZEMA", "ALLERGIC_RHINITIS", "HIV", "OSTEOPOROSIS", "OSTEOARTHRITIS",
    "RA", "GOUT", "SLE", "SJOGRENS", "SSc",
    "PMR_GCA", "ENDOMETRIOSIS", "HYPOTHYROIDISM", "HYPERTHYROIDISM",
    "ADDISON", "MS", "VISUAL_IMPAIRMENT", "MENIERES",
    "PERIPHERAL_NEUROPATHY", "DOWNS",
    "PERNICIOUS_ANAEMIA", "PSORIASIS", "PSORIATIC_ARTHRITIS", "ILD", "PTSD",
    "IBS", "HEARING_LOSS", "LEARNING_DISABILITY", "ENDOMETRIOSIS_V2", "PCOS",
    "SICKLE_CELL", "HAEMOCHROMATOSIS", "STROKE", "IHD", "DRUG_ALCOHOL",
    "IBD", "HAEM_CANCER", "BRONCHIECTASIS", "FIBROMYALGIA_CFS",
]
N_DISEASES = len(DISEASE_NAMES)

# Workbook sheet names.  All five must exist in every workbook.
SHEETS = ("Incidence", "Numerator", "Denominator", "Upper Limit", "Lower Limit")

# Demographic key columns the converter will recognise. Detection is by
# header match, not by position.
DEMO_COLS = ("ETHNICITY", "SEX", "IMD", "AGE_CATG")


# --------------------------------------------------------------------------
# Disease-header normalisation (same logic as the existing collect script,
# kept here so this converter has no project-internal dependencies).
# --------------------------------------------------------------------------
_HDR_STRIPS = (
    "_BHAM_CAM_FINAL", "_BIRM_CAM_V2", "_MM_BIRM_CAM", "_BHAM_CAM",
    "_BIRM_CAM", "_PUSHASTHMA", "_TOT", "_OPTIMAL",
    "_11_3_21", "_120421", "_SH_20092020", "_DRAFT_V1", "_V2",
    "_2021", "_STRICT",
)

def short_name(raw_header):
    """'BD_MEDI:TYPE2DIABETES_BHAM_CAM:8' -> 'TYPE2DIABETES'. The trailing
    ':N' (if any) is stripped along with the cohort-version tokens.

    Also handles the '_TOT' suffix used for diseases added later that don't
    have a trailing ':N' (e.g. 'BD_MEDI:BRONCHIECTASIS_TOT' -> 'BRONCHIECTASIS').
    Falls back to the raw string if parsing fails.

    This function does ONLY mechanical cleanup. Disease-specific renames
    (CANCER, SCHIZOPHRENIA, etc.) and collision resolution happen in
    build_canonical_disease_map(), where they are visible and auditable.
    """
    try:
        core = raw_header.split(":", 1)[1]
    except (IndexError, AttributeError):
        return str(raw_header)
    # strip any trailing ':N' that follows the disease-name token
    core = re.sub(r":\d+$", "", core)
    for tok in _HDR_STRIPS:
        core = core.replace(tok, "")
    # strip the '_TOT' totalled-cohort suffix used for later-added diseases
    if core.endswith("_TOT"):
        core = core[:-4]
    return core.strip("_")


# Explicit name overrides applied AFTER short_name() in
# build_canonical_disease_map(). Keys are the mechanically-cleaned names that
# short_name() produces; values are the canonical short names used in the
# parquet, the API, and the manuscript figures.
#
# Add to this map when a cohort-name convention differs from the manuscript
# label, NOT when there's actual semantic ambiguity in the underlying disease.
_NAME_OVERRIDES = {
    "ALLCA_NOBCC_VFINAL":              "CANCER",
    "SCHIZOPHRENIAMM":                 "SCHIZOPHRENIA",
    "PSORIATICARTHRITIS2021":          "PSORIATIC_ARTHRITIS",
    "DRUGALCOHOL":                     "DRUG_ALCOHOL",
    "ANY_DEAFNESS_HEARING_LOSS":       "HEARING_LOSS",
    "POLYCYSTIC_OVARIAN_SYNDROME_PCOS": "PCOS",
    "SICKLE_CELL_DISEASE":             "SICKLE_CELL",
    "ALL_DEMENTIA":                    "DEMENTIA",
    "TYPE1DM":                         "T1DM",
    "TYPE2DIABETES":                   "T2D",
    "CKDSTAGE3TO5":                    "CKD3-5",
    "VALVULARDISEASES":                "VALVULAR",
    "AORTICANEURYSM":                  "AORTIC_ANEURYSM",
    "EATINGDISORDERS":                 "EATING_DISORDERS",
    "ATOPICECZEMA":                    "ECZEMA",
    "ALLERGICRHINITISCONJ":            "ALLERGIC_RHINITIS",
    "HIVAIDS":                         "HIV",
    "RHEUMATOIDARTHRITIS":             "RA",
    "SYSTEMIC_LUPUS_ERYTHEMATOSUS":    "SLE",
    "SJOGRENSSYNDROME":                "SJOGRENS",
    "SYSTEMIC_SCLEROSIS":              "SSc",
    "PMRANDGCA":                       "PMR_GCA",
    "ENDOMETRIOSIS_ADENOMYOSIS":       "ENDOMETRIOSIS",  # collision: see below
    "ADDISON_DISEASE":                 "ADDISON",
    "MENIERESDISEASE":                 "MENIERES",
    "DOWNSSYNDROME":                   "DOWNS",
    "PERNICIOUSANAEMIA":               "PERNICIOUS_ANAEMIA",
    "PTSDDIAGNOSIS":                   "PTSD",
    "PREVALENT_IBS":                   "IBS",
    "LEARNINGDISABILITY":              "LEARNING_DISABILITY",
    "HAEMATOLOGICALCANCER":            "HAEM_CANCER",
    "PAD":                             "PAD",  # explicit no-change, for the audit trail
    "ILD":                             "ILD",
    "STROKE":                          "STROKE",
    "IHD":                             "IHD",
    "IBD":                             "IBD",
    "BRONCHIECTASIS":                  "BRONCHIECTASIS",
    "FIBROMYALGIA_CFS":                "FIBROMYALGIA_CFS",
}


# ----------------------------------------------------------------------
# Disease index = column-position in the CANONICAL workbook (1-based).
#
# The original InciGraph file tree uses column position as the disease
# identifier (S0_3 means "condition on whatever disease is at column
# position 3 in the canonical workbook"). The trailing ':N' in BD_MEDI
# headers is a legacy artifact of an earlier numbering scheme and is NOT
# reliable: indices can be 0-based, sparse (with gaps), and the most
# recently added diseases have no ':N' at all (just an '_TOT' suffix).
#
# We build the canonical map once from the first usable workbook and use
# it everywhere downstream. The map is also persisted in the JSON sidecar
# so users can reproduce the index <-> name mapping exactly.
# ----------------------------------------------------------------------
_DISEASE_INDEX_MAP: dict[str, int] = {}  # short_name -> 1-based position
_HEADER_TO_NAME: dict[str, str]    = {}  # raw header -> canonical short_name


def _canonicalise(raw: str, used: dict[str, int], pos: int) -> str:
    """Mechanical name + override + collision disambiguation.

    `used` is the mapping built so far (canonical_name -> position); we
    consult it to detect collisions. Returns the canonical short name to
    use for this raw header at this position."""
    mech = short_name(raw)
    name = _NAME_OVERRIDES.get(mech, mech)
    if name in used:
        # cohort-variant disambiguation
        if "_V2" in raw:
            name = f"{name}_V2"
        elif "_V3" in raw:
            name = f"{name}_V3"
        else:
            tail = re.sub(r"[^A-Z0-9]+", "_",
                          raw.split(":", 1)[1].upper())[-12:].strip("_")
            name = f"{name}_{tail}"
    return name


def build_canonical_disease_map(root: str) -> dict[str, int]:
    """Open the unstratified S0.xlsx (or the first workbook found with the
    full disease set) and record the short_name of every BD_MEDI column,
    in order. The 1-based position in this ordering IS the disease index
    we use everywhere.

    Strategy:
      1) Prefer the 'NONE' (unstratified) folder's S0.xlsx if present.
      2) Otherwise, scan all folders' root S0.xlsx workbooks and pick the
         one with the largest BD_MEDI column count.
      3) If multiple workbooks tie for the largest count, prefer the one
         from the folder with the simplest stratification.
    """
    best_path, best_n = None, 0
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        # try the top-level S0.xlsx in this folder
        candidate = os.path.join(folder, "S0.xlsx")
        if not os.path.exists(candidate):
            continue
        try:
            head = pd.read_excel(candidate, sheet_name="Incidence", nrows=0)
        except Exception:
            continue
        bd_cols = [c for c in head.columns if str(c).startswith("BD_MEDI:")]
        if len(bd_cols) > best_n:
            best_n, best_path = len(bd_cols), candidate
    if best_path is None:
        raise RuntimeError(
            "Could not find any S0.xlsx workbook to build the canonical "
            "disease list from."
        )

    head = pd.read_excel(best_path, sheet_name="Incidence", nrows=0)
    bd_cols = [c for c in head.columns if str(c).startswith("BD_MEDI:")]
    mapping = {}
    seen = []
    for pos, raw in enumerate(bd_cols, start=1):
        mech_before = short_name(raw)
        name = _canonicalise(raw, mapping, pos)
        if name != _NAME_OVERRIDES.get(mech_before, mech_before):
            sys.stderr.write(
                f"[info] disambiguated '{mech_before}' at canonical position "
                f"{pos} -> '{name}' (raw: {raw})\n"
            )
        mapping[name] = pos
        _HEADER_TO_NAME[raw] = name
        seen.append((pos, name, raw))
    sys.stderr.write(
        f"[info] canonical disease map built from {best_path}\n"
        f"[info] {len(mapping)} diseases indexed 1..{len(mapping)} by column position\n"
    )
    # print the first few and the last few so the user can spot anything odd
    if len(seen) > 8:
        preview = seen[:4] + [(0, "...", "...")] + seen[-4:]
    else:
        preview = seen
    for pos, name, raw in preview:
        if name == "...":
            sys.stderr.write("       ...\n")
        else:
            sys.stderr.write(f"       {pos:3d}  {name}  ({raw})\n")
    return mapping


def disease_index_for(raw_header: str) -> int | None:
    """Return the 1-based canonical disease index for a BD_MEDI header,
    by looking up its short_name in the canonical map. Returns None if
    the disease is not in the map (this indicates schema drift)."""
    return _DISEASE_INDEX_MAP.get(short_name(raw_header))


# --------------------------------------------------------------------------
# Folder / workbook discovery
# --------------------------------------------------------------------------
def discover_stratification_folders(root):
    """Return sorted list of immediate subdirectories that contain at least
    one S0*.xlsx workbook."""
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        # any workbook at depth 1 or 2 satisfies the convention
        has = False
        for entry in os.listdir(full):
            if entry.startswith("S0") and entry.endswith(".xlsx"):
                has = True; break
            sub = os.path.join(full, entry)
            if os.path.isdir(sub):
                for child in os.listdir(sub):
                    if child.startswith("S0") and child.endswith(".xlsx"):
                        has = True; break
                if has:
                    break
        if has:
            out.append(name)
    return out


def discover_workbooks(folder_root):
    """Walk a stratification folder and yield (path, sequence_length,
    sequence_indices) for every workbook found.

    The sequence convention:
      <folder>/S0.xlsx                        -> length 1
      <folder>/S0_<d1>/S0_<d1>.xlsx           -> length 2
      <folder>/S0_<d1>/S0_<d1>_<d2>.xlsx      -> length 3
    The leading file in length-2/3 folders is the length-2 workbook; the
    flatter file with a second underscore-separated index is length-3.
    `sequence_indices` is the post-cohort-root tuple, e.g. (3, 8) for
    HTN -> T2D.
    """
    pattern = re.compile(r"^S0(?:_(\d+))?(?:_(\d+))?\.xlsx$")
    for dirpath, _dirnames, filenames in os.walk(folder_root):
        for fname in filenames:
            m = pattern.match(fname)
            if not m:
                continue
            full = os.path.join(dirpath, fname)
            d1, d2 = m.group(1), m.group(2)
            if d1 is None:
                yield full, 1, ()
            elif d2 is None:
                yield full, 2, (int(d1),)
            else:
                yield full, 3, (int(d1), int(d2))


# --------------------------------------------------------------------------
# Schema fingerprint
# --------------------------------------------------------------------------
def fingerprint_disease_columns(disease_columns):
    """Hash the sorted list of BD_MEDI column headers in a workbook.  Used
    to detect schema drift between folders."""
    sorted_headers = sorted(disease_columns)
    h = hashlib.sha256()
    for c in sorted_headers:
        h.update(c.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------
# Per-workbook conversion: read 5 sheets, melt into long form
# --------------------------------------------------------------------------
def melt_one_workbook(path, sequence_length, seq_indices, stratification_key,
                     source_folder):
    """Read one workbook and return a long-form DataFrame: one row per
    (target disease column x stratum).  Returns None if the workbook is
    unreadable or has the wrong sheet set."""
    try:
        sheets = pd.read_excel(path, sheet_name=list(SHEETS), header=0)
    except Exception as e:
        sys.stderr.write(f"[warn] could not read {path}: {e}\n")
        return None
    if not all(s in sheets for s in SHEETS):
        sys.stderr.write(f"[warn] {path} missing one or more sheets; skipping\n")
        return None

    # find demographic key columns (those appearing before the first
    # BD_MEDI: column) in the Incidence sheet
    cols = list(sheets["Incidence"].columns)
    try:
        first_disease_col = next(c for c in cols if str(c).startswith("BD_MEDI:"))
    except StopIteration:
        sys.stderr.write(f"[warn] {path} has no BD_MEDI columns; skipping\n")
        return None
    first_disease_idx = cols.index(first_disease_col)
    key_cols = [c for c in cols[:first_disease_idx] if c.upper() in DEMO_COLS]
    disease_cols = cols[first_disease_idx:]

    # melt each sheet into long form, then join on (key_cols, target_disease)
    def melt(sheet_name, value_name):
        df = sheets[sheet_name][key_cols + disease_cols].melt(
            id_vars=key_cols,
            value_vars=disease_cols,
            var_name="target_disease_raw",
            value_name=value_name,
        )
        return df

    inc = melt("Incidence",     "incidence_rate")
    num = melt("Numerator",     "numerator")
    den = melt("Denominator",   "denominator")
    up  = melt("Upper Limit",   "upper_limit")
    lo  = melt("Lower Limit",   "lower_limit")

    df = inc.merge(num, on=key_cols + ["target_disease_raw"])
    df = df.merge(den, on=key_cols + ["target_disease_raw"])
    df = df.merge(up,  on=key_cols + ["target_disease_raw"])
    df = df.merge(lo,  on=key_cols + ["target_disease_raw"])

    # decode the target disease using the canonical column-position map.
    # Prefer the header->canonical-name cache (built from the canonical
    # workbook) for exact correctness on V2 and other collision cases.
    # Headers that weren't in the canonical workbook fall back to mechanical
    # cleanup + the override table.
    def _canonical_name(raw_header):
        if raw_header in _HEADER_TO_NAME:
            return _HEADER_TO_NAME[raw_header]
        mech = short_name(raw_header)
        return _NAME_OVERRIDES.get(mech, mech)

    df["target_disease_short"] = df["target_disease_raw"].map(_canonical_name)
    df["target_disease_idx"]   = df["target_disease_short"].map(_DISEASE_INDEX_MAP)
    # any rows whose disease isn't in the canonical map are skipped, with a
    # warning the first time we see each unknown name (schema drift signal).
    unknown = df[df["target_disease_idx"].isna()]
    if len(unknown):
        for name in unknown["target_disease_short"].unique():
            sys.stderr.write(
                f"[warn] {path}: disease {name!r} not in canonical map; "
                "rows dropped\n"
            )
        df = df[df["target_disease_idx"].notna()].copy()
    df["target_disease_idx"] = df["target_disease_idx"].astype("int16")

    # sequence columns: store the prior diseases (step_1, step_2) and the
    # final-step disease, which is the target.  sequence_length 1 has only
    # target; sequence_length 3 has step_1, step_2, target.
    n_prior = len(seq_indices)
    for i, idx in enumerate(seq_indices, start=1):
        df[f"step_{i}_idx"] = np.int16(idx)
    # the last step IS the target column we just melted
    df[f"step_{sequence_length}_idx"] = df["target_disease_idx"]
    # pad missing step columns with pd.NA for downstream schema uniformity
    for i in range(sequence_length + 1, 4):
        df[f"step_{i}_idx"] = pd.array([pd.NA] * len(df), dtype="Int16")

    df["sequence_length"] = np.int8(sequence_length)
    df["sequence"] = "0" + "".join(f" {i}" for i in seq_indices) + (
        " " + df["target_disease_idx"].astype(str) if sequence_length >= 1
        else ""
    )
    # rename / normalise key columns
    rename_map = {}
    if "ETHNICITY" in df.columns: rename_map["ETHNICITY"] = "ethnicity"
    if "SEX"       in df.columns: rename_map["SEX"]       = "sex"
    if "IMD"       in df.columns: rename_map["IMD"]       = "imd"
    if "AGE_CATG"  in df.columns: rename_map["AGE_CATG"]  = "age_catg"
    df = df.rename(columns=rename_map)

    # ensure all four demographic columns exist (null where absent so the
    # final parquet schema is uniform across folders)
    for col in ("ethnicity", "sex", "imd", "age_catg"):
        if col not in df.columns:
            df[col] = pd.NA

    # IMD: blank means MISSING (a real stratum, NOT a pooled total).  We
    # encode this with an explicit boolean flag, leaving `imd` numeric for
    # the 1..5 rows.
    df["imd_missing"] = df["imd"].isna() if "imd" in df.columns else True
    # The folder may not even stratify by IMD; in that case `imd_missing`
    # should be False (the row is not "an IMD-missing patient", it's
    # simply unstratified).  Detect via the stratification key.
    if "IMD" not in stratification_key:
        df["imd_missing"] = False
        df["imd"] = pd.NA

    # types and column ordering
    df["stratification_key"] = stratification_key
    df["source_folder"] = source_folder

    out_cols = [
        "sequence", "sequence_length",
        "step_1_idx", "step_2_idx", "step_3_idx",
        "target_disease_idx", "target_disease_short", "target_disease_raw",
        "stratification_key",
        "ethnicity", "sex", "imd", "imd_missing", "age_catg",
        "numerator", "denominator", "incidence_rate",
        "lower_limit", "upper_limit",
        "source_folder",
    ]
    return df[out_cols]


# --------------------------------------------------------------------------
# Canonicalise a folder name to a stratification key.  Conservative: read
# the workbook's key columns to confirm, rather than trust the folder name.
# --------------------------------------------------------------------------
def stratification_key_from_workbook(path):
    """Open one sheet's header row and return the canonical stratification
    key ('ETHNICITY+IMD', etc.) deduced from the demographic columns
    present BEFORE the first BD_MEDI column.  Returns 'NONE' if no demo
    columns are present."""
    try:
        head = pd.read_excel(path, sheet_name="Incidence", nrows=0)
    except Exception as e:
        sys.stderr.write(f"[warn] could not read header of {path}: {e}\n")
        return None
    cols = list(head.columns)
    try:
        first_disease_idx = next(
            i for i, c in enumerate(cols) if str(c).startswith("BD_MEDI:"))
    except StopIteration:
        return None
    demos = [c.upper() for c in cols[:first_disease_idx] if c.upper() in DEMO_COLS]
    # canonical alphabetical concat with '+' separator
    return "+".join(sorted(demos)) if demos else "NONE"


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Path to the InciGraph root containing stratification folders.")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write parquet outputs into.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip stratification folders whose chunk file already "
                         "exists in <out-dir>/_chunks/.  Default re-converts all.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the post-conversion round-trip verification "
                         "(faster, but no reviewer-armour against silent "
                         "schema drift).")
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    chunks_dir = os.path.join(args.out_dir, "_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Stage 1: folder discovery and canonical disease map
    folders = discover_stratification_folders(args.root)
    sys.stderr.write(f"[info] discovered {len(folders)} stratification folders: "
                     f"{folders}\n")

    # Build the canonical disease map ONCE. Every workbook's diseases are
    # resolved by short_name against this map, so the integer indices in the
    # parquet are stable column positions in the canonical (unstratified)
    # workbook -- which matches the convention the file tree itself uses
    # (S0_3 = "condition on disease at canonical position 3").
    global _DISEASE_INDEX_MAP
    _DISEASE_INDEX_MAP = build_canonical_disease_map(args.root)

    fingerprints = {}
    for f in folders:
        for path, sl, _ in discover_workbooks(os.path.join(args.root, f)):
            try:
                head = pd.read_excel(path, sheet_name="Incidence", nrows=0)
            except Exception:
                continue
            disease_cols = [c for c in head.columns if str(c).startswith("BD_MEDI:")]
            if disease_cols:
                # fingerprint by SHORT NAMES (column-order-independent) so
                # length-2/3 workbooks (which legitimately drop one disease)
                # don't trip the check
                short_names = sorted({short_name(c) for c in disease_cols})
                fingerprints[f] = (
                    fingerprint_disease_columns(short_names), len(short_names)
                )
                break

    # If folders disagree on the SET of diseases (not just the column order),
    # warn but continue: the canonical map governs final indexing anyway.
    unique_fps = {fp for fp, _ in fingerprints.values()}
    if len(unique_fps) > 1:
        sys.stderr.write(
            "[warn] folders disagree on the SET of diseases present:\n"
        )
        for f, (fp, n) in fingerprints.items():
            sys.stderr.write(f"        {f}: {n} diseases, fp {fp[:12]}...\n")
        sys.stderr.write(
            "        Proceeding -- the canonical map will resolve names "
            "across folders; rows whose disease isn't in the canonical map "
            "are dropped with a warning.\n"
        )
    schema_fp = next(iter(unique_fps)) if unique_fps else None
    sys.stderr.write(f"[info] schema fingerprint (sha256, sorted short_names): "
                     f"{schema_fp}\n")

    # Stage 2: per-folder streaming melt -> per-folder parquet chunk
    folder_chunk_paths = {}
    total_workbooks = total_rows = 0

    for f in folders:
        chunk_path = os.path.join(chunks_dir, f"{f}.parquet")
        folder_chunk_paths[f] = chunk_path
        if args.resume and os.path.exists(chunk_path):
            sys.stderr.write(f"[skip] {f} already converted -> {chunk_path}\n")
            continue

        sys.stderr.write(f"[info] converting folder: {f}\n")
        # pick the first workbook to deduce the canonical stratification key
        wb_iter = list(discover_workbooks(os.path.join(args.root, f)))
        if not wb_iter:
            sys.stderr.write(f"[warn] {f}: no workbooks found, skipping\n")
            continue
        strat_key = stratification_key_from_workbook(wb_iter[0][0])
        sys.stderr.write(f"       stratification key inferred: {strat_key}\n")

        parts = []
        for i, (path, sl, seq) in enumerate(wb_iter, 1):
            df = melt_one_workbook(path, sl, seq, strat_key, f)
            if df is not None and len(df):
                parts.append(df)
            if i % 50 == 0:
                sys.stderr.write(f"       {i}/{len(wb_iter)} workbooks processed\n")

        if not parts:
            sys.stderr.write(f"[warn] {f}: produced 0 rows, skipping\n")
            continue
        full = pd.concat(parts, ignore_index=True)
        # cast categoricals to keep the per-chunk file small
        for col in ("stratification_key", "source_folder", "target_disease_short",
                    "ethnicity", "sex", "age_catg"):
            if col in full.columns:
                full[col] = full[col].astype("category")

        full.to_parquet(chunk_path, engine="pyarrow", compression="zstd",
                        index=False)
        total_workbooks += len(wb_iter); total_rows += len(full)
        sys.stderr.write(f"       wrote {len(full):,} rows -> {chunk_path}\n")

    # Stage 3: concatenate chunks per sequence_length, write final files
    sys.stderr.write("[info] concatenating chunks per sequence length\n")
    by_length = defaultdict(list)
    for f, path in folder_chunk_paths.items():
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for sl in (1, 2, 3):
            sub = df[df["sequence_length"] == sl]
            if len(sub):
                by_length[sl].append(sub)

    final_paths = {}
    for sl in (1, 2, 3):
        if not by_length[sl]:
            continue
        # Drop any chunks that are empty -- this both silences the pandas
        # FutureWarning about all-NA chunks and is the recommended pattern
        # for forward compatibility with pandas 3.0.
        chunks = [c for c in by_length[sl] if len(c) > 0]
        if not chunks:
            continue
        big = pd.concat(chunks, ignore_index=True)
        # re-coerce categoricals (concat broke them across chunks)
        for col in ("stratification_key", "source_folder", "target_disease_short",
                    "ethnicity", "sex", "age_catg"):
            if col in big.columns:
                big[col] = big[col].astype("category")
        out_path = os.path.join(args.out_dir, f"incigraph_L{sl}.parquet")
        big.to_parquet(out_path, engine="pyarrow", compression="zstd",
                       index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        sys.stderr.write(f"       wrote L{sl}: {len(big):,} rows, "
                         f"{size_mb:.1f} MB -> {out_path}\n")
        final_paths[sl] = out_path

    # Stage 4: metadata table
    sys.stderr.write("[info] building metadata table\n")
    # Invert the canonical map so we can decode integer indices back to
    # short names. This is the source of truth for what each step_N_idx
    # means in the parquet.
    idx_to_name = {pos: name for name, pos in _DISEASE_INDEX_MAP.items()}
    max_known = max(idx_to_name) if idx_to_name else 0

    seqs = set()
    max_idx_seen = 0
    for sl, paths in by_length.items():
        for chunk in paths:
            for s in chunk["sequence"].unique():
                seqs.add((s, sl))
                for tok in str(s).split():
                    if tok != "0":
                        try:
                            n = int(tok)
                            if n > max_idx_seen:
                                max_idx_seen = n
                        except ValueError:
                            pass
    if max_idx_seen > max_known:
        sys.stderr.write(
            f"[warn] data has disease indices up to {max_idx_seen} but the "
            f"canonical map only covers 1..{max_known}. Indices "
            f"{max_known+1}..{max_idx_seen} will be labelled 'disease_NN' "
            "in the metadata.\n"
        )

    def _lookup_name(i):
        return idx_to_name.get(i, f"disease_{i:02d}")

    rows = []
    for s, sl in sorted(seqs):
        idxs = [int(x) for x in s.split() if x != "0"]
        decoded = " -> ".join(_lookup_name(i) for i in idxs)
        row = {"sequence": s, "sequence_length": sl, "sequence_decoded": decoded}
        for i in range(1, 4):
            row[f"step_{i}_short"] = (
                _lookup_name(idxs[i - 1]) if i <= len(idxs) else None
            )
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta["sequence_length"] = meta["sequence_length"].astype("int8")
    for col in (f"step_{i}_short" for i in (1, 2, 3)):
        meta[col] = meta[col].astype("category")
    meta_path = os.path.join(args.out_dir, "incigraph_metadata.parquet")
    meta.to_parquet(meta_path, engine="pyarrow", compression="zstd",
                    index=False)
    sys.stderr.write(f"       wrote metadata: {len(meta):,} sequences -> {meta_path}\n")

    # Stage 5: round-trip verification
    verify_log = []
    if not args.no_verify:
        sys.stderr.write("[info] running round-trip verification on a sample\n")
        for sl, path in final_paths.items():
            sample = pd.read_parquet(path).sample(min(50, len(by_length[sl][0])),
                                                  random_state=42)
            mismatches = 0
            for _, r in sample.iterrows():
                # reconstruct the source xlsx and pick the right column
                folder = r["source_folder"]
                idxs = [int(x) for x in r["sequence"].split() if x != "0"]
                prior = idxs[:-1]
                if len(prior) == 0:
                    wb = os.path.join(args.root, folder, "S0.xlsx")
                elif len(prior) == 1:
                    wb = os.path.join(args.root, folder,
                                       f"S0_{prior[0]}", f"S0_{prior[0]}.xlsx")
                else:
                    wb = os.path.join(args.root, folder,
                                       f"S0_{prior[0]}",
                                       f"S0_{prior[0]}_{prior[1]}.xlsx")
                try:
                    num_sheet = pd.read_excel(wb, sheet_name="Numerator")
                except Exception:
                    continue
                target = r["target_disease_raw"]
                if target not in num_sheet.columns:
                    continue
                # find the row matching the stratum
                mask = pd.Series([True] * len(num_sheet))
                for col, val in [("ETHNICITY", r["ethnicity"]),
                                 ("SEX",       r["sex"]),
                                 ("IMD",       r["imd"]),
                                 ("AGE_CATG",  r["age_catg"])]:
                    if col in num_sheet.columns:
                        if pd.isna(val):
                            mask &= num_sheet[col].isna()
                        else:
                            mask &= num_sheet[col] == val
                hits = num_sheet[mask]
                if not len(hits):
                    continue
                xlsx_n = hits[target].iloc[0]
                if pd.isna(xlsx_n) and pd.isna(r["numerator"]):
                    continue
                if pd.isna(xlsx_n) != pd.isna(r["numerator"]):
                    mismatches += 1
                elif abs(float(xlsx_n) - float(r["numerator"])) > 0.5:
                    mismatches += 1
            verify_log.append({"sequence_length": sl,
                               "sample_checked": len(sample),
                               "mismatches": mismatches})
            sys.stderr.write(f"       L{sl}: {len(sample)} cells checked, "
                             f"{mismatches} mismatches\n")

    # Stage 6: provenance sidecar
    sidecar = {
        "script_name": "incigraph_to_parquet.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "incigraph_root": os.path.abspath(args.root),
        "folders_scanned": folders,
        "schema_fingerprint_sha256": schema_fp,
        "workbooks_processed": total_workbooks,
        "long_estimates_total": total_rows,
        "canonical_disease_map":
            # 1-based position -> short_name; sorted so the JSON is readable
            {str(pos): name for name, pos in sorted(
                _DISEASE_INDEX_MAP.items(), key=lambda kv: kv[1])},
        "outputs": {f"L{sl}": p for sl, p in final_paths.items()},
        "metadata_path": meta_path,
        "verification": verify_log,
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "numpy":  np.__version__,
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    sidecar_path = os.path.join(args.out_dir, "incigraph_to_parquet.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    sys.stderr.write(f"[info] wrote provenance sidecar -> {sidecar_path}\n")

    # cleanup
    sys.stderr.write(f"[info] elapsed: {sidecar['elapsed_seconds']}s; "
                     "you can delete _chunks/ once verified.\n")


if __name__ == "__main__":
    main()
