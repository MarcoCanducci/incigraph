"""InciGraph Python API.

Programmatic access to the InciGraph multimorbidity estimates. The seven
public functions in this module are the only things most users will touch:

  load_estimates(length=None)         : the full long-form estimates table
  load_metadata()                     : sequence lookup table
  get_sequence(seq, stratification,   : a single tidy slice ready for plotting
                age_bands=None)
  compute_irr(df, comparison,         : crude IRR with 95% Wald CI
              reference)
  list_sequences(...)                 : filter the metadata for discovery
  decode_sequence(seq)                : "0 3 8" -> "HYPERTENSION -> T2D"
  available_stratifications()         : the 15 stratification keys

All functions accept a `parquet_dir` argument. By default the directory is
resolved from the INCIGRAPH_DATA environment variable, falling back to
'./incigraph_data' in the current working directory. Set
`incigraph.set_data_dir(path)` once at the top of a notebook to avoid
passing it on every call.

Data conventions preserved from the source workbooks (important):
- IMD blank means MISSING (a real stratum, NOT a pooled marginal). Rows
  flagged with `imd_missing=True` appear by default; filter them out
  explicitly if you only want the 1..5 quintiles.
- SEX = 'I' is its own demographic category, NOT a sex-pooled total. It is
  returned by default; filter it out explicitly if you want M/F only.
- Sequence strings always carry the leading '0' cohort-root marker for
  display ("0 3 8"); internally indices 1..64 reference the disease columns.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .ci import poisson_ci, irr_ci, RATE_DENOMINATOR
from .disease_index import (
    DISEASE_NAMES,
    decode_sequence as _decode_sequence_str,
    parse_sequence,
)

# --------------------------------------------------------------------------
# Data-directory resolution. A user can set this once with
# incigraph.set_data_dir(...) or via the INCIGRAPH_DATA env var, and then
# every other function picks it up.
# --------------------------------------------------------------------------
_DATA_DIR: Path | None = None


def set_data_dir(path: str | Path) -> None:
    """Tell the library where the parquet files live for this session."""
    global _DATA_DIR
    _DATA_DIR = Path(path)


def _resolve_dir(parquet_dir: str | Path | None) -> Path:
    if parquet_dir is not None:
        return Path(parquet_dir)
    if _DATA_DIR is not None:
        return _DATA_DIR
    env = os.environ.get("INCIGRAPH_DATA")
    if env:
        return Path(env)
    return Path("./incigraph_data")


def _parquet_path(parquet_dir: Path, length: int) -> Path:
    p = parquet_dir / f"incigraph_L{length}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {p}. Set the parquet directory via "
            "incigraph.set_data_dir(...) or the INCIGRAPH_DATA env var, "
            "and confirm the parquet files have been generated."
        )
    return p


def _metadata_path(parquet_dir: Path) -> Path:
    p = parquet_dir / "incigraph_metadata.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {p}. See set_data_dir() / INCIGRAPH_DATA."
        )
    return p


# --------------------------------------------------------------------------
# load_estimates
# --------------------------------------------------------------------------
def load_estimates(
    sequence_length: int | None = None,
    parquet_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load the full long-form estimates table.

    Parameters
    ----------
    sequence_length : 1, 2, 3, or None (default)
        If 1, 2, or 3, returns only that length's parquet (fastest). If
        None, concatenates all three. For large analyses prefer to call
        with a specific length and filter, rather than loading all 50 M
        rows into memory.
    parquet_dir : path, optional
        Override the resolved data directory.

    Returns
    -------
    DataFrame with the schema described in the package data dictionary.
    """
    pdir = _resolve_dir(parquet_dir)
    if sequence_length in (1, 2, 3):
        return pd.read_parquet(_parquet_path(pdir, sequence_length))
    if sequence_length is None:
        parts = [pd.read_parquet(_parquet_path(pdir, L)) for L in (1, 2, 3)]
        return pd.concat(parts, ignore_index=True)
    raise ValueError(
        f"sequence_length must be 1, 2, 3 or None; got {sequence_length!r}"
    )


# --------------------------------------------------------------------------
# load_metadata
# --------------------------------------------------------------------------
def load_metadata(parquet_dir: str | Path | None = None) -> pd.DataFrame:
    """The lookup table: every sequence in the data and its decoded name."""
    pdir = _resolve_dir(parquet_dir)
    return pd.read_parquet(_metadata_path(pdir))


# --------------------------------------------------------------------------
# available_stratifications
# --------------------------------------------------------------------------
def available_stratifications(
    parquet_dir: str | Path | None = None,
) -> list[str]:
    """Return the stratification keys present in the data, e.g.
    ['NONE', 'AGE_CATG', 'ETHNICITY+IMD', ...]."""
    pdir = _resolve_dir(parquet_dir)
    # read just the column we need from L1 (smallest file). All three lengths
    # share the same set of stratification folders, so L1 is sufficient.
    dset = ds.dataset(_parquet_path(pdir, 1), format="parquet")
    keys = (
        dset.to_table(columns=["stratification_key"])
            .column("stratification_key")
            .to_pandas()
            .astype(str)
            .unique()
            .tolist()
    )
    return sorted(set(keys))


# --------------------------------------------------------------------------
# decode_sequence — re-exported from disease_index for convenience
# --------------------------------------------------------------------------
def decode_sequence(seq) -> str:
    """'0 3 8' -> 'HYPERTENSION -> T2D'."""
    return _decode_sequence_str(seq)


# --------------------------------------------------------------------------
# list_sequences
# --------------------------------------------------------------------------
def list_sequences(
    sequence_length: int | None = None,
    starts_with_disease: str | None = None,
    ends_with_disease: str | None = None,
    contains_disease: str | None = None,
    parquet_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Filter the metadata table to find sequences matching criteria.

    All disease arguments take the short name (case-insensitive):
    e.g. "HYPERTENSION", "T2D", "ckd3-5". `starts_with` matches step_1,
    `ends_with` matches the final (target) step. `contains` matches any
    step in the trajectory.

    Returns a DataFrame with the same schema as load_metadata().
    """
    meta = load_metadata(parquet_dir)
    if sequence_length is not None:
        meta = meta[meta["sequence_length"] == sequence_length]

    def _match(col: str, name: str | None) -> pd.Series:
        if name is None:
            return pd.Series([True] * len(meta), index=meta.index)
        return meta[col].astype(str).str.upper() == name.upper()

    mask = pd.Series([True] * len(meta), index=meta.index)
    if starts_with_disease is not None:
        mask &= _match("step_1_short", starts_with_disease)
    if ends_with_disease is not None:
        # the last populated step in each row
        last = meta[["step_1_short", "step_2_short", "step_3_short"]].apply(
            lambda r: r.dropna().iloc[-1] if r.notna().any() else None, axis=1
        )
        mask &= last.astype(str).str.upper() == ends_with_disease.upper()
    if contains_disease is not None:
        wanted = contains_disease.upper()
        any_step = (
            meta[["step_1_short", "step_2_short", "step_3_short"]]
            .astype(str)
            .apply(lambda col: col.str.upper())
        )
        mask &= any_step.eq(wanted).any(axis=1)
    return meta[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# get_sequence — the headline query function
# --------------------------------------------------------------------------
def get_sequence(
    sequence,
    stratification: str,
    age_bands: list[str] | None = None,
    parquet_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return one tidy slice of estimates ready for plotting or further analysis.

    Parameters
    ----------
    sequence : str or list of int
        "0 3 8" (with or without the leading 0) or [3, 8].
    stratification : str
        Canonical key, e.g. "NONE", "IMD", "ETHNICITY+IMD". See
        `available_stratifications()` for the full set.
    age_bands : list of str, optional
        Age bands to include. If the source files stratify by age
        (`AGE_CATG` is in `stratification`):
          - None        -> all bands returned, one row per band per stratum.
          - ["51-60"]   -> only that band, no recomputation; bounds come
                           from the source.
          - ["51-60", "61-70"] -> bands pooled; numerator and denominator
                           summed within each (other) stratum and the rate
                           plus 95% CI recomputed (chi-squared for N<10,
                           Byar's for N>=10).
    parquet_dir : path, optional

    Returns
    -------
    DataFrame with columns: ethnicity, sex, imd, imd_missing, age_catg
    (NaN if pooled across or not stratified), numerator, denominator,
    incidence_rate, lower_limit, upper_limit, plus metadata columns
    sequence, sequence_length, target_disease_idx, target_disease_short.

    Notes
    -----
    The IMD-missing stratum (imd_missing=True) and the SEX='I' category are
    included by default because they are real strata in the source data.
    Filter them out explicitly if your analysis needs only the IMD 1..5
    quintiles or only M/F.
    """
    indices = parse_sequence(sequence)
    seq_len = len(indices)
    target_idx = indices[-1]
    canonical_seq = "0 " + " ".join(str(i) for i in indices)

    pdir = _resolve_dir(parquet_dir)
    df = pd.read_parquet(_parquet_path(pdir, seq_len))

    # filter to this sequence + stratification. The source_folder column
    # carries the stratification key (canonical, with '+' separators).
    df = df[
        (df["sequence"] == canonical_seq)
        & (df["stratification_key"].astype(str) == stratification)
        & (df["target_disease_idx"] == target_idx)
    ].copy()
    if df.empty:
        raise ValueError(
            f"No rows for sequence={canonical_seq!r}, "
            f"stratification={stratification!r}. Check the available "
            "stratifications and verify the sequence exists in metadata."
        )

    # If the source has age stratification, age_bands controls behaviour.
    has_age = "AGE_CATG" in stratification
    if has_age:
        if age_bands is None:
            pass  # return all bands, one row per (stratum, band)
        else:
            requested = [str(b).strip() for b in age_bands]
            present = set(df["age_catg"].dropna().astype(str).unique())
            missing = [b for b in requested if b not in present]
            if missing:
                raise ValueError(
                    f"age bands {missing!r} not present. Available: "
                    f"{sorted(present)}"
                )
            df = df[df["age_catg"].astype(str).isin(requested)].copy()
            if len(requested) > 1:
                df = _pool_age_bands(df, requested)
    elif age_bands is not None:
        # the stratification has no age axis; warn but proceed
        import warnings
        warnings.warn(
            f"stratification {stratification!r} has no age axis; "
            "age_bands is ignored",
            stacklevel=2,
        )

    # tidy column order, drop the per-row source provenance which is
    # implicit now that we've filtered to a single stratification
    keep_cols = [
        "sequence", "sequence_length",
        "target_disease_idx", "target_disease_short",
        "ethnicity", "sex", "imd", "imd_missing", "age_catg",
        "numerator", "denominator",
        "incidence_rate", "lower_limit", "upper_limit",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def _pool_age_bands(df: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Sum numerators and denominators across the given age bands within
    each (other-stratum) cell, then recompute rate and 95% CI from the
    cumulative counts. Mirrors the existing collect_incigraph behaviour."""
    # group keys are all the stratification columns OTHER than age_catg
    group_keys = [c for c in ("ethnicity", "sex", "imd", "imd_missing")
                  if c in df.columns]
    # carry the metadata through unchanged
    meta_cols = [c for c in ("sequence", "sequence_length",
                             "target_disease_idx", "target_disease_short")
                 if c in df.columns]
    # use dropna=False so the IMD-missing rows (NaN imd) and similar are
    # preserved as their own group
    agg = (
        df.groupby(group_keys + meta_cols, dropna=False)
          [["numerator", "denominator"]]
          .sum()
          .reset_index()
    )
    agg["incidence_rate"] = np.where(
        agg["denominator"] > 0,
        agg["numerator"] / agg["denominator"] * RATE_DENOMINATOR,
        np.nan,
    )
    lo, hi = poisson_ci(agg["numerator"].values, agg["denominator"].values)
    agg["lower_limit"] = lo
    agg["upper_limit"] = hi
    # the pooled rows do not have a single age_catg value
    agg["age_catg"] = pd.NA
    return agg


# --------------------------------------------------------------------------
# compute_irr — the contrast computation function
# --------------------------------------------------------------------------
def compute_irr(
    df: pd.DataFrame,
    comparison: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict:
    """Crude incidence-rate-ratio between two demographic groups.

    Each filter dict picks one row from `df` by exact match on the named
    columns. Example:
        compute_irr(
            df,
            comparison={"imd": 5, "ethnicity": "WHITE"},
            reference={"imd": 1, "ethnicity": "WHITE"},
        )

    Parameters
    ----------
    df : DataFrame
        Output of get_sequence() (or any DataFrame with numerator and
        denominator columns).
    comparison, reference : dict
        Column -> value filters identifying exactly one row each. The same
        column may appear in both with different values (typical).

    Returns
    -------
    dict with keys: irr, log_irr, se_log_irr, lower_ci, upper_ci, z, p_raw,
    n_comp, n_ref, comparison_rate, reference_rate. The CI is on the IRR
    itself (back-transformed from log scale). If either filter matches
    zero or more than one row, ValueError is raised.

    Notes
    -----
    The SE formula assumes independent Poisson counts with their own
    person-time. Adequate for n_comp >= 30 and n_ref >= 30; deteriorates
    at lower counts. Crude IRR — no adjustment for confounding, competing
    risks, or coding differentials.
    """
    def _pick_row(d: pd.DataFrame, filt: Mapping[str, object]) -> pd.Series:
        sub = d
        for col, val in filt.items():
            if col not in sub.columns:
                raise KeyError(
                    f"column {col!r} not in DataFrame; available: "
                    f"{list(sub.columns)}"
                )
            if val is None or (isinstance(val, float) and np.isnan(val)):
                sub = sub[sub[col].isna()]
            else:
                # match strings case-insensitively for ethnicity / sex
                if sub[col].dtype.kind in ("O", ) or hasattr(
                    sub[col], "cat"
                ):
                    sub = sub[
                        sub[col].astype(str).str.upper() == str(val).upper()
                    ]
                else:
                    sub = sub[sub[col] == val]
        if len(sub) == 0:
            raise ValueError(
                f"filter {dict(filt)} matched no rows."
            )
        if len(sub) > 1:
            raise ValueError(
                f"filter {dict(filt)} matched {len(sub)} rows; expected 1. "
                "Add more columns to disambiguate."
            )
        return sub.iloc[0]

    rc = _pick_row(df, comparison)
    rr = _pick_row(df, reference)

    result = irr_ci(
        n_comp=float(rc["numerator"]),
        y_comp=float(rc["denominator"]),
        n_ref=float(rr["numerator"]),
        y_ref=float(rr["denominator"]),
    )
    # add the per-100k rates for convenience
    result["comparison_rate"] = float(rc.get("incidence_rate", np.nan))
    result["reference_rate"] = float(rr.get("incidence_rate", np.nan))
    return result


__all__ = [
    "set_data_dir",
    "load_estimates",
    "load_metadata",
    "get_sequence",
    "compute_irr",
    "list_sequences",
    "decode_sequence",
    "available_stratifications",
]
