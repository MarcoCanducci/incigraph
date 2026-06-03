#!/usr/bin/env python3
"""
InciGraph contrast tool (Streamlit app).

A focused tool for comparing the incidence of an ordered disease trajectory
between two demographic groups, where each group can pool several
demographic cells.

Runs locally:
    streamlit run app/streamlit_app.py
or deployed to Streamlit Community Cloud (point it at this file).

Design
------
1. Pick a disease trajectory (1-3 ordered conditions).
2. Build a NUMERATOR group and a DENOMINATOR group, side by side. Each group:
     - picks a stratification scheme (which axes to break down by)
     - for each axis, a multiselect: pick ONE value to fix it, SEVERAL to
       pool them, or leave empty to pool across ALL values of that axis.
3. The app sums numerator and person-time within each group, then reports
   the incidence rate ratio with a 95% CI.

Because the two groups are built independently, both common contrasts fall
out naturally:
  - IMD 1+2 vs IMD 4+5            (both use the IMD scheme; pool {1,2} vs {4,5})
  - Black Female, age 0-41 vs 71+ (both use ETHNICITY_SEX_AGE; fix eth+sex,
                                    pool the two age ranges)

Every rate is crude. The tool is for hypothesis generation and service
planning, not causal inference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from incigraph.ci import irr_ci
from incigraph.disease_index import DISEASE_NAMES


# ======================================================================
# Page config + light clinical styling
# ======================================================================
st.set_page_config(
    page_title="InciGraph Contrast Tool",
    page_icon="\U0001FA7A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important;
                   color: #16314d; letter-spacing: -0.01em; }
      .caveat {
          background: #fff7e6; border-left: 4px solid #d98b00;
          padding: 10px 14px; border-radius: 4px; font-size: 0.86rem;
          color: #5c4400; margin: 12px 0;
      }
      .sparse {
          background: #fbe9e7; border-left: 4px solid #c0392b;
          padding: 10px 14px; border-radius: 4px; font-size: 0.9rem;
          color: #7b241c;
      }
      .metric-big { font-size: 2.6rem; font-weight: 700; color: #16314d;
                    line-height: 1.1; }
      .grp-num { border-top: 3px solid #2a6f97; padding-top: 6px; }
      .grp-den { border-top: 3px solid #99582a; padding-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# Data resolution + cached loaders
# ======================================================================
# ======================================================================
# Data resolution: direct reads from Zenodo with pyarrow filter pushdown
# ======================================================================
#
# Why not download the files locally?
# ---------------------------------------
# Streamlit Community Cloud's free tier has a memory cap around 1 GB. The
# full L3 parquet (94 MB on disk) decompresses to roughly 1.5 GB in memory
# once loaded into pandas, which OOM-kills the container. Downloading the
# files to local disk first only delays the problem -- the moment any user
# calls load_estimates(3) (e.g. picks a length-3 sequence), the load fills
# memory and the worker is killed.
#
# The solution: read directly from the Zenodo file URLs with pyarrow's
# filter pushdown enabled. For a single contrast query, this fetches only
# the row groups containing the relevant sequence (typically a few MB),
# never holds more than a few thousand rows in memory, and never writes
# to local disk.
#
# Streamlit's @st.cache_data wraps the actual reads, so repeated queries
# against the same sequence are instant (the row-group bytes are cached).

ZENODO_RECORD_ID = "20417249"
ZENODO_FILES = {
    "incigraph_L1.parquet":       "08dfb8d2842513a6cfc761a9b1307fc9",
    "incigraph_L2.parquet":       "7884588e7b267ee01cc751d7839181ff",
    "incigraph_L3.parquet":       "e924a952e9db99bdcfa2a66824464170",
    "incigraph_metadata.parquet": "8d0b7ae100db52abacf94bf358b8a1bf",
}


def _zenodo_url(fname: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fname}?download=1"


def _local_data_dir() -> Path | None:
    """If the user is running locally and has the parquets on disk, prefer
    that. Returns the local dir or None if not present."""
    try:
        if "INCIGRAPH_DATA" in st.secrets:
            p = Path(st.secrets["INCIGRAPH_DATA"])
            if p.exists():
                return p
    except Exception:
        pass
    env = os.environ.get("INCIGRAPH_DATA")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    candidate = here / "incigraph_data"
    if candidate.exists() and any(candidate.glob("incigraph_L*.parquet")):
        return candidate
    return None


def _source_for(fname: str, local_dir: Path | None) -> str:
    """Return a path or URL to read `fname` from. Prefer local file if
    present (no network), otherwise the Zenodo URL."""
    if local_dir is not None and (local_dir / fname).exists():
        return str(local_dir / fname)
    return _zenodo_url(fname)


@st.cache_data(show_spinner=False)
def _load_metadata(_local_dir_str: str | None) -> pd.DataFrame:
    """Load the small metadata parquet (~1 MB). Cached for the session."""
    import pyarrow.parquet as pq
    source = _source_for("incigraph_metadata.parquet",
                         Path(_local_dir_str) if _local_dir_str else None)
    if source.startswith("http"):
        import fsspec
        fs = fsspec.filesystem("https")
        with fs.open(source, mode="rb") as fh:
            return pq.read_table(fh).to_pandas()
    return pq.read_table(source).to_pandas()


@st.cache_data(show_spinner=False)
def _available_strats_from_l1(_local_dir_str: str | None) -> list[str]:
    """Read the stratification_key column of L1 (smallest data file) to
    enumerate the available schemes. This is one column over ~50K rows --
    less than 100 KB of data over the network."""
    import pyarrow.parquet as pq
    source = _source_for("incigraph_L1.parquet",
                         Path(_local_dir_str) if _local_dir_str else None)
    if source.startswith("http"):
        import fsspec
        fs = fsspec.filesystem("https")
        with fs.open(source, mode="rb") as fh:
            tbl = pq.read_table(fh, columns=["stratification_key"])
    else:
        tbl = pq.read_table(source, columns=["stratification_key"])
    keys = tbl.column("stratification_key").to_pylist()
    return sorted({str(k) for k in keys if k is not None})


@st.cache_data(show_spinner="Fetching data...")
def _read_sequence_filtered(sequence_length: int, sequence: str,
                            stratification: str,
                            _local_dir_str: str | None) -> pd.DataFrame:
    """Read only the rows matching this (sequence, stratification) tuple,
    using pyarrow's filter pushdown. The two equality filters become
    row-group statistics pruning at the parquet layer, so we typically
    fetch only a few MB even from the 94 MB L3 file.

    Streamlit's @st.cache_data caches the result, so repeated queries
    against the same sequence are instant.
    """
    import pyarrow.parquet as pq

    fname = f"incigraph_L{sequence_length}.parquet"
    source = _source_for(fname,
                         Path(_local_dir_str) if _local_dir_str else None)
    # Only the columns we actually need for the contrast UI.
    cols = ["sequence", "stratification_key", "target_disease_idx",
            "target_disease_short", "ethnicity", "sex",
            "imd", "imd_missing", "age_catg",
            "numerator", "denominator",
            "incidence_rate", "lower_limit", "upper_limit"]
    filters = [("sequence", "=", sequence),
               ("stratification_key", "=", stratification)]
    if source.startswith("http"):
        import fsspec
        fs = fsspec.filesystem("https")
        with fs.open(source, mode="rb") as fh:
            tbl = pq.read_table(fh, columns=cols, filters=filters)
    else:
        tbl = pq.read_table(source, columns=cols, filters=filters)
    return tbl.to_pandas()


# Map stratification keys to readable labels
STRAT_LABELS = {
    "NONE": "No breakdown (overall)",
    "AGE_CATG": "Age",
    "ETHNICITY": "Ethnicity",
    "IMD": "Deprivation (IMD)",
    "SEX": "Sex",
    "ETHNICITY+IMD": "Ethnicity \u00d7 deprivation",
    "AGE_CATG+ETHNICITY": "Age \u00d7 ethnicity",
    "AGE_CATG+IMD": "Age \u00d7 deprivation",
    "AGE_CATG+SEX": "Age \u00d7 sex",
    "ETHNICITY+SEX": "Ethnicity \u00d7 sex",
    "IMD+SEX": "Deprivation \u00d7 sex",
    "ETHNICITY+IMD+SEX": "Ethnicity \u00d7 deprivation \u00d7 sex",
    "AGE_CATG+ETHNICITY+IMD": "Age \u00d7 ethnicity \u00d7 deprivation",
    "AGE_CATG+ETHNICITY+SEX": "Age \u00d7 ethnicity \u00d7 sex",
    "AGE_CATG+IMD+SEX": "Age \u00d7 deprivation \u00d7 sex",
}

AXIS_COL = {"AGE_CATG": "age_catg", "ETHNICITY": "ethnicity",
            "IMD": "imd", "SEX": "sex"}
AXIS_LABEL = {"AGE_CATG": "Age band", "ETHNICITY": "Ethnicity",
              "IMD": "Deprivation (IMD)", "SEX": "Sex"}
POOLABLE_OK = {"AGE_CATG", "IMD"}


def strat_label(key):
    return STRAT_LABELS.get(key, key)


CAVEAT_HTML = (
    '<div class="caveat"><b>Interpretation note.</b> These are crude '
    "incidence rates and rate ratios, intended for hypothesis generation "
    "and service planning. They are <b>not</b> adjusted for confounding, "
    "competing risks, or differential recording between groups. Treat large "
    "ratios as signals to investigate, not as causal effects.</div>"
)


# ======================================================================
# Sidebar: data source
# ======================================================================
st.sidebar.title("InciGraph")
st.sidebar.caption("Multimorbidity incidence contrast tool")

# Resolve where the data lives. If a local copy is present (a collaborator
# running on their own machine, or a self-hosted deployment), prefer that.
# Otherwise we read directly from the public Zenodo deposit; no download
# happens up front -- each query fetches only the row group it needs.
local_dir = _local_data_dir()
local_dir_str = str(local_dir) if local_dir is not None else None

if local_dir is not None:
    st.sidebar.success(f"Reading data from local folder:\n`{local_dir}`")
else:
    st.sidebar.info(
        f"Reading data directly from Zenodo "
        f"(DOI 10.5281/zenodo.{ZENODO_RECORD_ID}). "
        "Each query fetches only the rows it needs."
    )

try:
    # Loading metadata is cheap (~1 MB). It's also the canary: if this
    # fails, we can't reach Zenodo and the rest of the app is unusable.
    META = _load_metadata(local_dir_str)
    STRATS = _available_strats_from_l1(local_dir_str)
except Exception as e:  # noqa: BLE001
    st.error(
        f"Could not load the InciGraph metadata: {e}\n\n"
        "If you are on Streamlit Community Cloud, the Zenodo record may be "
        "temporarily unreachable. Try refreshing in a minute. If the error "
        "persists, check that the Zenodo deposit "
        f"(10.5281/zenodo.{ZENODO_RECORD_ID}) is published."
    )
    st.stop()

st.sidebar.metric("Disease conditions", len(DISEASE_NAMES))
st.sidebar.metric("Stratification schemes", len(STRATS))

DISEASE_DISPLAY = [n.replace("_", " ").title() for n in DISEASE_NAMES]
NAME_TO_IDX = {disp: i + 1 for i, disp in enumerate(DISEASE_DISPLAY)}
IDX_TO_DISPLAY = {i + 1: disp for i, disp in enumerate(DISEASE_DISPLAY)}


# ======================================================================
# Display constants for axes (labels, references, ordering)
# ======================================================================
import numpy as np  # used by IRR tables and chart helpers

#
# Each demographic axis (Ethnicity, Sex, IMD, Age) has:
#   * a human label for the chart and dropdowns
#   * a default reference value (the manuscript's reporting convention)
#   * a label-renderer that adds context (e.g. "IMD 1 (least deprived)")
#   * an ordering hint so dropdowns/charts present values sensibly
#
# IMD direction follows the manuscript: 1 = least deprived, 5 = most deprived.
# The reference defaults are the standard UK health-inequalities choices.

AXIS_DISPLAY_NAME = {
    "ETHNICITY": "Ethnicity",
    "SEX":       "Sex",
    "IMD":       "Deprivation (IMD)",
    "AGE_CATG":  "Age band",
}

DEFAULT_REFERENCE = {
    "ETHNICITY": "WHITE",
    "SEX":       "M",
    "IMD":       1.0,         # IMD 1 = least deprived
    "AGE_CATG":  "41-50",
}

# Canonical complete universe of values for each demographic axis. The chart
# uses this list (rather than the observed-values list from the data) so that
# every group is always visible on the y-axis -- even if data is missing for
# that group in the current query. None / NaN denotes a missing-value bucket.
AXIS_UNIVERSE: dict[str, list] = {
    "ETHNICITY": ["WHITE", "SOUTH_ASIAN", "BLACK", "MIXED_RACE",
                  "OTHERS", "MISSING"],
    "SEX":       ["M", "F"],
    "IMD":       [1.0, 2.0, 3.0, 4.0, 5.0, None],   # None = IMD missing
    "AGE_CATG":  ["0-16", "17-30", "31-40", "41-50",
                  "51-60", "61-70", "71-80", "81+"],
}

# Three-tier suppression policy (matches the supplement section 2.2 update):
#   N >= EVENTS_RELIABLE         => normal display
#   EVENTS_DISCLOSURE <= N < EVENTS_RELIABLE => shown but de-emphasised
#                                              ("low-power" / shaded bar)
#   N < EVENTS_DISCLOSURE        => suppressed entirely (CPRD disclosure rule)
EVENTS_DISCLOSURE = 5    # below this, the cell must not be displayed
EVENTS_RELIABLE   = 10   # below this, the cell is flagged as low-power


def classify_cell(num_events: float, ref_events: float) -> str:
    """Return one of 'ok' / 'low_power' / 'suppressed' for an IRR cell.

    Suppression policy is two-sided: the strictest tier across the
    numerator and denominator wins. So an N=200 cell contrasted against
    a 6-event reference is 'low_power'; a 6-event cell contrasted
    against a 6-event reference is also 'low_power'; a 3-event cell
    against anything is 'suppressed'.
    """
    n = min(num_events, ref_events)
    if n < EVENTS_DISCLOSURE:
        return "suppressed"
    if n < EVENTS_RELIABLE:
        return "low_power"
    return "ok"

# Canonical age band order (used to sort the dropdown when AGE_CATG is the
# axis). The deposit may store any subset of these; we sort observed values
# according to this canonical list.
AGE_BAND_ORDER = ["0-16", "17-30", "31-40", "41-50",
                  "51-60", "61-70", "71-80", "81+"]


def render_imd_label(v) -> str:
    """Render an IMD value with its deprivation descriptor."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "IMD missing"
    iv = int(v)
    if iv == 1:
        return "IMD 1 (least deprived)"
    if iv == 5:
        return "IMD 5 (most deprived)"
    return f"IMD {iv}"


def render_axis_value(axis: str, v) -> str:
    """Render one value of one axis for display in dropdowns and charts."""
    if axis == "IMD":
        return render_imd_label(v)
    if axis == "SEX":
        if str(v) == "M":
            return "Male"
        if str(v) == "F":
            return "Female"
        if str(v) == "I":
            return "Indeterminate"
        return str(v)
    if axis == "ETHNICITY":
        return str(v).replace("_", " ").title()
    if axis == "AGE_CATG":
        return str(v)
    return str(v)


def order_axis_values(axis: str, observed_values: list) -> list:
    """Return the observed values of an axis in a sensible display order."""
    if axis == "AGE_CATG":
        # Use canonical order for the bands we recognise, then append any
        # other bands (e.g. older release with different cutpoints) sorted.
        canonical = [v for v in AGE_BAND_ORDER if v in observed_values]
        leftover = sorted(set(observed_values) - set(canonical),
                          key=lambda x: str(x))
        return canonical + leftover if canonical else leftover
    if axis == "IMD":
        # numeric ascending, with NaN (missing) at the end
        non_nan = sorted(v for v in observed_values if not pd.isna(v))
        return non_nan + [v for v in observed_values if pd.isna(v)]
    return sorted(observed_values, key=lambda x: str(x))


# ======================================================================
# Data helpers shared by both pages
# ======================================================================
def fetch_sequence_rows(seq: list[int], strat: str) -> pd.DataFrame | None:
    """Read parquet rows for one (sequence, stratification) tuple."""
    sequence_str = "0 " + " ".join(str(i) for i in seq)
    try:
        df = _read_sequence_filtered(
            sequence_length=len(seq), sequence=sequence_str,
            stratification=strat, _local_dir_str=local_dir_str,
        )
        if df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        st.markdown(f'<div class="sparse">No data: {e}</div>',
                    unsafe_allow_html=True)
        return None


def pick_sequence(key_prefix: str, default_first: str = "Hypertension",
                  min_length: int = 1, max_length: int = 3) -> list[int]:
    """Render up to three cascading disease pickers."""
    cols = st.columns(max_length)
    pickers = []
    none_label = "\u2014 none \u2014"
    with cols[0]:
        d1 = st.selectbox(
            "First condition", DISEASE_DISPLAY,
            index=DISEASE_DISPLAY.index(default_first)
            if default_first in DISEASE_DISPLAY else 0,
            key=f"{key_prefix}_d1")
    pickers.append(d1)
    if max_length >= 2:
        with cols[1]:
            d2 = st.selectbox(
                "Then" + (" (optional)" if min_length < 2 else ""),
                ([none_label] if min_length < 2 else []) + DISEASE_DISPLAY,
                index=0, key=f"{key_prefix}_d2")
        pickers.append(d2)
    if max_length >= 3:
        with cols[2]:
            disabled = (max_length >= 2 and pickers[1] == none_label)
            d3 = st.selectbox(
                "Then" + (" (optional)" if min_length < 3 else ""),
                ([none_label] if min_length < 3 else []) + DISEASE_DISPLAY,
                index=0, key=f"{key_prefix}_d3", disabled=disabled)
        pickers.append(d3)
    seq = [NAME_TO_IDX[pickers[0]]]
    for p in pickers[1:]:
        if p != none_label:
            seq.append(NAME_TO_IDX[p])
        else:
            break
    return seq


def render_fix_selections(df: pd.DataFrame, axes_to_use: list[str],
                          key_prefix: str) -> tuple[dict, str]:
    """Render dropdowns to FIX 1-2 demographic axes to single values.

    Returns (chosen_values_dict, human_label). The dict maps axis -> value
    (or None for IMD-missing). The widgets are rendered once; use
    apply_fix_selections to filter a dataframe afterwards.
    """
    chosen = {}
    label_bits = []
    for axis in axes_to_use:
        col = AXIS_COL[axis]
        if col not in df.columns:
            continue
        observed = list(df[col].dropna().unique().tolist())
        if col == "imd" and df.get("imd_missing", pd.Series(dtype=bool)).any():
            observed.append(None)
        ordered = order_axis_values(axis, observed)
        labels = {render_axis_value(axis, v): v for v in ordered}
        default_ref = DEFAULT_REFERENCE.get(axis)
        default_label = render_axis_value(axis, default_ref)
        default_index = (list(labels).index(default_label)
                         if default_label in labels else 0)
        chosen_label = st.selectbox(
            f"Fix {AXIS_DISPLAY_NAME.get(axis, axis)} to",
            list(labels), index=default_index,
            key=f"{key_prefix}_fix_{axis}",
        )
        chosen[axis] = labels[chosen_label]
        label_bits.append(chosen_label)
    return chosen, (", ".join(label_bits) if label_bits else "everyone")


def apply_fix_selections(df: pd.DataFrame,
                         chosen: dict) -> pd.DataFrame:
    """Apply the selections returned by render_fix_selections to a frame."""
    mask = pd.Series(True, index=df.index)
    for axis, value in chosen.items():
        col = AXIS_COL[axis]
        if col not in df.columns:
            continue
        if col == "imd":
            if value is None:
                mask &= df.get("imd_missing", False)
            else:
                mask &= (df["imd"] == value)
        else:
            mask &= (df[col].astype(str) == str(value))
    return df[mask].copy()


def fix_demographic_axes(df: pd.DataFrame, strat: str,
                         key_prefix: str,
                         axes_to_use: list[str]) -> tuple[pd.DataFrame, str]:
    """Convenience wrapper: render selections + apply them. Used by page 1
    where the same dataframe needs both."""
    chosen, label = render_fix_selections(df, axes_to_use, key_prefix)
    return apply_fix_selections(df, chosen), label


def gradient_irr_table(df: pd.DataFrame, gradient_axis: str,
                       reference_value, key_prefix: str
                       ) -> pd.DataFrame:
    """For each value of gradient_axis in the canonical AXIS_UNIVERSE,
    compute the IRR vs the reference_value. Returns a tidy table with
    columns:
      label, value, num_events, num_pt, ref_events, ref_pt,
      incidence_rate, ref_rate, irr, lower_ci, upper_ci, p_value,
      is_reference, power_class, status_note.

    A row exists for every value in AXIS_UNIVERSE[gradient_axis], even
    when the deposit has no matching cells for that group -- in which
    case the row is left with NaN numerator / denominator / IRR. This is
    deliberate: the chart shows the full universe of groups on the y-axis
    so missing data is visually evident.

    power_class is one of:
        'ok'          : both numerator and reference sides have >= 10 events
        'low_power'   : at least one side has 5..9 events (shown but flagged)
        'suppressed'  : at least one side has <5 events (no IRR shown;
                        CPRD disclosure rule)
        'no_data'     : the deposit has no matching row for this group
        'reference'   : this row is the reference value itself (IRR=1)
    """
    col = AXIS_COL[gradient_axis]
    if col not in df.columns:
        return pd.DataFrame()

    # find reference row totals
    def _select(v):
        if col == "imd" and v is None:
            return df[df.get("imd_missing", False)]
        return df[df[col] == v]

    ref_rows = _select(reference_value)
    ref_n = float(ref_rows["numerator"].fillna(0).sum())
    ref_t = float(ref_rows["denominator"].fillna(0).sum())
    ref_has_data = (len(ref_rows) > 0 and ref_t > 0)
    ref_class = classify_cell(ref_n, ref_n) if ref_has_data else "no_data"

    out: list[dict] = []
    for v in AXIS_UNIVERSE[gradient_axis]:
        rows = _select(v)
        n = float(rows["numerator"].fillna(0).sum())
        t = float(rows["denominator"].fillna(0).sum())
        has_data = (len(rows) > 0 and t > 0)
        is_ref = (v == reference_value
                  or (v is None and reference_value is None))

        base = {
            "label": render_axis_value(gradient_axis, v),
            "value": v,
            "num_events": int(n) if has_data else np.nan,
            "num_pt": t if has_data else np.nan,
            "ref_events": int(ref_n) if ref_has_data else np.nan,
            "ref_pt": ref_t if ref_has_data else np.nan,
            "incidence_rate": (n / t * 1e5) if has_data else np.nan,
            "ref_rate": (ref_n / ref_t * 1e5) if ref_has_data else np.nan,
        }

        if is_ref:
            out.append({
                **base,
                "irr": 1.0 if ref_has_data else np.nan,
                "lower_ci": np.nan, "upper_ci": np.nan, "p_value": np.nan,
                "is_reference": True,
                "power_class": "reference" if ref_has_data else "no_data",
                "status_note": ("reference group" if ref_has_data
                                else "no data in deposit"),
            })
            continue

        if not has_data or not ref_has_data:
            out.append({
                **base,
                "irr": np.nan, "lower_ci": np.nan, "upper_ci": np.nan,
                "p_value": np.nan, "is_reference": False,
                "power_class": "no_data",
                "status_note": "no data in deposit for this group",
            })
            continue

        cls = classify_cell(n, ref_n)
        if cls == "suppressed":
            out.append({
                **base,
                "irr": np.nan, "lower_ci": np.nan, "upper_ci": np.nan,
                "p_value": np.nan, "is_reference": False,
                "power_class": "suppressed",
                "status_note": ("suppressed: at least one side has fewer "
                                f"than {EVENTS_DISCLOSURE} events "
                                "(CPRD disclosure rule)"),
            })
            continue

        r = irr_ci(n, t, ref_n, ref_t)
        out.append({
            **base,
            "irr": r["irr"], "lower_ci": r["lower_ci"],
            "upper_ci": r["upper_ci"], "p_value": r["p_raw"],
            "is_reference": False,
            "power_class": cls,
            "status_note": ("lower precision (fewer than "
                            f"{EVENTS_RELIABLE} events on at least one side)"
                            if cls == "low_power" else ""),
        })
    return pd.DataFrame(out)


def history_irr_table(df_full: pd.DataFrame, df_parent: pd.DataFrame,
                      gradient_axis: str, reference_value
                      ) -> pd.DataFrame:
    """For each value of gradient_axis, compute the IRR of the full
    sequence's rate vs the parent sub-sequence's rate, both within that
    same value of the axis. Returns one row per gradient value with the
    same columns as gradient_irr_table (except is_reference is dropped --
    every row is its own contrast).
    """
    col = AXIS_COL[gradient_axis]
    if col not in df_full.columns or col not in df_parent.columns:
        return pd.DataFrame()

    observed_non_nan = sorted(set(df_full[col].dropna().unique().tolist())
                              | set(df_parent[col].dropna().unique().tolist()),
                              key=lambda x: str(x))
    has_missing_full = (col == "imd"
                        and df_full.get("imd_missing", pd.Series(dtype=bool)).any())
    has_missing_parent = (col == "imd"
                          and df_parent.get("imd_missing", pd.Series(dtype=bool)).any())
    observed = observed_non_nan + ([None] if (has_missing_full or has_missing_parent) else [])
    ordered = order_axis_values(gradient_axis, observed)

    def _sel(frame, v):
        if col == "imd" and v is None:
            return frame[frame.get("imd_missing", False)]
        return frame[frame[col] == v]

    out = []
    for v in ordered:
        rf = _sel(df_full, v)
        rp = _sel(df_parent, v)
        n_f = float(rf["numerator"].fillna(0).sum())
        t_f = float(rf["denominator"].fillna(0).sum())
        n_p = float(rp["numerator"].fillna(0).sum())
        t_p = float(rp["denominator"].fillna(0).sum())
        if t_f == 0 and t_p == 0:
            continue
        if n_f < 1 or n_p < 1 or t_f == 0 or t_p == 0:
            out.append({
                "label": render_axis_value(gradient_axis, v), "value": v,
                "num_events": int(n_f), "num_pt": t_f,
                "ref_events": int(n_p), "ref_pt": t_p,
                "incidence_rate": n_f / t_f * 1e5 if t_f > 0 else np.nan,
                "ref_rate": n_p / t_p * 1e5 if t_p > 0 else np.nan,
                "irr": np.nan, "lower_ci": np.nan, "upper_ci": np.nan,
                "p_value": np.nan,
                "is_reference_value": (v == reference_value),
            })
            continue
        r = irr_ci(n_f, t_f, n_p, t_p)
        out.append({
            "label": render_axis_value(gradient_axis, v), "value": v,
            "num_events": int(n_f), "num_pt": t_f,
            "ref_events": int(n_p), "ref_pt": t_p,
            "incidence_rate": n_f / t_f * 1e5,
            "ref_rate": n_p / t_p * 1e5,
            "irr": r["irr"], "lower_ci": r["lower_ci"],
            "upper_ci": r["upper_ci"], "p_value": r["p_raw"],
            "is_reference_value": (v == reference_value),
        })
    return pd.DataFrame(out)


def render_irr_chart(table: pd.DataFrame, title: str,
                     reference_label: str | None = None) -> None:
    """Render a horizontal bar chart of IRRs with 95% CI error bars and
    a reference line at IRR=1.0.

    The chart shows EVERY row of the input table, in row order. Rows
    where the IRR could not be computed (no data in the deposit, or
    suppressed under the CPRD <5-event disclosure rule) still appear on
    the y-axis but with no bar, so missing data is visually evident.
    Rows where one or both sides have 5-9 events ('low_power') are shown
    as pale-shaded bars to indicate reduced statistical precision; rows
    with >=10 events on both sides display normally.

    Bars are drawn on a LINEAR scale anchored at IRR = 1.0 (the
    reference). The x-axis range is auto-sized to include the full CI
    span of the rows that do have IRRs, with 10% padding on each side.
    """
    if table.empty:
        st.markdown('<div class="sparse">No data to chart.</div>',
                    unsafe_allow_html=True)
        return

    # The table already contains one row per universe value (with NaN IRR
    # for missing-data / suppressed groups). Show all of them in order.
    chart_df = table.copy().reset_index(drop=True)

    # Accept either column name for the reference flag
    ref_col = None
    if "is_reference" in chart_df.columns:
        ref_col = "is_reference"
    elif "is_reference_value" in chart_df.columns:
        ref_col = "is_reference_value"
    ref_mask = (chart_df[ref_col].fillna(False).astype(bool)
                if ref_col is not None
                else pd.Series(False, index=chart_df.index))
    chart_df["is_ref"] = ref_mask

    # power_class may not exist on tables produced by older code paths
    if "power_class" not in chart_df.columns:
        chart_df["power_class"] = chart_df["irr"].apply(
            lambda x: "no_data" if pd.isna(x) else "ok"
        )

    # Numeric placeholders for charting (will not be drawn for NaN IRR).
    chart_df["IRR"] = chart_df["irr"]
    chart_df["lo"] = chart_df["lower_ci"]
    chart_df["hi"] = chart_df["upper_ci"]
    chart_df["bar_start"] = 1.0
    chart_df["bar_end"] = chart_df["IRR"]

    # Build display labels: append "(N<10)" annotation for low_power rows,
    # "(no data)" for missing, "(suppressed)" for CPRD-suppressed.
    def _annotated_label(r):
        base = str(r["label"])
        pc = r.get("power_class", "ok")
        if pc == "low_power":
            return f"{base}  (low precision)"
        if pc == "no_data":
            return f"{base}  (no data)"
        if pc == "suppressed":
            return f"{base}  (<{EVENTS_DISCLOSURE} events; suppressed)"
        return base
    chart_df["display_label"] = chart_df.apply(_annotated_label, axis=1)

    # Auto-range the x-axis from CIs of rows that actually have data.
    finite_lo = chart_df["lo"].dropna()
    finite_hi = chart_df["hi"].dropna()
    if finite_lo.empty or finite_hi.empty:
        # Nothing to chart -- but we still render the y-axis so the
        # missing-data state is visible. Default range around 1.0.
        axis_lo, axis_hi = 0.5, 2.0
    else:
        lo_min = float(finite_lo.min())
        hi_max = float(finite_hi.max())
        axis_lo = min(lo_min, 1.0)
        axis_hi = max(hi_max, 1.0)
        span = max(axis_hi - axis_lo, 0.1)
        axis_lo = max(0.0, axis_lo - 0.1 * span)
        axis_hi = axis_hi + 0.1 * span

    # Colour policy. Reference is grey; bars >=1 are navy; bars <1 are
    # warm brown; low_power versions are paler shades of the same.
    # no_data and suppressed rows have NaN IRR so no bar is drawn -- the
    # colour assignment is irrelevant for those rows.
    COLOR_REF        = "#bbbbbb"
    COLOR_ABOVE      = "#2a6f97"   # navy
    COLOR_ABOVE_LOW  = "#9ec5dc"   # pale navy (low_power)
    COLOR_BELOW      = "#99582a"   # warm brown
    COLOR_BELOW_LOW  = "#d8b89b"   # pale warm brown (low_power)
    COLOR_NA         = "#dddddd"   # very pale grey (placeholder)

    def _row_color(r):
        pc = r.get("power_class", "ok")
        if r["is_ref"]:
            return COLOR_REF
        if pc in ("no_data", "suppressed"):
            return COLOR_NA
        irr = r["IRR"]
        if pd.isna(irr):
            return COLOR_NA
        if pc == "low_power":
            return COLOR_ABOVE_LOW if irr >= 1.0 else COLOR_BELOW_LOW
        return COLOR_ABOVE if irr >= 1.0 else COLOR_BELOW

    chart_df["bar_color"] = chart_df.apply(_row_color, axis=1)

    # Bar height: 32-40 px per row, with a generous floor so a 2-row
    # chart (e.g. SEX gradient) doesn't feel cramped.
    BAR_SIZE = 18              # bar thickness in px
    ROW_HEIGHT = 42            # vertical slot per row
    chart_height = max(160, ROW_HEIGHT * len(chart_df))

    import altair as alt
    x_scale = alt.Scale(domain=[axis_lo, axis_hi], nice=False)

    # Y axis: explicit category list so altair preserves our ordering.
    y_sort = chart_df["display_label"].tolist()
    base = alt.Chart(chart_df).encode(
        y=alt.Y("display_label:N", sort=y_sort, title=None,
                axis=alt.Axis(labelLimit=380, labelPadding=6)),
    )
    # Only emit bar / error marks for rows with valid IRR.
    bar_df = chart_df[chart_df["IRR"].notna()].copy()
    error_df = chart_df[chart_df["lo"].notna() & chart_df["hi"].notna()].copy()
    layers = []
    if not bar_df.empty:
        bars = alt.Chart(bar_df).mark_bar(size=BAR_SIZE).encode(
            y=alt.Y("display_label:N", sort=y_sort, title=None,
                    axis=alt.Axis(labelLimit=380, labelPadding=6)),
            x=alt.X("bar_start:Q", scale=x_scale,
                    title="Incidence rate ratio (reference = 1.0)"),
            x2="bar_end:Q",
            color=alt.Color("bar_color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Group"),
                alt.Tooltip("IRR:Q", format=".2f"),
                alt.Tooltip("lo:Q", format=".2f", title="95% CI lower"),
                alt.Tooltip("hi:Q", format=".2f", title="95% CI upper"),
                alt.Tooltip("num_events:Q", title="Events (this group)",
                            format=",d"),
                alt.Tooltip("ref_events:Q", title="Events (reference)",
                            format=",d"),
                alt.Tooltip("power_class:N", title="Precision tier"),
            ],
        )
        layers.append(bars)
    if not error_df.empty:
        errors = alt.Chart(error_df).mark_rule(
            color="#1f3147", strokeWidth=1.5).encode(
            y=alt.Y("display_label:N", sort=y_sort, title=None),
            x=alt.X("lo:Q", scale=x_scale, title=""),
            x2="hi:Q",
        )
        error_caps_lo = alt.Chart(error_df).mark_tick(
            color="#1f3147", thickness=1.5, size=8).encode(
            y=alt.Y("display_label:N", sort=y_sort, title=None),
            x=alt.X("lo:Q", scale=x_scale, title=""),
        )
        error_caps_hi = alt.Chart(error_df).mark_tick(
            color="#1f3147", thickness=1.5, size=8).encode(
            y=alt.Y("display_label:N", sort=y_sort, title=None),
            x=alt.X("hi:Q", scale=x_scale, title=""),
        )
        layers.extend([errors, error_caps_lo, error_caps_hi])

    refline = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
        strokeDash=[4, 4], color="#666").encode(
        x=alt.X("x:Q", scale=x_scale))
    layers.append(refline)

    # A "ghost" base layer that just renders the y-axis for every row,
    # even rows with no bar. This is what makes empty rows appear.
    yaxis_only = alt.Chart(chart_df).mark_point(opacity=0).encode(
        y=alt.Y("display_label:N", sort=y_sort, title=None,
                axis=alt.Axis(labelLimit=380, labelPadding=6)),
        x=alt.X("bar_start:Q", scale=x_scale,
                title="Incidence rate ratio (reference = 1.0)"),
    )
    layers.insert(0, yaxis_only)

    chart = alt.layer(*layers).properties(title=title, height=chart_height)
    st.altair_chart(chart, use_container_width=True)

    # Footnote describing the three-tier suppression policy.
    has_lowpower  = (chart_df["power_class"] == "low_power").any()
    has_nodata    = (chart_df["power_class"] == "no_data").any()
    has_suppress  = (chart_df["power_class"] == "suppressed").any()
    parts = []
    if reference_label:
        parts.append(
            f"Reference: **{reference_label}** (IRR = 1.0 by definition). "
            "Whiskers show the 95% confidence interval."
        )
    notes = []
    if has_lowpower:
        notes.append(
            f"pale bars indicate fewer than {EVENTS_RELIABLE} events on at "
            "least one side and should be interpreted with caution"
        )
    if has_suppress:
        notes.append(
            f"groups with fewer than {EVENTS_DISCLOSURE} events on either "
            "side are suppressed under the CPRD disclosure rule and have "
            "no bar drawn"
        )
    if has_nodata:
        notes.append(
            "groups with no rows in the deposit (e.g. excluded from this "
            "stratification) appear as empty rows"
        )
    if notes:
        parts.append("Notes: " + "; ".join(notes) + ".")
    if parts:
        st.caption(" ".join(parts))


def make_table_display(table: pd.DataFrame) -> pd.DataFrame:
    """Format the numeric table for display: rates rounded, p-values
    rendered, ordering preserved. Adds a 'Note' column reflecting the
    three-tier suppression policy: rows where data is missing entirely,
    suppressed under the CPRD <5-event rule, or shown with reduced
    statistical precision are flagged accordingly."""
    if table.empty:
        return table
    show = table.copy()
    show["IRR"] = show["irr"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "\u2014")
    show["95% CI"] = show.apply(
        lambda r: (f"{r['lower_ci']:.2f}\u2013{r['upper_ci']:.2f}"
                   if pd.notna(r["lower_ci"]) else "\u2014"),
        axis=1)
    show["p"] = show["p_value"].apply(
        lambda p: ("<0.001" if pd.notna(p) and p < 0.001
                   else (f"{p:.3f}" if pd.notna(p) else "\u2014")))
    show["Rate / 100k PY"] = show["incidence_rate"].apply(
        lambda x: f"{x:,.1f}" if pd.notna(x) else "\u2014")
    show["Events"] = show["num_events"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "\u2014")
    show["Person-years"] = show["num_pt"].apply(
        lambda x: f"{x:,.0f}" if pd.notna(x) else "\u2014")

    # Render the suppression/precision note. status_note is preferred (set
    # by gradient_irr_table); fall back to power_class for older tables.
    def _note(r):
        if "status_note" in r and isinstance(r.get("status_note"), str) and r["status_note"]:
            return r["status_note"]
        pc = r.get("power_class", "")
        if pc == "low_power":
            return f"lower precision (<{EVENTS_RELIABLE} events on at least one side)"
        if pc == "suppressed":
            return f"suppressed (<{EVENTS_DISCLOSURE} events; CPRD rule)"
        if pc == "no_data":
            return "no data in deposit"
        if pc == "reference":
            return "reference group"
        return ""
    show["Note"] = show.apply(_note, axis=1)

    cols = ["label", "Events", "Person-years", "Rate / 100k PY",
            "IRR", "95% CI", "p", "Note"]
    show = show[cols].rename(columns={"label": "Group"})
    return show


CAVEAT_BAR = (
    '<div class="caveat"><b>Note.</b> P-values shown are unadjusted for '
    "multiple comparisons. Confidence intervals (95%) reflect the Poisson "
    "variance of each rate. These are crude rate ratios for hypothesis "
    "generation and service planning, not adjusted causal effects.</div>"
)


# ======================================================================
# Main: radio chooses one of the two pages
# ======================================================================

st.title("InciGraph")
st.markdown(
    "A focused tool for asking two questions about the InciGraph "
    "multimorbidity deposit."
)

mode = st.radio(
    "Choose a question:",
    ["inequalities", "history"],
    format_func=lambda m: {
        "inequalities": "Demographic inequalities in a sequence",
        "history":      "Effect of prior history on a sequence",
    }[m],
    key="mode",
    horizontal=False,
)
st.divider()


# ----------------------------------------------------------------------
# PAGE 1 -- Demographic inequalities in a sequence
# ----------------------------------------------------------------------
if mode == "inequalities":

    st.subheader("1. Choose the disease sequence")
    seq = pick_sequence("p1_seq", min_length=1, max_length=3)
    endpoint = IDX_TO_DISPLAY[seq[-1]]
    traj = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq)
    st.markdown(f"**Sequence:** {traj}")
    if len(seq) > 1:
        prior = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq[:-1])
        st.caption(f"Incidence of {endpoint} after {prior} (first-ever "
                   "diagnoses in this order).")
    else:
        st.caption(f"Incidence of {endpoint} as a first-ever diagnosis.")

    st.subheader("2. Choose which demographic gradient to look at")
    all_axes = ["ETHNICITY", "SEX", "IMD", "AGE_CATG"]
    gradient_axis = st.radio(
        "Compare across:",
        all_axes,
        format_func=lambda a: AXIS_DISPLAY_NAME[a],
        horizontal=True,
        key="p1_axis",
    )

    st.subheader("3. (Optional) Fix other demographic axes")
    st.caption("You can fix up to two other axes to a single value, e.g. "
               "\"Asian women\". The gradient is then shown within that fixed "
               "stratum. Leave empty for the overall view across the gradient.")
    fixable = [a for a in all_axes if a != gradient_axis]
    fix_choice = st.multiselect(
        "Axes to fix",
        fixable,
        default=[],
        format_func=lambda a: AXIS_DISPLAY_NAME[a],
        key="p1_fix_choice",
    )
    if len(fix_choice) > 2:
        st.warning("The deposit supports at most three demographic axes at "
                   "once. Please fix at most two axes here (since the gradient "
                   "axis is the third).")
        st.stop()

    # Build the required stratification key from the chosen axes.
    needed_axes = set(fix_choice) | {gradient_axis}
    strat_key = "+".join(sorted(needed_axes))
    if strat_key not in STRATS:
        st.error(
            f"The combination you chose ({strat_key}) is not available in "
            "the deposit. The available schemes are: "
            f"{', '.join(sorted(STRATS))}."
        )
        st.stop()

    df = fetch_sequence_rows(seq, strat_key)
    if df is None:
        st.markdown('<div class="sparse">No data available for this '
                    "combination.</div>", unsafe_allow_html=True)
        st.stop()

    # Apply the fixed axes
    df_fixed, fixed_label = fix_demographic_axes(
        df, strat_key, "p1", fix_choice)

    st.subheader("4. Choose the reference group")
    # observed values on the gradient axis after fixing
    col = AXIS_COL[gradient_axis]
    observed_non_nan = list(df_fixed[col].dropna().unique().tolist())
    has_missing = (col == "imd"
                   and df_fixed.get("imd_missing", pd.Series(dtype=bool)).any())
    observed = observed_non_nan + ([None] if has_missing else [])
    ordered = order_axis_values(gradient_axis, observed)
    labels = {render_axis_value(gradient_axis, v): v for v in ordered}
    if not labels:
        st.markdown('<div class="sparse">No groups on the gradient axis '
                    "have data after the fixed selections.</div>",
                    unsafe_allow_html=True)
        st.stop()
    default_ref = DEFAULT_REFERENCE.get(gradient_axis)
    default_label = render_axis_value(gradient_axis, default_ref)
    default_index = (list(labels).index(default_label)
                     if default_label in labels else 0)
    ref_label = st.selectbox(
        f"Reference {AXIS_DISPLAY_NAME[gradient_axis]}",
        list(labels),
        index=default_index,
        key="p1_ref",
    )
    ref_value = labels[ref_label]

    st.subheader("5. Result")
    table = gradient_irr_table(df_fixed, gradient_axis, ref_value, "p1")
    if table.empty:
        st.markdown('<div class="sparse">No IRRs could be computed for '
                    "this combination.</div>", unsafe_allow_html=True)
    else:
        title = (f"IRR of {traj} across {AXIS_DISPLAY_NAME[gradient_axis]} "
                 f"in {fixed_label}")
        render_irr_chart(table, title, reference_label=ref_label)
        with st.expander("Underlying numbers (table)"):
            st.dataframe(make_table_display(table),
                         use_container_width=True, hide_index=True)
        st.download_button(
            "Download these IRRs (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name="incigraph_inequalities.csv", mime="text/csv",
        )

    st.markdown(CAVEAT_BAR, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE 2 -- Effect of prior history on a sequence (simplified)
# ----------------------------------------------------------------------
else:  # mode == "history"

    st.markdown(
        "Ask a single question of the form: *In this demographic group, does "
        "having one (or two) prior conditions in the patient's history "
        "elevate the rate of the later condition?* Pick a demographic "
        "stratum, then pick a 2- or 3-condition trajectory; the page "
        "returns one rate ratio with its 95% confidence interval and "
        "p-value."
    )

    st.subheader("1. Choose the trajectory (2 or 3 conditions)")
    seq = pick_sequence("p2_seq", min_length=2, max_length=3)
    if len(seq) < 2:
        st.info("Pick at least two conditions above to define a trajectory.")
        st.stop()

    endpoint = IDX_TO_DISPLAY[seq[-1]]
    traj_full = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq)
    parent_seq = seq[1:]                       # drop the earliest condition

    # Build human-readable phrases for "after <prior>" used in result sentences.
    # full_prior is the history that *precedes* the endpoint in the full
    # trajectory; parent_prior is the history that precedes the endpoint in
    # the parent (drop-earliest) sub-sequence.
    # For length 2 (A->B): full_prior = "A", parent_prior = None (=> "B alone")
    # For length 3 (A->B->C): full_prior = "A -> B", parent_prior = "B"
    prior_conds_full = [IDX_TO_DISPLAY[i] for i in seq[:-1]]
    prior_conds_parent = [IDX_TO_DISPLAY[i] for i in parent_seq[:-1]]
    full_prior = " \u2192 ".join(prior_conds_full) if prior_conds_full else ""
    parent_prior = (" \u2192 ".join(prior_conds_parent)
                    if prior_conds_parent else None)
    parent_phrase = (f"after **{parent_prior}**" if parent_prior
                     else "**alone**")
    traj_parent = (" \u2192 ".join(IDX_TO_DISPLAY[i] for i in parent_seq)
                   if len(parent_seq) > 1 else IDX_TO_DISPLAY[parent_seq[0]])

    st.subheader("2. Fix the demographic stratum")
    st.caption(
        "Pick 1\u20133 demographic axes to define the group of interest "
        "(e.g., for a White female patient aged 31\u201340 pick Ethnicity, "
        "Sex and Age, and fix each to its observed value)."
    )
    all_axes = ["ETHNICITY", "SEX", "IMD", "AGE_CATG"]
    fix_choice = st.multiselect(
        "Demographic axes to fix",
        all_axes,
        default=["ETHNICITY", "SEX", "AGE_CATG"],
        format_func=lambda a: AXIS_DISPLAY_NAME[a],
        key="p2_fix_choice",
    )
    if not fix_choice:
        st.info("Select at least one demographic axis to fix.")
        st.stop()
    if len(fix_choice) > 3:
        st.warning("Please fix at most three axes.")
        st.stop()

    strat_key = "+".join(sorted(fix_choice))
    if strat_key not in STRATS:
        st.error(
            f"The combination you chose ({strat_key}) is not available in "
            "the deposit. The available schemes are: "
            f"{', '.join(sorted(STRATS))}."
        )
        st.stop()

    df_full = fetch_sequence_rows(seq, strat_key)
    df_parent = fetch_sequence_rows(parent_seq, strat_key)
    if df_full is None or df_parent is None:
        st.markdown('<div class="sparse">No data available for one or both '
                    "sequences in this stratification.</div>",
                    unsafe_allow_html=True)
        st.stop()

    # Render the FIX selections (single value per axis), then apply
    fix_chosen, fixed_label = render_fix_selections(
        df_full, fix_choice, "p2")
    df_full_fixed = apply_fix_selections(df_full, fix_chosen)
    df_parent_fixed = apply_fix_selections(df_parent, fix_chosen)

    # Single-stratum totals
    n_full = float(df_full_fixed["numerator"].fillna(0).sum())
    t_full = float(df_full_fixed["denominator"].fillna(0).sum())
    n_par  = float(df_parent_fixed["numerator"].fillna(0).sum())
    t_par  = float(df_parent_fixed["denominator"].fillna(0).sum())

    st.subheader("3. Result")
    st.markdown(f"**Trajectory:** {traj_full}")
    st.markdown(f"**Demographic group:** {fixed_label}")

    # Headline contrast: full vs parent (drop earliest).
    # Three-tier suppression policy: <5 events on either side -> suppress;
    # 5..9 on either side -> show with a low-precision caveat; >=10 -> normal.
    headline_class = classify_cell(n_full, n_par)
    if headline_class == "suppressed":
        st.markdown(
            '<div class="sparse">Too few events in this group to display a '
            f"rate ratio (full sequence: {int(n_full):,} events; parent "
            f"sub-sequence: {int(n_par):,} events). Cells with fewer than "
            f"{EVENTS_DISCLOSURE} events on either side are suppressed "
            "under the CPRD disclosure rule. Try a broader group or a "
            "different trajectory.</div>",
            unsafe_allow_html=True)
    else:
        r1 = irr_ci(n_full, t_full, n_par, t_par)
        irr1 = r1["irr"]
        lo1, hi1 = r1["lower_ci"], r1["upper_ci"]
        p1 = r1["p_raw"]
        pstr1 = "<0.001" if p1 < 0.001 else f"{p1:.3f}"
        rate_full = (n_full / t_full * 1e5) if t_full > 0 else float("nan")
        rate_par  = (n_par / t_par * 1e5)  if t_par  > 0 else float("nan")

        st.markdown(
            f"In **{fixed_label}**, the rate of **{endpoint}** after "
            f"**{full_prior}** is **{irr1:.2f}\u00d7** the rate of "
            f"**{endpoint}** {parent_phrase} "
            f"(95% CI {lo1:.2f}\u2013{hi1:.2f}; p = {pstr1})."
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-big">{irr1:.2f}</div>',
                        unsafe_allow_html=True)
            st.caption("Incidence rate ratio")
        with m2:
            st.markdown(
                f'<div class="metric-big">{lo1:.2f}\u2013{hi1:.2f}</div>',
                unsafe_allow_html=True)
            st.caption("95% confidence interval")
        with m3:
            st.markdown(f'<div class="metric-big">{pstr1}</div>',
                        unsafe_allow_html=True)
            st.caption("p-value (unadjusted)")

        if headline_class == "low_power":
            st.markdown(
                '<div class="caveat">Lower precision: at least one side '
                f"has fewer than {EVENTS_RELIABLE} events "
                f"(full sequence: {int(n_full):,}; parent: {int(n_par):,}). "
                "The point estimate is reported but the Wald 95% CI may be "
                "wide and asymmetric \u2014 interpret with caution.</div>",
                unsafe_allow_html=True)

        st.caption(
            f"Underlying rates: **{traj_full}** \u2014 "
            f"{rate_full:,.1f} per 100,000 PY ({int(n_full):,} events over "
            f"{t_full:,.0f} person-years). **{traj_parent}** \u2014 "
            f"{rate_par:,.1f} per 100,000 PY ({int(n_par):,} events over "
            f"{t_par:,.0f} person-years)."
        )

        # Downloadable summary of this single contrast
        summary1 = pd.DataFrame([{
            "trajectory": traj_full,
            "reference_subsequence": traj_parent,
            "demographic_group": fixed_label,
            "stratification": strat_key,
            "events_full": int(n_full),
            "person_years_full": t_full,
            "events_parent": int(n_par),
            "person_years_parent": t_par,
            "rate_full_per_100k_PY": rate_full,
            "rate_parent_per_100k_PY": rate_par,
            "irr": irr1,
            "lower_95ci": lo1,
            "upper_95ci": hi1,
            "p_value": p1,
        }])
        st.download_button(
            "Download this result (CSV)",
            summary1.to_csv(index=False).encode("utf-8"),
            file_name="incigraph_history.csv", mime="text/csv",
        )

    # Optional second contrast for length-3: full vs endpoint alone
    if len(seq) == 3:
        with st.expander(
                f"Also show the contrast against **{endpoint}** alone "
                "(drop both earlier conditions)"):
            shortest_seq = [seq[-1]]
            traj_short = IDX_TO_DISPLAY[seq[-1]]
            df_short = fetch_sequence_rows(shortest_seq, strat_key)
            if df_short is None:
                st.markdown('<div class="sparse">No data for the endpoint-'
                            "alone sequence in this stratification.</div>",
                            unsafe_allow_html=True)
            else:
                df_short_fixed = apply_fix_selections(df_short, fix_chosen)
                n_sh = float(df_short_fixed["numerator"].fillna(0).sum())
                t_sh = float(df_short_fixed["denominator"].fillna(0).sum())
                endpoint_class = classify_cell(n_full, n_sh)
                if endpoint_class == "suppressed":
                    st.markdown(
                        '<div class="sparse">Too few events to display this '
                        f"contrast (full sequence: {int(n_full):,}; "
                        f"endpoint alone: {int(n_sh):,}). Cells with fewer "
                        f"than {EVENTS_DISCLOSURE} events on either side are "
                        "suppressed under the CPRD disclosure rule.</div>",
                        unsafe_allow_html=True)
                else:
                    r2 = irr_ci(n_full, t_full, n_sh, t_sh)
                    irr2 = r2["irr"]
                    lo2, hi2 = r2["lower_ci"], r2["upper_ci"]
                    p2 = r2["p_raw"]
                    pstr2 = "<0.001" if p2 < 0.001 else f"{p2:.3f}"
                    rate_sh = (n_sh / t_sh * 1e5) if t_sh > 0 else float("nan")
                    st.markdown(
                        f"In **{fixed_label}**, the rate of **{endpoint}** "
                        f"after **{full_prior}** is **{irr2:.2f}\u00d7** the "
                        f"rate of **{endpoint}** alone "
                        f"(95% CI {lo2:.2f}\u2013{hi2:.2f}; p = {pstr2})."
                    )
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-big">{irr2:.2f}</div>',
                                    unsafe_allow_html=True)
                        st.caption("Incidence rate ratio")
                    with m2:
                        st.markdown(
                            f'<div class="metric-big">{lo2:.2f}\u2013{hi2:.2f}</div>',
                            unsafe_allow_html=True)
                        st.caption("95% confidence interval")
                    with m3:
                        st.markdown(f'<div class="metric-big">{pstr2}</div>',
                                    unsafe_allow_html=True)
                        st.caption("p-value (unadjusted)")
                    if endpoint_class == "low_power":
                        st.markdown(
                            '<div class="caveat">Lower precision: at least '
                            f"one side has fewer than {EVENTS_RELIABLE} "
                            f"events (full: {int(n_full):,}; endpoint alone: "
                            f"{int(n_sh):,}). Interpret with caution.</div>",
                            unsafe_allow_html=True)
                    st.caption(
                        f"Underlying rate of **{endpoint}** alone: "
                        f"{rate_sh:,.1f} per 100,000 PY ({int(n_sh):,} "
                        f"events over {t_sh:,.0f} person-years)."
                    )
                    summary2 = pd.DataFrame([{
                        "trajectory": traj_full,
                        "reference_subsequence": traj_short,
                        "demographic_group": fixed_label,
                        "stratification": strat_key,
                        "events_full": int(n_full),
                        "person_years_full": t_full,
                        "events_endpoint_alone": int(n_sh),
                        "person_years_endpoint_alone": t_sh,
                        "rate_full_per_100k_PY": rate_full,
                        "rate_endpoint_alone_per_100k_PY": rate_sh,
                        "irr": irr2,
                        "lower_95ci": lo2,
                        "upper_95ci": hi2,
                        "p_value": p2,
                    }])
                    st.download_button(
                        "Download this result (CSV)",
                        summary2.to_csv(index=False).encode("utf-8"),
                        file_name="incigraph_history_endpoint_alone.csv",
                        mime="text/csv",
                        key="dl_endpoint_alone",
                    )

    st.markdown(CAVEAT_BAR, unsafe_allow_html=True)
