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
# Helpers used by both modes
# ======================================================================
def pick_sequence(key_prefix: str, default_first: str = "Hypertension"
                  ) -> list[int]:
    """Render three cascading disease pickers and return the chosen
    sequence as a list of 1-based canonical indices.

    `key_prefix` namespaces the widgets so multiple sequence pickers can
    co-exist on the page without colliding."""
    c1, c2, c3 = st.columns(3)
    with c1:
        d1 = st.selectbox(
            "First condition", DISEASE_DISPLAY,
            index=DISEASE_DISPLAY.index(default_first)
            if default_first in DISEASE_DISPLAY else 0,
            key=f"{key_prefix}_d1")
    with c2:
        d2 = st.selectbox(
            "Then (optional)", ["\u2014 none \u2014"] + DISEASE_DISPLAY,
            index=0, key=f"{key_prefix}_d2")
    with c3:
        d3 = st.selectbox(
            "Then (optional)", ["\u2014 none \u2014"] + DISEASE_DISPLAY,
            index=0, key=f"{key_prefix}_d3",
            disabled=(d2 == "\u2014 none \u2014"))
    seq = [NAME_TO_IDX[d1]]
    if d2 != "\u2014 none \u2014":
        seq.append(NAME_TO_IDX[d2])
        if d3 != "\u2014 none \u2014":
            seq.append(NAME_TO_IDX[d3])
    return seq


def fetch_sequence_rows(seq: list[int], strat: str) -> pd.DataFrame | None:
    """Read parquet rows for one (sequence, stratification) tuple. Returns
    a DataFrame on success or None on failure (with an inline error)."""
    sequence_str = "0 " + " ".join(str(i) for i in seq)
    try:
        df = _read_sequence_filtered(
            sequence_length=len(seq),
            sequence=sequence_str,
            stratification=strat,
            _local_dir_str=local_dir_str,
        )
        if df.empty:
            raise ValueError(
                f"No rows in the deposit for sequence {sequence_str!r} with "
                f"stratification {strat!r}.")
        return df
    except Exception as e:  # noqa: BLE001
        st.markdown(f'<div class="sparse">No data: {e}</div>',
                    unsafe_allow_html=True)
        return None


def pool_demographics(df: pd.DataFrame, strat: str, key_prefix: str
                      ) -> dict | None:
    """Render the multiselects for each axis of `strat` and return the
    pooled counts.

    Used by both modes: returns {num, pt, label, n_cells, strat}.
    `key_prefix` is the widget-key namespace (e.g. 'mode1_num',
    'mode2_shared'). Default selections pick the first value of each axis.
    """
    axes = [] if strat == "NONE" else strat.split("+")
    selections = {}
    for axis in axes:
        col = AXIS_COL[axis]
        if col not in df.columns:
            continue
        if col == "imd":
            raw_vals = sorted(v for v in df["imd"].dropna().unique())
            options = [f"{int(v)}" for v in raw_vals]
            if df.get("imd_missing", pd.Series(dtype=bool)).any():
                options.append("missing")
        else:
            options = sorted(str(v) for v in df[col].dropna().unique())

        poolable = axis in POOLABLE_OK
        help_txt = (
            "Pick one value to fix it, or several to pool them."
            if poolable else
            "Pick one value. Pooling several categories here is rarely "
            "meaningful."
        )
        picked = st.multiselect(
            AXIS_LABEL.get(axis, axis), options,
            default=options[:1],
            key=f"{key_prefix}_{axis}",
            help=help_txt,
        )
        if not poolable and len(picked) > 1:
            st.markdown(
                f'<div class="caveat">Pooling several '
                f"{AXIS_LABEL[axis].lower()} categories is unusual \u2014 the "
                "result sums incidence across them, which may not be "
                "interpretable.</div>",
                unsafe_allow_html=True)
        selections[axis] = picked

    mask = pd.Series(True, index=df.index)
    label_bits = []
    for axis, picked in selections.items():
        col = AXIS_COL[axis]
        if not picked:
            label_bits.append(f"all {AXIS_LABEL[axis].lower()}")
            continue
        if col == "imd":
            wanted_numeric = [float(v) for v in picked if v != "missing"]
            sub_mask = df["imd"].isin(wanted_numeric)
            if "missing" in picked:
                sub_mask = sub_mask | df.get("imd_missing", False)
            mask &= sub_mask
            label_bits.append(f"IMD {'+'.join(picked)}")
        else:
            mask &= df[col].astype(str).isin(picked)
            label_bits.append(f"{AXIS_LABEL[axis]} {'+'.join(picked)}")

    sel = df[mask]
    n_cells = len(sel)
    if n_cells == 0:
        st.markdown('<div class="sparse">No cells match this selection.</div>',
                    unsafe_allow_html=True)
        return None
    num = float(sel["numerator"].fillna(0).sum())
    pt = float(sel["denominator"].fillna(0).sum())
    label = ", ".join(label_bits) if label_bits else "everyone"
    st.caption(f"{n_cells} cell(s) pooled \u00b7 {int(num):,} events "
               f"over {pt:,.0f} person-years")
    return {"num": num, "pt": pt, "label": label, "n_cells": n_cells,
            "strat": strat}


def render_result(g_num: dict, g_den: dict, mode: str,
                  endpoint_num: str | None = None,
                  endpoint_den: str | None = None,
                  shared_group_label: str | None = None,
                  traj: str | None = None,
                  traj_num: str | None = None,
                  traj_den: str | None = None) -> None:
    """Render the IRR result block (metrics + sentence + rates + download).

    `mode` is 'demographics' (mode 1: same trajectory, two groups) or
    'sequences' (mode 2: two trajectories, same group)."""
    if g_num is None or g_den is None:
        st.info("Complete the inputs above to see the contrast.")
        return
    if g_num["num"] < 1 or g_den["num"] < 1:
        st.markdown(
            '<div class="sparse">One of the sides has no events, so the '
            "rate ratio is undefined. Widen the selection or pool more "
            "cells.</div>", unsafe_allow_html=True)
        return

    r = irr_ci(g_num["num"], g_num["pt"], g_den["num"], g_den["pt"])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-big">{r["irr"]:.2f}</div>',
                    unsafe_allow_html=True)
        st.caption("Incidence rate ratio")
    with m2:
        st.markdown(
            f'<div class="metric-big">{r["lower_ci"]:.2f}'
            f'\u2013{r["upper_ci"]:.2f}</div>',
            unsafe_allow_html=True)
        st.caption("95% confidence interval")
    with m3:
        p = r["p_raw"]
        pstr = "<0.001" if p < 0.001 else f"{p:.3f}"
        st.markdown(f'<div class="metric-big">{pstr}</div>',
                    unsafe_allow_html=True)
        st.caption("p-value (unadjusted)")

    times = "\u00d7"
    if mode == "demographics":
        st.markdown(
            f"In **{g_num['label']}**, the incidence of {endpoint_num} is "
            f"**{r['irr']:.2f}{times}** that in **{g_den['label']}**."
        )
    else:  # mode == 'sequences'
        st.markdown(
            f"In **{shared_group_label}**, the incidence of "
            f"**{traj_num}** is **{r['irr']:.2f}{times}** that of "
            f"**{traj_den}**."
        )

    rate_num = g_num["num"] / g_num["pt"] * 1e5
    rate_den = g_den["num"] / g_den["pt"] * 1e5
    st.caption(
        f"Numerator rate: {rate_num:,.1f} per 100,000 PY "
        f"({int(g_num['num']):,} events). "
        f"Denominator rate: {rate_den:,.1f} per 100,000 PY "
        f"({int(g_den['num']):,} events)."
    )

    # downloadable summary
    if mode == "demographics":
        row = {
            "mode": "compare demographic groups",
            "trajectory": traj,
            "numerator_group": g_num["label"],
            "numerator_stratification": g_num["strat"],
            "denominator_group": g_den["label"],
            "denominator_stratification": g_den["strat"],
        }
    else:
        row = {
            "mode": "compare sequences",
            "shared_group": shared_group_label,
            "shared_stratification": g_num["strat"],
            "numerator_trajectory": traj_num,
            "denominator_trajectory": traj_den,
        }
    row.update({
        "numerator_events": int(g_num["num"]),
        "numerator_person_years": g_num["pt"],
        "denominator_events": int(g_den["num"]),
        "denominator_person_years": g_den["pt"],
        "irr": r["irr"], "lower_95ci": r["lower_ci"],
        "upper_95ci": r["upper_ci"], "p_value": r["p_raw"],
    })
    summary = pd.DataFrame([row])
    st.download_button(
        "Download this result (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="incigraph_contrast.csv", mime="text/csv",
    )


# ======================================================================
# Main
# ======================================================================
st.title("InciGraph contrast tool")
st.markdown(
    "Compare incidence rates between groups or between sequences. "
    "Pick a mode below to get started."
)

mode = st.radio(
    "What do you want to compare?",
    ["demographics", "sequences"],
    format_func=lambda m: {
        "demographics":
            "Compare demographic groups (one trajectory, two groups)",
        "sequences":
            "Compare sequences (one demographic group, two trajectories)",
    }[m],
    key="mode",
    horizontal=False,
)
st.divider()

if mode == "demographics":
    # ---- Mode 1: today's flow, refactored to use the helpers ----
    st.subheader("1. Choose a trajectory")
    seq = pick_sequence("m1_seq")
    st.session_state["current_seq"] = seq

    endpoint = IDX_TO_DISPLAY[seq[-1]]
    traj = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq)
    st.markdown(f"**Trajectory:** {traj}")
    if len(seq) > 1:
        prior = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq[:-1])
        st.caption(f"Incidence of {endpoint} after {prior}.")
    else:
        st.caption(f"Incidence of {endpoint}.")

    st.subheader("2. Define the two groups")
    default_strat = STRATS.index("IMD") if "IMD" in STRATS else 0

    col_num, col_den = st.columns(2)
    with col_num:
        st.markdown('<div class="grp-num"><b>Numerator group</b> '
                    "(the rate on top of the ratio)</div>",
                    unsafe_allow_html=True)
        strat_num = st.selectbox("Break down by", STRATS,
                                  index=default_strat,
                                  format_func=strat_label,
                                  key="m1_num_strat")
        df_num = fetch_sequence_rows(seq, strat_num)
        g_num = (pool_demographics(df_num, strat_num, "m1_num")
                 if df_num is not None else None)
    with col_den:
        st.markdown('<div class="grp-den"><b>Denominator group</b> '
                    "(the reference rate)</div>",
                    unsafe_allow_html=True)
        strat_den = st.selectbox("Break down by", STRATS,
                                  index=default_strat,
                                  format_func=strat_label,
                                  key="m1_den_strat")
        df_den = fetch_sequence_rows(seq, strat_den)
        g_den = (pool_demographics(df_den, strat_den, "m1_den")
                 if df_den is not None else None)

    st.subheader("3. Result")
    if g_num is not None and g_den is not None \
            and g_num["strat"] != g_den["strat"]:
        st.markdown(
            '<div class="caveat">The two groups use different breakdown '
            "schemes. That is allowed, but make sure the comparison is "
            "meaningful.</div>", unsafe_allow_html=True)
    render_result(g_num, g_den, mode="demographics",
                  endpoint_num=endpoint, traj=traj)

else:
    # ---- Mode 2: same group, two trajectories ----
    st.subheader("1. Choose the demographic group")
    st.caption(
        "These characteristics are held the same for both trajectories. "
        "Pick the stratification scheme and the values that define the "
        "group you're studying."
    )
    default_strat = STRATS.index("IMD") if "IMD" in STRATS else 0
    shared_strat = st.selectbox(
        "Break down by", STRATS,
        index=default_strat, format_func=strat_label,
        key="m2_strat",
    )

    st.subheader("2. Choose the two trajectories to compare")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="grp-num"><b>Numerator trajectory</b></div>',
                    unsafe_allow_html=True)
        seq_a = pick_sequence("m2_seqA", default_first="Hypertension")
        traj_a = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq_a)
        st.markdown(f"&nbsp;&nbsp;{traj_a}", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="grp-den"><b>Denominator trajectory</b></div>',
                    unsafe_allow_html=True)
        seq_b = pick_sequence("m2_seqB", default_first="Hypertension")
        traj_b = " \u2192 ".join(IDX_TO_DISPLAY[i] for i in seq_b)
        st.markdown(f"&nbsp;&nbsp;{traj_b}", unsafe_allow_html=True)

    if seq_a == seq_b:
        st.info("The two trajectories are identical \u2014 the result will "
                "trivially be 1.0. Change one of them to get a meaningful "
                "contrast.")

    # Fetch each trajectory's data under the shared stratification
    df_a = fetch_sequence_rows(seq_a, shared_strat)
    df_b = fetch_sequence_rows(seq_b, shared_strat)
    if df_a is None or df_b is None:
        st.stop()

    st.subheader("3. Refine the demographic group")
    st.caption(
        "Pick one value to fix an axis, or several to pool. The same "
        "selections are applied to both trajectories."
    )
    # The two DataFrames share the schema (same stratification), so the
    # available values for each axis should be identical. We render one
    # set of multiselects and apply the resulting filter to both sides.
    # Using df_a as the "options source"; if any value is missing in
    # df_b the pooled sum simply contributes zero, which is correct.
    g_shared_label_holder = []

    # Render the demographic pickers ONCE, build the mask, apply to BOTH frames
    axes = [] if shared_strat == "NONE" else shared_strat.split("+")
    selections = {}
    for axis in axes:
        col = AXIS_COL[axis]
        if col not in df_a.columns:
            continue
        if col == "imd":
            raw_vals = sorted(v for v in df_a["imd"].dropna().unique())
            options = [f"{int(v)}" for v in raw_vals]
            if df_a.get("imd_missing", pd.Series(dtype=bool)).any():
                options.append("missing")
        else:
            options = sorted(str(v) for v in df_a[col].dropna().unique())
        poolable = axis in POOLABLE_OK
        help_txt = (
            "Pick one value to fix it, or several to pool them."
            if poolable else
            "Pick one value. Pooling several categories here is rarely "
            "meaningful.")
        picked = st.multiselect(
            AXIS_LABEL.get(axis, axis), options,
            default=options[:1],
            key=f"m2_shared_{axis}", help=help_txt)
        if not poolable and len(picked) > 1:
            st.markdown(
                f'<div class="caveat">Pooling several '
                f"{AXIS_LABEL[axis].lower()} categories is unusual.</div>",
                unsafe_allow_html=True)
        selections[axis] = picked

    def _apply_mask_and_pool(df: pd.DataFrame) -> tuple[float, float, int]:
        mask = pd.Series(True, index=df.index)
        for axis, picked in selections.items():
            col = AXIS_COL[axis]
            if not picked:
                continue
            if col == "imd":
                wanted = [float(v) for v in picked if v != "missing"]
                sub = df["imd"].isin(wanted)
                if "missing" in picked:
                    sub = sub | df.get("imd_missing", False)
                mask &= sub
            else:
                mask &= df[col].astype(str).isin(picked)
        sel = df[mask]
        n = float(sel["numerator"].fillna(0).sum())
        pt = float(sel["denominator"].fillna(0).sum())
        return n, pt, len(sel)

    label_bits = []
    for axis, picked in selections.items():
        if not picked:
            label_bits.append(f"all {AXIS_LABEL[axis].lower()}")
        elif AXIS_COL[axis] == "imd":
            label_bits.append(f"IMD {'+'.join(picked)}")
        else:
            label_bits.append(f"{AXIS_LABEL[axis]} {'+'.join(picked)}")
    shared_label = ", ".join(label_bits) if label_bits else "everyone"

    num_a, pt_a, ncells_a = _apply_mask_and_pool(df_a)
    num_b, pt_b, ncells_b = _apply_mask_and_pool(df_b)

    st.caption(
        f"For trajectory **{traj_a}**: {ncells_a} cell(s), {int(num_a):,} "
        f"events over {pt_a:,.0f} person-years."
    )
    st.caption(
        f"For trajectory **{traj_b}**: {ncells_b} cell(s), {int(num_b):,} "
        f"events over {pt_b:,.0f} person-years."
    )

    g_a = ({"num": num_a, "pt": pt_a, "label": shared_label,
            "n_cells": ncells_a, "strat": shared_strat}
           if ncells_a > 0 else None)
    g_b = ({"num": num_b, "pt": pt_b, "label": shared_label,
            "n_cells": ncells_b, "strat": shared_strat}
           if ncells_b > 0 else None)

    st.subheader("4. Result")
    render_result(g_a, g_b, mode="sequences",
                  shared_group_label=shared_label,
                  traj_num=traj_a, traj_den=traj_b)

st.markdown(CAVEAT_HTML, unsafe_allow_html=True)
