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

import incigraph as ig
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
def _resolve_data_dir(sidebar_override):
    if sidebar_override:
        p = Path(sidebar_override)
        if p.exists():
            return p
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
    if candidate.exists():
        return candidate
    return None


@st.cache_data(show_spinner=False)
def _available_strats(data_dir_str):
    ig.set_data_dir(data_dir_str)
    return ig.available_stratifications()


@st.cache_data(show_spinner=True)
def _get_sequence(data_dir_str, sequence, stratification):
    ig.set_data_dir(data_dir_str)
    return ig.get_sequence(list(sequence), stratification=stratification)


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

sidebar_path = st.sidebar.text_input(
    "Data folder (optional)", value="",
    help="Leave blank to use the bundled data.",
)
data_dir = _resolve_data_dir(sidebar_path or None)
if data_dir is None:
    st.error(
        "Could not find the InciGraph data. Place the parquet files in an "
        "`incigraph_data/` folder next to the repository, or type the path "
        "in the sidebar."
    )
    st.stop()

data_dir_str = str(data_dir)
ig.set_data_dir(data_dir_str)
st.sidebar.success(f"Data loaded from:\n`{data_dir}`")

try:
    STRATS = _available_strats(data_dir_str)
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to load the data: {e}")
    st.stop()

st.sidebar.metric("Disease conditions", len(DISEASE_NAMES))
st.sidebar.metric("Stratification schemes", len(STRATS))

DISEASE_DISPLAY = [n.replace("_", " ").title() for n in DISEASE_NAMES]
NAME_TO_IDX = {disp: i + 1 for i, disp in enumerate(DISEASE_DISPLAY)}
IDX_TO_DISPLAY = {i + 1: disp for i, disp in enumerate(DISEASE_DISPLAY)}


# ======================================================================
# Group builder
# ======================================================================
def group_builder(side_key, default_strat_idx=0):
    """Render one group's controls and return its pooled counts."""
    strat = st.selectbox(
        "Break down by",
        STRATS,
        index=default_strat_idx,
        format_func=strat_label,
        key=f"{side_key}_strat",
    )

    seq = tuple(st.session_state["current_seq"])
    try:
        df = _get_sequence(data_dir_str, seq, strat)
    except Exception as e:  # noqa: BLE001
        st.markdown(f'<div class="sparse">No data: {e}</div>',
                    unsafe_allow_html=True)
        return None

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
            AXIS_LABEL.get(axis, axis),
            options,
            default=options[:1],
            key=f"{side_key}_{axis}",
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
    num = float(sel["numerator"].fillna(0).sum())
    pt = float(sel["denominator"].fillna(0).sum())
    label = ", ".join(label_bits) if label_bits else "everyone"

    if n_cells == 0:
        st.markdown('<div class="sparse">No cells match this selection.</div>',
                    unsafe_allow_html=True)
        return None
    st.caption(f"{n_cells} cell(s) pooled \u00b7 {int(num):,} events "
               f"over {pt:,.0f} person-years")
    return {"num": num, "pt": pt, "label": label, "n_cells": n_cells,
            "strat": strat}


# ======================================================================
# Main
# ======================================================================
st.title("InciGraph contrast tool")
st.markdown(
    "Compare the incidence of a disease trajectory between two demographic "
    "groups. Each group can fix some characteristics and pool across others."
)

st.subheader("1. Choose a trajectory")
c1, c2, c3 = st.columns(3)
with c1:
    d1 = st.selectbox("First condition", DISEASE_DISPLAY,
                      index=DISEASE_DISPLAY.index("Hypertension")
                      if "Hypertension" in DISEASE_DISPLAY else 0,
                      key="seq_d1")
with c2:
    d2 = st.selectbox("Then (optional)", ["\u2014 none \u2014"] + DISEASE_DISPLAY,
                      index=0, key="seq_d2")
with c3:
    d3 = st.selectbox("Then (optional)", ["\u2014 none \u2014"] + DISEASE_DISPLAY,
                      index=0, key="seq_d3",
                      disabled=(d2 == "\u2014 none \u2014"))

seq = [NAME_TO_IDX[d1]]
if d2 != "\u2014 none \u2014":
    seq.append(NAME_TO_IDX[d2])
    if d3 != "\u2014 none \u2014":
        seq.append(NAME_TO_IDX[d3])
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
col_num, col_den = st.columns(2)
with col_num:
    st.markdown('<div class="grp-num"><b>Numerator group</b> '
                "(the rate on top of the ratio)</div>",
                unsafe_allow_html=True)
    default_num = STRATS.index("IMD") if "IMD" in STRATS else 0
    g_num = group_builder("num", default_num)
with col_den:
    st.markdown('<div class="grp-den"><b>Denominator group</b> '
                "(the reference rate)</div>",
                unsafe_allow_html=True)
    default_den = STRATS.index("IMD") if "IMD" in STRATS else 0
    g_den = group_builder("den", default_den)

st.subheader("3. Result")
if g_num is None or g_den is None:
    st.info("Complete both group definitions above to see the contrast.")
elif g_num["num"] < 1 or g_den["num"] < 1:
    st.markdown(
        '<div class="sparse">One of the groups has no events, so the rate '
        "ratio is undefined. Widen the selection or pool more cells.</div>",
        unsafe_allow_html=True)
else:
    if g_num["strat"] != g_den["strat"]:
        st.markdown(
            '<div class="caveat">The two groups use different breakdown '
            "schemes. That is allowed, but make sure the comparison is "
            "meaningful.</div>", unsafe_allow_html=True)

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
    st.markdown(
        f"In **{g_num['label']}**, the incidence of {endpoint} is "
        f"**{r['irr']:.2f}{times}** that in **{g_den['label']}**."
    )

    rate_num = g_num["num"] / g_num["pt"] * 1e5
    rate_den = g_den["num"] / g_den["pt"] * 1e5
    st.caption(
        f"Numerator rate: {rate_num:,.1f} per 100,000 PY "
        f"({int(g_num['num']):,} events). "
        f"Denominator rate: {rate_den:,.1f} per 100,000 PY "
        f"({int(g_den['num']):,} events)."
    )

    summary = pd.DataFrame([{
        "trajectory": traj,
        "numerator_group": g_num["label"],
        "numerator_stratification": g_num["strat"],
        "numerator_events": int(g_num["num"]),
        "numerator_person_years": g_num["pt"],
        "denominator_group": g_den["label"],
        "denominator_stratification": g_den["strat"],
        "denominator_events": int(g_den["num"]),
        "denominator_person_years": g_den["pt"],
        "irr": r["irr"], "lower_95ci": r["lower_ci"],
        "upper_95ci": r["upper_ci"], "p_value": r["p_raw"],
    }])
    st.download_button(
        "Download this result (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="incigraph_contrast.csv", mime="text/csv",
    )

st.markdown(CAVEAT_HTML, unsafe_allow_html=True)
