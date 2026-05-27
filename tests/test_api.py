"""Tests for incigraph.api.

These build a tiny synthetic parquet at fixture time and exercise the
seven public functions against it. They run in a few seconds and don't
need the real ~110 MB deposit.

The fixture is intentionally minimal:
- one stratification ('ETHNICITY+IMD')
- a few sequences of each length
- deterministic numerators and denominators

Tests against the real parquet deposit are marked `needs_parquet` and
require the actual files to be present at $INCIGRAPH_DATA; they're
skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import incigraph as ig
from incigraph.api import _resolve_dir
from incigraph.ci import poisson_ci, RATE_DENOMINATOR


# ----------------------------------------------------------------------
# Fixture: build a tiny synthetic parquet deposit for the test session
# ----------------------------------------------------------------------
def _build_synthetic_parquet(tmp_path: Path) -> Path:
    """Build a 3-file parquet that mimics the real deposit's schema."""
    rng = np.random.default_rng(42)
    ethnicities = ["WHITE", "SOUTH_ASIAN", "BLACK"]
    imd_quintiles = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]

    def make_rows(seq, seq_len, target_idx, target_short, stratification):
        rows = []
        for eth in ethnicities:
            for imd in imd_quintiles:
                n = int(rng.integers(0, 200))
                y = float(rng.uniform(1000, 100_000))
                rate = n / y * RATE_DENOMINATOR if y > 0 else np.nan
                lo, hi = poisson_ci(np.array([n]), np.array([y]))
                row = {
                    "sequence":             seq,
                    "sequence_length":      np.int8(seq_len),
                    "step_1_idx":           np.int16(int(seq.split()[1])),
                    "step_2_idx":           (np.int16(int(seq.split()[2]))
                                              if seq_len >= 2 else pd.NA),
                    "step_3_idx":           (np.int16(int(seq.split()[3]))
                                              if seq_len >= 3 else pd.NA),
                    "target_disease_idx":   np.int16(target_idx),
                    "target_disease_short": target_short,
                    "target_disease_raw":   f"BD_MEDI:{target_short}:{target_idx}",
                    "stratification_key":   stratification,
                    "ethnicity":            eth,
                    "sex":                  pd.NA,
                    "imd":                  imd,
                    "imd_missing":          bool(np.isnan(imd)),
                    "age_catg":             pd.NA,
                    "numerator":            np.int32(n),
                    "denominator":          y,
                    "incidence_rate":       rate,
                    "lower_limit":          float(lo[0]),
                    "upper_limit":          float(hi[0]),
                    "source_folder":        stratification,
                }
                rows.append(row)
        return rows

    # Build L1 (4 sequences -> targets HF, AF, HYPERTENSION, T2D)
    l1_rows = []
    for target in (1, 2, 3, 8):
        l1_rows.extend(make_rows(
            seq=f"0 {target}", seq_len=1,
            target_idx=target, target_short=ig.DISEASE_NAMES[target - 1],
            stratification="ETHNICITY+IMD",
        ))

    # Build L2 (HYPERTENSION -> {AF, T2D, COPD})
    l2_rows = []
    for target in (2, 8, 24):
        l2_rows.extend(make_rows(
            seq=f"0 3 {target}", seq_len=2,
            target_idx=target, target_short=ig.DISEASE_NAMES[target - 1],
            stratification="ETHNICITY+IMD",
        ))

    # Build L3 (HYPERTENSION -> T2D -> {CKD3-5, DEPRESSION})
    l3_rows = []
    for target in (9, 10):
        l3_rows.extend(make_rows(
            seq=f"0 3 8 {target}", seq_len=3,
            target_idx=target, target_short=ig.DISEASE_NAMES[target - 1],
            stratification="ETHNICITY+IMD",
        ))

    # Build metadata
    meta_rows = []
    for rows, sl in [(l1_rows, 1), (l2_rows, 2), (l3_rows, 3)]:
        for seq in sorted(set(r["sequence"] for r in rows)):
            idxs = [int(t) for t in seq.split() if t != "0"]
            row = {
                "sequence":           seq,
                "sequence_length":    np.int8(sl),
                "sequence_decoded":   " -> ".join(
                    ig.DISEASE_NAMES[i - 1] for i in idxs),
                "step_1_short":       ig.DISEASE_NAMES[idxs[0] - 1],
                "step_2_short":       (ig.DISEASE_NAMES[idxs[1] - 1]
                                        if len(idxs) >= 2 else None),
                "step_3_short":       (ig.DISEASE_NAMES[idxs[2] - 1]
                                        if len(idxs) >= 3 else None),
            }
            meta_rows.append(row)

    # Write
    for rows, sl in [(l1_rows, 1), (l2_rows, 2), (l3_rows, 3)]:
        pd.DataFrame(rows).to_parquet(
            tmp_path / f"incigraph_L{sl}.parquet", engine="pyarrow",
            compression="zstd", index=False)
    pd.DataFrame(meta_rows).to_parquet(
        tmp_path / "incigraph_metadata.parquet", engine="pyarrow",
        compression="zstd", index=False)

    return tmp_path


@pytest.fixture(scope="session")
def synthetic_parquet(tmp_path_factory):
    """Build the fixture once per test session."""
    tmp = tmp_path_factory.mktemp("synth_parquet")
    return _build_synthetic_parquet(tmp)


@pytest.fixture(autouse=True)
def _isolated_data_dir(synthetic_parquet, monkeypatch):
    """Every test in this module starts with set_data_dir pointing at
    the synthetic fixture."""
    monkeypatch.delenv("INCIGRAPH_DATA", raising=False)
    ig.set_data_dir(synthetic_parquet)


# ----------------------------------------------------------------------
# Tests for the seven public functions
# ----------------------------------------------------------------------
class TestSetDataDir:
    def test_set_data_dir_changes_resolution(self, tmp_path):
        ig.set_data_dir(tmp_path)
        assert _resolve_dir(None) == tmp_path

    def test_env_var_fallback(self, monkeypatch, tmp_path):
        """If set_data_dir hasn't been called, INCIGRAPH_DATA is used."""
        # Force the private state back to None
        from incigraph import api
        monkeypatch.setattr(api, "_DATA_DIR", None)
        monkeypatch.setenv("INCIGRAPH_DATA", str(tmp_path))
        assert _resolve_dir(None) == tmp_path


class TestLoadEstimates:
    def test_single_length(self):
        df = ig.load_estimates(2)
        assert "sequence" in df.columns
        assert (df["sequence_length"] == 2).all()
        assert len(df) > 0

    def test_all_lengths(self):
        df = ig.load_estimates(None)
        assert set(df["sequence_length"].unique()) >= {1, 2, 3}

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            ig.load_estimates(4)


class TestLoadMetadata:
    def test_returns_decoded_sequences(self):
        m = ig.load_metadata()
        assert "sequence_decoded" in m.columns
        assert m["sequence_decoded"].notna().all()
        # known anchor: 0 3 should decode to HYPERTENSION
        row = m[m["sequence"] == "0 3"]
        assert len(row) == 1
        assert row.iloc[0]["sequence_decoded"] == "HYPERTENSION"


class TestAvailableStratifications:
    def test_returns_list(self):
        keys = ig.available_stratifications()
        assert isinstance(keys, list)
        assert "ETHNICITY+IMD" in keys


class TestListSequences:
    def test_filter_by_length(self):
        df = ig.list_sequences(sequence_length=2)
        assert (df["sequence_length"] == 2).all()

    def test_filter_starts_with(self):
        df = ig.list_sequences(starts_with_disease="HYPERTENSION")
        # Every row should begin with HYPERTENSION
        assert (df["step_1_short"].astype(str).str.upper() == "HYPERTENSION").all()

    def test_filter_ends_with(self):
        df = ig.list_sequences(ends_with_disease="T2D")
        for _, row in df.iterrows():
            steps = [row[f"step_{i}_short"] for i in (1, 2, 3)
                     if pd.notna(row[f"step_{i}_short"])]
            assert steps[-1] == "T2D"

    def test_filter_contains(self):
        df = ig.list_sequences(contains_disease="T2D")
        for _, row in df.iterrows():
            steps = [row[f"step_{i}_short"] for i in (1, 2, 3)
                     if pd.notna(row[f"step_{i}_short"])]
            assert "T2D" in steps


class TestGetSequence:
    def test_basic_query(self):
        df = ig.get_sequence([3, 8], stratification="ETHNICITY+IMD")
        assert len(df) > 0
        assert "ethnicity" in df.columns
        assert "imd" in df.columns

    def test_imd_missing_rows_are_included_by_default(self):
        df = ig.get_sequence([3], stratification="ETHNICITY+IMD")
        assert df["imd_missing"].any()  # missing-IMD stratum is present

    def test_string_sequence_input(self):
        df_a = ig.get_sequence("0 3 8", stratification="ETHNICITY+IMD")
        df_b = ig.get_sequence([3, 8], stratification="ETHNICITY+IMD")
        assert len(df_a) == len(df_b)

    def test_unknown_stratification_raises(self):
        with pytest.raises(ValueError):
            ig.get_sequence([3], stratification="NONEXISTENT")

    def test_unknown_sequence_raises(self):
        with pytest.raises(ValueError):
            ig.get_sequence([3, 5, 7], stratification="ETHNICITY+IMD")


class TestComputeIRR:
    def test_returns_complete_dict(self):
        df = ig.get_sequence([3], stratification="ETHNICITY+IMD")
        r = ig.compute_irr(df,
            comparison={"ethnicity": "BLACK", "imd": 5.0},
            reference ={"ethnicity": "WHITE", "imd": 1.0})
        for key in ("irr", "log_irr", "se_log_irr",
                    "lower_ci", "upper_ci",
                    "z", "p_raw",
                    "n_comp", "n_ref",
                    "comparison_rate", "reference_rate"):
            assert key in r

    def test_ambiguous_filter_raises(self):
        """If the filter matches more than one row, we want a clean error."""
        df = ig.get_sequence([3], stratification="ETHNICITY+IMD")
        with pytest.raises(ValueError, match="matched"):
            ig.compute_irr(df,
                comparison={"ethnicity": "BLACK"},  # multiple IMDs!
                reference ={"ethnicity": "WHITE"})

    def test_no_match_raises(self):
        df = ig.get_sequence([3], stratification="ETHNICITY+IMD")
        with pytest.raises(ValueError, match="no rows"):
            ig.compute_irr(df,
                comparison={"ethnicity": "NONEXISTENT", "imd": 5.0},
                reference ={"ethnicity": "WHITE",       "imd": 1.0})


class TestDecodeSequence:
    def test_api_decoder_matches_module_decoder(self):
        from incigraph.disease_index import decode_sequence as raw
        assert ig.decode_sequence("0 3 8") == raw("0 3 8")
