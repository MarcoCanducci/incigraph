"""Tests for incigraph.disease_index.

These verify that the 64-disease canonical naming, sequence parsing,
and decode/encode round-trips are correct. They don't need parquet.
"""

from __future__ import annotations

import pytest

from incigraph.disease_index import (
    DISEASE_NAMES,
    N_DISEASES,
    decode_sequence,
    encode_sequence,
    parse_disease_index,
    parse_sequence,
    short_name,
)


class TestCanonicalList:
    def test_has_64_diseases(self):
        assert N_DISEASES == 64
        assert len(DISEASE_NAMES) == 64

    def test_known_position_anchors(self):
        """A few positions are anchored to specific diseases in the
        manuscript figures. If any of these drift, the figure scripts
        break silently."""
        assert DISEASE_NAMES[0]  == "HF"
        assert DISEASE_NAMES[1]  == "AF"
        assert DISEASE_NAMES[2]  == "HYPERTENSION"
        assert DISEASE_NAMES[7]  == "T2D"
        assert DISEASE_NAMES[8]  == "CKD3-5"
        assert DISEASE_NAMES[9]  == "DEPRESSION"
        assert DISEASE_NAMES[10] == "ANXIETY"
        assert DISEASE_NAMES[21] == "CANCER"
        assert DISEASE_NAMES[22] == "ASTHMA"
        assert DISEASE_NAMES[23] == "COPD"
        assert DISEASE_NAMES[29] == "OSTEOARTHRITIS"
        assert DISEASE_NAMES[53] == "ENDOMETRIOSIS_V2"  # V2 disambiguation
        assert DISEASE_NAMES[55] == "SICKLE_CELL"
        assert DISEASE_NAMES[59] == "DRUG_ALCOHOL"

    def test_no_duplicates(self):
        """Every disease name must be unique -- otherwise sequence
        encoding becomes ambiguous."""
        assert len(set(DISEASE_NAMES)) == len(DISEASE_NAMES)


class TestShortName:
    def test_strips_bd_medi_prefix_and_trailing_colon(self):
        assert short_name("BD_MEDI:HYPERTENSION_BHAM_CAM:5") == "HYPERTENSION"

    def test_strips_cohort_version_tokens(self):
        assert short_name("BD_MEDI:TYPE2DIABETES_11_3_21_BIRM_CAM:12") == "TYPE2DIABETES"

    def test_strips_tot_suffix(self):
        assert short_name("BD_MEDI:BRONCHIECTASIS_TOT") == "BRONCHIECTASIS"
        assert short_name("BD_MEDI:STROKE_TOT") == "STROKE"
        assert short_name("BD_MEDI:DRUGALCOHOL_TOT") == "DRUGALCOHOL"

    def test_handles_unparseable_input(self):
        """Falls back to the input string if there's no colon-separated form."""
        assert short_name("already_clean") == "already_clean"
        assert short_name("") == ""


class TestParseDiseaseIndex:
    def test_extracts_trailing_integer(self):
        assert parse_disease_index("BD_MEDI:T2D:8") == 8
        assert parse_disease_index("BD_MEDI:HF_FINAL:0") == 0
        assert parse_disease_index("BD_MEDI:SICKLE_CELL_V2:70") == 70

    def test_returns_none_for_unparseable(self):
        assert parse_disease_index("BD_MEDI:NO_TRAILING") is None
        assert parse_disease_index("BD_MEDI:STROKE_TOT") is None  # no trailing :N


class TestParseSequence:
    def test_accepts_string_with_root(self):
        assert parse_sequence("0 3 8") == (3, 8)

    def test_accepts_string_without_root(self):
        assert parse_sequence("3 8") == (3, 8)

    def test_accepts_list(self):
        assert parse_sequence([0, 3, 8]) == (3, 8)
        assert parse_sequence([3, 8]) == (3, 8)
        assert parse_sequence((3, 8)) == (3, 8)

    def test_handles_length_1(self):
        assert parse_sequence([3]) == (3,)
        assert parse_sequence("0 3") == (3,)

    def test_handles_length_3(self):
        assert parse_sequence([3, 8, 9]) == (3, 8, 9)
        assert parse_sequence("0 3 8 9") == (3, 8, 9)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_sequence([])
        with pytest.raises(ValueError):
            parse_sequence("0")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            parse_sequence([1, 2, 3, 4])

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            parse_sequence([3, 99])
        with pytest.raises(ValueError):
            parse_sequence([0])  # 0 after root-stripping means index 0, invalid

    def test_rejects_bad_strings(self):
        with pytest.raises(ValueError):
            parse_sequence("hello")


class TestDecodeSequence:
    def test_length_1(self):
        assert decode_sequence("0 3") == "HYPERTENSION"

    def test_length_2(self):
        assert decode_sequence("0 3 8") == "HYPERTENSION -> T2D"

    def test_length_3(self):
        assert decode_sequence("0 3 8 9") == "HYPERTENSION -> T2D -> CKD3-5"

    def test_accepts_list_input(self):
        assert decode_sequence([3, 8]) == "HYPERTENSION -> T2D"


class TestEncodeSequence:
    def test_basic(self):
        assert encode_sequence(["HYPERTENSION", "T2D"]) == "0 3 8"

    def test_case_insensitive(self):
        assert encode_sequence(["hypertension", "t2d"]) == "0 3 8"

    def test_without_root(self):
        assert encode_sequence(["HYPERTENSION"], add_root=False) == "3"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            encode_sequence(["NOT_A_DISEASE"])

    def test_round_trip(self):
        """encode then decode then encode should be a fixed point."""
        for seq_str in ["0 3", "0 8 9", "0 3 8 9"]:
            decoded = decode_sequence(seq_str)
            re_encoded = encode_sequence(decoded.split(" -> "))
            assert re_encoded == seq_str
