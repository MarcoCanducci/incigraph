"""Disease index <-> name mapping for InciGraph.

The InciGraph workbooks identify diseases by 1-based integer indices (1..64).
Disease N is the disease whose `BD_MEDI:NAME...:N` column appears at position
N+K in each workbook (where K is the number of stratification key columns).

This module exposes:
  DISEASE_NAMES         : the canonical 64-element list, position N-1 = disease N
  N_DISEASES            : 64
  short_name(raw)       : 'BD_MEDI:TYPE2DIABETES_BHAM_CAM:8' -> 'TYPE2DIABETES'
  decode_sequence(seq)  : "0 3 8" -> "HYPERTENSION -> T2D"
  encode_sequence(names): inverse of decode_sequence (lookup by short name)
  parse_sequence(seq)   : "0 3 8" or [3, 8] -> tuple of ints, root-stripped
"""

from __future__ import annotations

import re
from typing import Iterable

# 1-based disease index -> short readable name. Position N in this list is the
# disease referenced by index N in the sequence strings used throughout
# InciGraph. This list MUST stay in sync with what incigraph_to_parquet.py
# emits as the canonical_disease_map; the parquet's target_disease_idx
# column references it by integer position.
#
# If you regenerate the parquet against a different cohort, check
# incigraph_to_parquet.json's canonical_disease_map and update this list
# to match.
DISEASE_NAMES: list[str] = [
    "HF", "AF", "HYPERTENSION", "PAD", "VALVULAR",                       # 1-5
    "AORTIC_ANEURYSM", "T1DM", "T2D", "CKD3-5", "DEPRESSION",            # 6-10
    "ANXIETY", "BIPOLAR", "EATING_DISORDERS", "SCHIZOPHRENIA", "AUTISM", # 11-15
    "CHRONIC_LIVER_DISEASE_ALCOHOL", "NAFLD",
    "OTHER_CHRONIC_LIVER_DISEASE", "DEMENTIA", "PARKINSONS",             # 16-20
    "EPILEPSY", "CANCER", "ASTHMA", "COPD", "OSA",                       # 21-25
    "ECZEMA", "ALLERGIC_RHINITIS", "HIV", "OSTEOPOROSIS",
    "OSTEOARTHRITIS",                                                    # 26-30
    "RA", "GOUT", "SLE", "SJOGRENS", "SSc",                              # 31-35
    "PMR_GCA", "ENDOMETRIOSIS", "HYPOTHYROIDISM", "HYPERTHYROIDISM",
    "ADDISON",                                                           # 36-40
    "MS", "VISUAL_IMPAIRMENT", "MENIERES", "PERIPHERAL_NEUROPATHY",
    "DOWNS",                                                             # 41-45
    "PERNICIOUS_ANAEMIA", "PSORIASIS", "PSORIATIC_ARTHRITIS", "ILD",
    "PTSD",                                                              # 46-50
    "IBS", "HEARING_LOSS", "LEARNING_DISABILITY", "ENDOMETRIOSIS_V2",
    "PCOS",                                                              # 51-55
    "SICKLE_CELL", "HAEMOCHROMATOSIS", "STROKE", "IHD", "DRUG_ALCOHOL",   # 56-60
    "IBD", "HAEM_CANCER", "BRONCHIECTASIS", "FIBROMYALGIA_CFS",          # 61-64
]
N_DISEASES = len(DISEASE_NAMES)
assert N_DISEASES == 64, "DISEASE_NAMES must have exactly 64 entries"

# Inverse lookup, built once. We normalise on uppercase so casing differences
# in user input don't matter.
_NAME_TO_INDEX = {name.upper(): i + 1 for i, name in enumerate(DISEASE_NAMES)}

# Cohort-version tokens to strip when parsing a raw BD_MEDI: header. These
# accumulate over time as CPRD code lists are revised; the converter and the
# original `collect_incigraph.py` strip the same set.
_HDR_STRIPS = (
    "_BHAM_CAM_FINAL", "_BIRM_CAM_V2", "_MM_BIRM_CAM", "_BHAM_CAM",
    "_BIRM_CAM", "_PUSHASTHMA", "_TOT", "_OPTIMAL",
    "_11_3_21", "_120421", "_SH_20092020", "_DRAFT_V1", "_V2",
    "_2021", "_STRICT",
)


def short_name(raw_header: str) -> str:
    """Strip a raw BD_MEDI column header down to its disease name.

    'BD_MEDI:TYPE2DIABETES_BHAM_CAM:8'  ->  'TYPE2DIABETES'
    'BD_MEDI:HYPERTENSION_BIRM_CAM_V2:3' -> 'HYPERTENSION'

    Falls back to the input string if it doesn't follow the BD_MEDI convention,
    so this is safe to call on already-normalised names too.
    """
    try:
        core = raw_header.split(":")[1]
    except (IndexError, AttributeError):
        return str(raw_header)
    for tok in _HDR_STRIPS:
        core = core.replace(tok, "")
    return core.strip("_")


def parse_disease_index(raw_header: str) -> int | None:
    """Extract the trailing ':N' from a BD_MEDI header.

    Returns the integer disease index (1..64) or None if the header doesn't
    match the BD_MEDI:NAME:N pattern.
    """
    m = re.search(r":(\d+)$", str(raw_header))
    return int(m.group(1)) if m else None


def parse_sequence(seq) -> tuple[int, ...]:
    """Normalise a sequence specifier into a tuple of post-cohort-root indices.

    Accepts:
      "0 3 8"      ->  (3, 8)        # leading 0 is the cohort root, stripped
      "3 8"        ->  (3, 8)
      [0, 3, 8]    ->  (3, 8)
      [3, 8]       ->  (3, 8)
      (3, 8)       ->  (3, 8)

    Rejects sequences whose indices fall outside 1..64 or whose length is
    not 1, 2, or 3 after stripping the root.
    """
    if isinstance(seq, str):
        try:
            nums = [int(x) for x in seq.split()]
        except ValueError as e:
            raise ValueError(f"bad sequence string {seq!r}: {e}") from e
    else:
        try:
            nums = [int(x) for x in seq]
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad sequence {seq!r}: {e}") from e
    # strip the cohort-root 0 if present
    if nums and nums[0] == 0:
        nums = nums[1:]
    if not (1 <= len(nums) <= 3):
        raise ValueError(
            f"sequence must have 1, 2, or 3 post-root diseases; got {len(nums)} "
            f"in {seq!r}"
        )
    if any(not (1 <= n <= N_DISEASES) for n in nums):
        bad = [n for n in nums if not (1 <= n <= N_DISEASES)]
        raise ValueError(
            f"disease indices must be 1..{N_DISEASES}; got out-of-range {bad}"
        )
    return tuple(nums)


def decode_sequence(seq) -> str:
    """Turn a sequence string or list into a human-readable arrow chain.

    "0 3 8"     -> "HYPERTENSION -> T2D"
    "0 3 8 9"   -> "HYPERTENSION -> T2D -> CKD3-5"
    [3]         -> "HYPERTENSION"
    """
    idxs = parse_sequence(seq)
    return " -> ".join(DISEASE_NAMES[i - 1] for i in idxs)


def encode_sequence(names: Iterable[str], add_root: bool = True) -> str:
    """Turn a list of short disease names into the canonical sequence string.

    encode_sequence(["HYPERTENSION", "T2D"])
        -> "0 3 8"
    encode_sequence(["hypertension", "t2d", "ckd3-5"], add_root=False)
        -> "3 8 9"

    Case-insensitive on the names. Raises if any name isn't in DISEASE_NAMES.
    """
    idxs = []
    for n in names:
        key = str(n).strip().upper()
        if key not in _NAME_TO_INDEX:
            raise ValueError(
                f"unknown disease name {n!r}; must be one of "
                f"{sorted(DISEASE_NAMES)}"
            )
        idxs.append(_NAME_TO_INDEX[key])
    if not (1 <= len(idxs) <= 3):
        raise ValueError(
            f"need 1, 2, or 3 disease names; got {len(idxs)}"
        )
    parts = ([0] if add_root else []) + idxs
    return " ".join(str(i) for i in parts)
