"""InciGraph: a precomputed, intersectional, ordered-incidence platform
for primary-care multimorbidity.

This Python package provides programmatic access to the InciGraph estimates
(~50 million precomputed incidence rates across ordered disease sequences
and demographic strata in UK primary care).

Quick start
-----------
    import incigraph as ig

    # tell the library where the parquet files live
    ig.set_data_dir("/path/to/incigraph_data")

    # find a sequence
    matches = ig.list_sequences(starts_with_disease="HYPERTENSION",
                                ends_with_disease="T2D")

    # get the estimates
    df = ig.get_sequence("0 3 8", stratification="ETHNICITY+IMD",
                         age_bands=["51-60", "61-70"])

    # compute an IRR
    result = ig.compute_irr(
        df,
        comparison={"ethnicity": "SOUTH_ASIAN", "imd": 5},
        reference={"ethnicity": "WHITE", "imd": 1},
    )

See `incigraph.api` and the bundled notebooks for full documentation.
"""

from .api import (
    set_data_dir,
    load_estimates,
    load_metadata,
    get_sequence,
    compute_irr,
    list_sequences,
    decode_sequence,
    available_stratifications,
)
from .disease_index import DISEASE_NAMES, N_DISEASES
from .ci import poisson_ci, irr_ci

__version__ = "0.1.0"

__all__ = [
    "set_data_dir",
    "load_estimates",
    "load_metadata",
    "get_sequence",
    "compute_irr",
    "list_sequences",
    "decode_sequence",
    "available_stratifications",
    "poisson_ci",
    "irr_ci",
    "DISEASE_NAMES",
    "N_DISEASES",
    "__version__",
]
