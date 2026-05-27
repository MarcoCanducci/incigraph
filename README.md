# InciGraph

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

**InciGraph is open infrastructure for hypothesis generation and service-planning research on multimorbidity.** It is a precomputed library of ~50 million incidence-rate estimates of ordered disease trajectories in UK primary care, with intersectional demographic stratification, covering 64 long-term conditions and ~14 million patients in CPRD data.

A web interface for browsing the estimates lives at <https://incigraph.netlify.app/>. This repository provides programmatic access via a Python package plus the validation pipeline that produced the headline arithmetic guarantees in the accompanying manuscript.

---

## What this package gives you

Three things, in order of expected use:

- **A small Python API** (`incigraph.get_sequence`, `incigraph.compute_irr`, etc.) for pulling out estimates for any disease sequence × demographic stratum × age band combination, with confidence intervals computed correctly under the supplement's chi-squared / Byar's rule.
- **Five command-line tools** that wrap the API: pull a CSV slice for plotting, curate top-ranked contrasts, run the validation pipeline.
- **Three Jupyter notebooks** that rebuild the manuscript figures from scratch against the parquet deposit, demonstrating reproducibility end-to-end.

---

## Quick start

```bash
# 1. Install the package
pip install -e .

# 2. Download the data deposit (one-time, ~110 MB)
#    From Zenodo: https://zenodo.org/record/20417249

#    Expand into ./incigraph_data/
ls incigraph_data/
# incigraph_L1.parquet
# incigraph_L2.parquet
# incigraph_L3.parquet
# incigraph_metadata.parquet
# incigraph_to_parquet.json

# 3. Use it
python -c "
import incigraph as ig
ig.set_data_dir('./incigraph_data')

# Get incidence of T2D after hypertension, by ethnicity x IMD, ages 51-60
df = ig.get_sequence([3, 8], stratification='ETHNICITY+IMD+AGE_CATG', age_bands=['51-60'])

# Compute South Asian (IMD 5) vs White (IMD 1) IRR
result = ig.compute_irr(df,
    comparison={'ethnicity': 'SOUTH_ASIAN', 'imd': 5.0},
    reference ={'ethnicity': 'WHITE',       'imd': 1.0})
print(f\"IRR = {result['irr']:.2f}  [95% CI {result['lower_ci']:.2f} - {result['upper_ci']:.2f}]\")
"
```

The full walkthrough is in [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb).

---

## What's in the box

```
incigraph/
├── src/incigraph/             ← The Python API (the importable package)
│   ├── api.py                 ← Seven public functions
│   ├── ci.py                  ← Poisson and IRR confidence intervals
│   └── disease_index.py       ← The 64-disease canonical naming
│
├── scripts/                   ← Command-line tools
│   ├── collect_incigraph.py   ← Pull a tidy CSV for one sequence
│   ├── curate_ordered_transition_candidates.py
│   ├── plot_incigraph.py
│   ├── incigraph_to_parquet.py  ← Build the parquet deposit from raw xlsx
│   └── validation/            ← Reproducibility receipt for the manuscript
│       ├── step1_aggregation_consistency.py
│       ├── step2_sparsity_quantification.py
│       └── step3_contrast_scan.py
│
├── figures/                   ← Build the manuscript figures from parquet
│   ├── build_hero.py
│   └── build_l3.py
│
├── notebooks/
│   ├── 01_quickstart.ipynb              ← Start here
│   ├── 02_replicate_hero_figure.ipynb   ← Manuscript Figure 1 from parquet
│   └── 03_replicate_l3_figure.ipynb     ← Manuscript Figure 2 from parquet
│
├── tests/                     ← pytest suite
└── docs/                      ← Data dictionary, CI formulas, validation spec
```

---

## Data conventions you should know about

These are baked into the data and the API. Knowing them upfront prevents silent analysis errors:

- **Disease indices are 1-based** and correspond to **column position in the canonical unstratified workbook**. The mapping is at the top of [`src/incigraph/disease_index.py`](src/incigraph/disease_index.py) and in the JSON sidecar of each parquet deposit.
- **`IMD=blank` means MISSING**, NOT a pooled total. The parquet's `imd_missing` boolean flags these rows; filter them out explicitly if your analysis needs only the 1–5 quintiles.
- **`SEX='I'` is its own demographic category**, NOT a sex-pooled total. Same treatment: filter explicitly if you want only M/F.
- **The cohort-root marker `0`** appears at the start of every sequence string (`"0 3 8"`, `"0 3 8 9"`). It is a fixed marker, not a disease index, and is stripped automatically by `parse_sequence`.
- **Confidence intervals are computed per the supplement**: exact chi-squared for N < 10, Byar's cube-root for N ≥ 10. Verified against the workbook-stored CIs to ~10⁻¹⁴ relative tolerance.

---

## Verifying your installation

After installing the package and downloading the data, run:

```bash
pytest                                        # unit tests — fast, no parquet needed
python -m incigraph.api --selfcheck           # API smoke test against the parquet
scripts/validation/step1_aggregation_consistency.py \
    --parquet-dir ./incigraph_data --out-dir ./validation_output
```

These three commands cover unit-level correctness, integration with the parquet, and end-to-end reproducibility of the headline arithmetic guarantee. See [Verifying the install](#verifying-the-install) below for the full procedure.

---

## Citation

If you use InciGraph in academic work, please cite both the data deposit and the paper:

```bibtex
@article{incigraph2026,
  title   = {[manuscript title]},
  author  = {[authors]},
  journal = {PLOS Digital Health},
  year    = {2026},
  doi     = {10.xxxx/...}
}

@dataset{incigraph_data_2026,
  title     = {InciGraph parquet deposit},
  author    = {[authors]},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.<RECORD_ID>}
}
```

A machine-readable version is in [`CITATION.cff`](CITATION.cff). GitHub renders it as a "Cite this repository" button.

---

## Licensing

- **Code** is released under the MIT license. See [`LICENSE`](LICENSE).
- **Data** (the parquet files and the underlying xlsx tree) are released under CC-BY-4.0. See [`DATA_LICENSE`](DATA_LICENSE).

The two licenses are separate by design. Code can be freely incorporated into commercial work; data redistribution requires attribution.

---

## Verifying the install

The package ships with three independent verification levels. Each catches a different class of bug. Run them in this order:

### Level 1 — Unit tests (no data required)

```bash
pip install -e ".[dev]"
pytest
```

You should see something like:

```
tests/test_ci.py ...........               [ 30%]
tests/test_disease_index.py .......        [ 60%]
tests/test_api.py ........                 [100%]

================== 26 passed in 1.42s ==================
```

If any fails, the failure is in the library itself — likely a Python or dependency version mismatch. Open an issue with the pytest traceback.

### Level 2 — API smoke test (against your parquet)

```bash
python -c "
import incigraph as ig
ig.set_data_dir('./incigraph_data')

# Stratifications available
print('Stratifications:', ig.available_stratifications())

# 64 diseases in metadata
m = ig.load_metadata()
assert len(m[m.sequence_length == 1]) == 64
print('Metadata OK:', len(m), 'unique sequences')

# A specific query that should always succeed on a complete parquet
df = ig.get_sequence([3, 8], stratification='ETHNICITY+IMD')
print('HYPERTENSION -> T2D, ETHNICITY+IMD:', df.shape, 'rows')

# Compute an IRR with a known ballpark answer
result = ig.compute_irr(df,
    comparison={'ethnicity': 'SOUTH_ASIAN', 'imd': 5.0},
    reference ={'ethnicity': 'WHITE',       'imd': 1.0})
print(f'South Asian/IMD5 vs White/IMD1 IRR = {result[\"irr\"]:.2f}')
"
```

This confirms the API finds the parquet, the schema matches the API's expectations, and IRR computations produce finite numbers. Should take a few seconds.

### Level 3 — End-to-end pipeline (the manuscript's evidence)

```bash
mkdir -p validation_output

# Aggregation consistency check (Step 1 of the manuscript pipeline)
python scripts/validation/step1_aggregation_consistency.py \
    --parquet-dir ./incigraph_data \
    --out-dir ./validation_output

# Sparsity quantification (Step 2)
python scripts/validation/step2_sparsity_quantification.py \
    --parquet-dir ./incigraph_data \
    --out-dir ./validation_output

# Contrast scan (Step 3 — the expensive one, ~10-15 minutes)
python scripts/validation/step3_contrast_scan.py \
    --parquet-dir ./incigraph_data \
    --out-dir ./validation_output
```

Step 1 should report:

```
[gate] all marginalisable pairs passed.
```

If it doesn't, your parquet is internally inconsistent and downstream analyses can't be trusted. Step 1 exits non-zero in that case.

Step 2 produces `sparsity_by_stratification.csv` — the values behind Panel C of the hero figure. Open it and check that the `percent_n_ge_10` column matches what the manuscript reports.

Step 3 produces `demographic_contrast_scan_all.parquet` (and CSV). The top-ranked contrasts in this file should match the rows in `notebooks/02_replicate_hero_figure.ipynb`'s `PANEL_B_SPECS` list — that's the receipt that the published rows can be recovered from the data.

### Putting it together

If all three levels pass, you have an installation that reproduces the manuscript's analysis end-to-end. Send me a copy of `validation_output/aggregation_consistency.json` and any pytest output if anything looks off.

---

## Reporting issues

Open an issue on the GitHub repository with:

1. Python version and `pip freeze` output
2. The exact command you ran
3. The full error or unexpected output

Issues related to the **data** (e.g. "the IRR for sequence X looks wrong") should also include the output of `python -c "import incigraph; print(incigraph.__version__)"` and the SHA-256 of your parquet files (`shasum -a 256 incigraph_data/*.parquet`), so we can confirm you're using the published release.
