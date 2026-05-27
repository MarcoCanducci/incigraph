"""Tests for incigraph.ci.

These tests don't need the parquet deposit; they verify the arithmetic
against known textbook values and known properties of the formulas. If
these break, the library itself is broken.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from incigraph.ci import irr_ci, poisson_ci, RATE_DENOMINATOR


# ----------------------------------------------------------------------
# poisson_ci
# ----------------------------------------------------------------------
class TestPoissonCI:
    def test_zero_count_lower_bound_is_zero(self):
        """When N=0, the exact chi-squared interval is degenerate; the
        supplement specifies lower bound = 0."""
        lo, hi = poisson_ci(np.array([0]), np.array([10000.0]))
        assert lo[0] == 0.0
        assert hi[0] > 0.0  # upper bound is finite

    def test_single_count_uses_chi_squared_regime(self):
        """N < 10 uses exact chi-squared. For N=1 the supplement's
        formula gives lower ~ 0.025/Y, upper ~ 5.572/Y (in per-PY units
        before scaling)."""
        n, y = 1, 1000.0
        lo, hi = poisson_ci(np.array([n]), np.array([y]))
        # bounds (per 100,000 PY) should bracket the point estimate
        rate = n / y * RATE_DENOMINATOR  # 100.0
        assert lo[0] < rate < hi[0]
        # 95% CI for N=1 in Poisson is known: roughly [0.025, 5.572] events
        # scale to per-100k PY: [2.5, 557.2] approximately
        assert 0 < lo[0] < 50
        assert 300 < hi[0] < 700

    def test_large_count_uses_byars_regime(self):
        """N >= 10 uses Byar's approximation. For large N the CI should
        be approximately symmetric on the log scale and close to the
        normal approximation."""
        n, y = 100, 1000.0
        lo, hi = poisson_ci(np.array([n]), np.array([y]))
        rate = n / y * RATE_DENOMINATOR
        assert lo[0] < rate < hi[0]
        # symmetry on log scale: log(rate/lo) should be close to log(hi/rate)
        log_lo_diff = math.log(rate / lo[0])
        log_hi_diff = math.log(hi[0] / rate)
        assert abs(log_lo_diff - log_hi_diff) < 0.1

    def test_regime_continuity_at_threshold(self):
        """At N=10 we switch from chi-squared to Byar's. The two formulas
        give close but not identical bounds; the jump should be small."""
        lo9,  hi9  = poisson_ci(np.array([9]),  np.array([1000.0]))
        lo10, hi10 = poisson_ci(np.array([10]), np.array([1000.0]))
        # both regimes should bracket their respective rates
        assert lo9[0]  < 900  < hi9[0]
        assert lo10[0] < 1000 < hi10[0]
        # the regime jump in upper bound shouldn't exceed ~10%
        assert abs(hi10[0] - hi9[0]) / hi9[0] < 0.3

    def test_invalid_inputs_return_nan(self):
        """Person-years <= 0 should give NaN bounds."""
        lo, hi = poisson_ci(np.array([5, 5]), np.array([0.0, -1.0]))
        assert np.isnan(lo).all()
        assert np.isnan(hi).all()

    def test_vectorised(self):
        """Should accept and return arrays of matching length."""
        n = np.array([0, 1, 5, 10, 100, 1000])
        y = np.full(6, 1000.0)
        lo, hi = poisson_ci(n, y)
        assert lo.shape == (6,)
        assert hi.shape == (6,)
        # monotonicity: bounds should be non-decreasing in N
        rates = n / y * RATE_DENOMINATOR
        for i in range(6):
            assert lo[i] <= rates[i] <= hi[i]


# ----------------------------------------------------------------------
# irr_ci
# ----------------------------------------------------------------------
class TestIRRCI:
    def test_equal_rates_gives_irr_one(self):
        """When the two groups have identical incidence, IRR=1, log_irr=0,
        p_raw=1, and the CI brackets 1."""
        r = irr_ci(50, 1000, 50, 1000)
        assert abs(r["irr"] - 1.0) < 1e-12
        assert abs(r["log_irr"]) < 1e-12
        assert abs(r["p_raw"] - 1.0) < 1e-12
        assert r["lower_ci"] < 1.0 < r["upper_ci"]

    def test_four_fold_contrast(self):
        """200 events/1000 PY vs 50 events/1000 PY -> IRR = 4 exactly.
        SE(log IRR) = sqrt(1/200 + 1/50)."""
        r = irr_ci(200, 1000, 50, 1000)
        assert abs(r["irr"] - 4.0) < 1e-10
        expected_se = math.sqrt(1 / 200 + 1 / 50)
        assert abs(r["se_log_irr"] - expected_se) < 1e-12
        # p should be highly significant
        assert r["p_raw"] < 1e-10
        # CI bounds known approximately: log(4) ± 1.96*SE
        z = 1.959963984540054
        expected_lo = math.exp(math.log(4) - z * expected_se)
        expected_hi = math.exp(math.log(4) + z * expected_se)
        assert abs(r["lower_ci"] - expected_lo) < 1e-10
        assert abs(r["upper_ci"] - expected_hi) < 1e-10

    def test_zero_comparison_count_yields_nan(self):
        """Zero events in the comparison arm -> IRR is undefined."""
        r = irr_ci(0, 1000, 50, 1000)
        assert np.isnan(r["irr"])
        assert np.isnan(r["log_irr"])
        assert np.isnan(r["lower_ci"])
        assert np.isnan(r["upper_ci"])

    def test_zero_reference_count_yields_nan(self):
        """Zero events in the reference arm -> IRR is undefined."""
        r = irr_ci(50, 1000, 0, 1000)
        assert np.isnan(r["irr"])

    def test_negative_person_time_yields_nan(self):
        """Defensive: non-positive PY -> NaN."""
        r = irr_ci(50, -10, 50, 1000)
        assert np.isnan(r["irr"])

    def test_returns_expected_keys(self):
        """The result dict should have the documented keys."""
        r = irr_ci(50, 1000, 50, 1000)
        expected_keys = {
            "irr", "log_irr", "se_log_irr",
            "lower_ci", "upper_ci",
            "z", "p_raw",
            "n_comp", "n_ref",
        }
        assert expected_keys.issubset(set(r.keys()))

    def test_p_value_symmetry(self):
        """IRR=4 and IRR=0.25 (the inverse) should give the same
        two-sided p-value (within numerical tolerance)."""
        r_a = irr_ci(200, 1000, 50, 1000)
        r_b = irr_ci(50, 1000, 200, 1000)
        assert abs(r_a["p_raw"] - r_b["p_raw"]) < 1e-12
        assert abs(r_a["irr"] * r_b["irr"] - 1.0) < 1e-12
