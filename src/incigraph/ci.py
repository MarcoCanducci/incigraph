"""Confidence intervals for InciGraph incidence rates.

Implements the two-regime 95% CI rule specified in the InciGraph
supplementary material:

  N < 10  : exact chi-squared (Poisson) interval
              zeta_L = chi2(alpha/2,  2N)   / (2 Y)
              zeta_U = chi2(1-alpha/2, 2N+2) / (2 Y)
            With alpha = 0.05 the lower bound uses df = 2N at the 0.025
            quantile and the upper uses df = 2N+2 at the 0.975 quantile.
            When N = 0 the lower bound is fixed at 0 (the chi-squared is
            degenerate at df = 0).

  N >= 10 : Byar's cube-root approximation (z = 1.96 for alpha = 0.05)
              zeta_L = N     [1 - 1/(9 N)     - z/(3 sqrt(N))    ]^3 / Y
              zeta_U = (N+1) [1 - 1/(9(N+1))  + z/(3 sqrt(N+1)) ]^3 / Y

Both expressions are multiplied by `rate_per` so the bounds come out in the
same units as the incidence rate (per 100,000 person-years by default).

These formulas reproduce the workbook's stored CIs to machine precision (max
observed difference ~1.7e-15 over 7,623 cells), which is why the validation
pipeline can claim arithmetic exactness.

Also exposed:
  irr_ci(n_comp, y_comp, n_ref, y_ref)
    Crude incidence-rate-ratio with its 95% Wald CI on the log scale,
    standard error, raw two-sided Wald p-value, and IRR. The same
    computation used by `step3_contrast_scan.py` for the systematic scan.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import chi2, norm

RATE_DENOMINATOR = 100_000  # per how many person-years to report the rate


def poisson_ci(
    n_counts,
    person_years,
    alpha: float = 0.05,
    rate_per: float = RATE_DENOMINATOR,
):
    """Lower and upper 95% bounds for an incidence rate, per the supplement.

    Vectorised. Pass scalars or arrays for n_counts and person_years; the
    return is a pair of numpy arrays the same shape as the inputs.

    Cells with person_years <= 0 return (nan, nan). Cells with N = 0 return
    a lower bound of 0 (the exact chi-squared interval is degenerate at
    df = 0) and a finite upper bound.

    Parameters
    ----------
    n_counts : scalar or array-like of int
        Event counts (numerator).
    person_years : scalar or array-like of float
        Person-time at risk (denominator).
    alpha : float, default 0.05
        Two-sided significance level. 0.05 -> 95% CI.
    rate_per : float, default 100_000
        Multiplier so the bounds match the incidence rate's units.

    Returns
    -------
    (lower, upper) : tuple of numpy arrays
        Both in units of `rate_per` person-years.
    """
    n = np.asarray(n_counts, dtype=float)
    y = np.asarray(person_years, dtype=float)
    lo = np.full_like(n, np.nan, dtype=float)
    hi = np.full_like(n, np.nan, dtype=float)

    valid = (y > 0) & np.isfinite(n) & np.isfinite(y)
    if not valid.any():
        return lo, hi

    # standard normal quantile at 1 - alpha/2
    if abs(alpha - 0.05) < 1e-12:
        z = 1.959963984540054  # cached to spare a norm.ppf call
    else:
        z = norm.ppf(1.0 - alpha / 2.0)

    # --- regime split: chi-squared for sparse counts, Byar's otherwise ---
    chi_mask  = valid & (n < 10)
    byar_mask = valid & (n >= 10)

    # exact chi-squared interval. Degenerate at df = 0 (when N = 0); we
    # explicitly set the lower bound to 0 there.
    if chi_mask.any():
        nc = n[chi_mask]
        yc = y[chi_mask]
        lower_dof = 2.0 * nc
        upper_dof = 2.0 * nc + 2.0
        lo_chi = np.where(
            lower_dof > 0,
            chi2.ppf(alpha / 2.0, lower_dof) / (2.0 * yc),
            0.0,
        )
        hi_chi = chi2.ppf(1.0 - alpha / 2.0, upper_dof) / (2.0 * yc)
        lo[chi_mask] = lo_chi
        hi[chi_mask] = hi_chi

    # Byar's cube-root approximation. The (n+1) form for the upper bound is
    # part of the original Byar specification, not a typo.
    if byar_mask.any():
        nb = n[byar_mask]
        yb = y[byar_mask]
        lo[byar_mask] = (
            nb
            * (1.0 - 1.0 / (9.0 * nb) - z / (3.0 * np.sqrt(nb))) ** 3
            / yb
        )
        hi[byar_mask] = (
            (nb + 1.0)
            * (
                1.0
                - 1.0 / (9.0 * (nb + 1.0))
                + z / (3.0 * np.sqrt(nb + 1.0))
            ) ** 3
            / yb
        )

    return lo * rate_per, hi * rate_per


def irr_ci(
    n_comp: float,
    y_comp: float,
    n_ref: float,
    y_ref: float,
    alpha: float = 0.05,
) -> dict:
    """Crude incidence-rate-ratio with 95% Wald CI on the log scale.

    The SE formula is the standard approximation for the log of a ratio of
    Poisson rates with independent person-time:
        SE(log IRR) = sqrt(1/n_comp + 1/n_ref)
    Adequate for n_comp >= 30 and n_ref >= 30; deteriorates at low counts.

    Returns a dict with: irr, log_irr, se_log_irr, lower_ci, upper_ci, z,
    p_raw, n_comp, n_ref. If any count is zero or any person-time is
    non-positive, the result fields are NaN.
    """
    out = {
        "irr": np.nan, "log_irr": np.nan, "se_log_irr": np.nan,
        "lower_ci": np.nan, "upper_ci": np.nan,
        "z": np.nan, "p_raw": np.nan,
        "n_comp": n_comp, "n_ref": n_ref,
    }
    if n_comp <= 0 or n_ref <= 0 or y_comp <= 0 or y_ref <= 0:
        return out
    rate_c = n_comp / y_comp
    rate_r = n_ref / y_ref
    irr = rate_c / rate_r
    log_irr = math.log(irr)
    se = math.sqrt(1.0 / n_comp + 1.0 / n_ref)
    z_quant = norm.ppf(1.0 - alpha / 2.0)
    z_stat = log_irr / se
    p_raw = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    out.update({
        "irr": irr,
        "log_irr": log_irr,
        "se_log_irr": se,
        "lower_ci": math.exp(log_irr - z_quant * se),
        "upper_ci": math.exp(log_irr + z_quant * se),
        "z": z_stat,
        "p_raw": p_raw,
    })
    return out
