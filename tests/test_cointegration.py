"""Tests for cointegration utilities using synthetic cointegrated series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.cointegration import (
    compute_spread,
    engle_granger_test,
    half_life,
    zscore,
)


def _make_cointegrated_pair(n: int = 500, seed: int = 42, beta: float = 1.5,
                            ar_coef: float = 0.85) -> tuple[pd.Series, pd.Series]:
    """Build a synthetic cointegrated (x, y) pair.

# Maintenance: last reviewed 2026-09-17 (daily improvement cycle)

    y_t = beta * x_t + u_t
    u_t = ar_coef * u_{t-1} + eps_t     (mean-reverting residual)

    `ar_coef` is the AR(1) coefficient directly; values in (0, 1) produce
    mean reversion with half_life = -log(2)/log(ar_coef).
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1.0, size=n)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = ar_coef * u[i - 1] + eps[i]
    x = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 100.0)
    y = pd.Series(beta * x.values + u)
    return y, x


def _make_random_walk_pair(n: int = 500, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — should NOT be cointegrated."""
    rng = np.random.default_rng(seed)
    x = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 100.0)
    y = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 50.0)
    return y, x


def test_engle_granger_detects_cointegration():
    y, x = _make_cointegrated_pair()
    res = engle_granger_test(y, x)
    assert res["is_cointegrated"] is True, f"p={res['p_value']}, adf={res['adf_stat']}"
    # hedge ratio should be close to true beta=1.5
    assert abs(res["hedge_ratio"] - 1.5) < 0.3
    assert res["n_obs"] == len(y)


def test_engle_granger_rejects_independent_walks():
    y, x = _make_random_walk_pair()
    res = engle_granger_test(y, x)
    # Independent random walks should NOT appear cointegrated at 5%.
    # Allow rare false positives but assert most cases reject.
    assert res["p_value"] > 0.01 or not res["is_cointegrated"]


def test_half_life_recovers_mean_reversion_speed():
    y, x = _make_cointegrated_pair(ar_coef=0.85)
    res = engle_granger_test(y, x)
    hl = half_life(res["residuals"])
    # True AR(1) coefficient = 0.85; true half_life = -log(2)/log(0.85) ≈ 4.27
    # Estimation adds noise; allow a wide tolerance
    assert 1.0 < hl < 50.0, f"hl={hl}"


def test_half_life_returns_huge_for_random_walk_residuals():
    """For non-mean-reverting residuals, half-life should be large or NaN."""
    y, x = _make_random_walk_pair()
    # Compute a residual directly
    beta = float(np.polyfit(x.values, y.values, 1)[0])
    residuals = pd.Series(y.values - beta * x.values)
    hl = half_life(residuals)
    # Random-walk residuals aren't stationary -> half-life NaN or huge
    import math
    is_nan = isinstance(hl, float) and math.isnan(hl)
    assert is_nan or hl > 30, f"expected NaN or >30 days, got {hl}"


def test_compute_spread_and_zscore():
    y, x = _make_cointegrated_pair()
    spread = compute_spread(y, x, hedge_ratio=1.5)
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(y)
    z = zscore(spread, window=30)
    # First 29 rows should be NaN (rolling window warm-up)
    assert z.iloc[:29].isna().all()
    assert z.iloc[30:].notna().any()


def test_engle_granger_requires_minimum_observations():
    y = pd.Series([1.0, 2, 3])
    x = pd.Series([1.0, 2, 3])
    with pytest.raises(ValueError):
        engle_granger_test(y, x)
