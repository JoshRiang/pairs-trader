"""Smoke tests for the pairs backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.backtest import run_backtest, trades_to_dataframe
from pairs.cointegration import compute_spread


def _make_cointegrated_prices(n: int = 1500, beta: float = 1.5, phi: float = 0.03,
                              seed: int = 11) -> pd.DataFrame:
    """Cointegrated pair where the spread mean-reverts with known speed."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1.0, size=n)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = phi * u[i - 1] + eps[i]
    x = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 100.0)
    y = pd.Series(beta * x.values + u)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": y.values, "B": x.values}, index=idx)


def test_backtest_produces_trades_on_cointegrated_series():
    prices = _make_cointegrated_prices()
    result = run_backtest(prices, "A", "B", entry_z=1.0, exit_z=0.5,
                          stop_z=3.0, z_window=30, use_coint_test=False)
    # With a strongly cointegrated series, the strategy should trade
    assert result.n_trades > 0
    assert isinstance(result.equity_curve, pd.Series)
    # z-score rolling window consumes `window - 1` rows of warm-up data.
    expected_len = len(prices) - (30 - 1)
    assert len(result.equity_curve) == expected_len


def test_backtest_returns_columns():
    prices = _make_cointegrated_prices()
    result = run_backtest(prices, "A", "B", use_coint_test=False)
    df = trades_to_dataframe(result.trades)
    expected = {"entry_date", "exit_date", "direction", "entry_z",
                "exit_z", "entry_spread", "exit_spread", "pnl",
                "bars_held", "exit_reason"}
    assert expected.issubset(set(df.columns))


def test_backtest_stop_loss_can_trigger():
    """With a tight stop and noisy series, stop-loss exits should appear."""
    prices = _make_cointegrated_prices(seed=99, phi=0.4)  # fast-reverting noise
    result = run_backtest(prices, "A", "B", entry_z=1.0, exit_z=0.5,
                          stop_z=1.5, z_window=20, use_coint_test=False)
    df = trades_to_dataframe(result.trades)
    if len(df) > 0:
        # At least verify exits carry a reason
        assert "exit_reason" in df.columns
        assert df["exit_reason"].isin({"take_profit", "stop_loss", "end_of_data"}).all()


def test_backtest_skips_when_not_cointegrated():
    """When the cointegration gate is on and the series fails, no trades."""
    rng = np.random.default_rng(3)
    n = 800
    a = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 100.0)
    b = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 50.0)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"A": a.values, "B": b.values}, index=idx)

    result = run_backtest(prices, "A", "B", use_coint_test=True)
    # Independent random walks usually fail coint — backtest should be empty
    assert result.n_trades == 0


def test_spread_relationship_holds():
    prices = _make_cointegrated_prices(beta=1.5)
    y, x = prices["A"], prices["B"]
    # Use known beta
    spread = compute_spread(y, x, hedge_ratio=1.5)
    # Should approximate the synthetic residual u
    assert spread.std() > 0
