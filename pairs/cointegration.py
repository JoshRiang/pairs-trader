"""Cointegration utilities for pairs trading.

Provides:
- Engle-Granger cointegration test (OLS + ADF on residuals)
- Half-life of mean reversion from AR(1) on residuals
- Spread z-score helpers
"""

# Maintenance: last reviewed 2026-09-10 (daily improvement cycle)

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import log
from statsmodels.regression.linear_model import OLS
# statsmodels>=0.14 moved adfuller to statsmodels.tsa.stattools; keep a fallback
try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:  # pragma: no cover - older statsmodels
    from statsmodels.stattools import adfuller
from statsmodels.tools import add_constant


def engle_granger_test(y: pd.Series, x: pd.Series) -> dict:
    """Run the Engle-Granger cointegration test.

    1. Regress y on x (OLS).
    2. Test residuals for stationarity with Augmented Dickey-Fuller.
    3. Return hedge ratio, residual series, ADF stat, p-value, and a verdict.

    Parameters
    ----------
    y, x : pd.Series
        Price series aligned on the same index. y is the dependent leg,
        x is the independent leg.

    Returns
    -------
    dict with keys:
        hedge_ratio (beta from OLS y = alpha + beta*x)
        alpha (intercept)
        residuals (pd.Series)
        adf_stat
        p_value
        is_cointegrated (p_value < 0.05)
        n_obs
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("Not enough observations for Engle-Granger test (need >= 30).")

    y_aligned = df.iloc[:, 0].astype(float)
    x_aligned = df.iloc[:, 1].astype(float)

    X = add_constant(x_aligned.values)
    model = OLS(y_aligned.values, X).fit()

    alpha = float(model.params[0])
    beta = float(model.params[1])
    residuals = pd.Series(y_aligned.values - (alpha + beta * x_aligned.values),
                          index=y_aligned.index, name="residuals")

    adf_stat, p_value, _, _, crit, _ = adfuller(residuals.values, autolag="AIC",
                                               regresults=False)
    return {
        "hedge_ratio": beta,
        "alpha": alpha,
        "residuals": residuals,
        "adf_stat": float(adf_stat),
        "p_value": float(p_value),
        "is_cointegrated": bool(p_value < 0.05),
        "n_obs": int(len(df)),
        "adf_crit_5pct": float(crit["5%"]),
    }


def half_life(residuals: pd.Series) -> float:
    """Estimate half-life of mean reversion from an AR(1) fit on residuals.

    Model:  r_t = phi * r_{t-1} + eps_t
    half_life = -log(2) / log(phi)        (requires 0 < phi < 1)
    """
    r = residuals.dropna().astype(float).values
    if len(r) < 3:
        return float("nan")

    lagged = r[:-1]
    delta = np.diff(r)
    X = add_constant(lagged)
    phi = OLS(delta, X).fit().params[1]

    if phi >= 0 or phi >= 1.0:
        return float("nan")
    return float(-log(2) / log(1.0 + phi))


def compute_spread(y: pd.Series, x: pd.Series, hedge_ratio: float | None = None) -> pd.Series:
    """Compute spread: y - hedge_ratio * x (Engle-Granger residual form)."""
    if hedge_ratio is None:
        result = engle_granger_test(y, x)
        hedge_ratio = result["hedge_ratio"]
    spread = pd.Series(y.values - hedge_ratio * x.values, index=y.index, name="spread")
    return spread


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    s = series.astype(float)
    mean = s.rolling(window=window).mean()
    std = s.rolling(window=window).std()
    z = (s - mean) / std
    return z.rename(f"z_{window}")
