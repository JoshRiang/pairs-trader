"""Pairs trading backtest engine.

Strategy (canonical mean-reversion pairs trade):
- Long the spread (long y, short beta*x) when z-score > +entry_z
- Short the spread (short y, long beta*x) when z-score < -entry_z
- Exit when |z-score| < exit_z
- Stop loss when |z-score| > stop_z (spread diverged beyond entry)

Each position is held with a fixed notional per leg; PnL = change in
spread * dollar-beta notional. Trades are logged with entry/exit dates,
direction, and realized PnL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np
import pandas as pd

from .cointegration import compute_spread, engle_granger_test, half_life, zscore


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    direction: str          # "long_spread" or "short_spread"
    entry_z: float
    exit_z: float | None
    entry_spread: float
    exit_spread: float | None
    pnl: float               # dollars per unit notional
    bars_held: int
    exit_reason: str         # "take_profit" | "stop_loss" | "end_of_data"


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sharpe: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    params: dict = field(default_factory=dict)


def run_backtest(prices: pd.DataFrame,
                 ticker1: str,
                 ticker2: str,
                 entry_z: float = 1.0,
                 exit_z: float = 0.5,
                 stop_z: float = 3.0,
                 z_window: int = 30,
                 notional_per_leg: float = 10_000.0,
                 use_coint_test: bool = True) -> BacktestResult:
    """Run a pairs backtest on aligned price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Aligned price frame; must contain columns `ticker1`, `ticker2`.
    entry_z : float
        Open a trade when |z| crosses `entry_z`.
    exit_z : float
        Close the trade when |z| falls back below `exit_z`.
    stop_z : float
        Stop-loss when |z| widens beyond `stop_z` (adverse move).
    z_window : int
        Rolling window for z-score mean/std.
    notional_per_leg : float
        Dollar notional assumed for each leg (size of the trade in $).
    use_coint_test : bool
        If True, run the Engle-Granger test first; if it fails the series
        isn't cointegrated, skip the backtest (returns empty result).
    """
    if ticker1 not in prices.columns or ticker2 not in prices.columns:
        raise ValueError(f"prices must contain columns {ticker1}, {ticker2}")

    y = prices[ticker1].astype(float)
    x = prices[ticker2].astype(float)

    # Estimate hedge ratio (with optional cointegration gate).
    if use_coint_test:
        eg = engle_granger_test(y, x)
        if not eg["is_cointegrated"]:
            return BacktestResult(
                params={"entry_z": entry_z, "exit_z": exit_z, "stop_z": stop_z,
                        "z_window": z_window, "notional_per_leg": notional_per_leg,
                        "ticker1": ticker1, "ticker2": ticker2,
                        "coint_p_value": eg["p_value"]},
            )
        beta = eg["hedge_ratio"]
        coint_p = eg["p_value"]
    else:
        beta = float(np.polyfit(x.values, y.values, 1)[0])
        coint_p = float("nan")

    spread = compute_spread(y, x, hedge_ratio=beta)
    z = zscore(spread, window=z_window)

    # Drop warm-up NaNs
    df = pd.concat([spread.rename("spread"), z.rename("z")], axis=1).dropna()
    if df.empty:
        return BacktestResult()

    # Dollar-beta notional: $beta shares of x per 1 share of y, scaled by notional.
    # PnL per unit notional = change in spread divided by entry price (normalized).
    # For simplicity: assume 1 unit = notional dollars split evenly across both legs.
    dollar_per_unit = notional_per_leg  # total notional in the spread position

    trades: List[Trade] = []
    direction = None  # None | "long_spread" | "short_spread"
    entry_idx = None
    entry_spread = None
    entry_z_val = None

    equity = []
    running_pnl = 0.0
    realized = 0.0  # cumulative realized PnL

    prev_spread = None
    for i, (date, row) in enumerate(df.iterrows()):
        s = row["spread"]
        zv = row["z"]

        # Mark-to-market open position
        if direction is not None and prev_spread is not None and entry_spread is not None:
            if direction == "long_spread":
                unreal = (s - prev_spread) * dollar_per_unit / max(abs(entry_spread), 1e-9)
            else:  # short_spread
                unreal = (prev_spread - s) * dollar_per_unit / max(abs(entry_spread), 1e-9)
            equity.append(realized + unreal)
        else:
            equity.append(realized)

        prev_spread = s

        # ENTRY logic (only when flat)
        if direction is None:
            if zv > entry_z:
                direction = "short_spread"
                entry_idx = i
                entry_spread = s
                entry_z_val = zv
            elif zv < -entry_z:
                direction = "long_spread"
                entry_idx = i
                entry_spread = s
                entry_z_val = zv
            continue

        # EXIT logic
        abs_z = abs(zv)
        # Take profit
        if abs_z < exit_z:
            if direction == "long_spread":
                pnl = (s - entry_spread) * dollar_per_unit / max(abs(entry_spread), 1e-9)
            else:
                # short_spread PnL = entry - exit (delta reversed)
                pnl = (entry_spread - s) * dollar_per_unit / max(abs(entry_spread), 1e-9)
            realized += pnl
            trades.append(Trade(
                entry_date=df.index[entry_idx],
                exit_date=date,
                direction=direction,
                entry_z=float(entry_z_val),
                exit_z=float(zv),
                entry_spread=float(entry_spread),
                exit_spread=float(s),
                pnl=float(pnl),
                bars_held=int(i - entry_idx),
                exit_reason="take_profit",
            ))
            direction = None
            entry_idx = None
            entry_spread = None
            entry_z_val = None
            continue

        # Stop loss
        if abs_z > stop_z:
            if direction == "long_spread":
                pnl = (s - entry_spread) * dollar_per_unit / max(abs(entry_spread), 1e-9)
            else:
                pnl = (entry_spread - s) * dollar_per_unit / max(abs(entry_spread), 1e-9)
            realized += pnl
            trades.append(Trade(
                entry_date=df.index[entry_idx],
                exit_date=date,
                direction=direction,
                entry_z=float(entry_z_val),
                exit_z=float(zv),
                entry_spread=float(entry_spread),
                exit_spread=float(s),
                pnl=float(pnl),
                bars_held=int(i - entry_idx),
                exit_reason="stop_loss",
            ))
            direction = None
            entry_idx = None
            entry_spread = None
            entry_z_val = None
            continue

    # Force-close any open position at end of data
    if direction is not None and entry_idx is not None:
        s = df["spread"].iloc[-1]
        zv = df["z"].iloc[-1]
        if direction == "long_spread":
            pnl = (s - entry_spread) * dollar_per_unit / max(abs(entry_spread), 1e-9)
        else:
            pnl = (entry_spread - s) * dollar_per_unit / max(abs(entry_spread), 1e-9)
        realized += pnl
        trades.append(Trade(
            entry_date=df.index[entry_idx],
            exit_date=df.index[-1],
            direction=direction,
            entry_z=float(entry_z_val),
            exit_z=float(zv),
            entry_spread=float(entry_spread),
            exit_spread=float(s),
            pnl=float(pnl),
            bars_held=int(len(df) - 1 - entry_idx),
            exit_reason="end_of_data",
        ))

    equity_curve = pd.Series(equity, index=df.index, name="equity")
    # Sharpe on DAILY equity changes (not pct_change — PnL is additive, not multiplicative)
    rets = equity_curve.diff().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets) > 1 and rets.std() and rets.std() > 0 else 0.0
    total_return = float(equity_curve.iloc[-1]) if len(equity_curve) else 0.0
    pnls = [t.pnl for t in trades]
    win_rate = float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        sharpe=sharpe,
        total_return=total_return,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_pnl=avg_pnl,
        params={"entry_z": entry_z, "exit_z": exit_z, "stop_z": stop_z,
                "z_window": z_window, "notional_per_leg": notional_per_leg,
                "ticker1": ticker1, "ticker2": ticker2,
                "hedge_ratio": beta, "coint_p_value": coint_p,
                "half_life": half_life(spread)},
    )


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert list of Trade to DataFrame for inspection/analysis."""
    if not trades:
        return pd.DataFrame(columns=list(asdict(Trade("", None, "", 0.0, None, 0.0, None, 0.0, 0, "")).keys()))
    return pd.DataFrame([asdict(t) for t in trades])
