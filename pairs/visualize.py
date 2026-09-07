"""Visualization helpers for pairs-trading backtests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult


def plot_pairs(prices: pd.DataFrame, ticker1: str, ticker2: str,
               spread: pd.Series, z: pd.Series,
               result: BacktestResult | None = None,
               output: str | Path | None = None,
               show: bool = False) -> Path | None:
    """Three-panel pairs chart:

# Maintenance: last reviewed 2026-09-07 (daily improvement cycle)
        1. Normalized price of both legs
        2. Spread with entry/exit markers
        3. Z-score with entry/exit/stop-loss thresholds
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2]})
    fig.suptitle(f"Pairs: {ticker1} / {ticker2}", fontsize=14)

    # Panel 1: normalized prices
    norm1 = prices[ticker1] / prices[ticker1].iloc[0]
    norm2 = prices[ticker2] / prices[ticker2].iloc[0]
    axes[0].plot(prices.index, norm1, label=ticker1, color="#1f77b4")
    axes[0].plot(prices.index, norm2, label=ticker2, color="#d62728", alpha=0.8)
    axes[0].set_ylabel("Normalized price")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Panel 2: spread + trades
    axes[1].plot(spread.index, spread.values, color="#333333", linewidth=0.9)
    axes[1].axhline(spread.mean(), color="grey", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("Spread")

    # Panel 3: z-score with thresholds
    axes[2].plot(z.index, z.values, color="#2ca02c", linewidth=0.9)
    if result and result.params:
        axes[2].axhline(result.params.get("entry_z", 1.0), color="orange",
                        linestyle="--", linewidth=0.7, label="entry")
        axes[2].axhline(-result.params.get("entry_z", 1.0), color="orange",
                        linestyle="--", linewidth=0.7)
        axes[2].axhline(result.params.get("exit_z", 0.5), color="blue",
                        linestyle=":", linewidth=0.7, label="exit")
        axes[2].axhline(-result.params.get("exit_z", 0.5), color="blue",
                        linestyle=":", linewidth=0.7)
        axes[2].axhline(result.params.get("stop_z", 3.0), color="red",
                        linestyle=":", linewidth=0.7, label="stop")
        axes[2].axhline(-result.params.get("stop_z", 3.0), color="red",
                        linestyle=":", linewidth=0.7)
        axes[2].legend(loc="upper left", fontsize=8)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("Z-score")
    axes[2].grid(alpha=0.3)

    # Mark trades on panels 2 & 3
    if result and result.trades:
        for t in result.trades:
            color = "green" if t.direction == "long_spread" else "magenta"
            axes[1].axvline(t.entry_date, color=color, alpha=0.4, linewidth=0.7)
            axes[2].axvline(t.entry_date, color=color, alpha=0.4, linewidth=0.7)
            axes[1].axvline(t.exit_date, color="black", alpha=0.3, linewidth=0.5)
            axes[2].axvline(t.exit_date, color="black", alpha=0.3, linewidth=0.5)

    axes[2].set_xlabel("Date")

    fig.tight_layout()
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    if show:
        plt.show()
    plt.close(fig)
    return None
