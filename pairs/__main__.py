"""Command-line entry point: python -m pairs --ticker1 KO --ticker2 PEP --start 2015-01-01"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backtest import run_backtest, trades_to_dataframe
from .cointegration import engle_granger_test, half_life
from .data import get_pair_data
from .visualize import plot_pairs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m pairs",
        description="Pairs-trading backtester with cointegration.",
    )
    p.add_argument("--ticker1", required=True, help="Dependent leg (e.g. KO)")
    p.add_argument("--ticker2", required=True, help="Independent leg (e.g. PEP)")
    p.add_argument("--start", required=True, help="ISO start date (e.g. 2015-01-01)")
    p.add_argument("--end", default=None, help="ISO end date (default: today)")
    p.add_argument("--entry-z", type=float, default=1.0,
                   help="Z-score threshold to open a trade (default 1.0)")
    p.add_argument("--exit-z", type=float, default=0.5,
                   help="Z-score threshold to close a profitable trade (default 0.5)")
    p.add_argument("--stop-z", type=float, default=3.0,
                   help="Z-score stop-loss threshold (default 3.0)")
    p.add_argument("--z-window", type=int, default=30,
                   help="Rolling window for z-score mean/std (default 30)")
    p.add_argument("--notional", type=float, default=10_000.0,
                   help="Notional per leg in dollars (default 10000)")
    p.add_argument("--no-coint-test", action="store_true",
                   help="Skip the Engle-Granger gate before backtesting")
    p.add_argument("--plot", default=None,
                   help="Path to save PNG chart (default: no chart)")
    p.add_argument("--trades-csv", default=None,
                   help="Path to save trades log CSV")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"[pairs] downloading {args.ticker1}/{args.ticker2} from {args.start} ...")
    prices = get_pair_data(args.ticker1, args.ticker2, args.start, args.end)
    print(f"[pairs] {len(prices)} aligned bars")

    y = prices[args.ticker1]
    x = prices[args.ticker2]

    # Always report the cointegration test for transparency
    try:
        eg = engle_granger_test(y, x)
        print(f"[pairs] Engle-Granger: hedge_ratio={eg['hedge_ratio']:.4f}  "
              f"adf_stat={eg['adf_stat']:.3f}  p={eg['p_value']:.4f}  "
              f"cointegrated={eg['is_cointegrated']}")
        hl = half_life(eg["residuals"])
        print(f"[pairs] Half-life of mean reversion: {hl:.2f} days")
    except Exception as exc:
        print(f"[pairs] cointegration test failed: {exc}", file=sys.stderr)
        eg = None
        hl = float("nan")

    result = run_backtest(
        prices, args.ticker1, args.ticker2,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        z_window=args.z_window,
        notional_per_leg=args.notional,
        use_coint_test=not args.no_coint_test,
    )

    print(f"[pairs] Trades: {result.n_trades}")
    print(f"[pairs] Total PnL: ${result.total_return:,.2f}")
    print(f"[pairs] Sharpe (daily, ann=sqrt(252)): {result.sharpe:.3f}")
    print(f"[pairs] Win rate: {result.win_rate:.1%}")
    print(f"[pairs] Avg PnL per trade: ${result.avg_pnl:,.2f}")

    if result.params:
        print(f"[pairs] Params: {json.dumps(result.params, default=str)}")

    if args.trades_csv:
        trades_df = trades_to_dataframe(result.trades)
        trades_df.to_csv(args.trades_csv, index=False)
        print(f"[pairs] trades saved to {args.trades_csv}")

    if args.plot:
        # Recompute spread/z for plotting (cheap)
        from .cointegration import compute_spread, zscore
        spread = compute_spread(y, x, hedge_ratio=result.params.get("hedge_ratio"))
        z = zscore(spread, window=args.z_window)
        out_path = plot_pairs(prices, args.ticker1, args.ticker2, spread, z,
                              result=result, output=args.plot)
        print(f"[pairs] chart saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
