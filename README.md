# pairs-trader

A pairs-trading backtester with **Engle-Granger cointegration** testing,
**mean-reversion half-life** estimation, and a simple mean-reversion
backtest engine with stop-loss.

Built for studying statistical arbitrage on equity pairs (e.g. KO/PEP,
XOM/CVX, HD/LOW).

## Features

- **Cointegration test** — Engle-Granger (OLS + ADF on residuals) returns
  hedge ratio, ADF statistic, p-value, and verdict.
- **Half-life** — Estimated from an AR(1) fit on the cointegration
  residual: `half_life = -log(2) / log(1 + phi)`.
- **Z-score signal** — `(spread - rolling_mean) / rolling_std` over a
  rolling window.
- **Backtester** — Mean-reversion pairs strategy:
  - Long the spread when `z < -entry_z`
  - Short the spread when `z > +entry_z`
  - Exit (take-profit) when `|z| < exit_z`
  - Stop-loss when `|z| > stop_z`
  - Logs every trade and computes a daily Sharpe (annualized by √252).
- **Visualization** — Three-panel chart: normalized prices, spread,
  z-score with entry/exit/stop thresholds.
- **CLI** — `python -m pairs --ticker1 KO --ticker2 PEP --start 2015-01-01`
- **Tests** — Synthetic cointegrated series used to validate the
  detector and backtester.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

```bash
python -m pairs \
    --ticker1 KO --ticker2 PEP \
    --start 2015-01-01 \
    --entry-z 1.0 --exit-z 0.5 --stop-z 3.0 \
    --z-window 30 --notional 10000 \
    --plot chart.png --trades-csv trades.csv
```

Sample output:

```
[pairs] downloading KO/PEP from 2015-01-01 ...
[pairs] 2500 aligned bars
[pairs] Engle-Granger: hedge_ratio=0.9521  adf_stat=-3.412  p=0.0104  cointegrated=True
[pairs] Half-life of mean reversion: 87.32 days
[pairs] Trades: 27
[pairs] Total PnL: $1,842.11
[pairs] Sharpe (daily, ann=sqrt(252)): 0.873
[pairs] Win rate: 55.6%
```

## Programmatic use

```python
from pairs.data import get_pair_data
from pairs.cointegration import engle_granger_test, half_life
from pairs.backtest import run_backtest, trades_to_dataframe
from pairs.visualize import plot_pairs

prices = get_pair_data("KO", "PEP", "2015-01-01")
eg = engle_granger_test(prices["KO"], prices["PEP"])
print("cointegrated?", eg["is_cointegrated"])

result = run_backtest(prices, "KO", "PEP")
print(f"sharpe={result.sharpe:.2f}  trades={result.n_trades}")
trades_df = trades_to_dataframe(result.trades)
print(trades_df.head())
```

## Tests

```bash
pytest -v
```

## Project layout

```
pairs-trader/
├── pairs/
│   ├── __init__.py
│   ├── __main__.py        # CLI
│   ├── backtest.py        # mean-reversion backtest engine
│   ├── cointegration.py   # Engle-Granger + half-life + z-score
│   ├── data.py            # yfinance pair loader
│   └── visualize.py       # matplotlib chart
├── tests/
│   ├── test_cointegration.py
│   └── test_backtest.py
├── requirements.txt
├── README.md
└── .gitignore
```

## References

- Engle & Granger (1987), *Co-integration and Error Correction*.
- Chan, *Algorithmic Trading* (mean-reversion pairs chapter).
- Vidyamurthy, *Pairs Trading: Quantitative Methods and Analysis*.

## License

MIT.
