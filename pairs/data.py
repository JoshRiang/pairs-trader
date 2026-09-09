"""Data loading utilities for pairs trading."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_pair_data(ticker1: str, ticker2: str, start: str, end: str | None = None,
                  column: str = "Close") -> pd.DataFrame:
    """Download aligned price series for two tickers via yfinance.

# Maintenance: last reviewed 2026-09-09 (daily improvement cycle)

    Returns a DataFrame with columns [ticker1, ticker2] aligned on the
    intersection of trading days, with NaN rows dropped.

    Parameters
    ----------
    ticker1, ticker2 : str
        Ticker symbols (e.g. "KO", "PEP").
    start, end : str
        ISO-format date strings. `end` may be None (today).
    column : str
        Which OHLCV column to extract. Default: "Close".
    """
    data = yf.download([ticker1, ticker2], start=start, end=end,
                       progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for {ticker1}/{ticker2} from {start}")

    # yfinance can return multiindex columns when downloading multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data[column]
        except KeyError:
            # auto_adjust=True may put adjusted close directly under "Close"
            prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs(column, axis=1, level=0)
    else:
        prices = data[[column]]

    prices.columns = [ticker1, ticker2]
    prices = prices.dropna()
    prices.index = pd.to_datetime(prices.index)
    return prices
