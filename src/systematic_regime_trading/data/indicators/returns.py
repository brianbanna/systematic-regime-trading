"""
Functions to compute daily simple and log returns.
"""

import pandas as pd
import numpy as np

def compute_daily_returns_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns for a table with multiple tickers.

    Groups by ticker and computes simple and log returns per stock.

    Args:
        df: Table with 'ticker', 'Date', and 'Adj Close' columns

    Returns:
        Same table with 'simple_return' and 'log_return' columns added
    """
    # Ensure sorting by ticker and date
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # Compute simple returns within each ticker group
    df["simple_return"] = df.groupby("ticker")["Adj Close"].pct_change(fill_method=None)

    # Compute log returns safely, avoiding division by zero or invalid values
    def safe_log_return(group):
        shifted = group.shift(1)
        price_ratio = group / shifted
        price_ratio = price_ratio.replace([np.inf, -np.inf], np.nan)
        return np.log(price_ratio)

    df["log_return"] = df.groupby("ticker")["Adj Close"].transform(safe_log_return)

    return df
