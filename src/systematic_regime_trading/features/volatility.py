"""
Return computation, market volatility index, and volatility smoothing.

Consolidates daily return calculations, cross-sectional volatility,
and rolling smoothing functions.
"""

import pandas as pd
import numpy as np


def compute_daily_returns_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily simple and log returns for multiple tickers.

    Args:
        df: Table with 'ticker', 'Date', and 'Adj Close' columns

    Returns:
        Same table with 'simple_return' and 'log_return' columns added
    """
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    df["simple_return"] = df.groupby("ticker")["Adj Close"].pct_change()

    def safe_log_return(group):
        shifted = group.shift(1)
        price_ratio = group / shifted
        price_ratio = price_ratio.replace([np.inf, -np.inf], np.nan)
        return np.log(price_ratio)

    df["log_return"] = df.groupby("ticker")["Adj Close"].transform(safe_log_return)

    return df


def compute_market_volatility_index_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily market volatility as standard deviation of log returns across tickers.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns

    Returns:
        DataFrame with 'Date' and 'market_volatility' columns
    """
    returns_pivot = df.pivot_table(
        index="Date", columns="ticker", values="log_return", aggfunc="first"
    )

    volatility_df = pd.DataFrame(
        {
            "Date": returns_pivot.index,
            "market_volatility": returns_pivot.std(axis=1, skipna=True),
        }
    )

    return volatility_df


def compute_smoothed_volatility(
    df: pd.DataFrame, windows: list = None
) -> pd.DataFrame:
    """
    Compute rolling averages of market volatility.

    Args:
        df: Table with 'market_volatility' column
        windows: List of window sizes (default: [5, 10, 20, 30])

    Returns:
        Table with original columns plus 'volatility_ma_<window>d' columns
    """
    if windows is None:
        windows = [5, 10, 20, 30]

    smoothed_df = df.copy()
    for window in windows:
        smoothed_df[f"volatility_ma_{window}d"] = (
            df["market_volatility"].rolling(window=window).mean()
        )

    return smoothed_df


def assess_smoothed_volatility(smoothed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate smoothing using autocorrelation and variance reduction.

    Args:
        smoothed_df: Table with 'market_volatility' and rolling average columns

    Returns:
        DataFrame with 'window', 'autocorr_lag1', and 'variance_reduction'
    """
    metrics = []
    raw_var = smoothed_df["market_volatility"].var()

    for col in smoothed_df.columns:
        if col.startswith("volatility_ma_"):
            series = smoothed_df[col].dropna()
            autocorr = series.autocorr(lag=1)
            var_reduction = 1 - (series.var() / raw_var)
            metrics.append({
                "window": col,
                "autocorr_lag1": autocorr,
                "variance_reduction": var_reduction,
            })

    return pd.DataFrame(metrics)
