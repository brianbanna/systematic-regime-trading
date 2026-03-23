"""
Market indicator computation functions.

Computes cross-sectional market indicators: direction, volume, breadth,
ATR, correlation. Also computes the mood index (z-score standardization).
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_market_direction_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average daily return across all stocks.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns

    Returns:
        DataFrame with 'Date' as index and 'market_direction' column
    """
    direction_df = df.groupby("Date")["log_return"].mean().reset_index()
    direction_df = direction_df.rename(columns={"log_return": "market_direction"})
    direction_df = direction_df.set_index("Date")
    return direction_df


def compute_market_volume_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average daily change in trading volume across stocks.

    Args:
        df: Table with 'Date', 'ticker', and 'Volume' columns

    Returns:
        DataFrame with 'Date' as index and 'market_volume' column
    """
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    df["log_volume"] = np.log(df["Volume"].replace(0, np.nan))
    df["log_volume_diff"] = df.groupby("ticker")["log_volume"].diff()
    volume_df = df.groupby("Date")["log_volume_diff"].mean().reset_index()
    volume_df = volume_df.rename(columns={"log_volume_diff": "market_volume"})
    volume_df = volume_df.set_index("Date")
    return volume_df


def compute_market_breadth_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of stocks that went up each day.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns

    Returns:
        DataFrame with 'Date' as index and 'market_breadth' column
    """
    df["is_up"] = df["log_return"] > 0
    breadth_df = df.groupby("Date")["is_up"].mean().reset_index()
    breadth_df = breadth_df.rename(columns={"is_up": "market_breadth"})
    breadth_df = breadth_df.set_index("Date")
    return breadth_df


def compute_market_atr_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average True Range per stock, normalized by closing price.

    Args:
        df: Table with 'Date', 'ticker', 'High', 'Low', and 'Adj Close' columns

    Returns:
        DataFrame with 'Date' as index and 'market_atr' column
    """
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    df["prev_close"] = df.groupby("ticker")["Adj Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - df["prev_close"])
    tr3 = abs(df["Low"] - df["prev_close"])
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["normalized_tr"] = true_range / df["Adj Close"]
    atr_df = df.groupby("Date")["normalized_tr"].mean().reset_index()
    atr_df = atr_df.rename(columns={"normalized_tr": "market_atr"})
    atr_df = atr_df.set_index("Date")
    return atr_df


def _mean_pairwise_corr(window_df: pd.DataFrame) -> float:
    """Compute mean pairwise correlation for a window of returns."""
    corr_matrix = window_df.corr()
    n = corr_matrix.shape[0]
    return (corr_matrix.values.sum() - n) / (n * (n - 1)) if n > 1 else np.nan


def compute_market_correlation_dynamic(
    df: pd.DataFrame,
    window: int = 30,
    sample_size: int = 500,
    random_state: int = 42,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Rolling window mean pairwise correlation of stock returns.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns
        window: Number of days to look back
        sample_size: Number of stocks to sample for calculation
        random_state: Seed for reproducible sampling
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with 'Date' as index and 'market_correlation' column
    """
    rng = np.random.default_rng(random_state)

    pivot_df = df.pivot_table(
        index="Date", columns="ticker", values="log_return", aggfunc="first"
    )

    if start_date:
        pivot_df = pivot_df[pivot_df.index >= pd.to_datetime(start_date)]
    if end_date:
        pivot_df = pivot_df[pivot_df.index <= pd.to_datetime(end_date)]

    dates, correlations = [], []
    for i in tqdm(
        range(window, len(pivot_df) + 1), desc="Calculating rolling correlation"
    ):
        window_df = pivot_df.iloc[i - window : i]
        current_date = pivot_df.index[i - 1]

        active_tickers = window_df.dropna(axis=1).columns
        stds_in_window = window_df[active_tickers].std()
        volatile_tickers = stds_in_window[stds_in_window > 0].index

        if len(volatile_tickers) < sample_size:
            correlations.append(np.nan)
        else:
            sampled_tickers = rng.choice(
                volatile_tickers, sample_size, replace=False
            )
            correlations.append(_mean_pairwise_corr(window_df[sampled_tickers]))
        dates.append(current_date)

    return pd.DataFrame(
        data={"market_correlation": correlations},
        index=pd.Index(dates, name="Date"),
    )


def compute_mood_index(
    indicators_dict: dict, window: int = 252
) -> pd.DataFrame:
    """
    Standardize market indicators via rolling z-score.

    Converts each indicator to units of standard deviations from its
    recent rolling average, making all indicators comparable.

    Args:
        indicators_dict: Dict mapping indicator names to DataFrames with
            'Date' as index
        window: Rolling window for z-score normalization

    Returns:
        DataFrame with 'Date' as index and standardized indicators as columns
    """
    moodindex = pd.DataFrame(index=list(indicators_dict.values())[0].index)
    for name, df in indicators_dict.items():
        col = df.columns[0]
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std()
        moodindex[name] = (df[col] - rolling_mean) / rolling_std
    return moodindex
