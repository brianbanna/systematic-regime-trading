"""
Functions to compute market indicators and mood indices.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.indicators.returns import compute_daily_returns_unified
from src.data.indicators.volatility import compute_market_volatility_index_unified


def compute_market_direction_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average daily return across all stocks.

    For each day, takes the mean return from all stocks to see if the market
    went up or down overall.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns

    Returns:
        Table with 'Date' as index and 'market_direction' column
    """
    direction_df = df.groupby("Date")["log_return"].mean().reset_index()
    direction_df = direction_df.rename(columns={"log_return": "market_direction"})
    direction_df = direction_df.set_index("Date")
    return direction_df


def compute_market_volume_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average daily change in trading volume across stocks.

    Measures how much trading activity changed from the previous day,
    averaged across all stocks.

    Args:
        df: Table with 'Date', 'ticker', and 'Volume' columns

    Returns:
        Table with 'Date' as index and 'market_volume' column
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
    Calculate the fraction of stocks that went up each day.

    Shows how broad the market movement was - if many stocks went up,
    breadth is high; if only a few went up, breadth is low.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns

    Returns:
        Table with 'Date' as index and 'market_breadth' column
    """
    df["is_up"] = df["log_return"] > 0
    breadth_df = df.groupby("Date")["is_up"].mean().reset_index()
    breadth_df = breadth_df.rename(columns={"is_up": "market_breadth"})
    breadth_df = breadth_df.set_index("Date")
    return breadth_df


def compute_market_atr_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average price range per stock, normalized by closing price.

    Measures how much prices moved up and down during the day, averaged
    across all stocks and divided by the closing price to make it comparable.

    Args:
        df: Table with 'Date', 'ticker', 'High', 'Low', and 'Adj Close' columns

    Returns:
        Table with 'Date' as index and 'market_atr' column
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


def mean_pairwise_corr(window_df):
    corr_matrix = window_df.corr()
    n = corr_matrix.shape[0]
    return (corr_matrix.values.sum() - n) / (n * (n - 1)) if n > 1 else np.nan


def compute_market_correlation_dynamic(
    df: pd.DataFrame,
    window: int = 30,
    sample_size: int = 500,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Calculate how much stock returns move together, using a rolling window.

    For each day, looks at the previous N days and measures how correlated
    stock returns were. Samples a subset of stocks to make the calculation faster.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns
        window: Number of days to look back (default: 30)
        sample_size: Number of stocks to use for calculation (default: 500)
        start_date: Optional start date to filter data
        end_date: Optional end date to filter data

    Returns:
        Table with 'Date' as index and 'market_correlation' column
    """
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
            sampled_tickers = np.random.choice(
                volatile_tickers, sample_size, replace=False
            )
            correlations.append(mean_pairwise_corr(window_df[sampled_tickers]))
        dates.append(current_date)

    return pd.DataFrame(
        data={"market_correlation": correlations}, index=pd.Index(dates, name="Date")
    )


def compute_mood_index(indicators_dict: dict, window: int = 252) -> pd.DataFrame:
    """
    Standardize market indicators by subtracting the rolling mean and dividing by rolling std.

    Makes all indicators comparable by converting them to units of standard deviations
    from their recent average. High values mean the indicator is unusually high,
    low values mean it's unusually low.

    Args:
        indicators_dict: Dictionary mapping indicator names to tables with 'Date' as index
        window: Number of days to use for the rolling average (default: 252)

    Returns:
        Table with 'Date' as index and standardized indicators as columns
    """
    moodindex = pd.DataFrame(index=list(indicators_dict.values())[0].index)
    for name, df in indicators_dict.items():
        col = df.columns[0]
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std()
        moodindex[name] = (df[col] - rolling_mean) / rolling_std
    return moodindex


def compute_dynamic_range(moodindex: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how much each indicator varies over time.

    Computes two measures of spread: standard deviation (overall spread) and
    interquartile range (spread of the middle 50% of values).

    Args:
        moodindex: Table with standardized indicators as columns

    Returns:
        Table with 'std' and 'iqr' columns for each indicator
    """
    dynamic_metrics = pd.DataFrame(index=moodindex.columns)
    dynamic_metrics["std"] = moodindex.std()
    dynamic_metrics["iqr"] = moodindex.quantile(0.75) - moodindex.quantile(0.25)
    return dynamic_metrics


def compute_rolling_corr(
    moodindex: pd.DataFrame, volatility_col: str = "market_volatility", window: int = 30
) -> pd.DataFrame:
    """
    Calculate how each indicator moves together with market volatility over time.

    For each indicator, measures how correlated it is with volatility using
    a rolling window. High correlation means the indicator tends to move
    in the same direction as volatility.

    Args:
        moodindex: Table with 'Date' as index and standardized indicators as columns
        volatility_col: Name of the volatility column to compare against (default: 'market_volatility')
        window: Number of days to use for each correlation calculation (default: 30)

    Returns:
        Table with rolling correlations for each indicator
    """
    rolling_corrs = pd.DataFrame(index=moodindex.index)
    for col in moodindex.columns:
        if col != volatility_col:
            rolling_corrs[col] = (
                moodindex[col].rolling(window).corr(moodindex[volatility_col])
            )
    return rolling_corrs


def compute_subset_pc_scores(subset_df, global_pca_model, feature_cols, sample_size):
    """
    Computes PC1 and PC2 for a subset of stocks using Global PCA weights.

    Parameters:
    - subset_df: DataFrame of the subset (must include 'Adj Close', 'Volume', etc.)
    - global_pca_model: The PCA object fitted on the FULL market.
    - feature_cols: List of column names used in the Global PCA (must be exact order).

    Returns:
    - subset_mood: DataFrame with 'subset_stress_pc2' and 'subset_trend_pc1'
    """

    # 1. Compute Daily Indicators for the Subset
    # We reuse the logic from your Phase 2 notebook
    # Note: We assume the helper functions are imported (compute_daily_returns_unified, etc.)

    print("Computing indicators for subset...")
    df_feats = compute_daily_returns_unified(subset_df)

    # Compute the raw indicators just like before
    vol_df = compute_market_volatility_index_unified(df_feats)
    vol_df.set_index("Date", inplace=True)

    dir_df = compute_market_direction_unified(df_feats)
    vol_obj_df = compute_market_volume_unified(df_feats)
    brd_df = compute_market_breadth_unified(df_feats)
    atr_df = compute_market_atr_unified(df_feats)

    # Correlation needs special handling if subset is small (< sample_size)
    # If subset < 100 stocks, we just calculate correlation on all of them
    n_tickers = subset_df["ticker"].nunique()

    corr_df = compute_market_correlation_dynamic(
        df=df_feats, window=30, sample_size=sample_size
    )

    # 2. Package into Dictionary
    indicators_dict = {
        "market_direction": dir_df,
        "market_volume": vol_obj_df,
        "market_volatility": vol_df,
        "market_correlation": corr_df,
        "market_breadth": brd_df,
        "market_atr": atr_df,
    }

    # 3. Standardize (Compute Mood Index)
    # We MUST use the same window (252) to ensure Z-scores are comparable to Global
    print("Standardizing subset features...")
    subset_mood = compute_mood_index(indicators_dict, window=252)

    # 4. Project onto Global PCA Space
    # Drop NaNs to allow transformation
    valid_data = subset_mood[feature_cols].dropna()

    # CRITICAL: Use .transform(), NOT .fit_transform()
    # This applies the Global Weights to the Subset Data
    projected_scores = global_pca_model.transform(valid_data)

    # 5. Store Results
    subset_mood.loc[valid_data.index, "market_trend_pc1"] = projected_scores[:, 0]
    subset_mood.loc[valid_data.index, "systemic_stress_pc2"] = projected_scores[:, 1]

    return subset_mood
