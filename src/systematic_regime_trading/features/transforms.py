"""
Data transformation utilities.

Z-score normalization, rolling statistics, interpolation helpers,
and summary statistics computation.
"""

import pandas as pd
import numpy as np


def compute_z_score(series: pd.Series) -> pd.Series:
    """Normalize series to mean 0, std 1."""
    return (series - series.mean()) / series.std()


def rolling_window(
    series: pd.Series, window: int, stat: str = "mean"
) -> pd.Series:
    """
    Compute rolling statistics.

    Args:
        series: Input time series
        window: Rolling window size
        stat: Statistic to compute ('mean', 'std', 'sum', 'min', 'max')

    Returns:
        Series with rolling statistic applied
    """
    roller = series.rolling(window=window)
    return getattr(roller, stat)()


def interpolate_last_n(
    series: pd.Series, n: int = 5
) -> pd.Series:
    """Fill missing values using average of last N valid values."""
    result = series.copy()
    for i in range(len(result)):
        if pd.isna(result.iloc[i]):
            valid_prev = result.iloc[max(0, i - n) : i].dropna()
            if len(valid_prev) > 0:
                result.iloc[i] = valid_prev.mean()
    return result


def safe_interpolate(
    series: pd.Series,
    method: str = "linear",
    limit: int = 5,
    limit_direction: str = "forward",
) -> pd.Series:
    """
    Fill missing values via interpolation with configurable method.

    Args:
        series: Series with missing values
        method: Interpolation method ('linear', 'nearest', etc.)
        limit: Max consecutive NaNs to fill
        limit_direction: Direction to fill ('forward', 'backward', 'both')

    Returns:
        Interpolated series
    """
    # Clamp limit to series length to avoid numpy sliding_window_view error
    safe_limit = min(limit, max(len(series) - 1, 1))
    return series.interpolate(
        method=method, limit=safe_limit, limit_direction=limit_direction,
    )


def compute_statistics(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    """
    Compute summary statistics for a column.

    Args:
        df: Table with the column to analyze
        param_col: Name of the column

    Returns:
        DataFrame of statistics (count, mean, std, min, max, skew, kurtosis, percentiles)
    """
    param = df[param_col].replace([np.inf, -np.inf], np.nan).dropna()

    stats = {
        "count": len(param),
        "mean": param.mean(),
        "std": param.std(),
        "min": param.min(),
        "max": param.max(),
        "skewness": param.skew(),
        "kurtosis": param.kurtosis(),
        "q25": param.quantile(0.25),
        "q50": param.median(),
        "q75": param.quantile(0.75),
    }

    stats_df = pd.DataFrame(stats, index=[param_col]).T
    stats_df.columns = ["Value"]

    return stats_df
