"""
Helper utilities: z-score, rolling windows, interpolation, and I/O helpers.
"""

import pandas as pd
import numpy as np


def compute_z_score(series: pd.Series):
    """
    Normalize a series to have mean 0 and standard deviation 1.

    Args:
        series: Input series

    Returns:
        Normalized series
    """
    return (series - series.mean()) / series.std()


def rolling_window(series: pd.Series, window: int, func="mean"):
    """
    Compute rolling statistics over a window of values.

    Args:
        series: Input series
        window: Number of values to include in each window
        func: Function to apply ('mean', 'std', 'sum', etc.)

    Returns:
        Series with rolling values
    """
    return series.rolling(window=window).agg(func)


def interpolate_last_n(series: pd.Series, window: int = 5) -> pd.Series:
    """
    Fill missing values using the average of the last N valid values.

    Skips zeros and only fills if there are valid previous values.

    Args:
        series: Series with missing or zero values
        window: How many previous values to look back

    Returns:
        Series with filled values
    """
    series = series.copy()

    for i in range(len(series)):
        if pd.isna(series.iloc[i]) or series.iloc[i] == 0:
            # Get last N values before this index
            start_idx = max(0, i - window)
            prev_vals = series.iloc[start_idx:i].dropna()

            # Remove zeros from consideration
            prev_vals = prev_vals[prev_vals != 0]

            # Only interpolate if we have valid previous data
            if len(prev_vals) > 0:
                series.iloc[i] = prev_vals.mean()

    return series


def safe_interpolate(
    df: pd.DataFrame, columns: list, method: str = "linear", limit: int = 5
) -> pd.DataFrame:
    """
    Fill missing values by estimating between known values.

    Treats zeros as missing values before filling gaps.

    Args:
        df: DataFrame to process
        columns: Column names to fill
        method: How to estimate values ('linear', 'time', etc.)
        limit: Maximum consecutive missing values to fill

    Returns:
        DataFrame with filled values
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            # Replace zeros with NaN for proper interpolation
            df.loc[df[col] == 0, col] = np.nan

            # Interpolate with limit
            df[col] = df[col].interpolate(
                method=method,
                limit=limit,
                limit_area="inside",
                limit_direction="forward",
            )

    return df
