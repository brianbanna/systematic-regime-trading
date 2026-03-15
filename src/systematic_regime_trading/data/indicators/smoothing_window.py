"""
Functions to smooth market volatility and evaluate smoothing.
"""

import pandas as pd

def compute_smoothed_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages of market volatility for common window sizes.

    Args:
        df: Table with 'Date' and 'market_volatility' columns

    Returns:
        Table with the original columns plus 'volatility_ma_<window>d' columns
        for windows 5, 10, 20, and 30
    """
    smoothed_df = df.copy()

    # Compute rolling averages for predefined windows
    for window in [5, 10, 20, 30]:
        smoothed_df[f"volatility_ma_{window}d"] = df["market_volatility"].rolling(window=window).mean()

    return smoothed_df


def assess_smoothed_volatility(smoothed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate smoothing using autocorrelation and variance reduction.

    Args:
        smoothed_df: Table with 'market_volatility' and rolling average columns

    Returns:
        Table with 'window', 'autocorr_lag1', and 'variance_reduction'
    """
    metrics = []
    raw_var = smoothed_df["market_volatility"].var()

    # Loop through all rolling average columns
    for col in smoothed_df.columns:
        if col.startswith("volatility_ma_"):
            series = smoothed_df[col].dropna()

            # Compute autocorrelation at lag 1
            autocorr = series.autocorr(lag=1)

            # Compute variance reduction relative to raw volatility
            var_reduction = 1 - (series.var() / raw_var)

            metrics.append({
                "window": col,
                "autocorr_lag1": autocorr,
                "variance_reduction": var_reduction
            })

    metrics_df = pd.DataFrame(metrics)

    # Print nicely formatted metrics for user
    print("\nAssessment of Smoothed Volatility Metrics:\n")
    print(metrics_df.to_string(index=False))

    return metrics_df
