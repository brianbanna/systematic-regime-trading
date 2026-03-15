"""
Functions to compute simple summary statistics for a column.
"""

import pandas as pd
import numpy as np


def compute_statistics(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    """
    Compute summary stats for a column (count, mean, std, min, max, skew, kurtosis, percentiles).

    Args:
        df: Table with the column to analyze
        param_col: Name of the column

    Returns:
        One-column table of statistics named 'Value'
    """
    # Replace infinite values and drop NaNs to clean data
    param = df[param_col].replace([np.inf, -np.inf], np.nan).dropna()

    # Compute all relevant statistics
    stats_close = {
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

    # Convert to DataFrame for display
    stats_df = pd.DataFrame(stats_close, index=[param_col]).T
    stats_df.columns = ["Value"]

    # Display formatted summary
    print(f"Summary statistics for '{param_col}':\n")
    print(stats_df)

    return stats_df
