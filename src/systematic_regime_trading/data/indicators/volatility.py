"""
Functions to compute a daily market volatility index.
"""

import pandas as pd

def compute_market_volatility_index_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily market volatility from a unified table.

    Args:
        df: Table with 'Date', 'ticker', and 'log_return' columns

    Returns:
        Table with 'Date' and 'market_volatility'
    """
    # Pivot so each ticker has its own column
    returns_pivot = df.pivot_table(
        index="Date", columns="ticker", values="log_return", aggfunc="first"
    )

    # Compute daily market volatility as standard deviation across tickers
    volatility_df = pd.DataFrame(
        {
            "Date": returns_pivot.index,
            "market_volatility": returns_pivot.std(axis=1, skipna=True),
        }
    )

    return volatility_df
