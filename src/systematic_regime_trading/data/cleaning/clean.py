"""
Data cleaning functions.

Removes duplicates, fixes data types, and fills missing values.
Tickers with too much missing data (>10%) are removed.
"""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict

logger = logging.getLogger(__name__)

def clean_ticker_data(
    data_dict: Dict[str, pd.DataFrame], label: str = "ticker"
) -> Dict[str, pd.DataFrame]:
    """
    Clean each ticker's data.
    Removes duplicate dates, converts prices to numbers, sorts by date.
    Keeps the most recent entry when duplicates are found.

    Args:
        data_dict: Dictionary of ticker -> DataFrame
        label: Name shown in progress bar

    Returns:
        Dictionary with cleaned DataFrames
    """
    initial_count = len(data_dict)
    logger.info(f"Cleaning {initial_count} {label}s")

    for ticker, df in tqdm(data_dict.items(), desc=f"Cleaning {label} data"):
        df = df.drop_duplicates(subset="Date", keep="last")
        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        # Use simple assignment for speed if columns exist
        cols_present = [c for c in numeric_cols if c in df.columns]
        df[cols_present] = df[cols_present].apply(pd.to_numeric, errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        data_dict[ticker] = df

    logger.info(f"Cleaning complete. {len(data_dict)} {label}s processed")
    return data_dict

def clean_unified_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean combined DataFrame with all tickers.

    Removes duplicate ticker-date pairs, converts prices to numbers,
    sorts by ticker then date.

    Args:
        df: DataFrame with 'ticker' and 'Date' columns

    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    logger.info(f"Cleaning unified DataFrame: {initial_rows:,} rows")

    df = df.drop_duplicates(subset=["ticker", "Date"], keep="last")
    
    # Ensure numeric types
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate rows")

    return df

def fill_missing_data_unified(
    df: pd.DataFrame,
    max_tickers: int = None,
    interp_window: int = 5,
) -> pd.DataFrame:
    """
    Unified Single Source of Truth for data repair.
    
    1. Standardizes 'No Data' (0 -> NaN)
    2. Fills gaps via Interpolation
    3. Drops tickers that fail quality standards (>10% missing)
    4. Removes invalid negative prices
    """
    df = df.copy()
    
    # Optional debugging filter
    if max_tickers:
        logger.info(f"Processing limited to first {max_tickers} tickers")
        top_tickers = df["ticker"].unique()[:max_tickers]
        df = df[df["ticker"].isin(top_tickers)].copy()

    initial_rows = len(df)
    initial_tickers = df["ticker"].nunique()
    logger.info(f"Unified Cleaning: {initial_rows:,} rows, {initial_tickers} tickers")

    # 1. Setup
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    target_cols = [c for c in numeric_cols if c in df.columns]

    # 2. Vectorized Sanitization (0 -> NaN)
    # This replaces the logic that was previously in repair_adj_close
    for col in target_cols:
        if (df[col] == 0).any():
            df.loc[df[col] == 0, col] = np.nan

    # 3. Vectorized Interpolation
    logger.info(f"Interpolating gaps (Window: {interp_window} days)...")
    for col in target_cols:
        if df[col].isna().any():
            df[col] = df.groupby("ticker")[col].transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit=interp_window,
                    limit_area="inside",
                    limit_direction="forward",
                )
            )

    # 4. Quality Control (Drop Bad Tickers)
    # We judge a ticker by its worst performing column
    missing_ratios = df.groupby("ticker")[target_cols].apply(
        lambda x: x.isna().mean().max()
    )

    valid_tickers = missing_ratios[missing_ratios <= 0.1].index
    dropped_tickers = missing_ratios[missing_ratios > 0.1].index

    if not dropped_tickers.empty:
        logger.warning(
            f"Dropping {len(dropped_tickers)} tickers with >10% missing data"
        )

    # Filter the DataFrame
    result_df = df[df["ticker"].isin(valid_tickers)].copy()

    # 5. Sanity Check (Negative Prices)
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in result_df.columns]
    if price_cols:
        # Fast numpy check
        negative_mask = (result_df[price_cols].values < 0).any(axis=1)
        
        if negative_mask.any():
            negative_count = negative_mask.sum()
            affected_tickers = result_df.loc[negative_mask, "ticker"].unique()
            logger.warning(
                f"Removed {negative_count:,} rows with negative prices across {len(affected_tickers)} tickers"
            )
            result_df = result_df[~negative_mask].reset_index(drop=True)

    final_tickers = result_df["ticker"].nunique()
    logger.info(
        f"Cleanup Complete. Final: {final_tickers} tickers ({len(result_df):,} rows)"
    )

    return result_df