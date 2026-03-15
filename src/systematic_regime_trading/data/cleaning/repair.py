"""
Functions to fix common data quality problems in stock price data.
"""

import pandas as pd
import logging
from tqdm import tqdm
from typing import Dict

logger = logging.getLogger(__name__)

def repair_ohlcv_quality(
    data_dict: Dict[str, pd.DataFrame], label: str = "ticker"
) -> pd.DataFrame:
    """
    Fixes problems in Open, High, Low, Close, and Volume data.

    Steps:
    1. Fix missing or zero Open prices by using the previous day's Close
    2. Remove tickers where both Open and previous Close are missing
    3. Remove tickers where prices changed but volume was zero

    Args:
        data_dict: Dictionary mapping ticker names to their DataFrames
        label: Name for progress display (e.g., "ticker" or "ETF")

    Returns:
        DataFrame listing which tickers had issues and what was done
    """
    initial_count = len(data_dict)
    logger.info(f"Repairing OHLCV quality for {initial_count} {label}s")

    issues_summary = []
    tickers_to_drop = []

    for ticker, df in tqdm(data_dict.items(), desc=f"Repairing {label} OHLCV quality"):
        # We check for both 0 and NaN because we haven't standardized them yet
        bad_open_mask = (df["Open"] == 0) | (df["Open"].isna())

        if bad_open_mask.any():
            prev_close = df["Close"].shift(1)

            # Critical error: Open is bad AND we don't have a previous Close to fix it
            critical_error_mask = bad_open_mask & (
                (prev_close == 0) | prev_close.isna()
            )

            if critical_error_mask.any():
                tickers_to_drop.append(ticker)
                logger.debug(
                    f"{ticker}: Removed - both Open and previous Close are missing/zero"
                )
                continue # Skip the rest of the checks for this ticker

            # Fix repairable errors
            df.loc[bad_open_mask, "Open"] = prev_close[bad_open_mask]
            issues_summary.append(
                {
                    "ticker": ticker,
                    "issue": "Zero/NaN Open Price",
                    "action": "Replaced with previous close",
                }
            )

        # Volume checks
        bad_vol_mask = (df["Volume"] == 0) | (df["Volume"].isna())
        if bad_vol_mask.any():
            problematic_vol_df = df[bad_vol_mask]
            # If High != Low but Volume is 0, that's impossible price movement
            if (problematic_vol_df["High"] != problematic_vol_df["Low"]).any():
                issues_summary.append(
                    {
                        "ticker": ticker,
                        "issue": "Price movement on zero volume",
                        "action": "Flagged for removal",
                    }
                )
                tickers_to_drop.append(ticker)
                logger.debug(f"{ticker}: Removed - prices changed but volume was zero")

    # Drop bad tickers from the dictionary
    if tickers_to_drop:
        for ticker in tickers_to_drop:
            data_dict.pop(ticker, None)
        logger.info(
            f"Dropped {len(tickers_to_drop)} {label}s due to OHLCV quality issues"
        )

    summary_df = pd.DataFrame(issues_summary)
    
    if summary_df.empty:
        logger.info(f"Repair complete. No issues found. {len(data_dict)} {label}s remaining")
    else:
        logger.info(f"Repair complete. Processed {len(summary_df)} issues.")

    return summary_df