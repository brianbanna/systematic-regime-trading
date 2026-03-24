"""
Data cleaning, repair, and validation functions.

Consolidates per-ticker cleaning, unified DataFrame cleaning,
OHLCV repair, and data quality validation into a single module.
"""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-ticker cleaning
# ---------------------------------------------------------------------------

def clean_ticker_data(
    data_dict: Dict[str, pd.DataFrame], label: str = "ticker"
) -> Dict[str, pd.DataFrame]:
    """
    Clean each ticker's data.
    Removes duplicate dates, converts prices to numbers, sorts by date.
    """
    initial_count = len(data_dict)
    logger.info(f"Cleaning {initial_count} {label}s")

    for ticker, df in tqdm(data_dict.items(), desc=f"Cleaning {label} data"):
        df = df.drop_duplicates(subset="Date", keep="last")
        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols_present = [c for c in numeric_cols if c in df.columns]
        df[cols_present] = df[cols_present].apply(pd.to_numeric, errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        data_dict[ticker] = df

    logger.info(f"Cleaning complete. {len(data_dict)} {label}s processed")
    return data_dict


def clean_unified_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean combined DataFrame with all tickers.
    Removes duplicate ticker-date pairs, converts prices to numbers, sorts.
    """
    initial_rows = len(df)
    logger.info(f"Cleaning unified DataFrame: {initial_rows:,} rows")

    df = df.drop_duplicates(subset=["ticker", "Date"], keep="last")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate rows")

    return df


# ---------------------------------------------------------------------------
# Missing data repair
# ---------------------------------------------------------------------------

def fill_missing_data_unified(
    df: pd.DataFrame,
    max_missing_pct: float = 0.10,
    interp_window: int = 5,
    max_tickers: int = None,
) -> pd.DataFrame:
    """
    Unified data repair pipeline.

    1. Standardizes zeros to NaN
    2. Fills gaps via forward-only interpolation
    3. Drops tickers exceeding max_missing_pct
    4. Removes rows with negative prices
    """
    df = df.copy()

    if max_tickers:
        logger.info(f"Processing limited to first {max_tickers} tickers")
        top_tickers = df["ticker"].unique()[:max_tickers]
        df = df[df["ticker"].isin(top_tickers)].copy()

    initial_rows = len(df)
    initial_tickers = df["ticker"].nunique()
    logger.info(f"Unified Cleaning: {initial_rows:,} rows, {initial_tickers} tickers")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    target_cols = [c for c in numeric_cols if c in df.columns]

    # Vectorized sanitization (0 -> NaN)
    for col in target_cols:
        if (df[col] == 0).any():
            df.loc[df[col] == 0, col] = np.nan

    # Vectorized interpolation (forward-only to prevent lookahead)
    logger.info(f"Interpolating gaps (Window: {interp_window} days)...")
    for col in target_cols:
        if df[col].isna().any():
            df[col] = df.groupby("ticker")[col].transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit=min(interp_window, max(len(x) - 1, 1)),
                    limit_area="inside",
                    limit_direction="forward",
                )
            )

    # Quality control: drop tickers with too much missing data
    # Compute per-ticker missing ratio as the worst column's missing rate
    missing_by_col = df.groupby("ticker")[target_cols].apply(
        lambda x: x.isna().mean(), include_groups=False,
    )
    missing_ratios = missing_by_col.max(axis=1)

    valid_tickers = missing_ratios[missing_ratios <= max_missing_pct].index
    dropped_tickers = missing_ratios[missing_ratios > max_missing_pct].index

    if not dropped_tickers.empty:
        logger.warning(
            f"Dropping {len(dropped_tickers)} tickers with "
            f">{max_missing_pct:.0%} missing data"
        )

    result_df = df[df["ticker"].isin(valid_tickers)].copy()

    # Remove negative prices
    price_cols = [
        c for c in ["Open", "High", "Low", "Close", "Adj Close"]
        if c in result_df.columns
    ]
    if price_cols:
        negative_mask = (result_df[price_cols].values < 0).any(axis=1)
        if negative_mask.any():
            negative_count = negative_mask.sum()
            affected_tickers = result_df.loc[negative_mask, "ticker"].unique()
            logger.warning(
                f"Removed {negative_count:,} rows with negative prices "
                f"across {len(affected_tickers)} tickers"
            )
            result_df = result_df[~negative_mask].reset_index(drop=True)

    final_tickers = result_df["ticker"].nunique()
    logger.info(
        f"Cleanup Complete. Final: {final_tickers} tickers ({len(result_df):,} rows)"
    )

    return result_df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_pipeline(
    raw_data: dict[str, pd.DataFrame],
    config: dict = None,
) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    1. Per-ticker: dedup by date, convert types, sort
    2. Repair OHLCV quality (zero opens, impossible moves)
    3. Combine into unified DataFrame
    4. Cross-ticker dedup
    5. Interpolate missing (forward-only to prevent lookahead)
    6. Drop tickers exceeding missing threshold
    7. Remove negative prices
    8. Validate final output

    Args:
        raw_data: Dictionary mapping ticker symbols to raw DataFrames
        config: Data config dict (from configs/data.yaml). If None, uses defaults.

    Returns:
        Cleaned unified DataFrame
    """
    if config is None:
        from systematic_regime_trading.utils.config import load_config
        config = load_config("data")

    max_missing_pct = config["universe"]["max_missing_pct"]
    interp_window = config["universe"]["interpolation_window"]

    # Step 1: Per-ticker cleaning
    cleaned_data = clean_ticker_data(raw_data)

    # Step 2: Repair OHLCV quality
    repair_ohlcv_quality(cleaned_data)

    # Step 3: Combine into unified DataFrame
    from systematic_regime_trading.data.loaders import combine_dataframes_to_unified
    unified = combine_dataframes_to_unified(cleaned_data)

    # Step 4: Cross-ticker dedup + Step 5-7: Interpolate, drop, remove negatives
    cleaned = clean_unified_dataframe(unified)
    cleaned = fill_missing_data_unified(
        cleaned,
        max_missing_pct=max_missing_pct,
        interp_window=interp_window,
    )

    # Step 8: Validate
    report = generate_validation_report(
        cleaned,
        extreme_high_price=config["cleaning"]["extreme_high_price"],
        extreme_low_price=config["cleaning"]["extreme_low_price"],
        volume_tolerance_pct=config["cleaning"]["volume_tolerance_pct"],
        verbose=True,
    )

    logger.info(
        f"Pipeline complete: {cleaned['ticker'].nunique()} tickers, "
        f"{len(cleaned):,} rows. Valid: {report['is_valid']}"
    )

    return cleaned


# ---------------------------------------------------------------------------
# OHLCV repair
# ---------------------------------------------------------------------------

def repair_ohlcv_quality(
    data_dict: Dict[str, pd.DataFrame], label: str = "ticker"
) -> pd.DataFrame:
    """
    Fix OHLCV data quality issues.

    1. Fix missing/zero Open prices using previous day's Close
    2. Remove tickers where both Open and previous Close are missing
    3. Remove tickers where prices changed but volume was zero
    """
    initial_count = len(data_dict)
    logger.info(f"Repairing OHLCV quality for {initial_count} {label}s")

    issues_summary = []
    tickers_to_drop = []

    for ticker, df in tqdm(
        data_dict.items(), desc=f"Repairing {label} OHLCV quality"
    ):
        bad_open_mask = (df["Open"] == 0) | (df["Open"].isna())

        if bad_open_mask.any():
            prev_close = df["Close"].shift(1)
            critical_error_mask = bad_open_mask & (
                (prev_close == 0) | prev_close.isna()
            )

            if critical_error_mask.any():
                tickers_to_drop.append(ticker)
                logger.debug(
                    f"{ticker}: Removed - both Open and previous Close are missing/zero"
                )
                continue

            df.loc[bad_open_mask, "Open"] = prev_close[bad_open_mask]
            issues_summary.append(
                {"ticker": ticker, "issue": "Zero/NaN Open Price",
                 "action": "Replaced with previous close"}
            )

        bad_vol_mask = (df["Volume"] == 0) | (df["Volume"].isna())
        if bad_vol_mask.any():
            problematic_vol_df = df[bad_vol_mask]
            if (problematic_vol_df["High"] != problematic_vol_df["Low"]).any():
                issues_summary.append(
                    {"ticker": ticker, "issue": "Price movement on zero volume",
                     "action": "Flagged for removal"}
                )
                tickers_to_drop.append(ticker)
                logger.debug(
                    f"{ticker}: Removed - prices changed but volume was zero"
                )

    if tickers_to_drop:
        for ticker in tickers_to_drop:
            data_dict.pop(ticker, None)
        logger.info(
            f"Dropped {len(tickers_to_drop)} {label}s due to OHLCV quality issues"
        )

    summary_df = pd.DataFrame(issues_summary)

    if summary_df.empty:
        logger.info(
            f"Repair complete. No issues found. {len(data_dict)} {label}s remaining"
        )
    else:
        logger.info(f"Repair complete. Processed {len(summary_df)} issues.")

    return summary_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def generate_validation_report(
    df: pd.DataFrame,
    extreme_high_price: float = 10000,
    extreme_low_price: float = 0.01,
    volume_tolerance_pct: float = 0.01,
    verbose: bool = True,
) -> dict:
    """
    Check data quality and return a validation report.

    Returns dict with validation results and pass/fail status.
    """
    logger.info(
        f"Validating data: {len(df):,} rows, {df['ticker'].nunique()} tickers"
    )

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_counts = df[numeric_cols].isna().sum()

    zero_prices = (df["Adj Close"] == 0).sum()
    negative_prices = (df["Adj Close"] < 0).sum()
    invalid_dates = df["Date"].isna().sum()

    adj_close_stats = df["Adj Close"].describe()

    extreme_high = (df["Adj Close"] > extreme_high_price).sum()
    extreme_low = (df["Adj Close"] < extreme_low_price).sum()

    extreme_high_tickers = (
        df[df["Adj Close"] > extreme_high_price]["ticker"].unique()[:5]
    )
    extreme_low_tickers = (
        df[df["Adj Close"] < extreme_low_price]["ticker"].unique()[:5]
    )

    all_valid = (
        missing_counts[["Open", "High", "Low", "Close", "Adj Close"]].sum() == 0
        and missing_counts["Volume"] < len(df) * volume_tolerance_pct
        and zero_prices == 0
        and negative_prices == 0
        and invalid_dates == 0
    )

    report = {
        "total_rows": len(df),
        "total_tickers": df["ticker"].nunique(),
        "date_range": (df["Date"].min(), df["Date"].max()),
        "unique_dates": df["Date"].nunique(),
        "missing_counts": missing_counts.to_dict(),
        "zero_prices": zero_prices,
        "negative_prices": negative_prices,
        "invalid_dates": invalid_dates,
        "adj_close_stats": {
            "mean": adj_close_stats["mean"],
            "median": adj_close_stats["50%"],
            "min": adj_close_stats["min"],
            "max": adj_close_stats["max"],
        },
        "extreme_high_count": extreme_high,
        "extreme_high_tickers": extreme_high_tickers.tolist(),
        "extreme_low_count": extreme_low,
        "extreme_low_tickers": extreme_low_tickers.tolist(),
        "is_valid": all_valid,
    }

    if verbose:
        print(f"\nData validation report")
        print(f"Total rows: {len(df):,}")
        print(f"Total tickers: {df['ticker'].nunique():,}")
        print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Unique dates: {df['Date'].nunique():,}")

        print("\nMissing values:")
        for col in numeric_cols:
            count = missing_counts[col]
            if count > 0:
                pct = (count / len(df)) * 100 if len(df) > 0 else 0
                print(f"  {col}: {count:,} ({pct:.2f}%)")

        print(f"\nZero prices in Adj Close: {zero_prices:,}")
        print(f"Negative prices in Adj Close: {negative_prices:,}")
        print(f"Invalid dates: {invalid_dates:,}")

        print(f"\nAdj Close statistics:")
        print(f"  Mean: ${adj_close_stats['mean']:.2f}")
        print(f"  Median: ${adj_close_stats['50%']:.2f}")
        print(f"  Min: ${adj_close_stats['min']:.2f}")
        print(f"  Max: ${adj_close_stats['max']:.2f}")

        if extreme_high > 0:
            print(
                f"\n  Found {extreme_high:,} rows with "
                f"prices >${extreme_high_price:,.0f}"
            )
            print(f"  Examples: {', '.join(extreme_high_tickers.tolist())}")
        if extreme_low > 0:
            print(
                f"\n  Found {extreme_low:,} rows with "
                f"prices <${extreme_low_price}"
            )
            print(f"  Examples: {', '.join(extreme_low_tickers.tolist())}")

        status = "passed" if all_valid else "failed"
        print(f"\nValidation {status}")

    logger.info(
        f"Validation complete. Status: {'passed' if all_valid else 'failed'}"
    )
    return report
