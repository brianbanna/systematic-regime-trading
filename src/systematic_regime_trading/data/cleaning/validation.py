"""
Data validation functions.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_validation_report(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check data quality and return a validation report.

    Args:
        df: DataFrame with ticker, Date, and price columns
        verbose: Print report to console if True

    Returns:
        Dictionary with validation results and pass/fail status
    """
    logger.info(f"Validating data: {len(df):,} rows, {df['ticker'].nunique()} tickers")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_counts = df[numeric_cols].isna().sum()

    zero_prices = (df["Adj Close"] == 0).sum()
    negative_prices = (df["Adj Close"] < 0).sum()
    invalid_dates = df["Date"].isna().sum()

    adj_close_stats = df["Adj Close"].describe()

    extreme_high = (df["Adj Close"] > 10000).sum()
    extreme_low = (df["Adj Close"] < 0.01).sum()

    extreme_high_tickers = df[df["Adj Close"] > 10000]["ticker"].unique()[:5]
    extreme_low_tickers = df[df["Adj Close"] < 0.01]["ticker"].unique()[:5]

    all_valid = (
        missing_counts[["Open", "High", "Low", "Close", "Adj Close"]].sum() == 0
        and missing_counts["Volume"] < len(df) * 0.01
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
        print("\nData validation report")
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
        if negative_prices > 0:
            affected_tickers = df[df["Adj Close"] < 0]["ticker"].unique()
            print(f"  Warning: found {negative_prices:,} rows with negative prices")
            print(f"  Affected tickers: {', '.join(affected_tickers[:10].tolist())}")

        print(f"Invalid dates: {invalid_dates:,}")

        print(f"\nAdj Close statistics:")
        print(f"  Mean: ${adj_close_stats['mean']:.2f}")
        print(f"  Median: ${adj_close_stats['50%']:.2f}")
        print(f"  Min: ${adj_close_stats['min']:.2f}")
        print(f"  Max: ${adj_close_stats['max']:.2f}")

        if extreme_high > 0:
            print(f"\n  Found {extreme_high:,} rows with prices >$10,000")
            print(f"  Examples: {', '.join(extreme_high_tickers.tolist())}")
        if extreme_low > 0:
            print(f"\n  Found {extreme_low:,} rows with prices <$0.01 (penny stocks)")
            print(f"  Examples: {', '.join(extreme_low_tickers.tolist())}")

        print(f"\nFirst few rows:")
        sample_cols = [
            "ticker",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        print(df[sample_cols].head())

        if all_valid:
            print("\nValidation passed")
        else:
            print("\nValidation failed")

    status = "passed" if all_valid else "failed"
    logger.info(f"Validation complete. Status: {status}")

    return report
