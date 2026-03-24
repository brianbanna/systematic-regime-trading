"""
Data pipeline entry point.

Usage:
    python -m systematic_regime_trading.data [--fresh] [--validate-only]

Modes:
    default:        Load from existing CSV/Parquet, clean, save
    --fresh:        Download fresh data from yfinance
    --validate-only: Just validate existing processed data
"""

import argparse
import logging
import sys

from systematic_regime_trading.utils.config import load_config
from systematic_regime_trading.data.storage import (
    save_parquet,
    load_parquet,
    exists,
    save_parquet_subdir,
    update_data_catalog,
)
from systematic_regime_trading.data.loaders import (
    load_processed_csv,
    download_equity_data,
    download_vix,
    download_risk_free_rate,
    download_bond_returns,
)
from systematic_regime_trading.data.cleaning import (
    clean_pipeline,
    generate_validation_report,
)
from systematic_regime_trading.data.universe import (
    load_universe,
    validate_universe,
    filter_by_date_range,
)
from systematic_regime_trading.utils.config import get_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(fresh: bool = False, validate_only: bool = False):
    """Run the full data pipeline."""
    config = load_config("data")

    # --- Validate only mode ---
    if validate_only:
        if not exists("equity_data"):
            logger.error("No processed data found. Run pipeline first.")
            sys.exit(1)
        df = load_parquet("equity_data")
        report = generate_validation_report(df)
        sys.exit(0 if report["is_valid"] else 1)

    # --- Step 1: Load or download raw data ---
    if fresh:
        logger.info("=== FRESH DOWNLOAD MODE ===")
        try:
            tickers = load_universe(config)
        except FileNotFoundError:
            logger.error(
                "Universe file not found. Place ticker list at "
                f"{config['universe']['source']}"
            )
            sys.exit(1)

        start = config["universe"]["date_range"]["start"]
        end = config["universe"]["date_range"]["end"]

        raw_df = download_equity_data(tickers, start, end)
        save_parquet(raw_df, "equity_raw")
    else:
        logger.info("=== LOADING FROM EXISTING DATA ===")
        if exists("equity_raw"):
            raw_df = load_parquet("equity_raw")
        else:
            # Try loading from ADA-era processed CSV
            csv_candidates = [
                get_path("data/raw/nasdaq_processed.csv"),
                get_path("data/cache/nasdaq_processed.csv"),
                get_path("data/nasdaq_processed.csv"),
            ]
            csv_path = None
            for p in csv_candidates:
                if p.exists():
                    csv_path = p
                    break

            if csv_path is None:
                logger.error(
                    "No raw data found. Run with --fresh to download, "
                    "or place nasdaq_processed.csv in data/raw/"
                )
                sys.exit(1)

            raw_df = load_processed_csv(csv_path)
            save_parquet(raw_df, "equity_raw")

    # --- Step 2: Filter by date range ---
    raw_df = filter_by_date_range(raw_df, config=config)

    # --- Step 3: Clean data ---
    logger.info("=== CLEANING DATA ===")
    # Since raw_df is already unified, we apply unified cleaning directly
    from systematic_regime_trading.data.cleaning import (
        clean_unified_dataframe,
        fill_missing_data_unified,
    )

    cleaned = clean_unified_dataframe(raw_df)
    cleaned = fill_missing_data_unified(
        cleaned,
        max_missing_pct=config["universe"]["max_missing_pct"],
        interp_window=config["universe"]["interpolation_window"],
    )

    # --- Step 4: Validate universe ---
    try:
        universe_tickers = load_universe(config)
        valid_tickers = validate_universe(universe_tickers, cleaned, config=config)
        cleaned = cleaned[cleaned["ticker"].isin(valid_tickers)].copy()
        logger.info(f"Filtered to universe: {len(valid_tickers)} tickers")
    except FileNotFoundError:
        logger.warning(
            "Universe file not found. Using all available tickers."
        )

    # --- Step 5: Save processed data ---
    save_parquet(cleaned, "equity_data")

    # --- Step 6: Download auxiliary data ---
    _download_auxiliary_data(config)

    # --- Step 7: Update data catalog ---
    update_data_catalog()

    # --- Step 8: Print summary ---
    logger.info("=== PIPELINE COMPLETE ===")
    report = generate_validation_report(cleaned, verbose=True)

    return cleaned


def _download_auxiliary_data(config: dict):
    """Download and save auxiliary data (VIX, risk-free rate, bonds)."""
    start = config["universe"]["date_range"]["start"]
    end = config["universe"]["date_range"]["end"]

    # VIX
    try:
        vix = download_vix(start, end)
        save_parquet_subdir(vix, "vix", "auxiliary")
    except Exception as e:
        logger.warning(f"VIX download failed: {e}")

    # Risk-free rate
    try:
        rf = download_risk_free_rate(start, end)
        save_parquet_subdir(rf, "risk_free_rate", "auxiliary")
    except Exception as e:
        logger.warning(f"Risk-free rate download failed: {e}")

    # Bond returns (TLT)
    try:
        bonds = download_bond_returns(start, end)
        save_parquet_subdir(bonds, "tlt_bonds", "auxiliary")
    except Exception as e:
        logger.warning(f"Bond data download failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Data pipeline for systematic regime trading")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Download fresh data from yfinance instead of using cached data",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing processed data",
    )
    args = parser.parse_args()

    run_pipeline(fresh=args.fresh, validate_only=args.validate_only)


if __name__ == "__main__":
    main()
