"""
Functions to load stock and ETF data from CSV files.
"""

from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def load_all_etfs_and_stocks(folder_path: Path, label: str):
    """
    Load all CSV files from a folder.

    Each CSV file becomes one entry in the result dictionary.
    The filename (without .csv) is used as the ticker name.

    Args:
        folder_path: Path to folder with CSV files
        label: Name shown in progress bar (e.g., 'etf' or 'stock')

    Returns:
        Dictionary where keys are ticker names and values are their data tables
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    data_dict = {}

    for file in tqdm(all_files, desc=f"Loading {label} data"):
        ticker = file.split(".")[0]
        df = pd.read_csv(os.path.join(folder_path, file))
        df["ticker"] = ticker
        df["Date"] = pd.to_datetime(df["Date"])
        data_dict[ticker] = df

    return data_dict


def load_all_data(data_dir: Path = None):
    """
    Load all data: ETFs, stocks, symbol metadata, and company info.

    If no data directory is given, uses the default project data folder.

    Args:
        data_dir: Path to data folder (optional, uses default if not provided)

    Returns:
        Tuple containing:
        - Dictionary of ETF data (ticker -> data table)
        - Dictionary of stock data (ticker -> data table)
        - Symbol metadata table
        - Company info table (None if file doesn't exist)
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        raw_dir = project_root / "data" / "raw" / "stock-market-dataset"
    else:
        raw_dir = Path(data_dir)

    logger.info(f"Loading data from {raw_dir}")

    etfs_dict = load_all_etfs_and_stocks(raw_dir / "etfs", "etf")
    stocks_dict = load_all_etfs_and_stocks(raw_dir / "stocks", "stock")
    symbols_meta = pd.read_csv(raw_dir / "symbols_valid_meta.csv")
    companies_path = raw_dir / "companies.csv"
    companies = pd.read_csv(companies_path) if companies_path.exists() else None

    logger.info(f"Loaded {len(etfs_dict)} ETFs and {len(stocks_dict)} stocks")

    return etfs_dict, stocks_dict, symbols_meta, companies


def combine_dataframes_to_unified(data_dict: dict):
    """
    Combine all data tables into one big table.

    Takes all the separate ticker data tables and stacks them together.
    Each row keeps a 'ticker' column so you know which stock it came from.

    Args:
        data_dict: Dictionary mapping ticker names to their data tables

    Returns:
        Single table with all tickers combined
    """
    logger.info(f"Combining {len(data_dict)} tickers into one table")

    all_dfs = []
    for ticker, df in tqdm(data_dict.items(), desc="Combining tables"):
        all_dfs.append(df)

    unified_df = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"Combined table shape: {unified_df.shape}")

    return unified_df
