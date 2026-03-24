"""
Data downloaders and loaders.

Supports two modes:
1. Load from existing CSV files (ADA project continuity)
2. Fresh download from yfinance/FRED (for new projects)

All date ranges, tickers, and source configs come from configs/data.yaml.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV loaders (ADA compatibility)
# ---------------------------------------------------------------------------

def load_all_etfs_and_stocks(folder_path: Path, label: str) -> dict:
    """
    Load all CSV files from a folder.

    Each CSV file becomes one entry in the result dictionary.
    The filename (without .csv) is used as the ticker name.

    Args:
        folder_path: Path to folder with CSV files
        label: Name shown in progress bar (e.g., 'etf' or 'stock')

    Returns:
        Dictionary where keys are ticker names and values are DataFrames
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

    Args:
        data_dir: Path to data folder (optional, uses default if not provided)

    Returns:
        Tuple of (etfs_dict, stocks_dict, symbols_meta, companies)
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


def combine_dataframes_to_unified(data_dict: dict) -> pd.DataFrame:
    """
    Combine all ticker DataFrames into a single unified table.

    Args:
        data_dict: Dictionary mapping ticker names to DataFrames

    Returns:
        Single DataFrame with all tickers combined
    """
    logger.info(f"Combining {len(data_dict)} tickers into one table")

    all_dfs = []
    for ticker, df in tqdm(data_dict.items(), desc="Combining tables"):
        all_dfs.append(df)

    unified_df = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"Combined table shape: {unified_df.shape}")

    return unified_df


def load_processed_csv(path: Path) -> pd.DataFrame:
    """
    Load an already-processed CSV (e.g., nasdaq_processed.csv from ADA)
    and convert to the unified DataFrame format.

    Args:
        path: Path to the CSV file

    Returns:
        Unified DataFrame with ticker, Date, and OHLCV columns
    """
    logger.info(f"Loading processed CSV from {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    expected_cols = ["ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df


# ---------------------------------------------------------------------------
# yfinance downloaders
# ---------------------------------------------------------------------------

def download_equity_data(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download daily OHLCV from yfinance for a list of tickers.

    Args:
        tickers: List of ticker symbols
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        batch_size: Number of tickers to download at once

    Returns:
        Unified DataFrame with ticker, Date, OHLCV columns
    """
    import yfinance as yf

    logger.info(f"Downloading {len(tickers)} tickers from yfinance ({start} to {end})")

    all_dfs = []
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch, start=start, end=end,
                group_by="ticker", auto_adjust=False, progress=False,
            )

            if len(batch) == 1:
                ticker = batch[0]
                df = raw.copy()
                df["ticker"] = ticker
                df = df.reset_index()
                all_dfs.append(df)
            else:
                for ticker in batch:
                    try:
                        df = raw[ticker].dropna(how="all").copy()
                        df["ticker"] = ticker
                        df = df.reset_index()
                        all_dfs.append(df)
                    except (KeyError, AttributeError):
                        failed.append(ticker)
        except Exception as e:
            logger.warning(f"Batch download failed: {e}")
            failed.extend(batch)

    if failed:
        logger.warning(f"Failed to download {len(failed)} tickers: {failed[:10]}...")

    if not all_dfs:
        raise RuntimeError("No data downloaded successfully")

    result = pd.concat(all_dfs, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"])
    logger.info(f"Downloaded {result['ticker'].nunique()} tickers, {len(result):,} rows")
    return result


def download_vix(start: str, end: str) -> pd.DataFrame:
    """
    Download VIX and VIX3M from yfinance.

    Args:
        start: Start date string
        end: End date string

    Returns:
        DataFrame with Date, VIX, VIX3M columns
    """
    import yfinance as yf

    logger.info(f"Downloading VIX data ({start} to {end})")

    vix = yf.download("^VIX", start=start, end=end, progress=False)
    vix3m = yf.download("^VIX3M", start=start, end=end, progress=False)

    result = pd.DataFrame(index=vix.index)
    result["VIX"] = vix["Close"]
    result["VIX3M"] = vix3m["Close"].reindex(vix.index)
    result["VIX_contango"] = result["VIX3M"] / result["VIX"] - 1
    result = result.reset_index()
    result.columns = ["Date"] + list(result.columns[1:])
    result["Date"] = pd.to_datetime(result["Date"])

    logger.info(f"Downloaded VIX: {len(result):,} rows")
    return result


def download_risk_free_rate(start: str, end: str, api_key: str = None) -> pd.DataFrame:
    """
    Download 3-month T-bill rate from FRED.

    Args:
        start: Start date string
        end: End date string
        api_key: FRED API key (optional, reads from FRED_API_KEY env var)

    Returns:
        DataFrame with Date, risk_free_rate columns
    """
    logger.info(f"Downloading risk-free rate from FRED ({start} to {end})")

    try:
        from fredapi import Fred

        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            logger.warning("No FRED API key found, using fallback rate")
            return _fallback_risk_free_rate(start, end)

        fred = Fred(api_key=key)
        series = fred.get_series("DGS3MO", observation_start=start, observation_end=end)
        result = pd.DataFrame({"Date": series.index, "risk_free_rate": series.values / 100})
        result["Date"] = pd.to_datetime(result["Date"])
        result = result.dropna()
        logger.info(f"Downloaded risk-free rate: {len(result):,} rows")
        return result

    except Exception as e:
        logger.warning(f"FRED download failed: {e}. Using fallback rate.")
        return _fallback_risk_free_rate(start, end)


def _fallback_risk_free_rate(start: str, end: str, rate: float = 0.02) -> pd.DataFrame:
    """Create a constant risk-free rate series as fallback."""
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({"Date": dates, "risk_free_rate": rate})


def download_bond_returns(start: str, end: str) -> pd.DataFrame:
    """
    Download TLT (long-term Treasury ETF) for 60/40 benchmark.

    Args:
        start: Start date string
        end: End date string

    Returns:
        DataFrame with Date, bond_price, bond_return columns
    """
    import yfinance as yf

    logger.info(f"Downloading TLT bond data ({start} to {end})")

    tlt = yf.download("TLT", start=start, end=end, auto_adjust=False, progress=False)
    result = pd.DataFrame(index=tlt.index)
    result["bond_price"] = tlt["Adj Close"]
    result["bond_return"] = tlt["Adj Close"].pct_change()
    result = result.reset_index()
    result.columns = ["Date"] + list(result.columns[1:])
    result["Date"] = pd.to_datetime(result["Date"])

    logger.info(f"Downloaded TLT: {len(result):,} rows")
    return result
