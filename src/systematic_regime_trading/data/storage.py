"""
Parquet storage layer.

Read/write DataFrames to Parquet format in data/processed/.
Replaces CSV storage for faster I/O and type preservation.
"""

import pandas as pd
import logging
from pathlib import Path

from systematic_regime_trading.utils.config import get_path

logger = logging.getLogger(__name__)


def _get_processed_dir() -> Path:
    """Get the processed data directory, creating it if needed."""
    processed_dir = get_path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    """
    Save DataFrame to Parquet in data/processed/.

    Args:
        df: DataFrame to save
        name: File name (without .parquet extension)

    Returns:
        Path to saved file
    """
    path = _get_processed_dir() / f"{name}.parquet"
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info(f"Saved {len(df):,} rows to {path}")
    return path


def load_parquet(name: str) -> pd.DataFrame:
    """
    Load DataFrame from Parquet in data/processed/.

    Args:
        name: File name (without .parquet extension)

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = _get_processed_dir() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def exists(name: str) -> bool:
    """
    Check if a processed Parquet file exists.

    Args:
        name: File name (without .parquet extension)

    Returns:
        True if file exists
    """
    path = _get_processed_dir() / f"{name}.parquet"
    return path.exists()


def list_parquet_files() -> list:
    """List all Parquet files in data/processed/."""
    processed_dir = _get_processed_dir()
    return sorted([f.stem for f in processed_dir.glob("*.parquet")])
