"""
Parquet storage layer.

Read/write DataFrames to Parquet format in data/processed/.
Replaces CSV storage for faster I/O and type preservation.
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime

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


def save_parquet_subdir(df: pd.DataFrame, name: str, subdir: str) -> Path:
    """
    Save DataFrame to Parquet in a subdirectory of data/processed/.

    Args:
        df: DataFrame to save
        name: File name (without .parquet extension)
        subdir: Subdirectory name (e.g., 'auxiliary')

    Returns:
        Path to saved file
    """
    target_dir = _get_processed_dir() / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.parquet"
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info(f"Saved {len(df):,} rows to {path}")
    return path


def load_parquet_subdir(name: str, subdir: str) -> pd.DataFrame:
    """
    Load DataFrame from Parquet in a subdirectory of data/processed/.

    Args:
        name: File name (without .parquet extension)
        subdir: Subdirectory name (e.g., 'auxiliary')

    Returns:
        Loaded DataFrame
    """
    path = _get_processed_dir() / subdir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def update_data_catalog() -> dict:
    """
    Auto-generate data_catalog.yaml listing all processed files
    with creation dates, row counts, and column names.

    Returns:
        Catalog dictionary
    """
    processed_dir = _get_processed_dir()
    catalog = {"generated_at": datetime.now().isoformat(), "files": {}}

    # Scan main directory and subdirectories
    for pq_file in sorted(processed_dir.rglob("*.parquet")):
        rel_path = pq_file.relative_to(processed_dir)
        try:
            df = pd.read_parquet(pq_file, engine="pyarrow")
            entry = {
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "size_mb": round(pq_file.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(
                    pq_file.stat().st_mtime
                ).isoformat(),
            }

            # Add date range if Date column exists
            if "Date" in df.columns:
                entry["date_range"] = {
                    "start": str(df["Date"].min()),
                    "end": str(df["Date"].max()),
                }
            if "ticker" in df.columns:
                entry["n_tickers"] = df["ticker"].nunique()

            catalog["files"][str(rel_path)] = entry
        except Exception as e:
            catalog["files"][str(rel_path)] = {"error": str(e)}

    # Write catalog
    catalog_path = processed_dir / "data_catalog.yaml"
    with open(catalog_path, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)

    logger.info(
        f"Updated data catalog: {len(catalog['files'])} files at {catalog_path}"
    )
    return catalog
