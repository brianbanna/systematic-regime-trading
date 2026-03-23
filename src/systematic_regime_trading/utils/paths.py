"""
Path resolution utilities.

Provides consistent path resolution from project root for all modules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_data_dir(subdir: str = "") -> Path:
    """Get path to data directory (or subdirectory)."""
    path = PROJECT_ROOT / "data"
    if subdir:
        path = path / subdir
    return path


def get_raw_dir() -> Path:
    """Get path to raw data directory."""
    return get_data_dir("raw")


def get_processed_dir() -> Path:
    """Get path to processed data directory."""
    return get_data_dir("processed")


def get_cache_dir() -> Path:
    """Get path to cache directory."""
    return get_data_dir("cache")


def get_results_dir(subdir: str = "") -> Path:
    """Get path to results directory (or subdirectory)."""
    path = PROJECT_ROOT / "results"
    if subdir:
        path = path / subdir
    return path


def get_configs_dir() -> Path:
    """Get path to configs directory."""
    return PROJECT_ROOT / "configs"
