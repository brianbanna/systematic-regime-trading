"""
Universe construction and validation.

Loads the pre-selected ticker universe from a cached file.
The ADA project identified the optimal 295-ticker subset;
this module loads and validates it rather than re-running the search.
"""

import pandas as pd
import logging
from pathlib import Path

from systematic_regime_trading.utils.config import load_config, get_path

logger = logging.getLogger(__name__)


def load_universe(config: dict = None) -> list[str]:
    """
    Load the pre-selected ticker universe from cached file.

    Args:
        config: Data config dict. If None, loads from data.yaml.

    Returns:
        List of ticker symbols
    """
    if config is None:
        config = load_config("data")

    source = config["universe"]["source"]
    path = get_path(source)

    if not path.exists():
        # Try under data/ prefix
        path = get_path(f"data/{source}")

    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {path}. "
            f"Place the ticker list CSV at {get_path(source)} "
            f"or data/{source}"
        )

    df = pd.read_csv(path)

    # Support multiple column name conventions
    ticker_col = None
    for col in ["ticker", "Ticker", "Symbol", "symbol", "tickers"]:
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        # If single column, assume it's tickers
        if len(df.columns) == 1:
            ticker_col = df.columns[0]
        else:
            raise ValueError(
                f"Cannot identify ticker column in {path}. "
                f"Columns: {list(df.columns)}"
            )

    tickers = df[ticker_col].dropna().unique().tolist()
    logger.info(f"Loaded universe: {len(tickers)} tickers from {path.name}")
    return tickers


def validate_universe(
    tickers: list[str],
    data: pd.DataFrame,
    min_history_days: int = None,
    config: dict = None,
) -> list[str]:
    """
    Check which tickers have sufficient data in the date range.

    Args:
        tickers: List of ticker symbols to validate
        data: Unified DataFrame with 'ticker' and 'Date' columns
        min_history_days: Minimum number of trading days required
        config: Data config dict. If None, loads from data.yaml.

    Returns:
        List of valid tickers with enough history
    """
    if config is None:
        config = load_config("data")

    if min_history_days is None:
        min_history_days = config["universe"]["min_history_days"]

    available_tickers = set(data["ticker"].unique())
    missing = [t for t in tickers if t not in available_tickers]
    present = [t for t in tickers if t in available_tickers]

    if missing:
        logger.warning(
            f"{len(missing)} tickers not found in data: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Check history length
    history_counts = data[data["ticker"].isin(present)].groupby("ticker")["Date"].count()
    valid = history_counts[history_counts >= min_history_days].index.tolist()
    insufficient = history_counts[history_counts < min_history_days].index.tolist()

    if insufficient:
        logger.warning(
            f"{len(insufficient)} tickers have fewer than "
            f"{min_history_days} trading days: {insufficient[:10]}"
        )

    logger.info(
        f"Universe validation: {len(valid)}/{len(tickers)} tickers valid "
        f"({len(missing)} missing, {len(insufficient)} insufficient history)"
    )
    return valid


def filter_by_date_range(
    data: pd.DataFrame,
    start: str = None,
    end: str = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    Filter data to configured date range.

    Args:
        data: Unified DataFrame
        start: Start date (overrides config)
        end: End date (overrides config)
        config: Data config dict

    Returns:
        Filtered DataFrame
    """
    if config is None:
        config = load_config("data")

    start = start or config["universe"]["date_range"]["start"]
    end = end or config["universe"]["date_range"]["end"]

    mask = (data["Date"] >= pd.Timestamp(start)) & (data["Date"] <= pd.Timestamp(end))
    filtered = data[mask].copy()

    logger.info(
        f"Date filter [{start} to {end}]: "
        f"{len(data):,} -> {len(filtered):,} rows"
    )
    return filtered
