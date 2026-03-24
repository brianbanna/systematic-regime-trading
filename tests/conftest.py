"""
Shared test fixtures for the systematic-regime-trading test suite.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_unified_df():
    """Create a synthetic unified DataFrame with 10 tickers, 100 days each."""
    rng = np.random.default_rng(42)
    tickers = [f"TICK{i}" for i in range(10)]
    dates = pd.bdate_range("2020-01-01", periods=100)
    rows = []

    for ticker in tickers:
        base_price = rng.uniform(10, 200)
        prices = [base_price]
        for _ in range(99):
            ret = rng.normal(0.0005, 0.02)
            prices.append(prices[-1] * (1 + ret))

        for i, date in enumerate(dates):
            price = prices[i]
            high = price * (1 + rng.uniform(0, 0.03))
            low = price * (1 - rng.uniform(0, 0.03))
            volume = int(rng.uniform(100000, 10000000))
            rows.append({
                "ticker": ticker,
                "Date": date,
                "Open": price * (1 + rng.normal(0, 0.005)),
                "High": high,
                "Low": low,
                "Close": price,
                "Adj Close": price,
                "Volume": volume,
            })

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def sample_returns_df(sample_unified_df):
    """Unified DataFrame with returns already computed."""
    from systematic_regime_trading.features.volatility import (
        compute_daily_returns_unified,
    )
    return compute_daily_returns_unified(sample_unified_df)


@pytest.fixture
def sample_volatility_series():
    """Synthetic volatility series with 3 distinct regimes."""
    rng = np.random.default_rng(42)
    n = 500

    # Low vol regime (60%)
    low_vol = rng.normal(0.01, 0.002, size=int(n * 0.6))
    # Medium vol regime (25%)
    med_vol = rng.normal(0.025, 0.005, size=int(n * 0.25))
    # High vol regime (15%)
    high_vol = rng.normal(0.05, 0.01, size=int(n * 0.15))

    vol = np.concatenate([low_vol, med_vol, high_vol])
    vol = np.abs(vol)  # volatility must be positive
    rng.shuffle(vol)

    dates = pd.bdate_range("2018-01-01", periods=len(vol))
    return pd.Series(vol, index=dates, name="volatility")
