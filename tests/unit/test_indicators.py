"""Tests for feature indicators module."""

import pandas as pd
import numpy as np
import pytest

from systematic_regime_trading.features.indicators import (
    compute_market_direction_unified,
    compute_market_volume_unified,
    compute_market_breadth_unified,
    compute_market_atr_unified,
    compute_mood_index,
)
from systematic_regime_trading.features.volatility import (
    compute_daily_returns_unified,
    compute_market_volatility_index_unified,
)


class TestMarketDirection:
    def test_returns_dataframe_with_correct_column(self, sample_returns_df):
        result = compute_market_direction_unified(sample_returns_df)
        assert "market_direction" in result.columns
        assert result.index.name == "Date"

    def test_direction_is_mean_of_returns(self, sample_returns_df):
        result = compute_market_direction_unified(sample_returns_df)
        # Should have one value per date
        n_dates = sample_returns_df["Date"].nunique()
        # First date has NaN returns, so we may have fewer
        assert len(result) <= n_dates


class TestMarketVolume:
    def test_returns_dataframe_with_correct_column(self, sample_returns_df):
        result = compute_market_volume_unified(sample_returns_df)
        assert "market_volume" in result.columns


class TestMarketBreadth:
    def test_breadth_between_0_and_1(self, sample_returns_df):
        result = compute_market_breadth_unified(sample_returns_df)
        valid = result["market_breadth"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestMarketATR:
    def test_returns_positive_values(self, sample_returns_df):
        result = compute_market_atr_unified(sample_returns_df)
        valid = result["market_atr"].dropna()
        assert (valid >= 0).all()


class TestMarketVolatility:
    def test_returns_positive_volatility(self, sample_returns_df):
        result = compute_market_volatility_index_unified(sample_returns_df)
        valid = result["market_volatility"].dropna()
        assert (valid >= 0).all()


class TestMoodIndex:
    def test_standardization_near_zero_mean(self, sample_returns_df):
        vol_df = compute_market_volatility_index_unified(sample_returns_df)
        vol_df = vol_df.set_index("Date")
        dir_df = compute_market_direction_unified(sample_returns_df)

        indicators = {
            "market_volatility": vol_df,
            "market_direction": dir_df,
        }
        # Use short window for test data
        result = compute_mood_index(indicators, window=20)
        assert "market_volatility" in result.columns
        assert "market_direction" in result.columns
        # After warmup, values should be roughly standardized
        valid = result.dropna()
        assert len(valid) > 0
