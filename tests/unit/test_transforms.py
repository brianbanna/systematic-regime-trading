"""Tests for feature transforms module."""

import pandas as pd
import numpy as np
import pytest

from systematic_regime_trading.features.transforms import (
    compute_z_score,
    rolling_window,
    safe_interpolate,
    compute_statistics,
)


class TestZScore:
    def test_output_has_zero_mean(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        z = compute_z_score(series)
        assert abs(z.mean()) < 1e-10

    def test_output_has_unit_std(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        z = compute_z_score(series)
        assert abs(z.std() - 1.0) < 1e-10


class TestRollingWindow:
    def test_rolling_mean(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = rolling_window(series, window=3, stat="mean")
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 2.0
        assert result.iloc[3] == 3.0

    def test_rolling_std(self):
        series = pd.Series([1, 1, 1, 1, 1])
        result = rolling_window(series, window=3, stat="std")
        # Std of constant series is 0
        assert result.iloc[2] == 0.0


class TestSafeInterpolate:
    def test_fills_gaps(self):
        series = pd.Series([1, np.nan, 3, np.nan, 5])
        result = safe_interpolate(series)
        assert result.isna().sum() == 0
        assert result.iloc[1] == 2.0

    def test_respects_limit(self):
        series = pd.Series([1, np.nan, np.nan, np.nan, 5])
        result = safe_interpolate(series, limit=1)
        # Only 1 NaN should be filled from each direction
        assert result.isna().sum() > 0


class TestComputeStatistics:
    def test_returns_all_stats(self):
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = compute_statistics(df, "values")
        assert "count" in result.index
        assert "mean" in result.index
        assert "std" in result.index
        assert "skewness" in result.index
        assert result.loc["count", "Value"] == 10
        assert result.loc["mean", "Value"] == 5.5
