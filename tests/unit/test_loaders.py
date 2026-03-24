"""Tests for data loaders module."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from systematic_regime_trading.data.loaders import (
    combine_dataframes_to_unified,
    load_processed_csv,
    _fallback_risk_free_rate,
)


class TestCombineDataframes:
    def test_combines_multiple_tickers(self):
        data = {
            "AAPL": pd.DataFrame({
                "Date": pd.date_range("2020-01-01", periods=3),
                "ticker": "AAPL", "Close": [100, 101, 102],
            }),
            "GOOG": pd.DataFrame({
                "Date": pd.date_range("2020-01-01", periods=3),
                "ticker": "GOOG", "Close": [200, 201, 202],
            }),
        }
        result = combine_dataframes_to_unified(data)
        assert len(result) == 6
        assert set(result["ticker"].unique()) == {"AAPL", "GOOG"}


class TestLoadProcessedCSV:
    def test_loads_valid_csv(self, tmp_path):
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 3,
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [103, 104, 105],
            "Adj Close": [103, 104, 105],
            "Volume": [1000, 2000, 3000],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        result = load_processed_csv(csv_path)
        assert len(result) == 3
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_raises_on_missing_columns(self, tmp_path):
        df = pd.DataFrame({"ticker": ["AAPL"], "Date": ["2020-01-01"]})
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing expected columns"):
            load_processed_csv(csv_path)


class TestFallbackRiskFreeRate:
    def test_creates_constant_rate(self):
        result = _fallback_risk_free_rate("2020-01-01", "2020-01-31", rate=0.03)
        assert "risk_free_rate" in result.columns
        assert (result["risk_free_rate"] == 0.03).all()
        assert len(result) > 0
