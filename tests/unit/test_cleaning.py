"""Tests for data cleaning module."""

import pandas as pd
import numpy as np
import pytest

from systematic_regime_trading.data.cleaning import (
    clean_ticker_data,
    clean_unified_dataframe,
    fill_missing_data_unified,
    repair_ohlcv_quality,
    generate_validation_report,
)


class TestCleanTickerData:
    def test_removes_duplicate_dates(self):
        df = pd.DataFrame({
            "Date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "Open": [10, 11, 12],
            "Close": [10, 11, 12],
            "High": [10, 11, 12],
            "Low": [10, 11, 12],
            "Adj Close": [10, 11, 12],
            "Volume": [100, 200, 300],
        })
        result = clean_ticker_data({"TEST": df})
        assert len(result["TEST"]) == 2

    def test_converts_to_numeric(self):
        df = pd.DataFrame({
            "Date": ["2020-01-01"],
            "Open": ["10.5"],
            "Close": ["11.0"],
            "High": ["12.0"],
            "Low": ["9.0"],
            "Adj Close": ["11.0"],
            "Volume": ["1000"],
        })
        result = clean_ticker_data({"TEST": df})
        assert result["TEST"]["Open"].dtype in [np.float64, float]


class TestFillMissingData:
    def test_drops_tickers_with_too_much_missing(self):
        dates = pd.date_range("2020-01-01", periods=20)
        # Good ticker: no missing
        good = pd.DataFrame({
            "ticker": "GOOD", "Date": dates,
            "Open": range(20), "High": range(20), "Low": range(20),
            "Close": range(20), "Adj Close": range(20), "Volume": range(20),
        })
        # Bad ticker: large contiguous NaN block that can't be interpolated
        bad_vals = [1.0, 2.0] + [np.nan] * 16 + [19.0, 20.0]
        bad = pd.DataFrame({
            "ticker": "BAD", "Date": dates,
            "Open": bad_vals, "High": bad_vals, "Low": bad_vals,
            "Close": bad_vals, "Adj Close": bad_vals, "Volume": [100] * 20,
        })
        df = pd.concat([good, bad], ignore_index=True)
        result = fill_missing_data_unified(df, max_missing_pct=0.10, interp_window=2)
        assert "GOOD" in result["ticker"].values
        assert "BAD" not in result["ticker"].values

    def test_interpolates_small_gaps(self):
        dates = pd.date_range("2020-01-01", periods=5)
        df = pd.DataFrame({
            "ticker": "TEST", "Date": dates,
            "Open": [10, np.nan, 12, 13, 14],
            "High": [10, 11, 12, 13, 14],
            "Low": [10, 11, 12, 13, 14],
            "Close": [10, 11, 12, 13, 14],
            "Adj Close": [10, 11, 12, 13, 14],
            "Volume": [100, 100, 100, 100, 100],
        })
        result = fill_missing_data_unified(df, interp_window=5)
        assert result["Open"].isna().sum() == 0

    def test_removes_negative_prices(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({
            "ticker": "TEST", "Date": dates,
            "Open": [10, -5, 12],
            "High": [10, 11, 12],
            "Low": [10, 11, 12],
            "Close": [10, 11, 12],
            "Adj Close": [10, 11, 12],
            "Volume": [100, 100, 100],
        })
        result = fill_missing_data_unified(df)
        assert (result["Open"] >= 0).all()


class TestValidationReport:
    def test_valid_data_passes(self, sample_unified_df):
        report = generate_validation_report(sample_unified_df, verbose=False)
        assert report["total_tickers"] == 10
        assert report["total_rows"] == 1000
        assert report["negative_prices"] == 0
