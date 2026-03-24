"""Tests for universe construction module."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from systematic_regime_trading.data.universe import (
    load_universe,
    validate_universe,
    filter_by_date_range,
)


class TestLoadUniverse:
    def test_loads_from_csv(self, tmp_path):
        csv_path = tmp_path / "tickers.csv"
        pd.DataFrame({"ticker": ["AAPL", "GOOG", "MSFT"]}).to_csv(
            csv_path, index=False,
        )
        config = {"universe": {"source": str(csv_path)}}
        # Monkey-patch get_path to return the direct path
        import systematic_regime_trading.data.universe as uni_mod
        original_get_path = uni_mod.get_path
        uni_mod.get_path = lambda x: Path(x)
        try:
            tickers = load_universe(config)
            assert set(tickers) == {"AAPL", "GOOG", "MSFT"}
        finally:
            uni_mod.get_path = original_get_path

    def test_supports_symbol_column(self, tmp_path):
        csv_path = tmp_path / "tickers.csv"
        pd.DataFrame({"Symbol": ["AAPL", "GOOG"]}).to_csv(csv_path, index=False)
        config = {"universe": {"source": str(csv_path)}}
        import systematic_regime_trading.data.universe as uni_mod
        original_get_path = uni_mod.get_path
        uni_mod.get_path = lambda x: Path(x)
        try:
            tickers = load_universe(config)
            assert set(tickers) == {"AAPL", "GOOG"}
        finally:
            uni_mod.get_path = original_get_path


class TestValidateUniverse:
    def test_filters_missing_tickers(self):
        dates = pd.date_range("2020-01-01", periods=50)
        data = pd.DataFrame({
            "ticker": ["AAPL"] * 50 + ["GOOG"] * 50,
            "Date": list(dates) * 2,
        })
        config = {"universe": {"min_history_days": 30}}
        valid = validate_universe(
            ["AAPL", "GOOG", "MISSING"], data, config=config,
        )
        assert "AAPL" in valid
        assert "GOOG" in valid
        assert "MISSING" not in valid

    def test_filters_insufficient_history(self):
        data = pd.DataFrame({
            "ticker": ["AAPL"] * 50 + ["SHORT"] * 5,
            "Date": list(pd.date_range("2020-01-01", periods=50))
            + list(pd.date_range("2020-01-01", periods=5)),
        })
        config = {"universe": {"min_history_days": 30}}
        valid = validate_universe(["AAPL", "SHORT"], data, config=config)
        assert "AAPL" in valid
        assert "SHORT" not in valid


class TestFilterByDateRange:
    def test_filters_dates(self):
        dates = pd.date_range("2019-01-01", periods=500)
        data = pd.DataFrame({"Date": dates, "Close": range(500)})
        config = {
            "universe": {
                "date_range": {"start": "2019-06-01", "end": "2019-12-31"},
            },
        }
        result = filter_by_date_range(data, config=config)
        assert result["Date"].min() >= pd.Timestamp("2019-06-01")
        assert result["Date"].max() <= pd.Timestamp("2019-12-31")
