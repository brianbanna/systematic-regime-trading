"""Tests for signal generation modules."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.signals.regime_signal import (
    regime_to_allocation,
    regime_labels_to_allocation,
)
from systematic_regime_trading.signals.filters import (
    apply_confirmation_filter,
    apply_rate_limit,
    apply_execution_lag,
)
from systematic_regime_trading.signals.vol_target import apply_vol_target
from systematic_regime_trading.signals.generator import generate_all_signals


@pytest.fixture
def sample_probs():
    """Sample regime probabilities for 100 days."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    raw = rng.random((n, 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame({
        "prob_calm": raw[:, 0],
        "prob_moderate": raw[:, 1],
        "prob_turbulent": raw[:, 2],
    }, index=dates)


@pytest.fixture
def sample_labels():
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.integers(0, 3, n), index=dates)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=dates)


@pytest.fixture
def strategy_config():
    return {
        "strategies": {
            "binary_regime": {
                "allocation": {"calm": 1.0, "moderate": 0.5, "turbulent": 0.0},
            },
            "proportional_regime": {
                "method": "probability_weighted",
            },
            "vol_targeted": {
                "vol_target": 0.10,
                "vol_lookback_days": 20,
            },
            "regime_momentum": {
                "persistence_threshold_days": 5,
                "allocation": {
                    "calm_persistent": 1.2, "calm_new": 0.8,
                    "moderate": 0.5, "turbulent": 0.0,
                },
            },
        },
        "signal_filters": {
            "confirmation_days": 3,
            "max_daily_allocation_change": 0.25,
        },
    }


class TestRegimeToAllocation:
    def test_binary_regime(self, sample_probs, strategy_config):
        alloc = regime_to_allocation(sample_probs, "binary_regime", strategy_config)
        assert len(alloc) == 100
        assert (alloc >= 0).all()
        assert (alloc <= 1).all()

    def test_proportional_regime(self, sample_probs, strategy_config):
        alloc = regime_to_allocation(sample_probs, "proportional_regime", strategy_config)
        # Should equal P(calm)
        expected = sample_probs["prob_calm"]
        pd.testing.assert_series_equal(alloc, expected, check_names=False)

    def test_regime_momentum(self, sample_probs, strategy_config):
        alloc = regime_to_allocation(sample_probs, "regime_momentum", strategy_config)
        assert len(alloc) == 100
        # Can go up to 1.2 (calm_persistent)
        assert alloc.max() <= 1.2


class TestLabelsToAllocation:
    def test_maps_correctly(self):
        labels = pd.Series([0, 1, 2, 0, 2])
        alloc_map = {0: 1.0, 1: 0.5, 2: 0.0}
        result = regime_labels_to_allocation(labels, alloc_map)
        assert list(result) == [1.0, 0.5, 0.0, 1.0, 0.0]


class TestConfirmationFilter:
    def test_symmetric_suppresses_brief_regime_changes(self):
        # With symmetric 3-day confirmation, a 1-day blip is filtered
        labels = pd.Series([0]*10 + [2] + [0]*9)
        signal = pd.Series([1.0]*10 + [0.0] + [1.0]*9)
        filtered = apply_confirmation_filter(
            signal, labels,
            confirmation_days=3,
            confirmation_days_defensive=3,
            confirmation_days_risk_on=3,
        )
        assert filtered.iloc[10] == 1.0  # 1-day blip filtered out

    def test_asymmetric_fast_defensive(self):
        # With 1-day defensive confirmation, turbulent triggers immediately
        labels = pd.Series([0]*10 + [2] + [0]*9)
        signal = pd.Series([1.0]*10 + [0.0] + [1.0]*9)
        filtered = apply_confirmation_filter(
            signal, labels,
            confirmation_days=3,
            confirmation_days_defensive=1,
            confirmation_days_risk_on=5,
        )
        assert filtered.iloc[10] == 0.0  # fast defensive: immediate switch


class TestRateLimit:
    def test_limits_change_speed(self):
        signal = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])
        limited = apply_rate_limit(signal, max_daily_change=0.25)
        # Can't jump from 0 to 1 in one step
        assert limited.iloc[1] == 0.25
        assert limited.iloc[2] == 0.50  # approaching 1.0


class TestExecutionLag:
    def test_shifts_by_one_day(self):
        signal = pd.Series([1.0, 0.5, 0.0], index=pd.date_range("2020-01-01", periods=3))
        lagged = apply_execution_lag(signal, lag_days=1)
        assert np.isnan(lagged.iloc[0])
        assert lagged.iloc[1] == 1.0
        assert lagged.iloc[2] == 0.5

    def test_no_lookahead(self):
        """Signal at T should only use data up to T-1."""
        signal = pd.Series(range(5), index=pd.date_range("2020-01-01", periods=5), dtype=float)
        lagged = apply_execution_lag(signal, lag_days=1)
        # At time 1, we should see value from time 0
        assert lagged.iloc[1] == 0.0
        # At time 4, we should see value from time 3
        assert lagged.iloc[4] == 3.0


class TestVolTarget:
    def test_scales_allocation(self, sample_returns):
        allocation = pd.Series(1.0, index=sample_returns.index)
        result = apply_vol_target(
            allocation, sample_returns,
            vol_target=0.10, lookback_days=20, max_leverage=1.5,
        )
        # Should have NaN for warmup period
        assert result.iloc[:19].isna().all()
        # After warmup, values should be clipped to [0, 1.5]
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.5).all()


class TestGenerateAllSignals:
    def test_generates_all_strategies(
        self, sample_probs, sample_labels, sample_returns, strategy_config,
    ):
        backtest_config = {
            "execution": {"lag_days": 1},
            "constraints": {"max_leverage": 1.0, "min_allocation": 0.0},
        }
        result = generate_all_signals(
            sample_probs, sample_labels, sample_returns,
            strategy_config, backtest_config,
        )
        assert "binary_regime" in result.columns
        assert "proportional_regime" in result.columns
        assert "vol_targeted" in result.columns
        assert "regime_momentum" in result.columns

        # All should have NaN at first row (execution lag)
        assert result.iloc[0].isna().all()

        # All valid values should be within constraints
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1.0).all().all()
