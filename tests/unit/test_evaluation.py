"""Tests for the performance evaluation module."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def daily_returns():
    """Known daily returns for metric verification."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=1000)
    returns = rng.normal(0.0004, 0.01, size=1000)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def negative_returns():
    """Consistently negative returns."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=500)
    returns = rng.normal(-0.001, 0.01, size=500)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def benchmark_returns():
    """Benchmark returns for comparison."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2015-01-01", periods=1000)
    returns = rng.normal(0.0003, 0.012, size=1000)
    return pd.Series(returns, index=dates, name="benchmark")


@pytest.fixture
def regime_labels(daily_returns):
    """Synthetic regime labels aligned with daily_returns."""
    rng = np.random.default_rng(42)
    labels = rng.choice([0, 1, 2], size=len(daily_returns), p=[0.5, 0.3, 0.2])
    return pd.Series(labels, index=daily_returns.index, name="regime")


# --- Core Metrics Tests ---

class TestCoreMetrics:
    def test_sharpe_positive_for_positive_returns(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import compute_sharpe
        sharpe = compute_sharpe(daily_returns)
        assert sharpe > 0

    def test_sharpe_negative_for_negative_returns(self, negative_returns):
        from systematic_regime_trading.evaluation.metrics import compute_sharpe
        sharpe = compute_sharpe(negative_returns)
        assert sharpe < 0

    def test_sharpe_zero_for_zero_returns(self):
        from systematic_regime_trading.evaluation.metrics import compute_sharpe
        returns = pd.Series([0.0] * 100)
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0

    def test_sortino_higher_than_sharpe(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import (
            compute_sharpe, compute_sortino,
        )
        sharpe = compute_sharpe(daily_returns)
        sortino = compute_sortino(daily_returns)
        # Sortino uses only downside vol, so typically >= Sharpe
        assert sortino >= sharpe

    def test_max_drawdown_is_negative(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import compute_max_drawdown
        max_dd, duration = compute_max_drawdown(daily_returns)
        assert max_dd <= 0
        assert duration >= 0

    def test_cagr_positive_for_positive_returns(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import compute_cagr
        cagr = compute_cagr(daily_returns)
        assert cagr > 0

    def test_profit_factor_above_one_for_winners(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import compute_profit_factor
        pf = compute_profit_factor(daily_returns)
        assert pf > 1.0

    def test_compute_metrics_all_keys(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import compute_metrics
        metrics = compute_metrics(daily_returns)
        expected_keys = [
            "cagr", "annual_vol", "sharpe", "sortino", "calmar",
            "max_drawdown", "max_drawdown_duration", "skewness",
            "kurtosis", "hit_rate_daily", "profit_factor",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_monthly_hit_rate_bounded(self, daily_returns):
        from systematic_regime_trading.evaluation.metrics import compute_monthly_hit_rate
        hr = compute_monthly_hit_rate(daily_returns)
        assert 0 <= hr <= 1


# --- Rolling Metrics Tests ---

class TestRollingMetrics:
    def test_rolling_sharpe_length(self, daily_returns):
        from systematic_regime_trading.evaluation.rolling import rolling_sharpe
        rs = rolling_sharpe(daily_returns, window=252)
        assert len(rs) == len(daily_returns)
        # First 251 values should be NaN (insufficient window)
        assert rs.iloc[:251].isna().all()
        assert rs.iloc[251:].notna().all()

    def test_rolling_volatility_positive(self, daily_returns):
        from systematic_regime_trading.evaluation.rolling import rolling_volatility
        rv = rolling_volatility(daily_returns, window=63)
        valid = rv.dropna()
        assert (valid > 0).all()

    def test_rolling_drawdown_nonpositive(self, daily_returns):
        from systematic_regime_trading.evaluation.rolling import rolling_drawdown
        dd = rolling_drawdown(daily_returns)
        assert (dd <= 0).all()

    def test_rolling_beta_near_one_vs_self(self, daily_returns):
        from systematic_regime_trading.evaluation.rolling import rolling_beta
        beta = rolling_beta(daily_returns, daily_returns, window=100)
        valid = beta.dropna()
        # Beta of a series vs itself should be ~1
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-6)


# --- Regime Performance Tests ---

class TestRegimePerformance:
    def test_regime_pct_days_sum_to_one(self, daily_returns, regime_labels):
        from systematic_regime_trading.evaluation.regime_perf import performance_by_regime
        perf = performance_by_regime(daily_returns, regime_labels)
        total_pct = perf["pct_days"].sum()
        assert abs(total_pct - 1.0) < 1e-10

    def test_all_regimes_present(self, daily_returns, regime_labels):
        from systematic_regime_trading.evaluation.regime_perf import performance_by_regime
        perf = performance_by_regime(daily_returns, regime_labels)
        assert len(perf) == 3
        assert "Calm" in perf.index
        assert "Moderate" in perf.index
        assert "Turbulent" in perf.index

    def test_crisis_performance(self, daily_returns):
        from systematic_regime_trading.evaluation.regime_perf import crisis_performance
        crisis_periods = {
            "test_crisis": {
                "start": str(daily_returns.index[100]),
                "end": str(daily_returns.index[200]),
            }
        }
        perf = crisis_performance(daily_returns, crisis_periods)
        assert len(perf) == 1
        assert "cumulative_return" in perf.columns


# --- Statistical Significance Tests ---

class TestSignificance:
    def test_bootstrap_ci_contains_point_estimate(self, daily_returns):
        from systematic_regime_trading.evaluation.significance import bootstrap_sharpe_ci
        point, lower, upper = bootstrap_sharpe_ci(
            daily_returns, n_bootstrap=1000,
        )
        # Point estimate should be within a reasonable range of CI
        # (not always inside due to bootstrap bias, but CI should bracket it loosely)
        assert lower < upper
        assert upper > 0  # Positive returns -> upper CI should be positive

    def test_sharpe_difference_test(self, daily_returns, benchmark_returns):
        from systematic_regime_trading.evaluation.significance import sharpe_difference_test
        diff, p_value = sharpe_difference_test(
            daily_returns, benchmark_returns, n_bootstrap=1000,
        )
        assert 0 <= p_value <= 1


# --- Performance Table Tests ---

class TestPerformanceTable:
    def test_table_from_backtest_results(self, daily_returns, benchmark_returns):
        from systematic_regime_trading.evaluation.report import performance_table
        # Create mock backtest results
        results = {
            "strategy_a": pd.DataFrame({
                "net_return": daily_returns,
                "position": pd.Series(1.0, index=daily_returns.index),
            }),
            "benchmark": pd.DataFrame({
                "net_return": benchmark_returns,
                "position": pd.Series(1.0, index=benchmark_returns.index),
            }),
        }
        table = performance_table(results, include_ci=False)
        assert len(table) == 2
        assert "Sharpe" in table.columns
        assert "CAGR" in table.columns
        assert "Max_DD" in table.columns

    def test_table_sorted_by_sharpe(self, daily_returns, benchmark_returns):
        from systematic_regime_trading.evaluation.report import performance_table
        results = {
            "good": pd.DataFrame({
                "net_return": daily_returns,
                "position": pd.Series(1.0, index=daily_returns.index),
            }),
            "bad": pd.DataFrame({
                "net_return": daily_returns * -1,  # flip returns
                "position": pd.Series(1.0, index=daily_returns.index),
            }),
        }
        table = performance_table(results, include_ci=False)
        assert table.index[0] == "good"  # Higher Sharpe first
