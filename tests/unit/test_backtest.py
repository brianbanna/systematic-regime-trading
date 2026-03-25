"""Tests for the backtesting framework."""

import pandas as pd
import numpy as np
import pytest


# --- Fixtures ---

@pytest.fixture
def market_returns():
    """Synthetic daily market returns (500 days)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=500)
    returns = rng.normal(0.0004, 0.012, size=500)
    return pd.Series(returns, index=dates, name="market_return")


@pytest.fixture
def bond_returns():
    """Synthetic daily bond returns (500 days, lower vol)."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2015-01-01", periods=500)
    returns = rng.normal(0.0002, 0.005, size=500)
    return pd.Series(returns, index=dates, name="bond_return")


@pytest.fixture
def allocation_signal(market_returns):
    """Simple allocation signal: 1.0 for first half, 0.5 for second."""
    n = len(market_returns)
    signal = pd.Series(
        [1.0] * (n // 2) + [0.5] * (n - n // 2),
        index=market_returns.index,
    )
    return signal


@pytest.fixture
def cost_config():
    return {"cost_bps": 5, "slippage_bps": 2, "min_trade_bps": 1}


@pytest.fixture
def constraint_config():
    return {"max_leverage": 1.0, "min_allocation": 0.0, "max_turnover_annual": 20.0}


# --- Transaction Cost Tests ---

class TestTransactionCosts:
    def test_zero_turnover_zero_cost(self, market_returns, cost_config):
        from systematic_regime_trading.backtest.costs import compute_transaction_costs
        # Constant position -> only initial trade has cost
        position = pd.Series(1.0, index=market_returns.index)
        costs = compute_transaction_costs(position, cost_config)
        # After day 0, costs should be 0
        assert costs.iloc[1:].sum() == 0.0
        assert costs.iloc[0] > 0  # initial trade

    def test_costs_proportional_to_turnover(self, market_returns, cost_config):
        from systematic_regime_trading.backtest.costs import compute_transaction_costs
        # Two signals with different turnover
        pos_low = pd.Series(1.0, index=market_returns.index)
        pos_high = pd.Series(
            [1.0 if i % 2 == 0 else 0.0 for i in range(len(market_returns))],
            index=market_returns.index,
        )
        cost_low = compute_transaction_costs(pos_low, cost_config).sum()
        cost_high = compute_transaction_costs(pos_high, cost_config).sum()
        assert cost_high > cost_low

    def test_annualized_turnover(self, market_returns):
        from systematic_regime_trading.backtest.costs import annualized_turnover
        # Constant position = ~0 turnover (only initial)
        position = pd.Series(1.0, index=market_returns.index)
        at = annualized_turnover(position)
        assert at < 1.0  # Very low turnover


# --- Portfolio Constraint Tests ---

class TestPortfolioConstraints:
    def test_clips_leverage(self, constraint_config):
        from systematic_regime_trading.backtest.portfolio import construct_portfolio
        signal = pd.Series([1.5, 2.0, -0.5, 0.5])
        result = construct_portfolio(signal, constraint_config)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_preserves_valid_signal(self, constraint_config):
        from systematic_regime_trading.backtest.portfolio import construct_portfolio
        signal = pd.Series([0.5, 0.5, 0.5, 0.5])
        result = construct_portfolio(signal, constraint_config)
        pd.testing.assert_series_equal(result, signal)


# --- Backtest Engine Tests ---

class TestBacktestEngine:
    def test_buy_and_hold_matches_market(self, market_returns):
        from systematic_regime_trading.backtest.engine import run_backtest
        # Zero cost, full allocation = should match market returns exactly
        signal = pd.Series(1.0, index=market_returns.index)
        cost_config = {"cost_bps": 0, "slippage_bps": 0, "min_trade_bps": 0}
        constraint_config = {"max_leverage": 1.0, "min_allocation": 0.0, "max_turnover_annual": 100.0}

        bt = run_backtest(
            market_returns, signal,
            cost_config=cost_config,
            constraint_config=constraint_config,
        )

        # Gross return should equal market return
        np.testing.assert_allclose(
            bt["gross_return"].values, market_returns.values, atol=1e-10,
        )

    def test_zero_allocation_zero_return(self, market_returns):
        from systematic_regime_trading.backtest.engine import run_backtest
        signal = pd.Series(0.0, index=market_returns.index)
        cost_config = {"cost_bps": 0, "slippage_bps": 0, "min_trade_bps": 0}
        constraint_config = {"max_leverage": 1.0, "min_allocation": 0.0, "max_turnover_annual": 100.0}

        bt = run_backtest(
            market_returns, signal,
            cost_config=cost_config,
            constraint_config=constraint_config,
        )

        np.testing.assert_allclose(
            bt["gross_return"].values, 0.0, atol=1e-10,
        )

    def test_costs_reduce_returns(self, market_returns, allocation_signal):
        from systematic_regime_trading.backtest.engine import run_backtest
        constraint_config = {"max_leverage": 1.0, "min_allocation": 0.0, "max_turnover_annual": 100.0}

        # Zero cost
        bt_free = run_backtest(
            market_returns, allocation_signal,
            cost_config={"cost_bps": 0, "slippage_bps": 0, "min_trade_bps": 0},
            constraint_config=constraint_config,
        )

        # With costs
        bt_costly = run_backtest(
            market_returns, allocation_signal,
            cost_config={"cost_bps": 10, "slippage_bps": 5, "min_trade_bps": 0},
            constraint_config=constraint_config,
        )

        # Net return with costs should be lower
        assert bt_costly["cumulative_return"].iloc[-1] < bt_free["cumulative_return"].iloc[-1]

    def test_output_columns(self, market_returns, allocation_signal, cost_config, constraint_config):
        from systematic_regime_trading.backtest.engine import run_backtest
        bt = run_backtest(
            market_returns, allocation_signal, cost_config, constraint_config,
        )
        expected_cols = [
            "position", "gross_return", "turnover", "cost",
            "net_return", "cumulative_return", "drawdown",
        ]
        for col in expected_cols:
            assert col in bt.columns

    def test_drawdown_is_nonpositive(self, market_returns, allocation_signal, cost_config, constraint_config):
        from systematic_regime_trading.backtest.engine import run_backtest
        bt = run_backtest(
            market_returns, allocation_signal, cost_config, constraint_config,
        )
        assert (bt["drawdown"] <= 0).all()


# --- Benchmark Tests ---

class TestBenchmarks:
    def test_buy_and_hold_full_exposure(self, market_returns):
        from systematic_regime_trading.backtest.benchmarks import buy_and_hold
        bt = buy_and_hold(market_returns)
        assert (bt["position"] == 1.0).all()
        assert len(bt) == len(market_returns)

    def test_sixty_forty_weights(self, market_returns, bond_returns):
        from systematic_regime_trading.backtest.benchmarks import sixty_forty
        bt = sixty_forty(market_returns, bond_returns)
        # Equity weight should be around 0.6 (drifts between rebalances)
        assert bt["position"].mean() == pytest.approx(0.6, abs=0.05)

    def test_risk_parity_weights(self, market_returns, bond_returns):
        from systematic_regime_trading.backtest.benchmarks import risk_parity
        bt = risk_parity(market_returns, bond_returns, lookback=63)
        # Equity weight should shift toward bonds (lower vol asset)
        # Bond vol is lower, so bond gets higher weight -> equity weight < 0.5
        valid = bt["position"].iloc[63:]  # After warmup
        assert valid.mean() < 0.55  # Bonds have lower vol

    def test_all_benchmarks_produce_valid_output(self, market_returns, bond_returns):
        from systematic_regime_trading.backtest.benchmarks import buy_and_hold, sixty_forty, risk_parity
        for name, bt in [
            ("bah", buy_and_hold(market_returns)),
            ("60/40", sixty_forty(market_returns, bond_returns)),
            ("rp", risk_parity(market_returns, bond_returns)),
        ]:
            assert "cumulative_return" in bt.columns
            assert bt["cumulative_return"].iloc[-1] > 0
            assert (bt["drawdown"] <= 0).all()


# --- Cost Sensitivity Tests ---

class TestCostSensitivity:
    def test_higher_cost_lower_sharpe(self, market_returns, allocation_signal):
        from systematic_regime_trading.backtest.sensitivity import cost_sensitivity
        df = cost_sensitivity(
            market_returns, allocation_signal,
            cost_levels_bps=[0, 5, 10, 20],
        )
        # Sharpe should generally decrease with cost
        assert df.iloc[0]["sharpe"] >= df.iloc[-1]["sharpe"]

    def test_breakeven_between_bounds(self, market_returns, allocation_signal):
        from systematic_regime_trading.backtest.sensitivity import (
            cost_sensitivity, find_breakeven_cost,
        )
        df = cost_sensitivity(
            market_returns, allocation_signal,
            cost_levels_bps=[0, 5, 10, 20, 50, 100],
        )
        breakeven = find_breakeven_cost(df, metric="sharpe", threshold=0.0)
        assert breakeven >= 0
        assert breakeven <= 100
