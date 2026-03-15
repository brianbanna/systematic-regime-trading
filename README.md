# Market Regime Modeling for Systematic Trading

Detect market regimes. Generate allocation signals. Backtest with realistic costs.

A config-driven framework that combines Hidden Markov Models, GARCH volatility, and K-Means clustering into an ensemble regime detector, then translates regime probabilities into systematic equity allocation strategies evaluated via walk-forward backtesting.

<!--
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
-->

## Overview

Markets cycle between calm and turbulent states. This framework identifies those states in real time and trades accordingly: aggressive when conditions are favorable, defensive when they deteriorate.

**What it does:**

- Fits three independent regime models on daily US equity data (2000-2020)
- Combines them into a probability-weighted ensemble
- Maps regime probabilities to portfolio allocation targets
- Runs vectorized backtests with transaction costs, slippage, and execution lag
- Evaluates performance with bootstrap significance testing

**What it produces:**

- Out-of-sample regime predictions (walk-forward, no lookahead)
- Net-of-cost strategy returns for multiple allocation variants
- Full performance tearsheet (Sharpe, drawdown, rolling metrics, cost sensitivity)
- Static research website with key charts

## Research Pipeline

```
                    Market Data
                        |
                Feature Engineering
          (6 cross-sectional indicators + PCA)
                        |
                 Regime Detection
            (HMM | GARCH | K-Means)
                        |
             Ensemble Probabilities
          (performance-weighted voting)
                        |
              Signal Construction
         (filters, vol targeting, lag)
                        |
             Portfolio Allocation
           (4 strategy variants)
                        |
            Walk-Forward Backtest
          (transaction costs, slippage)
                        |
           Performance Evaluation
       (Sharpe, drawdown, significance)
```

## Models

| Model                   | Input               | What It Captures                                                                                    |
| ----------------------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| **Hidden Markov Model** | PCA stress index    | Latent state transitions via Gaussian emissions. Produces posterior regime probabilities.           |
| **GARCH(1,1)**          | Market returns      | Conditional volatility clustering. Student-t innovations. Expanding-window quantile classification. |
| **K-Means**             | Volatility + Volume | Cross-sectional regime clustering. Clusters ordered by centroid volatility.                         |
| **Ensemble**            | All three           | Probability-weighted voting. Reduces single-model noise.                                            |

All models are refitted quarterly using an expanding training window (minimum 5 years). Every prediction is strictly out-of-sample.

## Strategies

Regime probabilities drive four allocation strategies:

| Strategy        | Rule                                   | Characteristics                     |
| --------------- | -------------------------------------- | ----------------------------------- |
| Binary Regime   | 100% equity in calm, 0% in turbulent   | Sharp risk-off, higher turnover     |
| Proportional    | Allocation = P(calm)                   | Smooth transitions, lower turnover  |
| Vol-Targeted    | Scale to 10% annualized vol target     | Constant risk budget across regimes |
| Regime Momentum | Higher conviction when regime persists | Trend-following flavor              |

All strategies include:

- 1-day execution lag (signal at close T, trade at close T+1)
- Confirmation filter (regime must persist N days before switching)
- Rate limiter (max 25% allocation change per day)
- Transaction costs at 5 bps + 2 bps slippage, with sensitivity from 0 to 20 bps

Benchmarks: buy-and-hold, 60/40 equity-bond.

## Features

Six cross-sectional indicators computed daily from a 295-stock universe:

| Indicator          | Definition                               |
| ------------------ | ---------------------------------------- |
| Market Direction   | Mean log return across stocks            |
| Market Breadth     | Fraction of stocks with positive returns |
| Market Volatility  | Cross-sectional return dispersion        |
| Market ATR         | Average normalized true range            |
| Market Volume      | Mean change in log volume                |
| Market Correlation | Rolling pairwise correlation             |

PCA extracts two factors: **PC1** (market trend) and **PC2** (systemic stress). PC2 detects the dot-com crash, 2008 crisis, and COVID sell-off without using drawdown as an input.

## Data

| Dataset               | Source           | Frequency |
| --------------------- | ---------------- | --------- |
| US equity OHLCV       | yfinance         | Daily     |
| VIX / VIX3M           | yfinance         | Daily     |
| Risk-free rate        | FRED (3M T-bill) | Daily     |
| Treasury bond returns | yfinance (TLT)   | Daily     |

Data downloads automatically via `make data`. No manual preparation needed.

## Key Research Outputs

- Walk-forward regime predictions (strictly out-of-sample)
- Strategy backtests with realistic transaction costs
- Regime-conditional performance analysis
- Transaction cost sensitivity and breakeven analysis
- Full performance tearsheet with bootstrap confidence intervals
- Static research website presenting the results

## Quickstart

```bash
git clone https://github.com/brianbanna/systematic-regime-trading.git
cd systematic-regime-trading

pip install -e .

# Run the full pipeline
make all

# Or run stages individually
make data          # Download and clean market data
make features      # Compute indicators and PCA factors
make models        # Train regime models (walk-forward)
make signals       # Generate allocation signals
make backtest      # Run strategy backtests with costs
make evaluate      # Compute performance metrics
make report        # Generate tearsheet and charts

# Run tests
make test
```

Requires Python 3.10+.

## Project Structure

```
systematic-regime-trading/
├── configs/
│   ├── data.yaml              # Universe, date range, data sources
│   ├── features.yaml          # Indicator windows, PCA config
│   ├── models.yaml            # HMM, GARCH, KMeans, ensemble params
│   ├── strategy.yaml          # Allocation rules, filters, rebalancing
│   ├── backtest.yaml          # Costs, slippage, execution lag
│   └── evaluation.yaml        # Benchmarks, metrics, bootstrap config
│
├── src/
│   └── systematic_regime_trading/
│       ├── data/              # Loaders, cleaning, Parquet storage
│       ├── features/          # Indicators, PCA, transforms
│       ├── models/            # HMM, GARCH, KMeans, ensemble, walk-forward
│       ├── signals/           # Regime → allocation, filters, vol targeting
│       ├── backtest/          # Vectorized engine, costs, benchmarks
│       ├── evaluation/        # Metrics, rolling analysis, significance
│       ├── visualization/     # Performance charts, regime plots, tearsheet
│       └── utils/             # Config loader, paths, constants
│
├── notebooks/                 # Step-by-step research notebooks
├── data/                      # Raw + processed data (git-ignored)
├── results/                   # Figures, tables, tearsheets
├── website/                   # Static research site (GitHub Pages)
├── tests/                     # Unit and integration tests
├── Makefile                   # Pipeline orchestration
└── pyproject.toml             # Dependencies
```

All parameters are externalized to YAML configs. No magic numbers in source code.

## License

MIT

## Disclaimer

Research code for educational and demonstration purposes. Not investment advice. Backtested performance does not guarantee future results.
