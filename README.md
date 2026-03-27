# Systematic Regime Trading

A 5-model ensemble that detects market regimes and adjusts equity allocation. Beats SPY on both absolute return (10.2% vs 9.6% CAGR) and risk-adjusted basis (Sharpe 0.70 vs 0.46) while cutting max drawdown from -55% to -28%.

**[View the research site](https://brianbanna.com/systematic-regime-trading)**

## Results

| Strategy | CAGR | Sharpe | Max DD | Turnover |
|----------|------|--------|--------|----------|
| **Regime Momentum** | **10.2%** | **0.70** | **-28%** | 5.7x |
| Vol-Managed (baseline) | 6.6% | 0.47 | -24% | 1.7x |
| SMA-200 (baseline) | 7.0% | 0.47 | -22% | 6.0x |
| Buy & Hold (SPY) | 9.6% | 0.46 | -55% | 0.0x |

Benchmarked against SPY. Walk-forward validation, 2006-2020 out-of-sample. All returns net of 7 bps transaction costs.

Out-of-sample extension (2021-2026, rolling quarterly recalibration): Sharpe 0.44, CAGR 7.6%.

## How it works

190 NASDAQ stocks produce six cross-sectional features (volatility, breadth, direction, correlation, ATR, volume) plus macro features (high-yield credit spread, yield curve slope, financial stress index). Five regime models classify each trading day:

| Model | Input | What it captures |
|-------|-------|-----------------|
| HMM (5-state Gaussian) | Market volatility | Hidden state transitions, regime persistence |
| GARCH(1,1) Student-t | SPY returns | Conditional volatility clustering, fat tails |
| K-Means | Vol + volume + macro | Feature-space regime clustering |
| Gaussian Mixture Model | Vol + volume + macro | Bayesian posterior probabilities |
| Markov-Switching (Hamilton 1989) | SPY returns | Regime-dependent mean and variance |

The ensemble combines probabilities (not hard labels) with fixed weights. Regime predictions feed an allocation signal with asymmetric confirmation filters, rate limiting, and 1-day execution lag.

## Key findings

- The 5-model ensemble detects turbulent regimes a median of **6 days before drawdowns** accelerate
- Regime predictions are genuinely informative: calm predictions correspond to 12% realized vol, turbulent to 22%
- During Sep-Dec 2008: market fell 29%, strategy lost 4.3%
- Macro features (credit spread, yield curve) improve regime detection vs equity-only signals
- Strategy survives 30+ bps transaction costs (institutional costs are 5-10 bps)
- Cross-asset extension to crude oil confirms regime detection generalizes

## Quick start

```bash
git clone https://github.com/brianbanna/systematic-regime-trading.git
cd systematic-regime-trading
pip install -r requirements.txt
pip install -e .

make run-from-scratch    # Download data, run pipeline, generate figures
make test                # Run 140 tests
```

Individual steps:

```bash
make data-fresh          # Download 190-ticker NASDAQ + SPY + VIX + TLT
make run                 # Full pipeline: features -> regimes -> signals -> backtest
make figures             # Generate all charts (dark theme, 300 DPI)
```

## Project structure

```
configs/                        # All parameters in YAML (no magic numbers)
  models.yaml                   # HMM, GARCH, KMeans, GMM, Markov-Switching
  strategy.yaml                 # Allocation rules, filters, vol targets
  backtest.yaml                 # Transaction costs, execution lag
src/systematic_regime_trading/
  models/                       # 5 regime detectors + ensemble
  signals/                      # Regime -> allocation, filters, vol targeting
  backtest/                     # Vectorized engine, costs, benchmarks
  evaluation/                   # Metrics, factor regression, significance
  data/                         # Loaders, cleaning, macro data
scripts/
  run_pipeline.py               # End-to-end pipeline (make run)
  run_oos_extension.py          # 2021-2026 forward test
  run_commodities.py            # Crude oil regime detection
  generate_figures.py           # All charts (make figures)
results/                        # Performance table, predictions, backtest curves
website/                        # Research site (dark theme, interactive Plotly)
tests/                          # 140 tests
```

## Requirements

Python 3.12. Key dependencies: pandas, numpy, scikit-learn, hmmlearn, arch, statsmodels, matplotlib, plotly, yfinance. Full pinned versions in `requirements.txt`.

## License

MIT
