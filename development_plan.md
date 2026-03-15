# Development Plan: Market Regime Modeling for Systematic Trading

## Complete Technical Blueprint

---

# 1. Project Overview

## Research Question

> Can statistical regime detection models (Hidden Markov Models, GARCH, K-Means clustering) produce actionable allocation signals for systematic equity trading that survive realistic transaction costs and outperform static benchmarks on a risk-adjusted basis?

The existing ADA project answers "can we detect regimes?" — a classification question. This transformation answers "can we *trade* regimes?" — a P&L question. The deliverable is not a classification accuracy score. It is a Sharpe ratio, a drawdown chart, and a transaction cost sensitivity analysis.

## Why Regime Modeling Matters for Systematic Trading

Every systematic trading desk operates with some notion of market regime, whether explicit or implicit. Risk limits tighten during volatile periods. Position sizes shrink when correlations spike. Carry strategies get cut during crises. Regime detection formalizes this intuition into a quantitative signal.

The value proposition for a trading desk:

- **Risk management**: Reduce exposure before drawdowns deepen, not after.
- **Position sizing**: Scale positions by regime — aggressive in calm markets, defensive in turbulent ones.
- **Strategy selection**: Different strategies work in different regimes. Trend-following profits in trending regimes. Mean-reversion profits in range-bound regimes.
- **Cost awareness**: Regime switches trigger rebalancing. The signal must survive the cost of acting on it.

## Foundation for Future Projects

This project is architected to support two subsequent projects:

**Commodity Futures Curve Modeling and Factor Trading** will import:
- The HMM regime detection module (applied to commodity volatility regimes)
- The backtesting engine (extended for futures roll costs and margin)
- The performance evaluation library (Sharpe, drawdown, rolling metrics)
- The config-driven architecture pattern (YAML parameters, no magic numbers)

**Adaptive Statistical Arbitrage in Commodity Spreads** will import:
- The HMM module (applied to spread regime detection: mean-reverting vs broken)
- The GARCH module (applied to spread volatility for vol-targeting)
- The backtesting engine (extended for two-leg spread execution)
- The evaluation and visualization modules

The shared infrastructure lives in this project. Projects 2 and 3 are consumers, not forks.

---

# 2. Current Repository Assessment

## What Exists Today

The ADA project ("The Mood Swings of the NASDAQ") is a complete academic regime detection study with 8 sequential notebooks, a modular Python source tree, a Streamlit web app, and 80+ result figures.

### Components That Work Well

**Regime detection models** — all three are implemented and produce reasonable results:
- K-Means clustering on volatility+volume features (3 clusters, ~70% calm, ~19% moderate, ~11% turbulent)
- GARCH(1,1) conditional volatility with quantile-based regime classification (α=0.114, β=0.859, persistence=0.973)
- 5-state Gaussian HMM on PC2 stress index, mapped to 3 regimes (best individual method, composite score 0.612)

**Feature engineering pipeline** — 6 cross-sectional market indicators + PCA:
- market_direction, market_volume, market_breadth, market_atr, market_correlation, market_volatility
- PCA yields PC1 (trend, 29% variance) and PC2 (systemic stress, 24.5% variance)
- PC2 correctly identifies all major crises (dot-com, 2008, COVID) without using drawdown as input

**Ensemble framework** — weighted voting across methods with grid-search weight optimization.

**Statistical validation** — ANOVA testing, transition matrices, persistence metrics, crisis alignment scoring.

**Ticker subset selection** — stratified sampling of 295 tickers that mirrors full-market volatility (Pearson 0.65, Spearman 0.73).

### What Must Be Refactored

| Component | Problem | Action |
|-----------|---------|--------|
| Hard-coded parameters (30+) | Window sizes, thresholds, cluster counts scattered across files | Extract to YAML configs |
| `sampling.py` (500+ lines) | Monolithic file mixing index scraping, subset generation, evaluation | Split into focused modules |
| `display_section.py` (3,178 lines) | Massive Streamlit rendering file | Will not migrate — replaced by static website |
| `visualization.py` (42KB) | Single file for all plot types | Split by purpose: performance, regimes, diagnostics |
| CSV storage (1.38GB processed file) | Slow I/O, no type preservation | Migrate to Parquet |
| `np.random.seed()` global state | Non-reproducible across modules | Replace with `np.random.default_rng(seed)` |
| Hard-coded file paths | Brittle relative paths with `..` chains | Use config-based path resolution from project root |

### What Must Be Removed

| Component | Reason |
|-----------|--------|
| `src/utils/app/` (display_section.py, app_utils.py) | Streamlit app code — replaced by static research website |
| `webapp/` directory | Streamlit deployment artifacts |
| `src/scripts/get_ticker.py` | Non-functional data download script (loads from hard-coded CSV, not yfinance) |
| `src/data/sampling/` Wikipedia scraping logic | Fragile web scraping for index constituents — replace with static cached files |
| `datasets`, `transformers`, `sentencepiece`, `wandb` in requirements | NLP/ML dependencies not used by this project |

### What Is Missing for a Trading System

| Missing Component | Why It Matters |
|-------------------|---------------|
| **Signal generation** | Regime labels exist but no mapping to position targets |
| **Execution lag** | No delay between signal and trade — implies you can trade on today's regime |
| **Transaction costs** | No cost model — backtest Sharpe is meaningless without costs |
| **Backtesting engine** | No portfolio return series, no P&L computation |
| **Walk-forward validation** | All models trained and evaluated in-sample on 2000–2020 |
| **Performance metrics** | No Sharpe ratio, no max drawdown, no Sortino, no Calmar |
| **Benchmark comparison** | No buy-and-hold or 60/40 benchmark to compare against |
| **Regime probability output** | Models produce hard labels only; soft probabilities would reduce turnover |
| **Vol-targeting** | No mechanism to normalize risk exposure across regimes |

---

# 3. Target Final Architecture

```
systematic-regime-trading/
│
├── configs/
│   ├── data.yaml                 # Data sources, date ranges, universe definition
│   ├── features.yaml             # Indicator parameters, PCA config, normalization windows
│   ├── models.yaml               # HMM states, GARCH spec, KMeans clusters, ensemble weights
│   ├── strategy.yaml             # Allocation targets per regime, rebalance rules, constraints
│   ├── backtest.yaml             # Transaction costs, slippage, execution lag, vol target
│   └── evaluation.yaml           # Benchmark definitions, metrics list, rolling windows
│
├── src/
│   ├── data/
│   │   ├── loaders.py            # Load from CSV/Parquet/yfinance with config-driven paths
│   │   ├── cleaning.py           # Consolidated cleaning + repair + validation
│   │   ├── universe.py           # Ticker universe construction (from cached constituent files)
│   │   └── storage.py            # Parquet read/write, data catalog, versioning helpers
│   │
│   ├── features/
│   │   ├── indicators.py         # Market indicators: direction, breadth, ATR, volume, correlation
│   │   ├── volatility.py         # Volatility index, smoothing, GARCH conditional vol
│   │   ├── factors.py            # PCA components, composite scores, mood index
│   │   └── transforms.py         # Z-score, rolling normalization, expanding-window standardization
│   │
│   ├── models/
│   │   ├── hmm.py                # HMM fitting, prediction, state labeling, probability extraction
│   │   ├── garch.py              # GARCH fitting, conditional volatility, persistence metrics
│   │   ├── kmeans.py             # K-Means clustering pipeline with PCA option
│   │   ├── ensemble.py           # Weighted voting, optimization, combination logic
│   │   └── validation.py         # Walk-forward training, model selection, stability checks
│   │
│   ├── signals/
│   │   ├── regime_signal.py      # Regime probabilities → allocation targets
│   │   ├── filters.py            # Confirmation filters, smoothing, anti-whipsaw logic
│   │   └── vol_target.py         # Volatility targeting overlay
│   │
│   ├── backtest/
│   │   ├── engine.py             # Vectorized backtest: signals → positions → returns
│   │   ├── costs.py              # Transaction cost and slippage models
│   │   ├── portfolio.py          # Position sizing, rebalancing, exposure tracking
│   │   └── benchmarks.py         # Buy-and-hold, 60/40, equal-weight benchmark strategies
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # Sharpe, Sortino, Calmar, max DD, turnover, hit rate
│   │   ├── rolling.py            # Rolling Sharpe, rolling vol, rolling drawdown
│   │   ├── regime_perf.py        # Performance decomposed by regime
│   │   ├── significance.py       # Bootstrap confidence intervals, p-values
│   │   └── report.py             # Generate full performance tearsheet (HTML or PDF)
│   │
│   ├── visualization/
│   │   ├── performance.py        # Cumulative returns, drawdown, monthly heatmap
│   │   ├── regimes.py            # Regime timeline, transitions, distributions
│   │   ├── signals.py            # Signal evolution, allocation over time
│   │   ├── diagnostics.py        # Model residuals, QQ plots, ACF
│   │   └── tearsheet.py          # Combined strategy tearsheet (single call)
│   │
│   └── utils/
│       ├── config.py             # YAML config loader with defaults and overrides
│       ├── paths.py              # Project root detection, path resolution
│       └── constants.py          # Regime labels, color maps, trading days per year
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb        # Load, clean, store data
│   ├── 02_feature_engineering.ipynb   # Compute indicators, PCA, validate
│   ├── 03_regime_models.ipynb         # Train HMM, GARCH, KMeans; walk-forward
│   ├── 04_signal_construction.ipynb   # Regime → allocation signals, filters
│   ├── 05_backtest.ipynb              # Run strategies, compare to benchmarks
│   ├── 06_evaluation.ipynb            # Full performance analysis, significance
│   └── 07_tearsheet.ipynb             # Generate final research outputs
│
├── data/
│   ├── raw/                       # Original downloaded data (git-ignored)
│   ├── cache/                     # Cached index constituents, reference data
│   └── processed/                 # Parquet outputs from pipeline (git-ignored)
│
├── results/
│   ├── figures/                   # All generated plots (PNG, high-res)
│   ├── tables/                    # Performance tables (CSV)
│   └── tearsheets/                # Strategy tearsheets (HTML/PNG)
│
├── website/                       # Static research website (GitHub Pages)
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── assets/
│       └── figures/               # Copies of key result figures
│
├── tests/
│   ├── unit/
│   │   ├── test_cleaning.py
│   │   ├── test_indicators.py
│   │   ├── test_transforms.py
│   │   ├── test_hmm.py
│   │   ├── test_garch.py
│   │   ├── test_signals.py
│   │   ├── test_engine.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   └── test_pipeline.py       # End-to-end: data → signal → backtest → metrics
│   └── fixtures/
│       └── sample_data.parquet    # Small test dataset
│
├── Makefile                       # Pipeline orchestration targets
├── pyproject.toml                 # Project metadata + pinned dependencies
├── README.md                      # Professional project README
└── CLAUDE.md                      # Development context for AI assistance
```

### Module Responsibilities

| Module | Single Responsibility |
|--------|----------------------|
| `data/` | Raw data in, clean Parquet out. No modeling, no features. |
| `features/` | Clean data in, model-ready feature matrices out. Stateless transforms. |
| `models/` | Features in, regime labels and probabilities out. Training and prediction only. |
| `signals/` | Regime output in, position targets out. The bridge between models and trading. |
| `backtest/` | Signals in, portfolio returns out. Simulates execution with costs. |
| `evaluation/` | Returns in, metrics and reports out. Post-trade analysis only. |
| `visualization/` | Data in, figures out. No computation beyond what's needed for display. |
| `configs/` | All parameters. No magic numbers anywhere in source code. |

---

# 4. Phase-by-Phase Development Plan

## Phase 1: Repository Setup and Refactoring

**Objective**: Create the new repository structure, migrate reusable code from ADA, remove dead code, and establish the config-driven architecture. At the end of this phase, the existing regime detection pipeline must reproduce its original results in the new structure.

**Estimated time**: 4–5 days

### Task 1.1: Initialize Repository

- Create new git repository `systematic-regime-trading`
- Create the full directory structure from Section 3
- Write `pyproject.toml` with pinned dependencies:

```toml
[project]
name = "systematic-regime-trading"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.18",
    "arch>=6.0",            # GARCH models
    "hmmlearn>=0.3",        # Hidden Markov Models
    "statsmodels>=0.14",    # Statistical tests
    "pyarrow>=14.0",        # Parquet I/O
    "pyyaml>=6.0",          # Config loading
    "tqdm>=4.66",           # Progress bars
    "yfinance>=0.2.31",     # Market data
    "fredapi>=0.5",         # FRED macro data
    "quantstats>=0.0.62",   # Performance analytics (optional, for validation)
]
```

- Write initial `Makefile` with targets:

```makefile
.PHONY: data features models signals backtest evaluate report website clean all

data:         python -m src.data.loaders
features:     python -m src.features.indicators
models:       python -m src.models.validation
signals:      python -m src.signals.regime_signal
backtest:     python -m src.backtest.engine
evaluate:     python -m src.evaluation.report
report:       python -m src.visualization.tearsheet
website:      cd website && python build.py
clean:        rm -rf data/processed/* results/*
all:          data features models signals backtest evaluate report
test:         python -m pytest tests/ -v
```

- Write `CLAUDE.md` with project context for AI-assisted development

### Task 1.2: Create Configuration Files

Extract every hard-coded parameter identified in the audit into YAML configs.

**`configs/data.yaml`**:
```yaml
universe:
  source: "cache/nasdaq_constituents.csv"
  date_range:
    start: "2000-01-01"
    end: "2020-12-31"
  min_history_days: 30
  max_missing_pct: 0.10
  interpolation_window: 5

cleaning:
  extreme_high_price: 10000
  extreme_low_price: 0.01
  volume_tolerance_pct: 0.01

storage:
  format: "parquet"
  processed_dir: "data/processed"
  raw_dir: "data/raw"
```

**`configs/features.yaml`**:
```yaml
indicators:
  smoothing_window: 20
  correlation_window: 30
  correlation_sample_size: 500
  standardization_window: 252  # 1 trading year

pca:
  n_components: 2
  features:
    - market_volatility
    - market_atr
    - market_correlation
    - market_breadth
    - market_direction
    - market_volume

volatility:
  smoothing_windows: [5, 10, 20, 30]
  annualization_factor: 252
```

**`configs/models.yaml`**:
```yaml
hmm:
  n_states: 5             # Fit 5 states, remap to 3
  n_iter: 1000
  covariance_type: "full"
  random_state: 42
  input_feature: "systemic_stress_pc2"
  remap_to_3:
    calm: [0, 1]           # States 0,1 → calm
    moderate: [2]           # State 2 → moderate
    turbulent: [3, 4]       # States 3,4 → turbulent

garch:
  p: 1
  q: 1
  mean_model: "Constant"
  vol_model: "GARCH"
  distribution: "t"         # Student-t (fix from ADA's "normal")
  return_scaling: 100
  regime_quantiles: [0.33, 0.67]

kmeans:
  n_clusters: 3
  random_state: 42
  features:
    volvol: ["market_volatility", "market_volume"]
    mood: ["mood_index"]

ensemble:
  method: "weighted_vote"
  min_weight: 0.10
  max_weight: 0.80
  weight_step: 0.05
  n_regimes: 3

regime_labels:
  0: "Calm"
  1: "Moderate"
  2: "Turbulent"

walk_forward:
  min_train_years: 5
  test_window_days: 252     # 1 year
  step_days: 63             # ~quarterly refit
  expanding: true           # Expanding window (not rolling)
```

**`configs/strategy.yaml`**:
```yaml
strategies:
  binary_regime:
    description: "100% equity in calm, 0% in turbulent"
    allocation:
      calm: 1.0
      moderate: 0.5
      turbulent: 0.0

  proportional_regime:
    description: "Allocation proportional to P(calm)"
    method: "probability_weighted"

  vol_targeted:
    description: "Scale allocation to hit vol target per regime"
    vol_target: 0.10          # 10% annualized
    vol_lookback_days: 63     # Quarterly rolling vol estimate

  regime_momentum:
    description: "Overweight when regime is persistent"
    persistence_threshold_days: 10
    allocation:
      calm_persistent: 1.2     # Slight leverage
      calm_new: 0.8
      moderate: 0.5
      turbulent: 0.0

signal_filters:
  confirmation_days: 3        # Require N days in new regime before switching
  max_daily_allocation_change: 0.25  # Limit allocation change per day

rebalance:
  frequency: "daily"          # daily, weekly, monthly
  day_of_week: null           # For weekly: 0=Monday
```

**`configs/backtest.yaml`**:
```yaml
execution:
  lag_days: 1                 # Signal at close T, execute at close T+1
  cost_bps: 5                 # One-way transaction cost (basis points)
  slippage_bps: 2             # Additional slippage
  min_trade_bps: 1            # Don't trade if position change < threshold

constraints:
  max_leverage: 1.0           # No leverage (1.0 = fully invested max)
  min_allocation: 0.0         # No shorting
  max_turnover_annual: 20.0   # 2000% annual turnover cap (generous)

risk_free_rate:
  source: "FRED"
  series: "DGS3MO"            # 3-month T-bill
  fallback: 0.02              # 2% if FRED unavailable
```

**`configs/evaluation.yaml`**:
```yaml
benchmarks:
  buy_and_hold:
    allocation: 1.0            # 100% equity always

  sixty_forty:
    equity_weight: 0.60
    bond_etf: "TLT"           # Long-term Treasury ETF
    rebalance: "monthly"

metrics:
  - sharpe_ratio
  - sortino_ratio
  - calmar_ratio
  - max_drawdown
  - cagr
  - annual_volatility
  - turnover
  - hit_rate_monthly
  - profit_factor
  - skewness
  - kurtosis

rolling:
  window_days: 252             # 1-year rolling window

bootstrap:
  n_samples: 10000
  confidence_level: 0.95
  random_state: 42

crisis_periods:
  dotcom:
    start: "2000-03-10"
    end: "2002-10-09"
  financial_crisis:
    start: "2007-10-09"
    end: "2009-03-09"
  covid:
    start: "2020-02-19"
    end: "2020-03-23"
```

### Task 1.3: Migrate and Consolidate Source Code

For each source module, migrate from ADA to new structure:

**Data module** — consolidate 4 files into 2:
- `src/data/cleaning/clean.py` + `repair.py` + `validation.py` → `src/data/cleaning.py`
- `src/data/loaders/loader.py` → `src/data/loaders.py`
- Remove Wikipedia scraping from `sampling.py` — cache constituent lists as static CSV in `data/cache/`
- Subset selection logic → `src/data/universe.py` (simplified: load cached subset, not generate 200 random ones)
- Add `src/data/storage.py` for Parquet I/O

**Features module** — reorganize indicators:
- `src/data/indicators/mood_indicators.py` → split into `src/features/indicators.py` (individual indicators) and `src/features/factors.py` (PCA, composite scores)
- `src/data/indicators/returns.py` + `volatility.py` → `src/features/volatility.py`
- `src/data/indicators/smoothing_window.py` + `validation.py` → `src/features/transforms.py`
- All hard-coded windows → read from `configs/features.yaml`

**Models module** — clean up, keep mostly as-is:
- `src/models/hmm/hmm_model.py` → `src/models/hmm.py` (flatten, remove visualization)
- `src/models/garch/garch_model.py` → `src/models/garch.py` (change distribution to Student-t)
- `src/models/clustering/clustering.py` → `src/models/kmeans.py` (extract only clustering logic, remove plotting)
- `src/utils/analysis/ensemble.py` → `src/models/ensemble.py`
- All hard-coded model parameters → read from `configs/models.yaml`
- Add `src/models/validation.py` for walk-forward logic (new code)

**Visualization** — reorganize by purpose:
- Extract regime-specific plots from `visualization.py` → `src/visualization/regimes.py`
- Extract any performance plots → `src/visualization/performance.py`
- Diagnostic plots → `src/visualization/diagnostics.py`
- Remove all Streamlit-specific code

**Remove entirely**:
- `src/utils/app/` (all Streamlit code)
- `webapp/` directory
- `src/scripts/get_ticker.py`
- `src/data/sampling/historical_indices.py` (replace with cached file)
- `src/models/hmm/hmm_visualization.py` (merge into `src/visualization/regimes.py`)

### Task 1.4: Implement Config System

Write `src/utils/config.py`:

```python
"""
Config loader that reads YAML files from configs/ directory.
All parameters accessed via config object — no hard-coded values in source.
"""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_config(name: str) -> dict:
    """Load a YAML config file by name (without extension)."""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def get_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / relative
```

Every module imports config at the top:
```python
from src.utils.config import load_config, get_path
cfg = load_config("models")
```

### Task 1.5: Implement Parquet Storage

Write `src/data/storage.py`:
- `save_parquet(df, name)` — saves to `data/processed/{name}.parquet`
- `load_parquet(name)` — loads from `data/processed/{name}.parquet`
- `exists(name)` — checks if processed file exists
- All downstream code uses Parquet, not CSV

### Task 1.6: Fix Known Bugs

- Fix `test_indicators.py`: change import from `src.data.indicators.builder` to correct module paths
- Fix `test_models.py`: replace placeholder `assert True` with real tests for HMM and GARCH
- Replace all `np.random.seed()` calls with `np.random.default_rng(seed)` instances
- Remove deprecated `pct_change(fill_method=None)` kwarg in returns computation
- Fix GARCH distribution from `"normal"` to `"t"` (Student-t) to address the Jarque-Bera normality rejection

### Task 1.7: Write Initial Tests

Write unit tests for migrated code (target: every module has at least one test):
- `tests/unit/test_cleaning.py` — test dedup, interpolation, threshold dropping
- `tests/unit/test_indicators.py` — test each indicator computation on synthetic data
- `tests/unit/test_transforms.py` — test z-score, rolling normalization
- `tests/unit/test_hmm.py` — test fitting on synthetic 3-state data, state labeling
- `tests/unit/test_garch.py` — test fitting on synthetic volatile series, persistence computation
- `tests/fixtures/sample_data.parquet` — small synthetic dataset (100 rows, 10 tickers)

### Task 1.8: Validate Migration

- Run the full pipeline on existing ADA data through the new code structure
- Verify regime labels match original ADA outputs (within tolerance for stochastic models)
- Ensure all tests pass: `make test`

**Deliverables**:
- Clean repository with target architecture
- All configs externalized to YAML
- Parquet storage replacing CSV
- All known bugs fixed
- Passing test suite
- Regime results reproducible in new structure

---

## Phase 2: Data Pipeline

**Objective**: Build a robust, reproducible data pipeline that downloads, cleans, stores, and versions market data. The pipeline must be runnable with a single `make data` command and must support extending to new data sources later.

**Estimated time**: 3–4 days

### Task 2.1: Implement Data Downloaders

Write `src/data/loaders.py` with functions to fetch data from multiple sources:

```python
def download_equity_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from yfinance. Cache to Parquet."""

def download_vix(start: str, end: str) -> pd.DataFrame:
    """Download VIX and VIX3M from yfinance."""

def download_risk_free_rate(start: str, end: str) -> pd.DataFrame:
    """Download 3-month T-bill rate from FRED."""

def download_bond_returns(start: str, end: str) -> pd.DataFrame:
    """Download TLT (Treasury ETF) for 60/40 benchmark."""
```

All date ranges, tickers, and source configurations read from `configs/data.yaml`.

For the initial build, support two modes:
1. **From existing ADA data**: Load the existing `nasdaq_processed.csv` and convert to Parquet (for continuity with ADA results)
2. **Fresh download**: Fetch from yfinance for any universe + date range (for future projects)

### Task 2.2: Implement Cleaning Pipeline

Consolidate ADA's 3-file cleaning logic into `src/data/cleaning.py`:

```python
def clean_pipeline(raw_data: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Per-ticker: dedup by date, convert types, sort
    2. Repair OHLCV quality (zero opens, impossible moves)
    3. Combine into unified DataFrame
    4. Cross-ticker dedup
    5. Interpolate missing (config: window=5, limit forward-only)
    6. Drop tickers exceeding missing threshold (config: 10%)
    7. Remove negative prices
    8. Validate final output

    Returns: cleaned unified DataFrame
    """
```

Key fix: ensure interpolation is **forward-only** (`limit_direction='forward'`) to prevent any lookahead in gap-filling.

### Task 2.3: Implement Universe Construction

Write `src/data/universe.py`:

Instead of ADA's approach (generating 200 random subsets and scoring them), take a simpler approach:

```python
def load_universe(config: dict) -> list[str]:
    """
    Load the pre-selected 295-ticker universe from cached file.
    The ADA project already identified this as the optimal subset.
    For new projects, this can be overridden in config.
    """

def validate_universe(tickers: list, data: pd.DataFrame) -> list[str]:
    """
    Check which tickers have sufficient data in the date range.
    Return valid tickers with enough history.
    """
```

Cache the ADA-derived best subset (295 tickers) as `data/cache/best_subset_tickers.csv`. This avoids re-running the expensive subset search and ensures reproducibility.

### Task 2.4: Implement Storage Layer

Write `src/data/storage.py` — already outlined in Phase 1. Additionally:

- Add a `data_catalog.yaml` that gets auto-generated listing all processed files with their creation dates, row counts, and column names
- This catalog makes it trivial to check if data is stale or needs regeneration

### Task 2.5: Build Pipeline Entry Point

Create `src/data/__main__.py` so that `python -m src.data` runs the full pipeline:

```python
"""
Data pipeline entry point.
Usage: python -m src.data [--fresh] [--validate-only]
"""
# 1. Load config
# 2. Download or load raw data
# 3. Run cleaning pipeline
# 4. Load/validate universe
# 5. Save processed data to Parquet
# 6. Update data catalog
# 7. Print summary statistics
```

### Task 2.6: Download Auxiliary Data

Fetch and store:
- VIX daily (for vol term structure signal and regime confirmation)
- VIX3M daily (for VIX contango/backwardation signal)
- 3-month T-bill rate from FRED (for Sharpe ratio computation)
- TLT daily returns (for 60/40 benchmark)

Store all in `data/processed/auxiliary/` as Parquet.

**Deliverables**:
- `make data` downloads (or loads cached), cleans, and stores all data
- All outputs in Parquet format
- Universe cached and validated
- Auxiliary data (VIX, rates, bonds) available
- Data catalog auto-generated

---

## Phase 3: Regime Detection Models

**Objective**: Migrate, improve, and properly validate the three regime detection models. The critical addition is walk-forward training — no model should see future data during evaluation.

**Estimated time**: 5–6 days

### Task 3.1: Refactor HMM Module

Migrate `hmm_model.py` to `src/models/hmm.py`:

```python
class HMMRegimeDetector:
    """
    Gaussian HMM for regime detection.
    Fits N states, remaps to 3 trading regimes (calm/moderate/turbulent).

    Key improvement over ADA: extracts state probabilities, not just hard labels.
    """
    def __init__(self, config: dict):
        self.n_states = config['n_states']       # From models.yaml
        self.remap = config['remap_to_3']
        # ...

    def fit(self, X: np.ndarray) -> 'HMMRegimeDetector':
        """Fit HMM on training data."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard regime labels (0, 1, 2)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Regime probabilities [P(calm), P(moderate), P(turbulent)]."""

    def get_transition_matrix(self) -> np.ndarray:
        """3x3 transition probability matrix."""
```

Critical addition: `predict_proba()` — returns soft probabilities, not just hard labels. This enables the proportional allocation strategy and reduces turnover from regime-switching noise.

Fix the state-ordering problem: after each refit in walk-forward, states may swap. Sort states by emission mean (ascending = calm → turbulent) after every fit.

### Task 3.2: Refactor GARCH Module

Migrate to `src/models/garch.py`:

```python
class GARCHRegimeDetector:
    """
    GARCH(p,q) conditional volatility → quantile-based regime classification.

    Key improvement over ADA:
    - Student-t distribution (not normal) to handle fat tails
    - Expanding-window quantile thresholds (not full-sample)
    """
    def __init__(self, config: dict):
        self.p = config['p']
        self.q = config['q']
        self.dist = config['distribution']  # "t" for Student-t
        self.quantiles = config['regime_quantiles']

    def fit(self, returns: pd.Series) -> 'GARCHRegimeDetector':
        """Fit GARCH model on training returns."""

    def predict(self, returns: pd.Series) -> np.ndarray:
        """Regime labels using expanding-window quantile thresholds."""

    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """Pseudo-probabilities based on distance from quantile thresholds."""

    def get_conditional_volatility(self) -> pd.Series:
        """Time-varying conditional volatility."""
```

Key fix: quantile thresholds must use **expanding window** (`expanding().quantile()`), not full-sample quantiles. This prevents lookahead bias where today's regime classification uses tomorrow's volatility distribution.

### Task 3.3: Refactor K-Means Module

Migrate to `src/models/kmeans.py`:

```python
class KMeansRegimeDetector:
    """
    K-Means clustering on market features.
    Labels ordered by mean volatility (ascending).
    """
    def __init__(self, config: dict):
        self.n_clusters = config['n_clusters']
        self.features = config['features']

    def fit(self, X: pd.DataFrame) -> 'KMeansRegimeDetector':
        """Fit scaler + KMeans on training data."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Regime labels ordered by volatility (0=calm, 2=turbulent)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Pseudo-probabilities based on distance to cluster centers."""
```

K-Means doesn't naturally produce probabilities. Use inverse distance-to-centroid as pseudo-probabilities: `P(regime_k) = (1/d_k) / sum(1/d_j)`.

### Task 3.4: Refactor Ensemble

Migrate to `src/models/ensemble.py`:

```python
class EnsembleRegimeDetector:
    """
    Combines HMM, GARCH, and KMeans predictions.
    Uses probability-weighted voting (not just hard labels).
    """
    def __init__(self, weights: dict, config: dict):
        self.weights = weights  # e.g., {"hmm": 0.36, "garch": 0.36, "kmeans": 0.27}

    def combine(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted average of regime probabilities.
        Returns: [P(calm), P(moderate), P(turbulent)] per day.
        """

    def predict(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Hard labels from argmax of combined probabilities."""
```

Improvement over ADA: combine probabilities rather than hard labels. This produces smoother signals with less turnover.

### Task 3.5: Implement Walk-Forward Validation

This is the most important new code in Phase 3. Write `src/models/validation.py`:

```python
def walk_forward_train(
    data: pd.DataFrame,
    model_class: type,
    config: dict,
    min_train_years: int = 5,
    test_window_days: int = 252,
    step_days: int = 63,
    expanding: bool = True
) -> pd.DataFrame:
    """
    Walk-forward validation for regime models.

    Process:
    1. Train on [start, start + min_train_years]
    2. Predict on [end_of_train, end_of_train + test_window]
    3. Step forward by step_days
    4. Retrain (expanding or rolling window)
    5. Predict next test window
    6. Concatenate all out-of-sample predictions

    Returns DataFrame with columns:
        date, regime_label, regime_prob_calm, regime_prob_moderate, regime_prob_turbulent

    All predictions are strictly out-of-sample.
    """
```

Walk-forward schedule for 2000–2020 data with 5-year minimum training:
```
Train: 2000-01 to 2004-12 → Predict: 2005-01 to 2005-12
Train: 2000-01 to 2005-03 → Predict: 2005-04 to 2006-03
Train: 2000-01 to 2005-06 → Predict: 2005-07 to 2006-06
...
Train: 2000-01 to 2019-12 → Predict: 2020-01 to 2020-12
```

This produces ~15 years of out-of-sample regime predictions (2005–2020).

**Critical detail**: After each refit, check state ordering. HMM states may relabel. Sort by emission mean ascending. GARCH quantile thresholds must use expanding window up to current training end. K-Means clusters must be sorted by centroid volatility.

### Task 3.6: Model Selection and Diagnostics

Add model selection logic:
- For HMM: compare 2, 3, 4, 5 states using BIC on training data. Select best per window.
- For GARCH: compare GARCH(1,1) vs GARCH(2,1) using BIC. Compare Normal vs Student-t distribution.
- For K-Means: silhouette score for k=2, 3, 4 on training data.

Store model selection results for each walk-forward window for diagnostic reporting.

**Deliverables**:
- Three model classes with consistent `.fit()`, `.predict()`, `.predict_proba()` interface
- Ensemble combiner using probability-weighted voting
- Walk-forward validation producing 15 years of out-of-sample predictions
- Model selection diagnostics per window
- All parameters from config, no hard-coded values
- Tests for each model on synthetic data

---

## Phase 4: Strategy Construction

**Objective**: Build the signal generation layer that converts regime model outputs into portfolio allocation targets. This is the core new code that transforms regime detection into a trading system.

**Estimated time**: 3–4 days

### Task 4.1: Implement Regime Signal Module

Write `src/signals/regime_signal.py`:

```python
def regime_to_allocation(
    regime_probs: pd.DataFrame,
    strategy_config: dict
) -> pd.Series:
    """
    Convert regime probabilities to equity allocation target [0, 1].

    For binary_regime strategy:
        target = P(calm)*1.0 + P(moderate)*0.5 + P(turbulent)*0.0

    For proportional_regime strategy:
        target = P(calm)

    Returns: pd.Series of daily allocation targets.
    """
```

Each strategy variant in `configs/strategy.yaml` produces a different allocation series. All read from the same config structure.

### Task 4.2: Implement Signal Filters

Write `src/signals/filters.py`:

```python
def apply_confirmation_filter(
    raw_signal: pd.Series,
    regime_labels: pd.Series,
    confirmation_days: int = 3
) -> pd.Series:
    """
    Anti-whipsaw filter: only switch allocation when new regime
    persists for N consecutive days.

    Prevents: calm → turbulent (1 day) → calm from triggering 2 trades.
    The signal holds the previous allocation until confirmation.
    """

def apply_rate_limit(
    signal: pd.Series,
    max_daily_change: float = 0.25
) -> pd.Series:
    """
    Limit how fast allocation can change.
    Prevents: jumping from 100% to 0% in a single day.
    Smooths the transition over multiple days.
    """
```

### Task 4.3: Implement Volatility Targeting

Write `src/signals/vol_target.py`:

```python
def apply_vol_target(
    allocation: pd.Series,
    returns: pd.Series,
    vol_target: float = 0.10,
    lookback_days: int = 63
) -> pd.Series:
    """
    Scale allocation so that portfolio volatility targets a fixed level.

    realized_vol = returns.rolling(lookback).std() * sqrt(252)
    vol_scalar = vol_target / realized_vol
    adjusted_allocation = allocation * vol_scalar

    Clip to [0, max_leverage] from config.

    Uses only past data (rolling, not expanding) for vol estimate.
    """
```

Vol-targeting ensures that the portfolio carries approximately the same risk in all regimes. In calm regimes, allocation might increase above 1.0 (if leverage is allowed). In turbulent regimes, allocation shrinks automatically.

### Task 4.4: Apply Execution Lag

Every signal must be lagged before entering the backtest:

```python
def apply_execution_lag(signal: pd.Series, lag_days: int = 1) -> pd.Series:
    """
    Shift signal by lag_days to simulate execution delay.
    Signal generated at close on day T → position entered at close on day T+1.
    """
    return signal.shift(lag_days)
```

This is a one-liner but it is **the single most important anti-lookahead measure in the entire project**. Without this shift, the backtest assumes you can trade on information you don't yet have.

### Task 4.5: Generate All Strategy Signals

Create a pipeline that produces allocation signals for all strategy variants:

```python
def generate_all_signals(
    regime_probs: pd.DataFrame,
    returns: pd.Series,
    config: dict
) -> pd.DataFrame:
    """
    Generate allocation signals for each strategy defined in config.
    Apply filters, vol-targeting, and execution lag.

    Returns DataFrame with columns:
        date, binary_regime, proportional_regime, vol_targeted, regime_momentum

    Each column is a daily allocation target [0, max_leverage].
    """
```

**Deliverables**:
- Signal generation module converting probabilities to allocations
- Confirmation filter and rate limiter
- Volatility targeting overlay
- Execution lag applied to all signals
- All strategy variants generated from single config
- Tests validating that no lookahead exists (signal at T uses only data ≤ T-1)

---

## Phase 5: Backtesting Framework

**Objective**: Build a vectorized backtesting engine that computes portfolio returns with realistic transaction costs. The engine must be reusable for Projects 2 and 3.

**Estimated time**: 4–5 days

### Task 5.1: Implement Backtest Engine

Write `src/backtest/engine.py`:

```python
def run_backtest(
    market_returns: pd.Series,
    allocation_signal: pd.Series,
    cost_config: dict
) -> pd.DataFrame:
    """
    Vectorized backtest.

    Steps:
    1. Position = allocation_signal (already lagged)
    2. Gross return = position * market_return (per day)
    3. Position change = |position_t - position_{t-1}|
    4. Transaction cost = position_change * (cost_bps + slippage_bps) / 10000
    5. Net return = gross_return - transaction_cost
    6. Cumulative return = (1 + net_return).cumprod()
    7. Drawdown = cumulative / cumulative.cummax() - 1

    Returns DataFrame:
        date, position, gross_return, turnover, cost, net_return,
        cumulative_return, drawdown
    """
```

The engine is intentionally simple — a vectorized computation, not an event-driven simulator. This is appropriate for daily equity allocation strategies.

### Task 5.2: Implement Transaction Cost Model

Write `src/backtest/costs.py`:

```python
def compute_transaction_costs(
    positions: pd.Series,
    config: dict
) -> pd.Series:
    """
    Compute daily transaction costs from position changes.

    Cost model:
    - Fixed cost: cost_bps per unit traded (one-way)
    - Slippage: slippage_bps per unit traded
    - Minimum trade filter: skip trades smaller than min_trade_bps

    Total one-way cost = (cost_bps + slippage_bps) / 10000
    Daily cost = |position_change| * total_one_way_cost
    """
```

### Task 5.3: Implement Portfolio Module

Write `src/backtest/portfolio.py`:

```python
def construct_portfolio(
    signal: pd.Series,
    config: dict
) -> pd.Series:
    """
    Apply portfolio constraints to raw signal.

    Constraints (from config):
    - Max leverage: cap allocation at max_leverage
    - Min allocation: floor at 0 (no shorting) or -1 (if shorting allowed)
    - Turnover cap: if annualized turnover exceeds limit, smooth signal
    """
```

### Task 5.4: Implement Benchmarks

Write `src/backtest/benchmarks.py`:

```python
def buy_and_hold(market_returns: pd.Series) -> pd.DataFrame:
    """100% equity, always. The simplest benchmark."""

def sixty_forty(
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    rebalance: str = "monthly"
) -> pd.DataFrame:
    """Classic 60/40 portfolio, rebalanced monthly."""

def risk_parity(
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    lookback: int = 63
) -> pd.DataFrame:
    """Inverse-vol weighted equity/bond allocation."""
```

### Task 5.5: Run All Backtests

Create orchestration that runs every strategy through the backtest engine:

```python
def run_all_backtests(
    signals: pd.DataFrame,       # Columns: strategy variants
    market_returns: pd.Series,
    bond_returns: pd.Series,
    config: dict
) -> dict[str, pd.DataFrame]:
    """
    Run backtest for each strategy + benchmarks.
    Returns dict mapping strategy_name → backtest_results DataFrame.
    """
```

### Task 5.6: Transaction Cost Sensitivity Analysis

Run each strategy at multiple cost levels: 0, 2, 5, 10, 15, 20 bps one-way.

```python
def cost_sensitivity(
    market_returns: pd.Series,
    signal: pd.Series,
    cost_levels_bps: list = [0, 2, 5, 10, 15, 20]
) -> pd.DataFrame:
    """
    Returns DataFrame: cost_bps, sharpe, cagr, max_dd, turnover
    One row per cost level.

    The breakeven cost (where Sharpe drops to 0 or equals benchmark) is
    the most important output of this analysis.
    """
```

This is the chart that tells a trader "this strategy works up to X bps of costs."

**Deliverables**:
- Vectorized backtest engine with transaction costs
- Benchmark strategies (buy-and-hold, 60/40)
- Cost sensitivity analysis
- All strategies backtested with net returns
- Turnover tracking per strategy
- Tests: verify that a zero-cost backtest of buy-and-hold matches raw market returns exactly

---

## Phase 6: Performance Evaluation

**Objective**: Compute comprehensive performance metrics, statistical significance tests, and regime-conditional analysis. Produce a full strategy tearsheet.

**Estimated time**: 3–4 days

### Task 6.1: Implement Core Metrics

Write `src/evaluation/metrics.py`:

```python
def compute_metrics(returns: pd.Series, risk_free: pd.Series = None) -> dict:
    """
    Compute all standard performance metrics.

    Returns dict:
        cagr, annual_vol, sharpe, sortino, calmar,
        max_drawdown, max_drawdown_duration_days,
        skewness, kurtosis,
        hit_rate_monthly, profit_factor,
        avg_monthly_return, worst_month, best_month,
        turnover_annual (if position data provided)
    """

def compute_sharpe(returns: pd.Series, risk_free: pd.Series = None) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - (risk_free or 0)
    return excess.mean() / excess.std() * np.sqrt(252)

def compute_max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """Max drawdown magnitude and duration in trading days."""

def compute_sortino(returns: pd.Series, risk_free: pd.Series = None) -> float:
    """Sortino ratio: penalizes downside vol only."""

def compute_calmar(returns: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
```

### Task 6.2: Implement Rolling Metrics

Write `src/evaluation/rolling.py`:

```python
def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """1-year rolling Sharpe ratio."""

def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Quarterly rolling annualized volatility."""

def rolling_drawdown(returns: pd.Series) -> pd.Series:
    """Continuous drawdown series (underwater curve)."""

def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
    """Rolling beta to benchmark."""
```

### Task 6.3: Implement Regime-Conditional Performance

Write `src/evaluation/regime_perf.py`:

```python
def performance_by_regime(
    returns: pd.Series,
    regime_labels: pd.Series
) -> pd.DataFrame:
    """
    Compute metrics separately for each regime.

    Returns DataFrame indexed by regime:
        regime | mean_return | vol | sharpe | max_dd | pct_days | contribution

    'contribution' = regime's share of total cumulative return.
    """
```

This answers the question every trader asks: "Where does the alpha come from? Does it work in all regimes or just calm markets?"

### Task 6.4: Implement Statistical Significance

Write `src/evaluation/significance.py`:

```python
def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for Sharpe ratio.
    Returns: (sharpe, lower_ci, upper_ci)

    If lower_ci > 0: strategy is statistically significant at given confidence.
    """

def sharpe_difference_test(
    returns_strategy: pd.Series,
    returns_benchmark: pd.Series,
    n_bootstrap: int = 10000
) -> tuple[float, float]:
    """
    Test whether strategy Sharpe is significantly different from benchmark.
    Returns: (sharpe_diff, p_value)

    Uses Ledoit-Wolf (2008) HAC-adjusted bootstrap.
    """
```

### Task 6.5: Generate Performance Comparison Table

```python
def performance_table(
    backtest_results: dict[str, pd.DataFrame],
    risk_free: pd.Series
) -> pd.DataFrame:
    """
    Master comparison table across all strategies and benchmarks.

    Columns: CAGR, Vol, Sharpe, Sharpe CI, Sortino, Calmar,
             Max DD, Turnover, Hit Rate
    Rows: one per strategy + benchmarks

    This is the single most important output of the project.
    """
```

### Task 6.6: Generate Report

Write `src/evaluation/report.py`:

```python
def generate_tearsheet(
    backtest_results: dict[str, pd.DataFrame],
    regime_labels: pd.Series,
    config: dict,
    output_dir: str = "results/tearsheets"
) -> str:
    """
    Generate complete HTML tearsheet:
    1. Performance summary table
    2. Cumulative return chart
    3. Drawdown chart
    4. Rolling Sharpe
    5. Regime-conditional performance
    6. Monthly return heatmap
    7. Cost sensitivity chart
    8. Turnover analysis

    Returns path to generated HTML file.
    """
```

**Deliverables**:
- Complete metrics library (Sharpe, Sortino, Calmar, max DD, hit rate, profit factor)
- Rolling metrics (Sharpe, vol, drawdown)
- Regime-conditional performance analysis
- Bootstrap confidence intervals for Sharpe
- Performance comparison table
- HTML tearsheet generator
- Tests: verify metrics against known values on synthetic returns

---

## Phase 7: Visualization and Research Outputs

**Objective**: Generate all charts needed for the research website and tearsheet. Every figure must be publication-quality, clearly labeled, and interpretable by a trader who has 30 seconds to look at it.

**Estimated time**: 3–4 days

### Task 7.1: Performance Visualizations

Write `src/visualization/performance.py`:

**Chart 1: Cumulative Return Comparison**
```
- X-axis: Date (2005–2020, out-of-sample period only)
- Y-axis: Growth of $1
- Lines: Each strategy + buy-and-hold + 60/40
- Background: Light colored bands for regime periods
- Annotations: Major events (2008 crisis, COVID)
```

**Chart 2: Drawdown Chart**
```
- X-axis: Date
- Y-axis: Drawdown from peak (negative %)
- Area fill for best strategy, line for benchmark
- Regime bands in background
```

**Chart 3: Monthly Return Heatmap**
```
- X-axis: Month (Jan–Dec)
- Y-axis: Year (2005–2020)
- Color: Return magnitude (red=negative, green=positive)
- One heatmap per strategy
```

**Chart 4: Rolling 1-Year Sharpe**
```
- X-axis: Date
- Y-axis: Rolling Sharpe ratio
- Lines: Strategy + benchmark
- Shaded area: Bootstrap confidence band
- Horizontal line at Sharpe = 0
```

**Chart 5: Cost Sensitivity**
```
- X-axis: Transaction cost (bps)
- Y-axis: Sharpe ratio
- Lines: Each strategy
- Horizontal line at benchmark Sharpe
- Vertical annotation: "Breakeven cost"
```

### Task 7.2: Regime Visualizations

Write `src/visualization/regimes.py`:

**Chart 6: Regime Timeline**
```
- X-axis: Date (2000–2020)
- Three-color horizontal bands: calm (green), moderate (amber), turbulent (red)
- Top panel: S&P 500 price overlay
- Bottom panel: Regime probabilities stacked area
```

**Chart 7: Regime Transition Matrix**
```
- 3×3 heatmap with transition probabilities
- Annotated with percentages
- Color intensity proportional to probability
```

**Chart 8: Regime Distribution**
```
- Bar chart: % of days in each regime
- Stacked by time period (pre-2008, 2008-2009, post-2009, 2020)
```

**Chart 9: Regime-Conditional Performance**
```
- Grouped bar chart
- Groups: Calm, Moderate, Turbulent
- Bars: Mean return, volatility, Sharpe (each metric as separate subplot)
```

### Task 7.3: Signal Visualizations

Write `src/visualization/signals.py`:

**Chart 10: Allocation Over Time**
```
- X-axis: Date
- Y-axis: Allocation level (0 to 1)
- Area fill for allocation level
- Overlay: regime bands in background
- Shows when strategy goes defensive vs aggressive
```

**Chart 11: Signal vs Drawdown**
```
- Dual axis: allocation signal (left), market drawdown (right, inverted)
- Shows: did the signal reduce exposure before/during drawdowns?
- Most compelling chart for demonstrating the strategy's value
```

### Task 7.4: Diagnostic Visualizations

Write `src/visualization/diagnostics.py`:

**Chart 12: Model Selection Over Time**
```
- X-axis: Walk-forward window index
- Y-axis: BIC / selected n_states
- Shows model stability through time
```

**Chart 13: Ensemble Agreement**
```
- Heatmap: pairwise agreement rates between HMM, GARCH, K-Means
- Shows where methods agree and disagree
```

### Task 7.5: Tearsheet Composition

Write `src/visualization/tearsheet.py`:

```python
def generate_full_tearsheet(results: dict, output_path: str):
    """
    Generate a single multi-page figure combining all key charts.
    Layout:
        Page 1: Performance summary + cumulative returns + drawdown
        Page 2: Regime timeline + allocation over time
        Page 3: Rolling metrics + cost sensitivity
        Page 4: Regime-conditional performance + monthly heatmap
    """
```

**Deliverables**:
- 13 publication-quality charts
- Consistent style across all plots (dark theme option for website, light theme for PDF)
- Tearsheet generator combining key charts
- All figures saved as high-res PNG (300 DPI) in `results/figures/`
- Chart-generation code separated from data computation

---

## Phase 8: Research Website

**Objective**: Build a clean static website presenting the research results. The website must be deployable via GitHub Pages and linkable from a CV.

**Estimated time**: 2–3 days

### Task 8.1: Design Website Structure

Single-page scrolling layout. Dark theme. No JavaScript framework — plain HTML/CSS with minimal JS for smooth scrolling and lazy image loading.

```
URL: yourname.com/research/regimes

Sections:
├── Hero: Title + tagline + key metric badges (Sharpe, Max DD, CAGR)
├── Research Question: 2–3 sentences
├── Methodology: Pipeline diagram (Data → Features → Regimes → Signal → Backtest)
├── Regime Detection: Regime timeline chart + transition matrix
├── Strategy Design: Allocation logic + signal chart
├── Performance: Cumulative return chart + drawdown chart
├── Performance Table: All strategies + benchmarks
├── Rolling Sharpe: Stability analysis
├── Cost Sensitivity: Breakeven analysis
├── Regime Performance: Where does alpha come from?
├── Limitations: Honest assessment
├── Footer: Links to GitHub, other projects
```

### Task 8.2: Build HTML/CSS

Write `website/index.html` and `website/css/style.css`:

Design principles:
- Dark background (`#0a0a0a`), light text (`#e0e0e0`)
- Trading blue accent (`#4fc3f7`)
- Maximum width 900px for readability
- Responsive (works on mobile)
- Charts as `<img>` tags loading from `assets/figures/`
- Performance table as styled HTML `<table>`
- No external dependencies beyond Google Fonts (optional)

### Task 8.3: Build Chart Assets

Copy key figures from `results/figures/` to `website/assets/figures/`.

Charts for website (subset of full results):
1. Regime timeline with price overlay
2. Cumulative return comparison
3. Drawdown chart
4. Rolling Sharpe
5. Cost sensitivity
6. Allocation over time
7. Regime-conditional performance
8. Performance summary table (rendered as HTML, not image)

### Task 8.4: Write Website Content

Write concise, trader-friendly text for each section. Not academic prose. Not blog writing. Think: research note from a bank quant team.

Example tone:
> "The HMM detects regime shifts 3–5 days before volatility spikes become obvious in rolling metrics. During the 2008 crisis, the model moved to full defensive allocation by October 2007 — four months before the worst drawdowns. The strategy's max drawdown of -X% compares to buy-and-hold's -55%."

### Task 8.5: Deploy

- Configure GitHub Pages to serve from `website/` directory (or `docs/` if GitHub requires it)
- Verify all images load correctly
- Test on mobile
- Add link to README

**Deliverables**:
- Static HTML/CSS website
- Deployed on GitHub Pages
- All key charts and metrics visible
- Mobile-responsive
- Linked from README and suitable for CV

---

# 5. Data Requirements

## Primary Data

| Dataset | Source | Frequency | Date Range | Format |
|---------|--------|-----------|------------|--------|
| US equity OHLCV (NASDAQ universe) | Existing ADA dataset or yfinance | Daily | 2000–2020 | Parquet |
| S&P 500 / NASDAQ constituents | Cached from ADA project | Point-in-time | 2000, 2010, 2019 | CSV |
| Best subset ticker list (295) | Cached from ADA analysis | Static | — | CSV |

## Auxiliary Data

| Dataset | Source | Frequency | Date Range | Purpose |
|---------|--------|-----------|------------|---------|
| VIX | yfinance (`^VIX`) | Daily | 2000–2020 | Vol regime confirmation |
| VIX3M | yfinance (`^VIX3M`) | Daily | 2007–2020 | VIX term structure signal |
| 3-month T-bill rate | FRED (`DGS3MO`) | Daily | 2000–2020 | Risk-free rate for Sharpe |
| TLT (20+ Year Treasury ETF) | yfinance (`TLT`) | Daily | 2002–2020 | Bond leg for 60/40 benchmark |
| S&P 500 Total Return | yfinance (`^SP500TR`) or calculated | Daily | 2000–2020 | Market return series |

## Data Pipeline Flow

```
Step 1: Download / Load Raw Data
    ├── Load existing ADA CSVs OR download from yfinance
    ├── Download VIX, VIX3M, T-bill rate, TLT from yfinance/FRED
    └── Load cached ticker universe (295 tickers)

Step 2: Clean
    ├── Deduplicate, type-convert, sort
    ├── Repair OHLCV quality
    ├── Forward-only interpolation (5-day window)
    ├── Drop tickers with >10% missing
    └── Remove negative prices

Step 3: Store
    ├── Save cleaned equity data as Parquet
    ├── Save auxiliary data as Parquet
    └── Generate data catalog

Step 4: Validate
    ├── Row counts match expected
    ├── No NaN in critical columns
    ├── Date range coverage complete
    └── Print summary statistics
```

All parameters in this pipeline are read from `configs/data.yaml`. The pipeline runs end-to-end with `make data`.

---

# 6. Modeling Approach

## Model 1: Hidden Markov Model (Primary)

**Input**: PC2 systemic stress index (from PCA of market indicators).

**Architecture**: 5-state Gaussian HMM with full covariance. States remapped to 3 trading regimes by emission mean ordering.

**Training**: Maximum likelihood via EM algorithm (1000 iterations, convergence checked). Walk-forward with expanding window, quarterly refits.

**Output**: Daily regime probabilities [P(calm), P(moderate), P(turbulent)].

**Why HMM is the primary model**: It was the best individual method in ADA (composite score 0.612) and naturally produces posterior probabilities — essential for smooth allocation signals.

**Validation**:
- BIC model selection for n_states at each refit
- State ordering check after each fit (sort by emission mean)
- Transition matrix stability across windows
- Crisis period detection accuracy (2008, 2020)

## Model 2: GARCH(1,1) Conditional Volatility

**Input**: Market direction returns (mean daily log return of subset).

**Architecture**: GARCH(1,1) with Student-t innovations (fixing ADA's normality issue). Conditional volatility → expanding-window quantile classification.

**Training**: Maximum likelihood. Walk-forward with expanding window.

**Output**: Daily conditional volatility + regime label. Pseudo-probabilities from quantile distance.

**Validation**:
- Ljung-Box test on standardized residuals (should be insignificant)
- Persistence < 1.0 (stationarity check)
- Expanding quantile thresholds must use only past data

## Model 3: K-Means Clustering (Baseline)

**Input**: Market volatility + market volume (2 features for VolVol variant) or mood composite (1 feature for MoodIndex variant).

**Architecture**: StandardScaler → KMeans(3 clusters). Labels sorted by centroid volatility.

**Training**: Walk-forward with expanding window. Scaler and cluster centers refitted.

**Output**: Daily cluster labels + pseudo-probabilities from inverse centroid distance.

**Validation**:
- Silhouette score at each refit
- Cluster centroid stability
- Label ordering consistency

## Ensemble Combination

Probability-weighted average across all three models. Weights from `configs/models.yaml` (default: roughly equal, based on ADA's performance-based optimization).

Walk-forward note: ensemble weights could also be optimized walk-forward (train on in-sample composite score), but for simplicity start with fixed weights. Optimize weights as a later enhancement.

---

# 7. Strategy Design

## How Regime Signals Become Trading Decisions

```
HMM, GARCH, KMeans → Regime Probabilities → Ensemble → Allocation Target
                                                            ↓
                                               Confirmation Filter (3 days)
                                                            ↓
                                               Rate Limiter (max 25%/day)
                                                            ↓
                                               Vol-Target Overlay (optional)
                                                            ↓
                                               Execution Lag (1 day)
                                                            ↓
                                               Final Position → Backtest
```

## Strategy Variants

### Strategy 1: Binary Regime Allocation
- Calm → 100% equity
- Moderate → 50% equity
- Turbulent → 0% equity (100% cash)
- Allocation = weighted sum using regime probabilities
- Expected character: aggressive risk-off in crises, high turnover at regime boundaries

### Strategy 2: Proportional Allocation
- Allocation = P(calm) directly
- Smoother than binary — avoids discrete jumps
- Lower turnover, potentially lower Sharpe but higher Calmar

### Strategy 3: Volatility-Targeted Regime
- Base allocation from binary regime
- Scaled by vol-target overlay (target: 10% annualized)
- In calm low-vol regimes: allocation > 1.0 possible (if leverage permitted)
- In turbulent high-vol regimes: allocation shrinks further
- Expected character: most stable risk profile

### Strategy 4: Regime Momentum
- If regime has persisted > 10 days, increase conviction
- Calm + persistent → 120% (slight leverage)
- Calm + new → 80%
- Turbulent → 0% regardless
- Expected character: trend-following flavor within regime framework

## Signal Storage

All signals saved to `data/processed/signals.parquet` with columns:
```
date | binary_regime | proportional | vol_targeted | regime_momentum |
      regime_label | regime_prob_calm | regime_prob_moderate | regime_prob_turbulent
```

This file is the single input to the backtesting engine.

---

# 8. Backtesting Methodology

## Walk-Forward Structure

```
|-------- Training ---------|-- Test --|
2000 ======================== 2005      2006
2000 ============================ 2006      2007
2000 ================================ 2007      2008
...
2000 ====================================================== 2019      2020

Total out-of-sample: 2005-01 to 2020-12 (~15 years)
```

- **Training**: expanding window, minimum 5 years
- **Test**: 1-year forward prediction
- **Refit frequency**: quarterly (63 trading days)
- **All signals evaluated only on out-of-sample predictions**

## Execution Assumptions

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Execution lag | 1 day | Signal at close T, trade at close T+1 |
| Transaction cost | 5 bps one-way | Realistic for institutional equity (ETF) trading |
| Slippage | 2 bps one-way | Conservative for liquid equities |
| Total round-trip cost | 14 bps | (5+2) × 2 = 14 bps per round trip |
| Rebalance frequency | Daily (but filtered) | Only trade when signal changes materially |
| Min trade size | 1% allocation change | Skip tiny trades to reduce cost drag |

## Avoiding Common Mistakes

| Mistake | How We Prevent It |
|---------|-------------------|
| **Lookahead bias** | Walk-forward training. Execution lag. Expanding-window features only. |
| **Survivorship bias** | Ticker universe from S&P 500 temporal union (includes past members). |
| **Data leakage** | Models trained on [0, T], predict T+1 only. No future data in features. |
| **Overfitting** | Walk-forward (not in-sample). Bootstrap Sharpe CI. Cost sensitivity. |
| **Unrealistic costs** | 5+2 bps model. Sensitivity analysis from 0 to 20 bps. |
| **Selection bias** | Report ALL strategy variants, not just the best one. |
| **Data snooping** | Pre-register strategies in config before running backtests. |

## Portfolio Return Computation

```python
# Vectorized daily return computation
position = signal.shift(1)                           # Lag 1 day
gross_return = position * market_return               # Daily P&L
turnover = position.diff().abs()                      # Daily turnover
cost = turnover * (cost_bps + slippage_bps) / 10000   # Daily cost
net_return = gross_return - cost                       # Net of costs
cumulative = (1 + net_return).cumprod()                # Growth of $1
drawdown = cumulative / cumulative.cummax() - 1        # Underwater
```

---

# 9. Evaluation and Performance Metrics

## Metrics Computed

| Category | Metrics |
|----------|---------|
| **Return** | CAGR, mean monthly return, best month, worst month |
| **Risk** | Annualized volatility, max drawdown, max drawdown duration, skewness, kurtosis |
| **Risk-adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio |
| **Execution** | Annual turnover, average trade size, number of trades |
| **Consistency** | Hit rate (% positive months), profit factor |
| **Statistical** | Bootstrap Sharpe 95% CI, Sharpe difference p-value vs benchmark |

## Performance Report Structure

The final performance report is a single HTML file containing:

```
1. Executive Summary
   - Best strategy name and key metrics
   - Comparison to buy-and-hold: "Strategy produced Sharpe X vs benchmark Y"

2. Performance Table
   - All strategies + benchmarks
   - All metrics in columns
   - Color-coded: green for better than benchmark, red for worse

3. Cumulative Return Chart
   - All strategies + benchmarks overlaid

4. Drawdown Analysis
   - Drawdown chart for best strategy vs benchmark
   - Top 5 drawdown events: start date, end date, recovery date, magnitude

5. Rolling Analysis
   - Rolling 1-year Sharpe
   - Rolling 1-year volatility

6. Regime Analysis
   - Performance by regime
   - Contribution by regime

7. Cost Analysis
   - Sharpe vs transaction cost
   - Breakeven cost level

8. Statistical Significance
   - Bootstrap Sharpe confidence intervals
   - p-value for Sharpe difference vs benchmark
```

---

# 10. Visualizations

## Complete Chart List

| # | Chart | Purpose | File |
|---|-------|---------|------|
| 1 | Cumulative return comparison | Core performance proof | `cumulative_returns.png` |
| 2 | Drawdown chart | Risk visualization | `drawdown.png` |
| 3 | Rolling 1Y Sharpe | Strategy stability | `rolling_sharpe.png` |
| 4 | Monthly return heatmap | Consistency analysis | `monthly_heatmap.png` |
| 5 | Cost sensitivity | Practical viability | `cost_sensitivity.png` |
| 6 | Regime timeline | Regime detection quality | `regime_timeline.png` |
| 7 | Regime transition matrix | Regime dynamics | `transition_matrix.png` |
| 8 | Regime distribution | Regime balance | `regime_distribution.png` |
| 9 | Regime-conditional performance | Alpha source | `regime_performance.png` |
| 10 | Allocation over time | Signal behavior | `allocation_timeline.png` |
| 11 | Signal vs drawdown overlay | Signal quality | `signal_vs_drawdown.png` |
| 12 | Model selection over time | Model stability | `model_selection.png` |
| 13 | Ensemble agreement | Method consensus | `ensemble_agreement.png` |
| 14 | Performance summary table | Master comparison | `performance_table.png` |

## Style Guidelines

- **DPI**: 300 for all figures
- **Color palette**: Consistent across all charts
  - Strategy: `#4fc3f7` (blue)
  - Benchmark (buy-hold): `#78909c` (gray)
  - Benchmark (60/40): `#b0bec5` (light gray)
  - Calm regime: `#66bb6a` (green)
  - Moderate regime: `#ffa726` (amber)
  - Turbulent regime: `#ef5350` (red)
  - Positive returns: `#66bb6a`
  - Negative returns: `#ef5350`
- **Figure size**: 12×6 for time series, 10×8 for heatmaps
- **Font**: Sans-serif, 12pt labels, 14pt titles
- **Background**: White for PDF/PNG export, dark option for website

---

# 11. Research Website

## Hosting

- **Platform**: GitHub Pages
- **URL**: `yourname.github.io/systematic-regime-trading` (or custom domain subdirectory)
- **Build**: No build step — static HTML/CSS/JS checked into `website/` directory

## Content Structure

```html
<!-- Section 1: Hero -->
<header>
  <h1>Market Regime Modeling for Systematic Trading</h1>
  <p class="tagline">Regime-Based Allocation on US Equities, 2005–2020</p>
  <div class="metrics-badges">
    <span class="badge">Sharpe: X.XX</span>
    <span class="badge">Max DD: -XX%</span>
    <span class="badge">CAGR: X.X%</span>
  </div>
  <a href="https://github.com/..." class="button">View on GitHub</a>
</header>

<!-- Section 2: Research Question -->
<section>
  <h2>Research Question</h2>
  <p>Can unsupervised market regime detection produce systematic allocation
     signals that survive transaction costs?</p>
</section>

<!-- Section 3: Methodology -->
<section>
  <h2>Methodology</h2>
  <img src="assets/pipeline_diagram.svg" alt="Research pipeline">
  <p>Brief description of HMM + GARCH + KMeans ensemble.</p>
</section>

<!-- Section 4: Regime Detection -->
<section>
  <h2>Regime Detection</h2>
  <img src="assets/figures/regime_timeline.png">
  <img src="assets/figures/transition_matrix.png">
</section>

<!-- Section 5: Strategy Performance -->
<section>
  <h2>Performance</h2>
  <img src="assets/figures/cumulative_returns.png">
  <img src="assets/figures/drawdown.png">
  <table><!-- Performance comparison table --></table>
</section>

<!-- Section 6: Rolling Analysis -->
<section>
  <h2>Stability</h2>
  <img src="assets/figures/rolling_sharpe.png">
</section>

<!-- Section 7: Cost Analysis -->
<section>
  <h2>Transaction Cost Sensitivity</h2>
  <img src="assets/figures/cost_sensitivity.png">
  <p>Strategy remains profitable up to X bps one-way costs.</p>
</section>

<!-- Section 8: Limitations -->
<section>
  <h2>Limitations</h2>
  <ul>
    <li>Equity-only universe; commodity extension in progress</li>
    <li>Daily rebalancing assumes institutional execution</li>
    <li>Walk-forward avoids in-sample bias but parameter choices still explored ex-post</li>
  </ul>
</section>

<!-- Footer -->
<footer>
  <p>Part of the Systematic Trading Research portfolio</p>
  <a href="/research/curves">CurveTrade: Commodity Factor Trading</a>
  <a href="/research/spreads">SpreadHunter: Statistical Arbitrage</a>
</footer>
```

## Design Requirements

- Maximum width: 900px centered
- Clean typography (system fonts or Inter/JetBrains Mono)
- Charts full-width within container
- Performance table: sticky header, alternating row colors
- Responsive: images scale on mobile
- No JavaScript frameworks — vanilla JS only for scroll behavior
- Total page weight: < 5MB (optimize chart PNGs)

---

# 12. Estimated Timeline

| Phase | Description | Days | Cumulative |
|-------|-------------|------|------------|
| **Phase 1** | Repository setup, refactoring, config system, bug fixes | 4–5 | Week 1 |
| **Phase 2** | Data pipeline (loaders, cleaning, storage, universe) | 3–4 | Week 2 |
| **Phase 3** | Regime models (HMM, GARCH, KMeans, ensemble, walk-forward) | 5–6 | Week 3 |
| **Phase 4** | Strategy construction (signals, filters, vol-target) | 3–4 | Week 3–4 |
| **Phase 5** | Backtesting framework (engine, costs, benchmarks) | 4–5 | Week 4 |
| **Phase 6** | Performance evaluation (metrics, significance, reporting) | 3–4 | Week 5 |
| **Phase 7** | Visualization (all charts, tearsheet) | 3–4 | Week 5–6 |
| **Phase 8** | Research website (HTML/CSS, content, deploy) | 2–3 | Week 6 |
| **Buffer** | Testing, debugging, polish | 2–3 | Week 6–7 |
| **Total** | | **30–38 days** | **~6–7 weeks** |

### Suggested Weekly Schedule

```
Week 1: Phase 1 — Repository restructuring, configs, bug fixes
        Milestone: `make test` passes, regime results reproduce

Week 2: Phase 2 + Phase 3 start — Data pipeline, begin model refactoring
        Milestone: `make data` produces clean Parquet, HMM walk-forward runs

Week 3: Phase 3 complete + Phase 4 — All models walk-forward, signal construction
        Milestone: Out-of-sample regime predictions, allocation signals generated

Week 4: Phase 5 — Backtesting framework
        Milestone: All strategies backtested, performance metrics computed

Week 5: Phase 6 + Phase 7 — Evaluation, all visualizations
        Milestone: Tearsheet generated, all 14 charts produced

Week 6: Phase 8 + polish — Website, README, final review
        Milestone: Website deployed, repository clean, v1.0 tagged
```

---

# 13. Risks and Pitfalls

## Risk 1: Unstable HMM State Labels Across Walk-Forward Windows

**Problem**: When the HMM is refitted on a new training window, the EM algorithm may converge to a different label ordering. State 0 in window 1 might correspond to State 2 in window 2.

**Mitigation**: After every refit, sort states by emission mean (ascending). This ensures State 0 is always the lowest-volatility state. Verify by checking that transition matrix diagonal values are > 0.5 (states should be persistent).

**Detection**: Plot emission means across walk-forward windows. If they jump erratically, the sorting is failing.

## Risk 2: Overfitting the Strategy to In-Sample Regimes

**Problem**: Choosing strategy parameters (allocation targets, confirmation days, vol target) based on backtest results is a form of data snooping.

**Mitigation**:
- Pre-register strategy variants in `configs/strategy.yaml` before running backtests
- Report ALL variants, not just the best one
- Use bootstrap Sharpe CI to assess whether differences are significant
- Run cost sensitivity to show robustness across assumptions
- Keep strategy logic simple (fewer parameters = less room to overfit)

## Risk 3: GARCH Quantile Thresholds and Lookahead

**Problem**: If GARCH regime thresholds (33rd/67th percentile of conditional volatility) are computed on the full sample, today's classification uses future information.

**Mitigation**: Use expanding-window quantiles: `expanding().quantile(q)`. At time T, the threshold uses only volatility values from [0, T]. This means early-period classifications are noisier (small sample), which is realistic.

## Risk 4: Transaction Cost Assumptions Too Generous

**Problem**: If the cost model is too low (e.g., 1 bps), the strategy looks better than it would in practice. If too high, the strategy looks worse than necessary.

**Mitigation**: Use 5+2 bps as the central estimate (reasonable for institutional equity/ETF trading). Run sensitivity from 0 to 20 bps. Report the breakeven cost. Let the reader decide whether their execution is better or worse than the assumption.

## Risk 5: Regime Model Produces No Signal

**Problem**: If the HMM decides everything is "moderate" for years, the strategy holds 50% equity for years and underperforms buy-and-hold in calm markets.

**Mitigation**: This is actually an acceptable outcome — it means the model is uncertain, and the strategy's response is caution. Report regime distribution. If >80% of days are classified as a single regime, the model may be too conservative. Consider reducing the number of HMM states or adjusting the remapping thresholds.

## Risk 6: Walk-Forward Produces Insufficient Out-of-Sample Data

**Problem**: With 5-year minimum training and 2000–2020 data, you only get 15 years out-of-sample. That's ~3,750 trading days — decent but not enormous for statistical significance.

**Mitigation**: Bootstrap confidence intervals account for sample size. Report the CI width honestly. If the 95% CI for Sharpe includes 0, say so. Consider extending the data range using yfinance (1990–2024 is available for many stocks).

## Risk 7: Website Performance with Large Images

**Problem**: 14 high-res PNG charts at 300 DPI can create a 20MB+ page.

**Mitigation**: Save website versions of charts at 150 DPI or as optimized PNGs (using `pngquant` or similar). Keep full-res versions in `results/figures/` for the PDF report. Target total page weight < 5MB.

## Risk 8: Ensemble Underperforms Best Individual Model

**Problem**: The ADA project already showed that the ensemble (0.579) scored lower than HMM alone (0.612). Adding complexity without improvement hurts credibility.

**Mitigation**: Report individual models AND ensemble. If HMM alone is best, present it as the primary strategy and show the ensemble as a robustness check. Don't force the ensemble to be the answer if it isn't. Honesty about model selection is more impressive to traders than claiming everything works.

---

# Appendix A: Key Files to Migrate from ADA

| ADA File | New Location | Migration Notes |
|----------|-------------|-----------------|
| `src/data/cleaning/clean.py` | `src/data/cleaning.py` | Merge with repair.py and validation.py |
| `src/data/cleaning/repair.py` | `src/data/cleaning.py` | Merge |
| `src/data/cleaning/validation.py` | `src/data/cleaning.py` | Merge |
| `src/data/loaders/loader.py` | `src/data/loaders.py` | Add yfinance support, config paths |
| `src/data/indicators/mood_indicators.py` | `src/features/indicators.py` | Extract transforms to separate file |
| `src/data/indicators/returns.py` | `src/features/volatility.py` | Merge with volatility.py |
| `src/data/indicators/volatility.py` | `src/features/volatility.py` | Merge |
| `src/data/indicators/smoothing_window.py` | `src/features/transforms.py` | Merge with validation |
| `src/models/hmm/hmm_model.py` | `src/models/hmm.py` | Add predict_proba, config params |
| `src/models/garch/garch_model.py` | `src/models/garch.py` | Fix: Student-t, expanding quantiles |
| `src/models/clustering/clustering.py` | `src/models/kmeans.py` | Remove plotting, add predict_proba |
| `src/utils/analysis/ensemble.py` | `src/models/ensemble.py` | Probability-based combination |
| `src/utils/analysis/regime_analysis.py` | `src/evaluation/regime_perf.py` | Extend with performance metrics |
| `src/utils/analysis/regime_characterization.py` | `src/evaluation/metrics.py` | Extract risk metrics |
| `src/utils/visualization/visualization.py` | Split across `src/visualization/` | Split by purpose |
| `src/utils/visualization/regime_comparison_plots.py` | `src/visualization/regimes.py` | Clean up |
| `src/utils/helpers/helpers.py` | `src/features/transforms.py` | Merge into transforms |

## Files NOT Migrated (Removed)

| ADA File | Reason |
|----------|--------|
| `src/utils/app/display_section.py` (159KB) | Streamlit-specific, replaced by static website |
| `src/utils/app/app_utils.py` | Streamlit-specific |
| `webapp/` | Streamlit deployment |
| `src/scripts/get_ticker.py` | Non-functional, hard-coded paths |
| `src/data/sampling/sampling.py` (19KB) | Subset search not needed (result cached) |
| `src/data/sampling/historical_indices.py` | Wikipedia scraping replaced by cached file |
| `src/models/hmm/hmm_visualization.py` | Visualization separated from models |

---

# Appendix B: Config Parameter Reference

Complete list of every parameter that was hard-coded in ADA and now lives in YAML configs:

| Parameter | ADA Value | Config File | Config Key |
|-----------|-----------|-------------|------------|
| Missing data threshold | 10% | data.yaml | `universe.max_missing_pct` |
| Interpolation window | 5 days | data.yaml | `universe.interpolation_window` |
| Min ticker history | 30 days | data.yaml | `universe.min_history_days` |
| Extreme high price | $10,000 | data.yaml | `cleaning.extreme_high_price` |
| Extreme low price | $0.01 | data.yaml | `cleaning.extreme_low_price` |
| Smoothing window | 20 days | features.yaml | `indicators.smoothing_window` |
| Correlation window | 30 days | features.yaml | `indicators.correlation_window` |
| Correlation sample size | 500 | features.yaml | `indicators.correlation_sample_size` |
| Standardization window | 252 days | features.yaml | `indicators.standardization_window` |
| PCA components | 2 | features.yaml | `pca.n_components` |
| Volatility windows | [5,10,20,30] | features.yaml | `volatility.smoothing_windows` |
| HMM states | 5 (remapped to 3) | models.yaml | `hmm.n_states` |
| HMM iterations | 1000 | models.yaml | `hmm.n_iter` |
| HMM covariance type | "full" | models.yaml | `hmm.covariance_type` |
| HMM random state | 42 | models.yaml | `hmm.random_state` |
| GARCH p | 1 | models.yaml | `garch.p` |
| GARCH q | 1 | models.yaml | `garch.q` |
| GARCH distribution | "normal" → "t" | models.yaml | `garch.distribution` |
| GARCH regime quantiles | [0.33, 0.67] | models.yaml | `garch.regime_quantiles` |
| GARCH return scaling | 100 | models.yaml | `garch.return_scaling` |
| KMeans clusters | 3 | models.yaml | `kmeans.n_clusters` |
| KMeans random state | 42 | models.yaml | `kmeans.random_state` |
| Ensemble min weight | 0.10 | models.yaml | `ensemble.min_weight` |
| Ensemble max weight | 0.80 | models.yaml | `ensemble.max_weight` |
| Ensemble weight step | 0.05 | models.yaml | `ensemble.weight_step` |
| Walk-forward min train | 5 years | models.yaml | `walk_forward.min_train_years` |
| Walk-forward test window | 252 days | models.yaml | `walk_forward.test_window_days` |
| Walk-forward step | 63 days | models.yaml | `walk_forward.step_days` |
| Confirmation filter | 3 days | strategy.yaml | `signal_filters.confirmation_days` |
| Max allocation change | 0.25/day | strategy.yaml | `signal_filters.max_daily_allocation_change` |
| Vol target | 10% | strategy.yaml | `strategies.vol_targeted.vol_target` |
| Execution lag | 1 day | backtest.yaml | `execution.lag_days` |
| Transaction cost | 5 bps | backtest.yaml | `execution.cost_bps` |
| Slippage | 2 bps | backtest.yaml | `execution.slippage_bps` |
| Max leverage | 1.0 | backtest.yaml | `constraints.max_leverage` |
| Risk-free rate source | FRED DGS3MO | backtest.yaml | `risk_free_rate.series` |
| Bootstrap samples | 10,000 | evaluation.yaml | `bootstrap.n_samples` |
| Bootstrap confidence | 0.95 | evaluation.yaml | `bootstrap.confidence_level` |
| Rolling window | 252 days | evaluation.yaml | `rolling.window_days` |
| Crisis: dot-com | 2000-03 to 2002-10 | evaluation.yaml | `crisis_periods.dotcom` |
| Crisis: 2008 | 2007-10 to 2009-03 | evaluation.yaml | `crisis_periods.financial_crisis` |
| Crisis: COVID | 2020-02 to 2020-03 | evaluation.yaml | `crisis_periods.covid` |
| DPI for figures | 300 | constants.py | `FIGURE_DPI` |
| Regime colors | green/amber/red | constants.py | `REGIME_COLORS` |
| Trading days per year | 252 | constants.py | `TRADING_DAYS_PER_YEAR` |
| Bear market threshold | -20% | evaluation.yaml | (if used) |
| Correction threshold | -10% | evaluation.yaml | (if used) |

Total: **50+ parameters** externalized from source code to configuration.
