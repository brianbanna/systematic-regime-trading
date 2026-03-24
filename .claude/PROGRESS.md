# Project Progress Tracker

## Current Phase: Phase 2 — Data Pipeline
## Current Task: 2.1 — Implement Data Downloaders

---

## Phase 1: Repository Setup and Refactoring
- [x] **1.1** Initialize Repository (directory structure, pyproject.toml, Makefile, CLAUDE.md)
- [x] **1.2** Create Configuration Files (data.yaml, features.yaml, models.yaml, strategy.yaml, backtest.yaml, evaluation.yaml)
- [x] **1.3** Migrate and Consolidate Source Code (data, features, models, visualization modules)
- [x] **1.4** Implement Config System (src/utils/config.py, path resolution)
- [x] **1.5** Implement Parquet Storage (src/data/storage.py)
- [x] **1.6** Fix Known Bugs (imports, np.random.seed, pct_change, GARCH distribution)
- [x] **1.7** Write Initial Tests (unit tests for migrated modules + fixtures)
- [x] **1.8** Validate Migration (45 tests passing, all imports verified)

## Phase 2: Data Pipeline
- [ ] **2.1** Implement Data Downloaders (yfinance, FRED, CSV loader)
- [ ] **2.2** Implement Cleaning Pipeline (consolidated clean.py)
- [ ] **2.3** Implement Universe Construction (cached subset loading)
- [ ] **2.4** Implement Storage Layer (Parquet I/O + data catalog)
- [ ] **2.5** Build Pipeline Entry Point (src/data/__main__.py)
- [ ] **2.6** Download Auxiliary Data (VIX, T-bill, TLT)

## Phase 3: Regime Detection Models
- [ ] **3.1** Refactor HMM Module (predict_proba, state ordering, config-driven)
- [ ] **3.2** Refactor GARCH Module (Student-t, expanding quantiles, predict_proba)
- [ ] **3.3** Refactor K-Means Module (inverse-distance probabilities, config-driven)
- [ ] **3.4** Refactor Ensemble (probability-weighted voting)
- [ ] **3.5** Implement Walk-Forward Validation (expanding window, quarterly refit)
- [ ] **3.6** Model Selection and Diagnostics (BIC, silhouette, stability checks)

## Phase 4: Strategy Construction
- [ ] **4.1** Implement Regime Signal Module (probabilities to allocations)
- [ ] **4.2** Implement Signal Filters (confirmation, rate limiter, anti-whipsaw)
- [ ] **4.3** Implement Volatility Targeting (vol-target overlay)
- [ ] **4.4** Apply Execution Lag (1-day shift)
- [ ] **4.5** Generate All Strategy Signals (binary, proportional, vol-targeted, momentum)

## Phase 5: Backtesting Framework
- [ ] **5.1** Implement Backtest Engine (vectorized: signals to returns)
- [ ] **5.2** Implement Transaction Cost Model (fixed + slippage + min trade filter)
- [ ] **5.3** Implement Portfolio Module (constraints, leverage, turnover cap)
- [ ] **5.4** Implement Benchmarks (buy-and-hold, 60/40, risk parity)
- [ ] **5.5** Run All Backtests (orchestration across strategies)
- [ ] **5.6** Transaction Cost Sensitivity Analysis (0-20 bps sweep)

## Phase 6: Performance Evaluation
- [ ] **6.1** Implement Core Metrics (Sharpe, Sortino, Calmar, max DD, hit rate)
- [ ] **6.2** Implement Rolling Metrics (rolling Sharpe, vol, drawdown, beta)
- [ ] **6.3** Implement Regime-Conditional Performance (alpha decomposition)
- [ ] **6.4** Implement Statistical Significance (bootstrap CI, Sharpe difference test)
- [ ] **6.5** Generate Performance Comparison Table (master table)
- [ ] **6.6** Generate Report (HTML tearsheet)

## Phase 7: Visualization and Research Outputs
- [ ] **7.1** Performance Visualizations (cumulative returns, drawdown, heatmap, rolling Sharpe, cost sensitivity)
- [ ] **7.2** Regime Visualizations (timeline, transition matrix, distribution, conditional performance)
- [ ] **7.3** Signal Visualizations (allocation over time, signal vs drawdown)
- [ ] **7.4** Diagnostic Visualizations (model selection, ensemble agreement)
- [ ] **7.5** Tearsheet Composition (multi-page combined figure)

## Phase 8: Research Website
- [ ] **8.1** Design Website Structure (single-page scrolling layout)
- [ ] **8.2** Build HTML/CSS (dark theme, responsive, 900px max-width)
- [ ] **8.3** Build Chart Assets (optimized PNGs for web)
- [ ] **8.4** Write Website Content (trader-friendly research note)
- [ ] **8.5** Deploy (GitHub Pages)

---

## Session Log
<!-- Each session appends a brief log entry here -->

### 2026-03-23
- **Task 1.1 completed**: Restructured src/ from nested sub-packages to flat-module target architecture. Created empty placeholder modules for all 33 target files. Fixed pyproject.toml build backend and added quantstats dependency. Set up `regime-trading` conda environment with all dependencies installed.
- **Task 1.2 completed**: Created all 6 YAML config files (data, features, models, strategy, backtest, evaluation) with every hard-coded parameter from the ADA project externalized.
- **Task 1.3 completed**: Migrated and consolidated all ADA source code into flat-module target architecture. Data: 3 files -> 1 cleaning.py + loaders.py. Features: indicators split into indicators.py, volatility.py, factors.py, transforms.py. Models: HMM/GARCH/K-Means flattened, plotting removed, ensemble consolidated. Visualization: split by purpose (regimes, performance, diagnostics). Fixed np.random.seed -> default_rng, GARCH default dist -> Student-t.

### 2026-03-24
- **Tasks 1.4-1.5 completed**: Implemented config system (utils/config.py, paths.py, constants.py) and Parquet storage layer (data/storage.py).
- **Task 1.6 completed**: All known bugs (np.random.seed, pct_change fill_method, GARCH normal dist) were already fixed during migration in task 1.3. Also fixed pandas 3.0 interpolation limit edge case.
- **Tasks 1.7-1.8 completed**: Wrote 45 unit tests across 6 test files (cleaning, indicators, transforms, HMM, GARCH, config/storage). All 45 tests passing. Phase 1 complete.
