# PROJECT TRACKER

## Market Regime Modeling for Systematic Trading

---

## Project Overview

**Objective**: Build a systematic trading system that uses HMM, GARCH, and K-Means regime detection to generate allocation signals for US equities. The system must produce signals that survive realistic transaction costs and outperform static benchmarks on a risk-adjusted basis.

**Research Question**: Can statistical regime detection models produce actionable allocation signals for systematic equity trading that survive realistic transaction costs and outperform static benchmarks on a risk-adjusted basis?

---

## Current Status

| Field                    | Value                                          |
|--------------------------|-------------------------------------------------|
| Repository status        | Phase 2 complete — 55 tests passing              |
| Current phase            | Phase 3 — Regime Detection Models                |
| Last completed milestone | Phase 2 — Data Pipeline                          |
| Current task             | 3.1 — Refactor HMM Module                        |
| Next task                | 3.2 — Refactor GARCH Module                      |

---

## Master Development Roadmap

### Phase 1 — Repository Setup and Refactoring

**Goal**: Create the new repository structure, migrate reusable code from ADA, remove dead code, and establish the config-driven architecture. Reproduce original ADA results in the new structure.

**Key Deliverables**:
- Clean repository with target architecture
- All configs externalized to YAML
- Parquet storage replacing CSV
- All known bugs fixed
- Passing test suite
- Regime results reproducible in new structure

**Tasks**:
- [x] **1.1** Initialize repository (directory structure, pyproject.toml, Makefile, CLAUDE.md)
- [x] **1.2** Create configuration files (data.yaml, features.yaml, models.yaml, strategy.yaml, backtest.yaml, evaluation.yaml)
- [x] **1.3** Migrate and consolidate source code (data, features, models, visualization modules)
- [x] **1.4** Implement config system (src/utils/config.py, path resolution)
- [x] **1.5** Implement Parquet storage (src/data/storage.py)
- [x] **1.6** Fix known bugs (imports, np.random.seed, pct_change, GARCH distribution)
- [x] **1.7** Write initial tests (unit tests for migrated modules + fixtures)
- [x] **1.8** Validate migration (45 tests passing, all imports verified)

---

### Phase 2 — Data Pipeline

**Goal**: Build a robust, reproducible data pipeline that downloads, cleans, stores, and versions market data. Runnable with a single `make data` command.

**Key Deliverables**:
- `make data` downloads, cleans, and stores all data
- All outputs in Parquet format
- Universe cached and validated
- Auxiliary data (VIX, rates, bonds) available
- Data catalog auto-generated

**Tasks**:
- [x] **2.1** Implement data downloaders (yfinance, FRED, CSV loader)
- [x] **2.2** Implement cleaning pipeline (consolidated clean.py)
- [x] **2.3** Implement universe construction (cached subset loading)
- [x] **2.4** Implement storage layer (Parquet I/O + data catalog)
- [x] **2.5** Build pipeline entry point (src/data/__main__.py)
- [x] **2.6** Download auxiliary data (VIX, T-bill, TLT)

---

### Phase 3 — Regime Detection Models

**Goal**: Migrate, improve, and properly validate the three regime detection models. Add walk-forward training so no model sees future data during evaluation.

**Key Deliverables**:
- Three model classes with consistent `.fit()`, `.predict()`, `.predict_proba()` interface
- Ensemble combiner using probability-weighted voting
- Walk-forward validation producing ~15 years of out-of-sample predictions
- Model selection diagnostics per window
- All parameters from config

**Tasks**:
- [ ] **3.1** Refactor HMM module (predict_proba, state ordering, config-driven)
- [ ] **3.2** Refactor GARCH module (Student-t, expanding quantiles, predict_proba)
- [ ] **3.3** Refactor K-Means module (inverse-distance probabilities, config-driven)
- [ ] **3.4** Refactor ensemble (probability-weighted voting)
- [ ] **3.5** Implement walk-forward validation (expanding window, quarterly refit)
- [ ] **3.6** Model selection and diagnostics (BIC, silhouette, stability checks)

---

### Phase 4 — Strategy Construction

**Goal**: Build the signal generation layer that converts regime model outputs into portfolio allocation targets. This is the core new code that transforms regime detection into a trading system.

**Key Deliverables**:
- Signal generation module converting probabilities to allocations
- Confirmation filter and rate limiter
- Volatility targeting overlay
- Execution lag applied to all signals
- All strategy variants generated from single config

**Tasks**:
- [ ] **4.1** Implement regime signal module (probabilities to allocations)
- [ ] **4.2** Implement signal filters (confirmation, rate limiter, anti-whipsaw)
- [ ] **4.3** Implement volatility targeting (vol-target overlay)
- [ ] **4.4** Apply execution lag (1-day shift)
- [ ] **4.5** Generate all strategy signals (binary, proportional, vol-targeted, momentum)

---

### Phase 5 — Backtesting Framework

**Goal**: Build a vectorized backtesting engine that computes portfolio returns with realistic transaction costs. Reusable for future projects.

**Key Deliverables**:
- Vectorized backtest engine with transaction costs
- Benchmark strategies (buy-and-hold, 60/40, risk parity)
- Cost sensitivity analysis (0–20 bps sweep)
- All strategies backtested with net returns
- Turnover tracking per strategy

**Tasks**:
- [ ] **5.1** Implement backtest engine (vectorized: signals to returns)
- [ ] **5.2** Implement transaction cost model (fixed + slippage + min trade filter)
- [ ] **5.3** Implement portfolio module (constraints, leverage, turnover cap)
- [ ] **5.4** Implement benchmarks (buy-and-hold, 60/40, risk parity)
- [ ] **5.5** Run all backtests (orchestration across strategies)
- [ ] **5.6** Transaction cost sensitivity analysis (0–20 bps sweep)

---

### Phase 6 — Performance Evaluation

**Goal**: Compute comprehensive performance metrics, statistical significance tests, and regime-conditional analysis. Produce a full strategy tearsheet.

**Key Deliverables**:
- Complete metrics library (Sharpe, Sortino, Calmar, max DD, hit rate, profit factor)
- Rolling metrics (Sharpe, vol, drawdown)
- Regime-conditional performance analysis
- Bootstrap confidence intervals for Sharpe
- Performance comparison table
- HTML tearsheet generator

**Tasks**:
- [ ] **6.1** Implement core metrics (Sharpe, Sortino, Calmar, max DD, hit rate)
- [ ] **6.2** Implement rolling metrics (rolling Sharpe, vol, drawdown, beta)
- [ ] **6.3** Implement regime-conditional performance (alpha decomposition)
- [ ] **6.4** Implement statistical significance (bootstrap CI, Sharpe difference test)
- [ ] **6.5** Generate performance comparison table (master table)
- [ ] **6.6** Generate report (HTML tearsheet)

---

### Phase 7 — Visualization and Research Outputs

**Goal**: Generate all charts needed for the research website and tearsheet. Every figure must be publication-quality, clearly labeled, and interpretable by a trader in 30 seconds.

**Key Deliverables**:
- 13 publication-quality charts
- Consistent style across all plots (dark + light themes)
- Tearsheet generator combining key charts
- All figures saved as high-res PNG (300 DPI)

**Tasks**:
- [ ] **7.1** Performance visualizations (cumulative returns, drawdown, heatmap, rolling Sharpe, cost sensitivity)
- [ ] **7.2** Regime visualizations (timeline, transition matrix, distribution, conditional performance)
- [ ] **7.3** Signal visualizations (allocation over time, signal vs drawdown)
- [ ] **7.4** Diagnostic visualizations (model selection, ensemble agreement)
- [ ] **7.5** Tearsheet composition (multi-page combined figure)

---

### Phase 8 — Research Website

**Goal**: Build a clean static website presenting the research results. Deployable via GitHub Pages and linkable from a CV.

**Key Deliverables**:
- Static HTML/CSS website (dark theme, responsive, 900px max-width)
- Deployed on GitHub Pages
- All key charts and metrics visible
- Mobile-responsive

**Tasks**:
- [ ] **8.1** Design website structure (single-page scrolling layout)
- [ ] **8.2** Build HTML/CSS (dark theme, responsive, 900px max-width)
- [ ] **8.3** Build chart assets (optimized PNGs for web)
- [ ] **8.4** Write website content (trader-friendly research note)
- [ ] **8.5** Deploy (GitHub Pages)

---

## Active Work Log

<!-- Each entry records a development session. Append new entries at the bottom. -->

| Date | Task | What Was Implemented | Issues Encountered | Resolution | Next Step |
|------|------|----------------------|--------------------|------------|-----------|
| 2026-03-23 | 1.1 Initialize Repository | Restructured src/ to flat-module target architecture (33 modules). Fixed pyproject.toml build backend, added quantstats. Created regime-trading conda env. | setuptools._legacy backend didn't exist | Changed to setuptools.build_meta | 1.2 Create Configuration Files |
| 2026-03-23 | 1.2 Create Configuration Files | Created 6 YAML configs (data, features, models, strategy, backtest, evaluation) with all hard-coded parameters externalized from ADA. | None | — | 1.3 Migrate and Consolidate Source Code |
| 2026-03-23 | 1.3 Migrate Source Code | Consolidated ADA code into flat modules: data (cleaning, loaders), features (indicators, volatility, factors, transforms), models (hmm, garch, kmeans, ensemble), visualization (regimes, performance, diagnostics). Removed plotting from models, fixed np.random.seed, changed GARCH default to Student-t. | None | — | 1.4 Implement Config System |
| 2026-03-24 | 1.4-1.5 Config + Storage | Implemented config system (utils/config.py, paths.py, constants.py) and Parquet storage layer (data/storage.py). | None | — | 1.6 Fix Known Bugs |
| 2026-03-24 | 1.6 Fix Known Bugs | All bugs already fixed in task 1.3. Fixed pandas 3.0 interpolation limit edge case (limit > series length). | pandas 3.0 sliding_window_view error | Clamp interpolation limit to len(series)-1 | 1.7 Write Tests |
| 2026-03-24 | 1.7-1.8 Tests + Validation | Wrote 45 unit tests across 7 test files. All passing. Phase 1 complete. | None | — | 2.1 Implement Data Downloaders |
| 2026-03-24 | 2.1-2.6 Full Data Pipeline | Implemented yfinance/FRED/CSV downloaders, clean_pipeline wrapper, universe construction, data catalog, pipeline entry point (__main__.py), auxiliary data support (VIX, T-bill, TLT). 10 new tests (55 total). | None | — | 3.1 Refactor HMM Module |

---

## Blockers and Risks

<!-- Track technical issues encountered during development. -->

| Problem | Affected Component | Status | Notes |
|---------|--------------------|--------|-------|
| — | — | — | — |

<!-- Status options: Open / Investigating / Resolved -->

---

## Technical Decisions

<!-- Record important architecture or modeling decisions. -->

### Decision Template

> **Decision**: [What was decided]
> **Context**: [Why the decision was needed]
> **Options Considered**: [Alternatives evaluated]
> **Chosen Approach**: [What was selected]
> **Reason**: [Why this option was chosen]

---

## Ideas and Future Improvements

<!-- Record new ideas discovered during development. -->

- [ ] Alternative regime detection models (e.g., Markov-switching GARCH, regime-switching regression)
- [ ] Additional features (e.g., credit spreads, yield curve slope, put/call ratio)
- [ ] Multi-asset extension (commodities, FX, fixed income)
- [ ] Dynamic ensemble weight optimization (walk-forward weight tuning)
- [ ] Online/incremental model updates (avoid full refits)
- [ ] Alternative strategy variants (e.g., long/short, sector rotation by regime)
- [ ] Regime-dependent risk parity
- [ ] Out-of-sample extension beyond 2020
- [ ] Integration with live data feeds for paper trading

---

## Metrics Tracking

<!-- Update once backtests are running. One row per strategy variant + benchmarks. -->

| Strategy | Sharpe | CAGR | Max Drawdown | Volatility | Turnover | Breakeven Cost (bps) |
|----------|--------|------|--------------|------------|----------|----------------------|
| Binary Regime | — | — | — | — | — | — |
| Proportional Regime | — | — | — | — | — | — |
| Vol-Targeted Regime | — | — | — | — | — | — |
| Regime Momentum | — | — | — | — | — | — |
| Buy-and-Hold (benchmark) | — | — | — | — | — | — |
| 60/40 (benchmark) | — | — | — | — | — | — |
| Risk Parity (benchmark) | — | — | — | — | — | — |

---

## Project Completion Checklist

- [ ] All phases completed
- [ ] Backtests fully reproducible
- [ ] All charts generated
- [ ] Tearsheet created
- [ ] Website deployed
- [ ] README finalized
- [ ] Repository cleaned
- [ ] v1.0 tag created
