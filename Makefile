.PHONY: data features models signals backtest evaluate report website clean all test run figures run-from-scratch website-build

PKG = systematic_regime_trading

data:
	python -m $(PKG).data

data-fresh:
	python -m $(PKG).data --fresh

data-validate:
	python -m $(PKG).data --validate-only

features:
	python -m $(PKG).features.indicators

models:
	python -m $(PKG).models.validation

signals:
	python -m $(PKG).signals.regime_signal

backtest:
	python -m $(PKG).backtest.engine

evaluate:
	python -m $(PKG).evaluation.report

report:
	python -m $(PKG).visualization.tearsheet

website-build:
	mkdir -p website/assets/figures
	cp results/figures/*.png website/assets/figures/ 2>/dev/null || true

clean:
	rm -rf data/processed/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf results/tearsheets/*
	rm -rf results/backtest_results/*

run:
	python scripts/run_pipeline.py

figures:
	python scripts/generate_figures.py

run-from-scratch: data-fresh run figures website-build
	@echo "Full pipeline complete. Results in results/, website in website/"

all: data features models signals backtest evaluate report

test:
	python -m pytest tests/ -v
