"""
Project-wide constants.

Regime labels, color maps, trading calendar constants.
"""

# Trading calendar
TRADING_DAYS_PER_YEAR = 252

# Regime labels
REGIME_LABELS = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

# Color scheme for regime visualization
REGIME_COLORS = {
    0: "#2ecc71",   # green (Calm)
    1: "#f1c40f",   # yellow (Moderate)
    2: "#e74c3c",   # red (Turbulent)
}

# Extended 5-state colors (for ensemble intermediates)
REGIME_COLORS_5 = {
    0: "#2ecc71",     # green (Calm)
    0.5: "#ccff00",   # yellow-green (Calm-Moderate)
    1: "#f1c40f",     # yellow (Moderate)
    1.5: "#f39c12",   # orange (Moderate-Turbulent)
    2: "#e74c3c",     # red (Turbulent)
}

# OHLCV column names
OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]
