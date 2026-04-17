"""
Project-wide constants.

Regime labels, color maps, trading calendar constants.
"""

# Trading calendar
TRADING_DAYS_PER_YEAR = 252

# Regime labels
REGIME_LABELS = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

# Editorial palette for regime visualization (matched to brianbanna.com)
REGIME_COLORS = {
    0: "#8ca891",   # sage (Calm)
    1: "#c5b58c",   # warm tan (Moderate)
    2: "#b87c6c",   # terracotta (Turbulent)
}

# Extended 5-state colors (interpolated)
REGIME_COLORS_5 = {
    0:   "#8ca891",   # sage (Calm)
    0.5: "#a8b08e",   # sage-tan blend
    1:   "#c5b58c",   # warm tan (Moderate)
    1.5: "#c19a7c",   # tan-terracotta blend
    2:   "#b87c6c",   # terracotta (Turbulent)
}

# OHLCV column names
OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]
