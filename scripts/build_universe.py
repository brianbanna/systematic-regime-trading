"""
Build the full NASDAQ universe for the regime trading system.

Downloads NASDAQ constituent tickers, validates data availability,
and saves the universe to data/cache/nasdaq_constituents.csv.

This replaces the ADA project's random subset search with a
principled approach: take all NASDAQ-listed stocks that have
sufficient history (2000-2020) and pass data quality checks.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Full NASDAQ-100 + extended large/mid-cap NASDAQ tickers
# These are stocks that were significant NASDAQ constituents during 2000-2020
NASDAQ_UNIVERSE = [
    # NASDAQ-100 core (current and historical)
    "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "NVDA",
    "AVGO", "COST", "PEP", "CSCO", "ADBE", "CMCSA", "NFLX", "INTC",
    "AMD", "TXN", "QCOM", "AMGN", "INTU", "AMAT", "ISRG", "BKNG",
    "ADI", "MDLZ", "REGN", "VRTX", "LRCX", "PYPL", "SNPS", "KLAC",
    "PANW", "CDNS", "MNST", "MELI", "NXPI", "ORLY", "FTNT", "CTAS",
    "MCHP", "KDP", "DXCM", "ADP", "PAYX", "KHC", "MRNA", "ODFL",
    "CPRT", "WDAY", "PCAR", "LULU", "FAST", "EA", "ROST", "CTSH",
    "BIIB", "VRSK", "CSGP", "BKR", "WBD", "ZS", "ANSS", "IDXX",
    "DLTR", "FANG", "ILMN", "ALGN", "EBAY", "WBA", "SIRI", "SWKS",
    "TEAM", "DDOG", "CRWD", "ZM", "DOCU",

    # Historical NASDAQ-100 members (may have been removed/acquired)
    "YHOO", "SBUX", "GILD", "CELG", "ESRX", "CA", "DISH", "TMUS",
    "MAR", "EXPE", "CHKP", "NTAP", "AKAM", "VRSN", "JBHT", "HSIC",
    "HOLX", "MXIM", "XLNX", "ALXN", "WYNN", "ULTA", "SGEN", "BMRN",
    "TTWO", "CHTR", "FOX", "FOXA", "ATVI", "MTCH",

    # Large-cap NASDAQ tech and growth
    "CRM", "SQ", "SHOP", "UBER", "LYFT", "SNAP", "PINS", "ROKU",
    "NET", "OKTA", "TWLO", "MDB", "SPLK", "VEEV", "COUP",
    "ESTC", "BILL", "HUBS", "PCTY", "PAYC", "WIX",

    # Biotech / pharma on NASDAQ
    "GILD", "REGN", "VRTX", "SGEN", "BMRN", "ALNY", "IONS",
    "NBIX", "EXEL", "TECH", "BIO", "IOVA", "RARE", "SRPT",
    "UTHR", "JAZZ", "HZNP", "MYOK", "FOLD",

    # Semiconductor
    "MU", "MRVL", "ON", "WOLF", "MPWR", "SLAB", "CREE",
    "RMBS", "DIOD", "POWI", "AMBA", "SYNA", "SITM",

    # Internet / software / SaaS
    "ZEN", "FIVN", "MIME", "QLYS", "RPD", "TENB", "SAIL",
    "API", "PLAN", "PING", "SUMO", "BIGC", "NCNO",

    # Financial services on NASDAQ
    "NDAQ", "SBNY", "SIVB", "HBAN", "FITB", "ZION", "CATY",
    "TREE", "LPLA", "IBKR", "VIRT",

    # Consumer / retail on NASDAQ
    "TSCO", "POOL", "FIVE", "ETSY", "CHWY", "ABNB",
    "BURL", "JACK", "CAKE", "SHAK", "WING", "DPZ",
    "SBUX", "LULU", "ROST", "DLTR",

    # Media / entertainment
    "NFLX", "ROKU", "EA", "TTWO", "ATVI", "ZNGA",
    "MTCH", "IAC", "TRIP", "EXPE", "BKNG",

    # Industrial / other NASDAQ
    "ODFL", "JBHT", "CHRW", "XEL", "AEP",
    "CTLT", "TECH", "WST", "TER", "KEYS",
    "ZBRA", "TRMB", "GNRC", "ENPH",

    # Additional breadth for market representation
    "VIAV", "JNPR", "FFIV", "CIEN", "CAVM", "INFN",
    "ORCL", "DELL", "HPE", "HPQ", "LSCC", "MTSI",
    "FORM", "CRUS", "SMTC", "SGH",
    "VRNT", "CALX", "LITE", "IIVI",
    "MANH", "EPAM", "GLOB", "LPSN",
    "FEYE", "CYBR", "SAIL", "PRGS",
    "MIDD", "STE", "HELE", "FOXF",
    "MGLN", "LHCG", "HALO", "IRTC",
    "PODD", "NVCR", "TNDM", "SILK",
    "PCRX", "SUPN", "CORT", "ACAD",
    "PTCT", "INSM", "RVNC", "ZGNX",
    "LGND", "IRWD", "ARWR", "SAGE",
]


def deduplicate_tickers(tickers: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for t in tickers:
        t_upper = t.upper().strip()
        if t_upper not in seen:
            seen.add(t_upper)
            result.append(t_upper)
    return result


def validate_ticker_availability(
    tickers: list,
    start: str = "2000-01-01",
    end: str = "2020-12-31",
    min_history_days: int = 1000,
    batch_size: int = 50,
) -> tuple[list, list]:
    """
    Check which tickers have sufficient data on yfinance.

    Returns:
        (valid_tickers, invalid_tickers)
    """
    valid = []
    invalid = []

    for i in tqdm(range(0, len(tickers), batch_size), desc="Validating tickers"):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, start=start, end=end,
                group_by="ticker", auto_adjust=False,
                progress=False, threads=True,
            )

            if len(batch) == 1:
                ticker = batch[0]
                if len(data.dropna(how="all")) >= min_history_days:
                    valid.append(ticker)
                else:
                    invalid.append(ticker)
            else:
                for ticker in batch:
                    try:
                        ticker_data = data[ticker].dropna(how="all")
                        if len(ticker_data) >= min_history_days:
                            valid.append(ticker)
                        else:
                            invalid.append(ticker)
                    except (KeyError, AttributeError):
                        invalid.append(ticker)
        except Exception as e:
            logger.warning(f"Batch validation failed: {e}")
            invalid.extend(batch)

        time.sleep(0.5)  # Rate limiting

    return valid, invalid


def main():
    logger.info("Building NASDAQ universe for regime trading system")

    # Step 1: Deduplicate
    tickers = deduplicate_tickers(NASDAQ_UNIVERSE)
    logger.info(f"Starting with {len(tickers)} unique tickers")

    # Step 2: Validate availability on yfinance
    logger.info("Validating ticker availability (2000-2020)...")
    valid, invalid = validate_ticker_availability(tickers)

    logger.info(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
    if invalid:
        logger.info(f"Invalid tickers (first 20): {invalid[:20]}")

    # Step 3: Save universe
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    universe_df = pd.DataFrame({"ticker": valid})
    universe_path = CACHE_DIR / "nasdaq_constituents.csv"
    universe_df.to_csv(universe_path, index=False)
    logger.info(f"Saved {len(valid)} tickers to {universe_path}")

    # Also save the invalid list for reference
    if invalid:
        invalid_df = pd.DataFrame({"ticker": invalid})
        invalid_path = CACHE_DIR / "invalid_tickers.csv"
        invalid_df.to_csv(invalid_path, index=False)

    print(f"\nUniverse built: {len(valid)} tickers saved to {universe_path}")
    print(f"Next step: run 'make data-fresh' to download full dataset")


if __name__ == "__main__":
    main()
