"""
Fama-French factor regression.

Regresses strategy excess returns on standard risk factors to determine
whether alpha survives after controlling for known risk premia.

If alpha (intercept) is significant, the strategy captures something
beyond market, size, value, momentum, and profitability exposures.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from io import StringIO

logger = logging.getLogger(__name__)


def download_ff_factors(start: str = "2000-01-01", end: str = "2020-12-31") -> pd.DataFrame:
    """
    Download Fama-French 5 factors + Momentum from Kenneth French's website.

    Returns:
        DataFrame with daily factor returns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    """
    try:
        import urllib.request

        # FF 5 factors (daily)
        ff5_url = (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        )
        # Momentum factor (daily)
        mom_url = (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Momentum_Factor_daily_CSV.zip"
        )

        ff5 = _download_ff_csv(ff5_url, "F-F_Research_Data_5_Factors_2x3_daily.CSV")
        mom = _download_ff_csv(mom_url, "F-F_Momentum_Factor_daily.CSV")

        # Merge
        factors = ff5.join(mom, how="inner")

        # Filter date range
        factors = factors.loc[start:end]

        # Convert from percent to decimal
        factors = factors / 100

        logger.info(f"Downloaded FF factors: {len(factors)} days, {list(factors.columns)}")
        return factors

    except Exception as e:
        logger.warning(f"Could not download FF factors: {e}")
        logger.info("Using synthetic factors as fallback (NOT for production)")
        return _synthetic_ff_factors(start, end)


def _download_ff_csv(url: str, inner_filename: str) -> pd.DataFrame:
    """Download and parse a Fama-French CSV from a zip file."""
    import zipfile
    import io
    import urllib.request

    response = urllib.request.urlopen(url)
    zip_data = io.BytesIO(response.read())

    with zipfile.ZipFile(zip_data) as zf:
        # Find the CSV file
        csv_name = None
        for name in zf.namelist():
            if name.endswith(".CSV") or name.endswith(".csv"):
                csv_name = name
                break
        if csv_name is None:
            raise ValueError(f"No CSV found in {url}")

        with zf.open(csv_name) as f:
            content = f.read().decode("utf-8")

    # Parse: skip header rows until we find numeric data
    lines = content.strip().split("\n")
    data_start = None
    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                int(parts[0].strip())
                data_start = i
                break
            except ValueError:
                continue

    if data_start is None:
        raise ValueError("Could not find data start in FF CSV")

    # Read header (one line before data)
    header_line = lines[data_start - 1].strip().split(",")
    header = [h.strip() for h in header_line]

    # Read data until empty line or non-numeric
    data_lines = []
    for line in lines[data_start:]:
        parts = line.strip().split(",")
        if len(parts) < 2:
            break
        try:
            int(parts[0].strip())
            data_lines.append(parts)
        except ValueError:
            break

    df = pd.DataFrame(data_lines, columns=header[:len(data_lines[0])])

    # Parse date column
    date_col = df.columns[0]
    df[date_col] = df[date_col].str.strip()
    df.index = pd.to_datetime(df[date_col], format="%Y%m%d")
    df = df.drop(columns=[date_col])

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

    return df


def _synthetic_ff_factors(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic FF factors as fallback (for testing only)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    return pd.DataFrame({
        "Mkt-RF": rng.normal(0.0003, 0.01, n),
        "SMB": rng.normal(0.0001, 0.003, n),
        "HML": rng.normal(0.0001, 0.003, n),
        "RMW": rng.normal(0.0001, 0.002, n),
        "CMA": rng.normal(0.0001, 0.002, n),
        "Mom": rng.normal(0.0002, 0.005, n),
        "RF": np.full(n, 0.02 / 252),
    }, index=dates)


def fama_french_regression(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    risk_free_col: str = "RF",
) -> dict:
    """
    Regress strategy excess returns on Fama-French factors.

    Model: R_strategy - RF = alpha + beta_MKT*(Mkt-RF) + beta_SMB*SMB
           + beta_HML*HML + beta_RMW*RMW + beta_CMA*CMA + beta_MOM*Mom + epsilon

    Args:
        strategy_returns: Daily strategy net returns
        factors: DataFrame with FF factor columns and RF
        risk_free_col: Name of risk-free rate column

    Returns:
        Dict with alpha, alpha_tstat, alpha_pvalue, r_squared,
        factor_betas, factor_tstats, factor_pvalues
    """
    import statsmodels.api as sm

    # Align dates
    common = strategy_returns.index.intersection(factors.index)
    if len(common) < 60:
        logger.warning(f"Only {len(common)} common dates for factor regression")
        return {"alpha": np.nan, "alpha_tstat": np.nan, "r_squared": np.nan}

    ret = strategy_returns.loc[common]
    ff = factors.loc[common]

    # Excess return
    if risk_free_col in ff.columns:
        rf = ff[risk_free_col]
        excess_ret = ret - rf
    else:
        excess_ret = ret

    # Factor columns
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
                   if c in ff.columns]
    X = ff[factor_cols]
    X = sm.add_constant(X)

    # OLS with Newey-West HAC standard errors
    model = sm.OLS(excess_ret, X, missing="drop")
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 10})

    alpha = results.params.get("const", np.nan)
    alpha_annual = alpha * 252

    output = {
        "alpha_daily": alpha,
        "alpha_annual": alpha_annual,
        "alpha_tstat": results.tvalues.get("const", np.nan),
        "alpha_pvalue": results.pvalues.get("const", np.nan),
        "r_squared": results.rsquared,
        "r_squared_adj": results.rsquared_adj,
        "n_obs": int(results.nobs),
    }

    # Factor exposures
    for factor in factor_cols:
        output[f"beta_{factor}"] = results.params.get(factor, np.nan)
        output[f"tstat_{factor}"] = results.tvalues.get(factor, np.nan)
        output[f"pvalue_{factor}"] = results.pvalues.get(factor, np.nan)

    return output


def run_factor_regressions(
    backtest_results: dict,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run factor regression for all strategies.

    Args:
        backtest_results: Dict mapping strategy_name -> backtest DataFrame
        factors: Fama-French factor DataFrame

    Returns:
        DataFrame with one row per strategy, columns for alpha, betas, stats
    """
    rows = []
    for name, bt in backtest_results.items():
        logger.info(f"Factor regression: {name}")
        result = fama_french_regression(bt["net_return"], factors)
        result["strategy"] = name
        rows.append(result)

    df = pd.DataFrame(rows).set_index("strategy")
    return df
