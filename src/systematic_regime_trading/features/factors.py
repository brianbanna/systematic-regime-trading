"""
PCA factor computation, composite scores, and subset projection.

Handles Principal Component Analysis on market indicators, producing
trend (PC1) and systemic stress (PC2) factors.
"""

import pandas as pd
import numpy as np

from systematic_regime_trading.features.volatility import (
    compute_daily_returns_unified,
    compute_market_volatility_index_unified,
)
from systematic_regime_trading.features.indicators import (
    compute_market_direction_unified,
    compute_market_volume_unified,
    compute_market_breadth_unified,
    compute_market_atr_unified,
    compute_market_correlation_dynamic,
    compute_mood_index,
)


def compute_dynamic_range(moodindex: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate standard deviation and IQR for each indicator.

    Args:
        moodindex: Table with standardized indicators as columns

    Returns:
        DataFrame with 'std' and 'iqr' columns for each indicator
    """
    dynamic_metrics = pd.DataFrame(index=moodindex.columns)
    dynamic_metrics["std"] = moodindex.std()
    dynamic_metrics["iqr"] = moodindex.quantile(0.75) - moodindex.quantile(0.25)
    return dynamic_metrics


def compute_rolling_corr(
    moodindex: pd.DataFrame,
    volatility_col: str = "market_volatility",
    window: int = 30,
) -> pd.DataFrame:
    """
    Rolling correlation between each indicator and market volatility.

    Args:
        moodindex: DataFrame with standardized indicators
        volatility_col: Name of the volatility column
        window: Rolling window size

    Returns:
        DataFrame with rolling correlations for each indicator
    """
    rolling_corrs = pd.DataFrame(index=moodindex.index)
    for col in moodindex.columns:
        if col != volatility_col:
            rolling_corrs[col] = (
                moodindex[col].rolling(window).corr(moodindex[volatility_col])
            )
    return rolling_corrs


def compute_subset_pc_scores(
    subset_df: pd.DataFrame,
    global_pca_model,
    feature_cols: list,
    sample_size: int,
    correlation_window: int = 30,
    standardization_window: int = 252,
) -> pd.DataFrame:
    """
    Compute PC1 and PC2 for a subset of stocks using global PCA weights.

    Projects subset indicators into the global PCA space without refitting.
    Uses transform() (not fit_transform()) to preserve global weights.

    Args:
        subset_df: DataFrame of the subset (must include price/volume columns)
        global_pca_model: PCA object fitted on the full market
        feature_cols: Feature column names used in global PCA (exact order)
        sample_size: Number of stocks for correlation calculation
        correlation_window: Window for rolling correlation
        standardization_window: Window for z-score normalization

    Returns:
        DataFrame with 'market_trend_pc1' and 'systemic_stress_pc2' columns
    """
    df_feats = compute_daily_returns_unified(subset_df)

    vol_df = compute_market_volatility_index_unified(df_feats)
    vol_df = vol_df.set_index("Date")

    dir_df = compute_market_direction_unified(df_feats)
    vol_obj_df = compute_market_volume_unified(df_feats)
    brd_df = compute_market_breadth_unified(df_feats)
    atr_df = compute_market_atr_unified(df_feats)
    corr_df = compute_market_correlation_dynamic(
        df=df_feats, window=correlation_window, sample_size=sample_size,
    )

    indicators_dict = {
        "market_direction": dir_df,
        "market_volume": vol_obj_df,
        "market_volatility": vol_df,
        "market_correlation": corr_df,
        "market_breadth": brd_df,
        "market_atr": atr_df,
    }

    subset_mood = compute_mood_index(indicators_dict, window=standardization_window)

    valid_data = subset_mood[feature_cols].dropna()

    # Use transform (not fit_transform) to apply global PCA weights
    projected_scores = global_pca_model.transform(valid_data)

    subset_mood.loc[valid_data.index, "market_trend_pc1"] = projected_scores[:, 0]
    subset_mood.loc[valid_data.index, "systemic_stress_pc2"] = projected_scores[:, 1]

    return subset_mood
