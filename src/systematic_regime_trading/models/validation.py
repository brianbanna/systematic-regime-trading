"""
Walk-forward validation for regime detection models.

Produces strictly out-of-sample predictions by training on expanding
(or rolling) windows and predicting on held-out test periods.

For 2000-2020 data with 5-year minimum training, this produces
~15 years of out-of-sample regime predictions (2005-2020).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
import logging
from datetime import timedelta

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


def walk_forward_train(
    data: pd.DataFrame,
    feature_col: str,
    model_class: Type,
    model_config: dict,
    wf_config: dict = None,
) -> pd.DataFrame:
    """
    Walk-forward validation for a single regime model.

    Process:
    1. Train on [start, start + min_train_years]
    2. Predict on [end_of_train, end_of_train + test_window]
    3. Step forward by step_days
    4. Retrain (expanding or rolling window)
    5. Predict next test window
    6. Concatenate all out-of-sample predictions

    Args:
        data: DataFrame with 'Date' column and feature columns
        feature_col: Column name(s) to use as model input
        model_class: Model class with .fit(), .predict(), .predict_proba()
        model_config: Config dict for model constructor
        wf_config: Walk-forward config. If None, loads from models.yaml

    Returns:
        DataFrame with columns:
            Date, regime_label, regime_prob_calm, regime_prob_moderate,
            regime_prob_turbulent, window_id
    """
    if wf_config is None:
        wf_config = load_config("models")["walk_forward"]

    min_train_days = wf_config["min_train_years"] * 252
    test_window_days = wf_config["test_window_days"]
    step_days = wf_config["step_days"]
    expanding = wf_config.get("expanding", True)

    # Sort by date
    data = data.sort_values("Date").reset_index(drop=True)
    dates = data["Date"].values
    n_total = len(data)

    if n_total < min_train_days + test_window_days:
        raise ValueError(
            f"Insufficient data: {n_total} rows, need at least "
            f"{min_train_days + test_window_days}"
        )

    all_predictions = []
    window_id = 0
    train_end_idx = min_train_days

    while train_end_idx + step_days <= n_total:
        test_end_idx = min(train_end_idx + test_window_days, n_total)

        if expanding:
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - min_train_days)

        train_data = data.iloc[train_start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]

        if len(test_data) == 0:
            break

        # Extract features
        train_X = _extract_features(train_data, feature_col)
        test_X = _extract_features(test_data, feature_col)

        # Fit model
        model = model_class(model_config)
        model.fit(train_X)

        # Predict on test period
        labels = model.predict(test_X)
        probs = model.predict_proba(test_X)

        # Build results
        window_results = pd.DataFrame({
            "Date": test_data["Date"].values,
            "regime_label": labels,
            "regime_prob_calm": probs[:, 0],
            "regime_prob_moderate": probs[:, 1],
            "regime_prob_turbulent": probs[:, 2],
            "window_id": window_id,
            "train_start": dates[train_start_idx],
            "train_end": dates[train_end_idx - 1],
        })

        all_predictions.append(window_results)

        logger.debug(
            f"Window {window_id}: train [{train_start_idx}:{train_end_idx}] "
            f"-> test [{train_end_idx}:{test_end_idx}] "
            f"({len(test_data)} days)"
        )

        window_id += 1
        train_end_idx += step_days

    if not all_predictions:
        raise RuntimeError("No walk-forward windows produced predictions")

    result = pd.concat(all_predictions, ignore_index=True)

    # For overlapping windows, keep the most recent prediction
    result = result.sort_values(["Date", "window_id"])
    result = result.drop_duplicates(subset="Date", keep="last")
    result = result.sort_values("Date").reset_index(drop=True)

    logger.info(
        f"Walk-forward complete: {window_id} windows, "
        f"{len(result)} out-of-sample predictions "
        f"({result['Date'].min()} to {result['Date'].max()})"
    )

    return result


def walk_forward_ensemble(
    data: pd.DataFrame,
    models: Dict[str, dict],
    ensemble_weights: dict = None,
    wf_config: dict = None,
) -> pd.DataFrame:
    """
    Walk-forward validation with ensemble of models.

    Args:
        data: DataFrame with Date and feature columns
        models: Dict mapping model_name -> {
            "class": ModelClass, "config": dict, "feature_col": str
        }
        ensemble_weights: Dict mapping model_name -> weight
        wf_config: Walk-forward config

    Returns:
        DataFrame with ensemble predictions and per-model probabilities
    """
    from systematic_regime_trading.models.ensemble import EnsembleRegimeDetector

    if wf_config is None:
        wf_config = load_config("models")["walk_forward"]

    # Run walk-forward for each model
    model_results = {}
    for name, spec in models.items():
        logger.info(f"Running walk-forward for {name}...")
        result = walk_forward_train(
            data=data,
            feature_col=spec["feature_col"],
            model_class=spec["class"],
            model_config=spec["config"],
            wf_config=wf_config,
        )
        model_results[name] = result

    # Align dates across models
    common_dates = None
    for name, result in model_results.items():
        dates_set = set(result["Date"].values)
        if common_dates is None:
            common_dates = dates_set
        else:
            common_dates = common_dates.intersection(dates_set)

    common_dates = sorted(common_dates)
    n = len(common_dates)
    logger.info(f"Common dates across all models: {n}")

    # Build probability arrays
    prob_arrays = {}
    for name, result in model_results.items():
        result_aligned = result[result["Date"].isin(common_dates)].sort_values("Date")
        probs = result_aligned[
            ["regime_prob_calm", "regime_prob_moderate", "regime_prob_turbulent"]
        ].values
        prob_arrays[name] = probs

    # Combine with ensemble
    ensemble_config = load_config("models")["ensemble"]
    ensemble = EnsembleRegimeDetector(
        weights=ensemble_weights, config=ensemble_config,
    )
    combined_probs = ensemble.combine(prob_arrays)
    ensemble_labels = np.argmax(combined_probs, axis=1)

    # Build output
    output = pd.DataFrame({
        "Date": common_dates,
        "ensemble_label": ensemble_labels,
        "ensemble_prob_calm": combined_probs[:, 0],
        "ensemble_prob_moderate": combined_probs[:, 1],
        "ensemble_prob_turbulent": combined_probs[:, 2],
    })

    # Add per-model labels
    for name, result in model_results.items():
        aligned = result[result["Date"].isin(common_dates)].sort_values("Date")
        output[f"{name}_label"] = aligned["regime_label"].values

    return output


def _extract_features(data: pd.DataFrame, feature_col) -> np.ndarray:
    """Extract feature array from DataFrame."""
    if isinstance(feature_col, str):
        if feature_col in data.columns:
            return data[feature_col].values
        raise KeyError(f"Feature column '{feature_col}' not in data")
    elif isinstance(feature_col, list):
        missing = [c for c in feature_col if c not in data.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}")
        return data[feature_col].values
    return feature_col
