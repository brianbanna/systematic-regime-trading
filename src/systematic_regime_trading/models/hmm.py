"""
Hidden Markov Model regime detection.

Fitting, prediction, state labeling, transition analysis,
and probability extraction for Gaussian HMMs.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from typing import Tuple, Dict
import warnings


def fit_hmm(
    data: pd.Series,
    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
    covariance_type: str = "full",
) -> Tuple:
    """
    Fit a Gaussian Hidden Markov Model to volatility data.

    Args:
        data: Time series of volatility or returns
        n_states: Number of hidden states
        n_iter: Maximum EM algorithm iterations
        random_state: Random seed for reproducibility
        covariance_type: Covariance structure ('full', 'diag', 'spherical', 'tied')

    Returns:
        Tuple of (fitted_model, state_sequence)
    """
    X = data.values.reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmc",
        params="stmc",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X)

    states = model.predict(X)

    return model, states


def predict_states(model, data: pd.Series) -> np.ndarray:
    """
    Predict hidden states for new data using Viterbi algorithm.

    Args:
        model: Fitted GaussianHMM model
        data: Time series to predict states for

    Returns:
        Array of predicted states
    """
    X = data.values.reshape(-1, 1)
    return model.predict(X)


def get_transition_matrix(model) -> pd.DataFrame:
    """
    Extract transition probability matrix from fitted HMM.

    Args:
        model: Fitted GaussianHMM model

    Returns:
        DataFrame with transition probabilities (rows=from, columns=to)
    """
    trans_mat = model.transmat_
    state_labels = [f"State {i}" for i in range(model.n_components)]
    return pd.DataFrame(trans_mat, index=state_labels, columns=state_labels)


def compute_state_persistence(model) -> Dict[str, dict]:
    """
    Compute persistence probability and expected duration per state.

    Expected duration = 1 / (1 - persistence_probability).

    Args:
        model: Fitted GaussianHMM model

    Returns:
        Dict mapping state_i to {'persistence_prob', 'expected_duration'}
    """
    trans_mat = model.transmat_
    persistence_dict = {}

    for i in range(model.n_components):
        persistence = trans_mat[i, i]
        if persistence < 1.0:
            expected_duration = 1.0 / (1.0 - persistence)
        else:
            expected_duration = np.inf

        persistence_dict[f"state_{i}"] = {
            "persistence_prob": persistence,
            "expected_duration": expected_duration,
        }

    return persistence_dict


def get_state_statistics(data: pd.Series, states: np.ndarray) -> pd.DataFrame:
    """
    Compute summary statistics for each identified state.

    Args:
        data: Original time series data
        states: State assignments from HMM

    Returns:
        DataFrame with mean, std, min, max, count, percentage per state
    """
    df = pd.DataFrame({"value": data.values, "state": states})

    stats = df.groupby("state")["value"].agg(
        [
            ("mean", "mean"),
            ("std", "std"),
            ("min", "min"),
            ("max", "max"),
            ("count", "count"),
        ]
    )
    stats = stats.reset_index()
    stats["percentage"] = (stats["count"] / len(data)) * 100

    return stats


def label_states_by_volatility(
    model, states: np.ndarray, data: pd.Series
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Relabel states by ascending mean volatility.

    State 0 = lowest volatility (calm), State N = highest (turbulent).

    Args:
        model: Fitted HMM model
        states: Original state assignments
        data: Volatility data

    Returns:
        Tuple of (relabeled_states, mapping_dict, label_names)
    """
    state_means = []
    for i in range(model.n_components):
        state_mask = states == i
        mean_vol = data[state_mask].mean()
        state_means.append((i, mean_vol))

    state_means.sort(key=lambda x: x[1])

    mapping = {old: new for new, (old, _) in enumerate(state_means)}
    relabeled_states = np.array([mapping[s] for s in states])

    if model.n_components == 3:
        label_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}
    else:
        label_names = {i: f"State {i}" for i in range(model.n_components)}

    return relabeled_states, mapping, label_names


def compute_log_likelihood(model, data: pd.Series) -> float:
    """
    Compute log-likelihood of data given the model.

    Args:
        model: Fitted HMM model
        data: Time series data to evaluate

    Returns:
        Log-likelihood score (higher is better)
    """
    X = data.values.reshape(-1, 1)
    return model.score(X)


def get_emission_parameters(model) -> pd.DataFrame:
    """
    Extract emission parameters (means and covariances) for each state.

    Args:
        model: Fitted HMM model

    Returns:
        DataFrame with state, mean, std, variance columns
    """
    params = []
    for i in range(model.n_components):
        mean = model.means_[i, 0]

        if model.covariance_type == "full":
            var = model.covars_[i, 0, 0]
        elif model.covariance_type == "diag":
            var = model.covars_[i, 0]
        elif model.covariance_type == "spherical":
            var = model.covars_[i]
        elif model.covariance_type == "tied":
            var = model.covars_[0, 0]
        else:
            var = np.nan

        std = np.sqrt(var)
        params.append({"state": i, "mean": mean, "std": std, "variance": var})

    return pd.DataFrame(params)


def compute_state_probabilities(model, data: pd.Series) -> pd.DataFrame:
    """
    Compute posterior probabilities of being in each state at each time.

    Uses forward-backward algorithm for full probability distribution
    over all states (unlike predict which gives only the most likely).

    Args:
        model: Fitted HMM model
        data: Time series data

    Returns:
        DataFrame with P(State 0), P(State 1), ... columns, dates as index
    """
    X = data.values.reshape(-1, 1)
    posteriors = model.predict_proba(X)

    state_labels = [f"P(State {i})" for i in range(model.n_components)]
    return pd.DataFrame(posteriors, index=data.index, columns=state_labels)
