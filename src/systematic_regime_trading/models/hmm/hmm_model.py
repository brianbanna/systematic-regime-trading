"""
Functions for Hidden Markov Model regime detection and analysis.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


def fit_hmm(
    data: pd.Series,
    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
    covariance_type: str = "full",
) -> Tuple:
    """
    Fit a Gaussian Hidden Markov Model to volatility data.

    HMM assumes the market operates in discrete hidden states with transition
    probabilities between them. Uses Expectation-Maximization to learn:
    - Emission parameters: mean and variance of volatility in each state
    - Transition matrix: probabilities of switching between states
    - Initial state distribution: starting probabilities

    Args:
        data: Time series of volatility or returns
        n_states: Number of hidden states (default: 3 for calm/moderate/turbulent)
        n_iter: Maximum EM algorithm iterations (default: 1000)
        random_state: Random seed for reproducibility (default: 42)
        covariance_type: Covariance structure ('full', 'diag', 'spherical', 'tied')

    Returns:
        Tuple of (fitted_model, state_sequence)
        - fitted_model: Trained GaussianHMM object with learned parameters
        - state_sequence: Most likely state at each time point (Viterbi path)
    """
    # Reshape data for hmmlearn (requires 2D array: n_samples x n_features)
    X = data.values.reshape(-1, 1)

    # Initialize HMM with Gaussian emissions
    # init_params='stmc': Initialize start probs, transition, means, covariances
    # params='stmc': Update all parameters during training
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmc",
        params="stmc",
    )

    # Fit model using Baum-Welch (EM) algorithm
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X)

    # Decode most likely state sequence using Viterbi algorithm
    # This gives the single best path through states given the observations
    states = model.predict(X)

    return model, states


def predict_states(model, data: pd.Series) -> np.ndarray:
    """
    Predict hidden states for new data using fitted HMM.

    Uses the Viterbi algorithm to find the most likely state sequence
    given the learned model parameters and new observations.

    Args:
        model: Fitted GaussianHMM model
        data: Time series to predict states for

    Returns:
        Array of predicted states (integers 0 to n_states-1)
    """
    # Reshape for hmmlearn format
    X = data.values.reshape(-1, 1)

    # Apply Viterbi decoding
    states = model.predict(X)

    return states


def get_transition_matrix(model) -> pd.DataFrame:
    """
    Extract transition probability matrix from fitted HMM.

    Element (i,j) represents P(state_t = j | state_{t-1} = i):
    - Each row sums to 1 (valid probability distribution)
    - Diagonal elements show state persistence (self-transition probability)
    - Off-diagonal elements show regime switching probabilities

    Args:
        model: Fitted GaussianHMM model

    Returns:
        DataFrame with transition probabilities (rows=from, columns=to)
    """
    # Extract learned transition matrix
    trans_mat = model.transmat_

    # Create labeled DataFrame for readability
    state_labels = [f"State {i}" for i in range(model.n_components)]
    trans_df = pd.DataFrame(trans_mat, index=state_labels, columns=state_labels)

    return trans_df


def compute_state_persistence(model) -> Dict[str, float]:
    """
    Compute persistence metrics for each state.

    Persistence measures how long the market tends to stay in each regime:
    - Persistence probability: P(stay in state | currently in state)
    - Expected duration: Average number of periods before switching out

    Formula: Expected duration = 1 / (1 - persistence_probability)
    Example: If persistence = 0.9, expected duration = 10 periods

    Args:
        model: Fitted GaussianHMM model

    Returns:
        Dict mapping state_i to {'persistence_prob', 'expected_duration'}
    """
    trans_mat = model.transmat_
    n_states = model.n_components

    persistence_dict = {}

    for i in range(n_states):
        # Diagonal element = probability of staying in same state
        persistence = trans_mat[i, i]

        # Compute expected duration using geometric distribution formula
        # If persistence = 1.0, state is absorbing (infinite duration)
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

    Characterizes each regime by its volatility distribution and frequency.
    Useful for understanding what each state represents (calm vs turbulent).

    Args:
        data: Original time series data (volatility or returns)
        states: State assignments from HMM (integers 0 to n_states-1)

    Returns:
        DataFrame with columns: state, mean, std, min, max, count, percentage
    """
    # Combine data and states for grouping
    df = pd.DataFrame({"value": data.values, "state": states})

    # Compute statistics per state
    # mean/std: characterize volatility level and spread
    # min/max: show range of values in each regime
    # count: number of periods in each state
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

    # Add percentage of time spent in each state
    stats["percentage"] = (stats["count"] / len(data)) * 100

    return stats


def label_states_by_volatility(
    model, states: np.ndarray, data: pd.Series
) -> Tuple[np.ndarray, Dict]:
    """
    Relabel states based on mean volatility level.

    HMM assigns arbitrary state numbers (0, 1, 2, ...). This function
    reorders them by volatility for interpretability:
    - State 0 = lowest volatility (calm market)
    - State 1 = medium volatility (moderate market)
    - State 2 = highest volatility (turbulent market)

    Args:
        model: Fitted HMM model
        states: Original state assignments (arbitrary numbering)
        data: Volatility data used to compute state means

    Returns:
        Tuple of (relabeled_states, mapping_dict, label_names)
        - relabeled_states: States ordered by volatility (0=low, 1=mid, 2=high)
        - mapping_dict: Mapping from old to new state labels
        - label_names: Descriptive names for each state
    """
    # Compute mean volatility for each state
    state_means = []
    for i in range(model.n_components):
        state_mask = states == i
        mean_vol = data[state_mask].mean()
        state_means.append((i, mean_vol))

    # Sort states by mean volatility (ascending)
    state_means.sort(key=lambda x: x[1])

    # Create mapping: old_state_number -> new_state_number
    # Example: if state 2 has lowest vol, mapping[2] = 0
    mapping = {old: new for new, (old, _) in enumerate(state_means)}

    # Apply mapping to relabel all state assignments
    relabeled_states = np.array([mapping[s] for s in states])

    # Create descriptive labels based on number of states
    if model.n_components == 3:
        label_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}
    else:
        label_names = {i: f"State {i}" for i in range(model.n_components)}

    return relabeled_states, mapping, label_names


def compute_log_likelihood(model, data: pd.Series) -> float:
    """
    Compute log-likelihood of data given the model.

    Measures how well the model explains the observed data. Higher values
    indicate better fit. Useful for model selection and comparison.

    Args:
        model: Fitted HMM model
        data: Time series data to evaluate

    Returns:
        Log-likelihood score (higher is better)
    """
    # Reshape for hmmlearn format
    X = data.values.reshape(-1, 1)

    # Compute log P(observations | model) using forward algorithm
    return model.score(X)


def get_emission_parameters(model) -> pd.DataFrame:
    """
    Extract emission parameters (means and covariances) for each state.

    Each state has a Gaussian distribution over observations. This function
    extracts the learned mean and variance for each state's distribution.

    Args:
        model: Fitted HMM model

    Returns:
        DataFrame with columns: state, mean, std, variance
    """
    n_states = model.n_components

    params = []
    for i in range(n_states):
        # Extract mean for this state
        mean = model.means_[i, 0]

        # Extract variance based on covariance type
        # Different covariance types store variance in different formats
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

        # Compute standard deviation from variance
        std = np.sqrt(var)

        params.append({"state": i, "mean": mean, "std": std, "variance": var})

    return pd.DataFrame(params)


def compute_state_probabilities(model, data: pd.Series) -> pd.DataFrame:
    """
    Compute posterior probabilities of being in each state at each time.

    Unlike predict() which gives the single most likely state, this returns
    the full probability distribution over all states at each time point.
    Useful for measuring regime uncertainty and transition periods.

    Args:
        model: Fitted HMM model
        data: Time series data

    Returns:
        DataFrame with columns P(State 0), P(State 1), ... and dates as index
        Each row sums to 1 (valid probability distribution)
    """
    # Reshape for hmmlearn format
    X = data.values.reshape(-1, 1)

    # Compute posterior probabilities using forward-backward algorithm
    # This considers all possible state sequences, not just the most likely one
    posteriors = model.predict_proba(X)

    # Create DataFrame with dates as index for time series analysis
    state_labels = [f"P(State {i})" for i in range(model.n_components)]
    prob_df = pd.DataFrame(posteriors, index=data.index, columns=state_labels)

    return prob_df
