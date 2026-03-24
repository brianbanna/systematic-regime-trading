"""
Hidden Markov Model regime detection.

Fits N states via Gaussian HMM, remaps to 3 trading regimes
(calm/moderate/turbulent). Extracts state probabilities for
proportional allocation strategies.

Key improvements over ADA:
- Class-based interface with .fit(), .predict(), .predict_proba()
- State ordering by emission mean after every fit (fixes relabeling)
- Config-driven parameters from models.yaml
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from typing import Tuple, Dict, Optional
import warnings
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    Gaussian HMM for regime detection.
    Fits N states, remaps to 3 trading regimes (calm/moderate/turbulent).
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config("models")["hmm"]

        self.n_states = config["n_states"]
        self.n_iter = config.get("n_iter", 1000)
        self.covariance_type = config.get("covariance_type", "full")
        self.random_state = config.get("random_state", 42)
        self.remap_config = config.get("remap_to_3", None)

        self.model_ = None
        self.state_mapping_ = None
        self.label_names_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray) -> "HMMRegimeDetector":
        """
        Fit HMM on training data.

        Args:
            X: 1D array or Series of volatility/feature values

        Returns:
            self
        """
        X_2d = self._to_2d(X)

        self.model_ = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",
            params="stmc",
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model_.fit(X_2d)

        # Sort states by emission mean ascending (calm -> turbulent)
        self._order_states_by_mean()
        self.is_fitted_ = True

        logger.info(
            f"HMM fitted: {self.n_states} states, "
            f"converged={self.model_.monitor_.converged}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hard regime labels (0=calm, 1=moderate, 2=turbulent).

        If n_states > 3, remaps to 3 regimes using remap_to_3 config.
        """
        self._check_fitted()
        X_2d = self._to_2d(X)
        raw_states = self.model_.predict(X_2d)

        if self.n_states <= 3:
            return raw_states

        return self._remap_to_3(raw_states)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Regime probabilities [P(calm), P(moderate), P(turbulent)].

        Uses forward-backward algorithm for posterior state probabilities.
        If n_states > 3, aggregates probabilities per remap config.

        Returns:
            Array of shape (n_samples, 3) with regime probabilities
        """
        self._check_fitted()
        X_2d = self._to_2d(X)
        raw_probs = self.model_.predict_proba(X_2d)

        if self.n_states <= 3:
            return raw_probs

        return self._aggregate_probs_to_3(raw_probs)

    def get_transition_matrix(self) -> np.ndarray:
        """3x3 (or NxN) transition probability matrix."""
        self._check_fitted()
        return self.model_.transmat_.copy()

    def get_state_persistence(self) -> dict:
        """Expected duration per state from transition matrix diagonal."""
        self._check_fitted()
        trans = self.model_.transmat_
        result = {}
        for i in range(self.n_states):
            p = trans[i, i]
            duration = 1.0 / (1.0 - p) if p < 1.0 else np.inf
            result[i] = {"persistence_prob": p, "expected_duration": duration}
        return result

    def get_emission_params(self) -> pd.DataFrame:
        """Emission means and standard deviations per state."""
        self._check_fitted()
        params = []
        for i in range(self.n_states):
            mean = self.model_.means_[i, 0]
            if self.covariance_type == "full":
                var = self.model_.covars_[i, 0, 0]
            elif self.covariance_type == "diag":
                var = self.model_.covars_[i, 0]
            elif self.covariance_type == "spherical":
                var = self.model_.covars_[i]
            else:
                var = self.model_.covars_[0, 0]
            params.append({"state": i, "mean": mean, "std": np.sqrt(var)})
        return pd.DataFrame(params)

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of data given the model."""
        self._check_fitted()
        return self.model_.score(self._to_2d(X))

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion (lower is better)."""
        self._check_fitted()
        X_2d = self._to_2d(X)
        n = X_2d.shape[0]
        ll = self.model_.score(X_2d)
        k = self._count_params()
        return -2 * ll * n + k * np.log(n)

    # --- Private helpers ---

    def _to_2d(self, X) -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def _order_states_by_mean(self):
        """Reorder model states so means are ascending (calm -> turbulent)."""
        means = self.model_.means_.flatten()
        order = np.argsort(means)

        if np.array_equal(order, np.arange(self.n_states)):
            return  # Already sorted

        self.model_.means_ = self.model_.means_[order]
        self.model_.startprob_ = self.model_.startprob_[order]
        self.model_.transmat_ = self.model_.transmat_[order][:, order]

        # Reorder covars — handle different shapes per covariance_type
        raw_covars = self.model_.covars_
        reordered = raw_covars[order]

        if self.covariance_type == "diag":
            # hmmlearn stores diag as 3D internally but validates as 2D
            n_dim = self.model_.means_.shape[1]
            reordered = reordered.reshape(self.n_states, n_dim)

        self.model_.covars_ = reordered

    def _remap_to_3(self, states: np.ndarray) -> np.ndarray:
        """Remap N-state labels to 3 regimes using config mapping."""
        if self.remap_config is None:
            raise ValueError("remap_to_3 config required for n_states > 3")

        mapping = {}
        for regime_idx, (regime_name, state_list) in enumerate(
            self.remap_config.items()
        ):
            for s in state_list:
                mapping[s] = regime_idx

        return np.array([mapping.get(s, 1) for s in states])

    def _aggregate_probs_to_3(self, probs: np.ndarray) -> np.ndarray:
        """Sum probabilities of states that map to same regime."""
        if self.remap_config is None:
            raise ValueError("remap_to_3 config required for n_states > 3")

        n_samples = probs.shape[0]
        result = np.zeros((n_samples, 3))

        for regime_idx, (regime_name, state_list) in enumerate(
            self.remap_config.items()
        ):
            for s in state_list:
                result[:, regime_idx] += probs[:, s]

        return result

    def _count_params(self) -> int:
        """Count free parameters for BIC."""
        k = self.n_states  # means
        if self.covariance_type == "full":
            k += self.n_states  # variances (1D)
        elif self.covariance_type == "diag":
            k += self.n_states
        else:
            k += 1
        k += self.n_states * (self.n_states - 1)  # transition matrix
        k += self.n_states - 1  # start probs
        return k


# --- Convenience functions (backwards compatible) ---

def fit_hmm(
    data: pd.Series,
    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
    covariance_type: str = "full",
) -> Tuple:
    """Fit HMM and return (model, states). Backwards-compatible wrapper."""
    config = {
        "n_states": n_states, "n_iter": n_iter,
        "random_state": random_state, "covariance_type": covariance_type,
    }
    detector = HMMRegimeDetector(config)
    detector.fit(data)
    states = detector.predict(data)
    return detector.model_, states


def predict_states(model, data: pd.Series) -> np.ndarray:
    X = data.values.reshape(-1, 1)
    return model.predict(X)


def get_transition_matrix(model) -> pd.DataFrame:
    trans_mat = model.transmat_
    labels = [f"State {i}" for i in range(model.n_components)]
    return pd.DataFrame(trans_mat, index=labels, columns=labels)


def compute_state_persistence(model) -> Dict[str, dict]:
    trans_mat = model.transmat_
    result = {}
    for i in range(model.n_components):
        p = trans_mat[i, i]
        dur = 1.0 / (1.0 - p) if p < 1.0 else np.inf
        result[f"state_{i}"] = {"persistence_prob": p, "expected_duration": dur}
    return result


def get_state_statistics(data: pd.Series, states: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"value": data.values, "state": states})
    stats = df.groupby("state")["value"].agg(
        [("mean", "mean"), ("std", "std"), ("min", "min"),
         ("max", "max"), ("count", "count")]
    ).reset_index()
    stats["percentage"] = (stats["count"] / len(data)) * 100
    return stats


def label_states_by_volatility(
    model, states: np.ndarray, data: pd.Series,
) -> Tuple[np.ndarray, Dict, Dict]:
    state_means = []
    for i in range(model.n_components):
        mask = states == i
        state_means.append((i, data[mask].mean()))
    state_means.sort(key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(state_means)}
    relabeled = np.array([mapping[s] for s in states])
    if model.n_components == 3:
        names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}
    else:
        names = {i: f"State {i}" for i in range(model.n_components)}
    return relabeled, mapping, names


def compute_log_likelihood(model, data: pd.Series) -> float:
    return model.score(data.values.reshape(-1, 1))


def get_emission_parameters(model) -> pd.DataFrame:
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
        params.append({"state": i, "mean": mean, "std": np.sqrt(var), "variance": var})
    return pd.DataFrame(params)


def compute_state_probabilities(model, data: pd.Series) -> pd.DataFrame:
    X = data.values.reshape(-1, 1)
    posteriors = model.predict_proba(X)
    labels = [f"P(State {i})" for i in range(model.n_components)]
    return pd.DataFrame(posteriors, index=data.index, columns=labels)
