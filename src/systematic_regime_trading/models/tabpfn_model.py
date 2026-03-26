"""
TabPFN regime detector.

Pretrained transformer foundation model for tabular classification.
No hyperparameter tuning needed. Approximates Bayesian inference
on small datasets in a single forward pass.

TabPFN excels at exactly this problem: 1000-2000 training samples,
10-15 features, 3 classes. It captures non-linear regime boundaries
that HMM (Gaussian emissions) and GMM (Gaussian mixtures) miss.
"""

import numpy as np
import pandas as pd
import logging
import warnings

logger = logging.getLogger(__name__)


class TabPFNRegimeDetector:
    """
    TabPFN-based regime classifier.

    Uses regime labels from the ensemble (or any labeling scheme) as
    training targets. Learns to predict regime from tabular features
    in a single forward pass.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        self.n_regimes = config.get("n_regimes", 3)
        self.device = config.get("device", "cpu")

        self.classifier_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        self._available = None

    def _check_available(self):
        """Check if TabPFN is importable."""
        if self._available is None:
            try:
                from tabpfn import TabPFNClassifier
                self._available = True
            except ImportError:
                self._available = False
                logger.warning("TabPFN not installed. Run: pip install tabpfn")
        return self._available

    def fit(self, X, y=None) -> "TabPFNRegimeDetector":
        """
        Fit TabPFN classifier on labeled regime data.

        Args:
            X: Feature array or DataFrame (n_samples, n_features)
            y: Regime labels (0, 1, 2). If None, cannot train.

        Returns:
            self
        """
        if not self._check_available():
            logger.warning("TabPFN not available, model will return uniform probs")
            self.is_fitted_ = True
            return self

        if y is None:
            logger.warning("TabPFN requires labels (y). Cannot fit without labels.")
            self.is_fitted_ = True
            return self

        from tabpfn import TabPFNClassifier

        X_arr = self._to_array(X)
        y_arr = np.asarray(y, dtype=int)

        # Remove NaN rows
        valid = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr.astype(float)))
        X_clean = X_arr[valid]
        y_clean = y_arr[valid]

        if len(X_clean) < 30:
            logger.warning(f"TabPFN: only {len(X_clean)} valid samples, too few")
            self.is_fitted_ = True
            return self

        # TabPFN has a sample limit; subsample if needed
        max_train = 3000
        if len(X_clean) > max_train:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_clean), max_train, replace=False)
            X_clean = X_clean[idx]
            y_clean = y_clean[idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            try:
                self.classifier_ = TabPFNClassifier(
                    device=self.device,
                    n_estimators=4,
                )
                self.classifier_.fit(X_clean, y_clean)
                self.is_fitted_ = True

                logger.info(
                    f"TabPFN fitted: {len(X_clean)} samples, "
                    f"{X_clean.shape[1]} features, {len(np.unique(y_clean))} classes"
                )
            except Exception as e:
                logger.warning(f"TabPFN fit failed: {e}")
                self.classifier_ = None
                self.is_fitted_ = True

        return self

    def predict(self, X) -> np.ndarray:
        """Hard regime labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """
        Calibrated regime probabilities.

        Returns:
            Array of shape (n_samples, n_regimes)
        """
        X_arr = self._to_array(X)
        n = len(X_arr)

        if not self.is_fitted_ or self.classifier_ is None:
            return np.full((n, self.n_regimes), 1.0 / self.n_regimes)

        try:
            # Handle NaN in test data
            valid = ~np.isnan(X_arr).any(axis=1)
            probs = np.full((n, self.n_regimes), 1.0 / self.n_regimes)

            if valid.sum() > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    pred_probs = self.classifier_.predict_proba(X_arr[valid])

                # TabPFN might return fewer columns if not all classes seen
                if pred_probs.shape[1] < self.n_regimes:
                    padded = np.full((pred_probs.shape[0], self.n_regimes), 0.01)
                    padded[:, :pred_probs.shape[1]] = pred_probs
                    padded /= padded.sum(axis=1, keepdims=True)
                    pred_probs = padded

                probs[valid] = pred_probs[:, :self.n_regimes]

            return probs

        except Exception as e:
            logger.warning(f"TabPFN predict failed: {e}")
            return np.full((n, self.n_regimes), 1.0 / self.n_regimes)

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        if isinstance(X, pd.Series):
            return X.values.reshape(-1, 1).astype(np.float32)
        return np.asarray(X, dtype=np.float32)
