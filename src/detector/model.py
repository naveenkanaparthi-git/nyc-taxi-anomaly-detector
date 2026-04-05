"""Isolation Forest anomaly detection wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.config import get_settings
from src.utils.logging_config import setup_logger
from src.utils.schema import AnomalyResult, TaxiTrip

logger = setup_logger(__name__)


class TaxiAnomalyDetector:
    """Wraps scikit-learn IsolationForest for NYC taxi trip anomaly detection.

    The model scores each trip based on 7 engineered features:
        fare_amount, trip_distance, duration_seconds, passenger_count,
        tip_amount, fare_per_mile, fare_per_minute

    Scores < 0 indicate anomalies; the decision threshold is set by
    the contamination parameter at training time.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialise detector, loading a pre-trained model if available.

        Args:
            model_path: Path to a serialised IsolationForest pickle file.
                        Falls back to settings.model_path if None.
        """
        self._settings = get_settings()
        self._model_path = Path(model_path or self._settings.model_path)
        self._model: Optional[IsolationForest] = None

        if self._model_path.exists():
            self._load_model()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, trip: TaxiTrip) -> AnomalyResult:
        """Score a single taxi trip for anomalousness.

        Args:
            trip: A validated TaxiTrip instance.

        Returns:
            AnomalyResult with is_anomaly flag and raw score.

        Raises:
            RuntimeError: If the model has not been trained or loaded yet.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call train() or ensure model file exists at "
                f"{self._model_path}"
            )

        features = trip.to_feature_vector()
        X = np.array(features).reshape(1, -1)
        prediction = self._model.predict(X)[0]   # 1 = normal, -1 = anomaly
        score = float(self._model.score_samples(X)[0])

        return AnomalyResult(
            trip_id=trip.trip_id,
            is_anomaly=(prediction == -1),
            anomaly_score=score,
            features=features,
        )

    def predict_batch(self, trips: list[TaxiTrip]) -> list[AnomalyResult]:
        """Score a batch of trips efficiently using vectorised inference.

        Args:
            trips: List of TaxiTrip instances.

        Returns:
            Corresponding list of AnomalyResult objects.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        feature_matrix = np.array([t.to_feature_vector() for t in trips])
        predictions = self._model.predict(feature_matrix)
        scores = self._model.score_samples(feature_matrix)

        return [
            AnomalyResult(
                trip_id=trip.trip_id,
                is_anomaly=(pred == -1),
                anomaly_score=float(score),
                features=trip.to_feature_vector(),
            )
            for trip, pred, score in zip(trips, predictions, scores)
        ]

    def train(self, X: np.ndarray) -> None:
        """Fit the Isolation Forest on the provided feature matrix.

        Args:
            X: numpy array of shape (n_samples, 7) — feature vectors.
        """
        logger.info("Training Isolation Forest", extra={"n_samples": len(X)})
        self._model = IsolationForest(
            n_estimators=self._settings.model_n_estimators,
            contamination=self._settings.anomaly_contamination,
            random_state=self._settings.model_random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        logger.info("Training complete")

    def save(self) -> None:
        """Serialise the trained model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("Model saved", extra={"path": str(self._model_path)})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Deserialise the IsolationForest model from disk."""
        with open(self._model_path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("Model loaded", extra={"path": str(self._model_path)})
