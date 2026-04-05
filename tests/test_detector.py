"""Unit tests for the TaxiAnomalyDetector model wrapper."""

from __future__ import annotations

import pytest

from src.detector.model import TaxiAnomalyDetector
from src.utils.schema import AnomalyResult


class TestTaxiAnomalyDetector:
    """Tests for TaxiAnomalyDetector."""

    def test_train_and_predict_normal(self, feature_matrix, normal_trip):
        """A trained detector should classify a typical trip as normal."""
        detector = TaxiAnomalyDetector(model_path="/tmp/test_model.pkl")
        detector.train(feature_matrix)

        result = detector.predict(normal_trip)

        assert isinstance(result, AnomalyResult)
        assert result.trip_id == normal_trip.trip_id
        assert result.is_anomaly is False, "Normal trip should not be flagged as anomaly"
        assert result.anomaly_score > -0.5, "Normal trip score should not be highly negative"

    def test_train_and_predict_anomaly(self, feature_matrix, anomalous_trip):
        """A trained detector should flag a trip with absurdly high fare as anomalous."""
        detector = TaxiAnomalyDetector(model_path="/tmp/test_model_anom.pkl")
        detector.train(feature_matrix)

        result = detector.predict(anomalous_trip)

        assert isinstance(result, AnomalyResult)
        assert result.is_anomaly is True, "High-fare low-distance trip should be anomalous"

    def test_predict_raises_without_model(self, normal_trip):
        """predict() must raise RuntimeError when no model is loaded."""
        detector = TaxiAnomalyDetector(model_path="/tmp/does_not_exist_xyz.pkl")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.predict(normal_trip)

    def test_predict_batch_returns_correct_count(self, feature_matrix, trip_factory):
        """predict_batch() should return one result per input trip."""
        detector = TaxiAnomalyDetector(model_path="/tmp/test_batch.pkl")
        detector.train(feature_matrix)

        trips = [trip_factory(fare=10 + i, distance=2.0 + i * 0.1) for i in range(5)]
        results = detector.predict_batch(trips)

        assert len(results) == 5
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_feature_vector_length(self, normal_trip):
        """TaxiTrip.to_feature_vector() should always return 7 features."""
        features = normal_trip.to_feature_vector()
        assert len(features) == 7

    def test_save_and_reload(self, feature_matrix, normal_trip, tmp_path):
        """Saving and reloading a model should produce identical predictions."""
        path = str(tmp_path / "model.pkl")
        detector = TaxiAnomalyDetector(model_path=path)
        detector.train(feature_matrix)
        detector.save()

        reloaded = TaxiAnomalyDetector(model_path=path)
        result_original = detector.predict(normal_trip)
        result_reloaded = reloaded.predict(normal_trip)

        assert result_original.is_anomaly == result_reloaded.is_anomaly
        assert abs(result_original.anomaly_score - result_reloaded.anomaly_score) < 1e-9
