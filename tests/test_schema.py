"""Unit tests for TaxiTrip and AnomalyResult Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from src.utils.schema import AnomalyResult, TaxiTrip


class TestTaxiTripSchema:
    """Tests for TaxiTrip validation and computed properties."""

    def test_valid_trip_construction(self, normal_trip):
        """A fully specified trip should construct without errors."""
        assert normal_trip.trip_id.startswith("TXN-")
        assert normal_trip.fare_amount == 15.0
        assert normal_trip.trip_distance == 3.5

    def test_duration_seconds_computed(self, normal_trip):
        """duration_seconds should equal dropoff minus pickup in seconds."""
        expected = 20 * 60  # 20 minutes
        assert normal_trip.duration_seconds == expected

    def test_invalid_passenger_count_raises(self):
        """Passenger count outside [1, 9] should raise ValidationError."""
        pickup = datetime(2024, 1, 1, 10, 0)
        with pytest.raises(ValidationError):
            TaxiTrip(
                pickup_datetime=pickup,
                dropoff_datetime=pickup + timedelta(minutes=10),
                pickup_longitude=-73.98,
                pickup_latitude=40.75,
                dropoff_longitude=-73.96,
                dropoff_latitude=40.76,
                passenger_count=10,  # invalid
                trip_distance=2.0,
                fare_amount=12.0,
                tip_amount=2.0,
                total_amount=14.0,
                payment_type="card",
                vendor_id="CMT",
            )

    def test_negative_distance_raises(self):
        """Negative trip_distance should raise ValidationError."""
        pickup = datetime(2024, 1, 1, 10, 0)
        with pytest.raises(ValidationError):
            TaxiTrip(
                pickup_datetime=pickup,
                dropoff_datetime=pickup + timedelta(minutes=10),
                pickup_longitude=-73.98,
                pickup_latitude=40.75,
                dropoff_longitude=-73.96,
                dropoff_latitude=40.76,
                passenger_count=1,
                trip_distance=-1.0,  # invalid
                fare_amount=8.0,
                tip_amount=1.0,
                total_amount=9.0,
                payment_type="cash",
                vendor_id="VTS",
            )

    def test_feature_vector_contains_expected_values(self, normal_trip):
        """Feature vector should include fare, distance, duration, passengers, tip."""
        fv = normal_trip.to_feature_vector()
        assert fv[0] == pytest.approx(15.0)   # fare_amount
        assert fv[1] == pytest.approx(3.5)    # trip_distance
        assert fv[2] == pytest.approx(1200.0) # duration_seconds (20 min)
        assert fv[3] == pytest.approx(2.0)    # passenger_count
        assert fv[4] == pytest.approx(3.0)    # tip_amount

    def test_fare_per_mile_in_feature_vector(self, normal_trip):
        """fare_per_mile feature should equal fare / distance."""
        fv = normal_trip.to_feature_vector()
        expected_fare_per_mile = 15.0 / 3.5
        assert fv[5] == pytest.approx(expected_fare_per_mile, rel=1e-3)


class TestAnomalyResult:
    """Tests for AnomalyResult schema."""

    def test_anomaly_result_construction(self, normal_trip):
        """AnomalyResult should store all required fields."""
        result = AnomalyResult(
            trip_id=normal_trip.trip_id,
            is_anomaly=False,
            anomaly_score=-0.12,
            features=normal_trip.to_feature_vector(),
        )
        assert result.is_anomaly is False
        assert result.anomaly_score == pytest.approx(-0.12)
        assert len(result.features) == 7
        assert result.processed_at is not None
