"""Shared pytest fixtures for the NYC Taxi Anomaly Detector test suite."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.utils.schema import TaxiTrip


def _make_trip(
    fare: float = 15.0,
    distance: float = 3.5,
    duration_min: int = 20,
    passenger_count: int = 2,
    tip: float = 3.0,
) -> TaxiTrip:
    """Helper to build a TaxiTrip with controllable key fields."""
    pickup = datetime(2024, 6, 15, 14, 30, 0)
    return TaxiTrip(
        pickup_datetime=pickup,
        dropoff_datetime=pickup + timedelta(minutes=duration_min),
        pickup_longitude=-73.985,
        pickup_latitude=40.748,
        dropoff_longitude=-73.961,
        dropoff_latitude=40.762,
        passenger_count=passenger_count,
        trip_distance=distance,
        fare_amount=fare,
        tip_amount=tip,
        total_amount=fare + tip,
        payment_type="card",
        vendor_id="CMT",
    )


@pytest.fixture
def normal_trip() -> TaxiTrip:
    """A plausible normal taxi trip."""
    return _make_trip()


@pytest.fixture
def anomalous_trip() -> TaxiTrip:
    """A trip with anomalous fare (high fare, very short distance)."""
    return _make_trip(fare=450.0, distance=0.2, duration_min=8)


@pytest.fixture
def trip_factory():
    """Factory fixture returning the _make_trip helper."""
    return _make_trip


@pytest.fixture
def feature_matrix() -> np.ndarray:
    """Small synthetic feature matrix for training tests (95% normal, 5% anomaly)."""
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=[15, 3, 1200, 2, 2, 5, 0.8], scale=[5, 1, 300, 1, 1, 2, 0.3], size=(200, 7))
    anomaly = rng.normal(loc=[400, 0.2, 500, 1, 5, 2000, 50], scale=[50, 0.1, 100, 0, 2, 200, 10], size=(10, 7))
    return np.vstack([normal, anomaly])
