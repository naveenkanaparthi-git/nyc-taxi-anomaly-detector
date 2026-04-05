"""Model trainer — generates synthetic training data and fits the Isolation Forest."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.detector.model import TaxiAnomalyDetector
from src.utils.logging_config import setup_logger
from src.utils.schema import TaxiTrip

logger = setup_logger(__name__)

# NYC bounding box (Manhattan + surrounds)
LAT_RANGE = (40.68, 40.85)
LON_RANGE = (-74.02, -73.93)


def _random_trip(inject_anomaly: bool = False) -> TaxiTrip:
    """Generate a single synthetic NYC taxi trip.

    Args:
        inject_anomaly: If True, inject an obviously anomalous trip.

    Returns:
        A TaxiTrip with plausible field values.
    """
    pickup = datetime(2024, 1, 1) + timedelta(
        days=random.randint(0, 364),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    if inject_anomaly:
        # Anomaly patterns: zero distance huge fare, or huge distance tiny fare
        anomaly_type = random.choice(["high_fare_low_dist", "low_fare_high_dist", "long_duration"])
        if anomaly_type == "high_fare_low_dist":
            distance = round(random.uniform(0.1, 0.5), 2)
            fare = round(random.uniform(200.0, 500.0), 2)
            duration_min = random.randint(5, 15)
        elif anomaly_type == "low_fare_high_dist":
            distance = round(random.uniform(40.0, 80.0), 2)
            fare = round(random.uniform(1.0, 5.0), 2)
            duration_min = random.randint(10, 20)
        else:
            distance = round(random.uniform(1.0, 3.0), 2)
            fare = round(random.uniform(10.0, 25.0), 2)
            duration_min = random.randint(180, 360)
    else:
        distance = round(random.uniform(0.5, 20.0), 2)
        fare = round(2.50 + distance * random.uniform(2.0, 4.0), 2)
        duration_min = random.randint(5, 60)

    dropoff = pickup + timedelta(minutes=duration_min)
    tip = round(fare * random.uniform(0.0, 0.25), 2)

    return TaxiTrip(
        pickup_datetime=pickup,
        dropoff_datetime=dropoff,
        pickup_longitude=round(random.uniform(*LON_RANGE), 6),
        pickup_latitude=round(random.uniform(*LAT_RANGE), 6),
        dropoff_longitude=round(random.uniform(*LON_RANGE), 6),
        dropoff_latitude=round(random.uniform(*LAT_RANGE), 6),
        passenger_count=random.randint(1, 4),
        trip_distance=distance,
        fare_amount=fare,
        tip_amount=tip,
        total_amount=round(fare + tip, 2),
        payment_type=random.choice(["cash", "card"]),
        vendor_id=random.choice(["CMT", "VTS", "DDS"]),
    )


def generate_training_data(n_normal: int = 5000, n_anomaly: int = 250) -> np.ndarray:
    """Generate a labelled synthetic dataset of taxi trips.

    Args:
        n_normal: Number of normal trips to generate.
        n_anomaly: Number of anomalous trips to inject.

    Returns:
        Feature matrix of shape (n_normal + n_anomaly, 7).
    """
    logger.info("Generating training data", extra={"n_normal": n_normal, "n_anomaly": n_anomaly})
    trips = [_random_trip(inject_anomaly=False) for _ in range(n_normal)]
    trips += [_random_trip(inject_anomaly=True) for _ in range(n_anomaly)]
    random.shuffle(trips)
    return np.array([t.to_feature_vector() for t in trips])


def generate_sample_file(path: str = "data/sample/taxi_trips_sample.json", n: int = 50) -> None:
    """Persist n synthetic trips as JSON for demo / testing purposes.

    Args:
        path: Output file path.
        n: Number of sample trips to write.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    trips = [_random_trip(inject_anomaly=(i % 10 == 0)) for i in range(n)]
    records = [t.model_dump(mode="json") for t in trips]
    with open(out, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info("Sample file written", extra={"path": path, "n": n})


if __name__ == "__main__":
    generate_sample_file()
    X = generate_training_data()
    detector = TaxiAnomalyDetector()
    detector.train(X)
    detector.save()
    logger.info("Model training pipeline complete")
