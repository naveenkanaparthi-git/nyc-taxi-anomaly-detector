"""Pydantic schema for NYC taxi trip events."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class TaxiTrip(BaseModel):
    """Represents a single NYC taxi trip event.

    All monetary values are in USD. Distance in miles. Duration in seconds.
    """

    trip_id: str = Field(default_factory=lambda: f"TXN-{uuid.uuid4().hex[:8].upper()}")
    pickup_datetime: datetime
    dropoff_datetime: datetime
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int = Field(ge=1, le=9)
    trip_distance: float = Field(ge=0.0)
    fare_amount: float
    tip_amount: float = Field(ge=0.0)
    total_amount: float
    payment_type: str  # "cash" | "card"
    vendor_id: str

    @field_validator("trip_distance", "fare_amount", "total_amount", mode="before")
    @classmethod
    def round_two(cls, v: float) -> float:
        """Round monetary/distance fields to 2 decimal places."""
        return round(float(v), 2)

    @property
    def duration_seconds(self) -> float:
        """Trip duration in seconds."""
        return (self.dropoff_datetime - self.pickup_datetime).total_seconds()

    def to_feature_vector(self) -> list[float]:
        """Return numeric feature vector for anomaly detection model.

        Features: [fare_amount, trip_distance, duration_seconds, passenger_count,
                   tip_amount, fare_per_mile, fare_per_minute]
        """
        duration_min = self.duration_seconds / 60.0
        fare_per_mile = self.fare_amount / max(self.trip_distance, 0.01)
        fare_per_min = self.fare_amount / max(duration_min, 0.01)
        return [
            self.fare_amount,
            self.trip_distance,
            self.duration_seconds,
            float(self.passenger_count),
            self.tip_amount,
            fare_per_mile,
            fare_per_min,
        ]


class AnomalyResult(BaseModel):
    """Anomaly detection output for a single trip."""

    trip_id: str
    is_anomaly: bool
    anomaly_score: float  # negative = more anomalous (Isolation Forest convention)
    features: list[float]
    processed_at: datetime = Field(default_factory=datetime.utcnow)
