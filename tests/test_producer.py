"""Unit tests for the taxi Kafka producer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from src.utils.schema import TaxiTrip


class TestTaxiProducer:
    """Tests for producer trip generation logic."""

    def test_build_trip_returns_valid_taxi_trip(self):
        """_build_trip should always return a valid TaxiTrip instance."""
        from src.producer.taxi_producer import _build_trip
        trip = _build_trip()
        assert isinstance(trip, TaxiTrip)
        assert trip.fare_amount > 0
        assert trip.trip_distance >= 0
        assert 1 <= trip.passenger_count <= 9

    def test_build_trip_generates_unique_ids(self):
        """Each call to _build_trip should produce a unique trip_id."""
        from src.producer.taxi_producer import _build_trip
        ids = {_build_trip().trip_id for _ in range(100)}
        assert len(ids) == 100, "trip_ids should be unique across 100 generated trips"

    def test_build_trip_anomaly_injection_rate(self):
        """Roughly 5% of generated trips should have anomalously high fares."""
        from src.producer.taxi_producer import _build_trip
        trips = [_build_trip() for _ in range(1000)]
        # An anomaly injection produces fare >= 200
        high_fare = [t for t in trips if t.fare_amount >= 200]
        rate = len(high_fare) / len(trips)
        # Allow wide tolerance since it's probabilistic
        assert 0.01 <= rate <= 0.15, f"Anomaly injection rate {rate:.2%} outside expected range"

    def test_producer_calls_kafka_send(self):
        """produce() should call KafkaProducer.send() for each trip."""
        from src.producer.taxi_producer import produce
        import asyncio

        mock_producer = MagicMock()
        mock_producer.send = MagicMock()

        with patch("src.producer.taxi_producer.KafkaProducer", return_value=mock_producer):
            with patch("src.producer.taxi_producer.settings") as mock_settings:
                mock_settings.kafka_input_topic = "taxi.trips"
                mock_settings.producer_rate_per_sec = 1000  # fast
                asyncio.run(produce(num_trips=3))

        assert mock_producer.send.call_count == 3
        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()
