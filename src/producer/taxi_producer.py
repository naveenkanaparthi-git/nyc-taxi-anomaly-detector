"""Async Kafka producer that streams synthetic NYC taxi trip events."""

from __future__ import annotations

import asyncio
import json
import random
import signal
from datetime import datetime, timedelta

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.utils.config import get_settings
from src.utils.logging_config import setup_logger
from src.utils.schema import TaxiTrip

logger = setup_logger(__name__)
settings = get_settings()

LAT_RANGE = (40.68, 40.85)
LON_RANGE = (-74.02, -73.93)

_running = True


def _build_trip() -> TaxiTrip:
    """Construct a random taxi trip event with occasional injected anomalies."""
    pickup = datetime.utcnow() - timedelta(minutes=random.randint(0, 30))
    distance = round(random.uniform(0.3, 25.0), 2)
    fare = round(2.50 + distance * random.uniform(2.0, 4.5), 2)

    # 5% chance of anomaly injection
    if random.random() < 0.05:
        fare = round(random.uniform(200.0, 600.0), 2)
        distance = round(random.uniform(0.1, 0.4), 2)

    tip = round(fare * random.uniform(0.0, 0.30), 2)
    duration_min = random.randint(5, 90)

    return TaxiTrip(
        pickup_datetime=pickup,
        dropoff_datetime=pickup + timedelta(minutes=duration_min),
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


def _create_producer() -> KafkaProducer:
    """Initialise and return a KafkaProducer with JSON serialisation.

    Returns:
        Configured KafkaProducer instance.

    Raises:
        KafkaError: If connection to the broker cannot be established.
    """
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
        max_block_ms=10_000,
    )


async def produce(num_trips: int = 0) -> None:
    """Stream taxi trip events to the Kafka input topic.

    Args:
        num_trips: Total trips to publish (0 = run indefinitely).
    """
    global _running
    producer = _create_producer()
    interval = 1.0 / settings.producer_rate_per_sec
    count = 0

    logger.info(
        "Producer started",
        extra={
            "topic": settings.kafka_input_topic,
            "rate": settings.producer_rate_per_sec,
            "limit": num_trips or "unlimited",
        },
    )

    try:
        while _running and (num_trips == 0 or count < num_trips):
            trip = _build_trip()
            payload = trip.model_dump(mode="json")

            producer.send(
                settings.kafka_input_topic,
                value=payload,
                key=trip.trip_id.encode(),
            )
            count += 1

            if count % 500 == 0:
                logger.info("Published trips", extra={"count": count})

            await asyncio.sleep(interval)
    except KafkaError as exc:
        logger.error("Kafka error during produce", extra={"error": str(exc)})
        raise
    finally:
        producer.flush()
        producer.close()
        logger.info("Producer shut down", extra={"total_published": count})


def _handle_signal(sig: int, frame: object) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _running
    logger.info("Shutdown signal received", extra={"signal": sig})
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    asyncio.run(produce(num_trips=settings.producer_num_trips))
