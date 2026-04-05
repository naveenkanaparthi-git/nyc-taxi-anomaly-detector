# Architecture — NYC Taxi Trip Anomaly Detector

## Overview

This pipeline follows the **Lambda-lite** pattern: a real-time streaming layer handles low-latency anomaly detection, while all processed data is persisted to Parquet for batch analytics.

## Components

### 1. Kafka Producer (`src/producer/taxi_producer.py`)

- Simulates the NYC TLC real-time feed using synthetic trip data
- Publishes JSON-serialised `TaxiTrip` objects to `taxi.trips` at a configurable rate (default 50/s)
- Injects ~5% anomalous trips (extreme fares, impossible distances) to provide signal for the detector
- Uses `asyncio` for non-blocking rate-controlled publishing
- Graceful shutdown on SIGINT/SIGTERM

### 2. Kafka Topics

| Topic | Description |
|-------|-------------|
| `taxi.trips` | Raw trip events (input) |
| `taxi.normal` | Normal-classified trips |
| `taxi.anomalies` | Anomalous trips — downstream alert consumers subscribe here |

### 3. PySpark Structured Streaming (`src/consumer/spark_consumer.py`)

- Reads from `taxi.trips` using the Spark-Kafka connector
- Parses JSON messages against a strict `StructType` schema
- Applies `mapInPandas` to run Isolation Forest scoring per partition
- Forwards results to Kafka output topics and sinks to Parquet

**Trigger**: `processingTime=30 seconds` (configurable via `SPARK_TRIGGER_INTERVAL`)

### 4. Isolation Forest Model (`src/detector/model.py`)

**Feature engineering** (7 features per trip):

| # | Feature | Rationale |
|---|---------|-----------|
| 1 | `fare_amount` | Primary monetary signal |
| 2 | `trip_distance` | Sanity-check for fare |
| 3 | `duration_seconds` | Long idle trips are anomalous |
| 4 | `passenger_count` | Group rides differ from solo |
| 5 | `tip_amount` | Tip outliers correlate with fraud |
| 6 | `fare_per_mile` | Normalised rate (catches zero-distance high fares) |
| 7 | `fare_per_minute` | Time-normalised rate |

**Training**: `src/detector/trainer.py` generates 5,000 synthetic normal trips + 250 anomaly-injected trips and fits an `IsolationForest(contamination=0.05)`.

### 5. Parquet Data Lake (`data/output/`)

- All scored trips (normal + anomaly) are persisted partitioned by `payment_type`
- Schema is Delta Lake compatible (add `delta-spark` dependency to enable ACID and time travel)

## Data Flow

```
Producer (asyncio) ──┬──► [Kafka: taxi.trips]
                     │           │
                     │    PySpark Structured Streaming (micro-batch, 30s)
                     │           │
                     │    Isolation Forest scoring (mapInPandas)
                     │           │
                     │    ┌──────┴──────────────────────────┐
                     │    │                                  │
                     │  [Kafka: taxi.normal]     [Kafka: taxi.anomalies]
                     │                                       │
                     │                             [Alert Consumer / Dashboard]
                     │
                     └──► [Parquet: data/output/ partitioned by payment_type]
```

## Scaling Notes

- **Kafka**: Add partitions to `taxi.trips` (recommend 1 partition per Spark executor core)
- **Spark**: Switch `SPARK_MASTER` to a Spark cluster URL or use Databricks / EMR
- **Model**: Replace pickle-based loading with MLflow Model Registry for versioned deploys
- **Storage**: Enable Delta Lake for ACID transactions and time-travel queries
