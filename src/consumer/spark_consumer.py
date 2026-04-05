"""PySpark Structured Streaming consumer with Isolation Forest anomaly detection."""

from __future__ import annotations

from typing import Iterator

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from src.detector.model import TaxiAnomalyDetector
from src.utils.config import get_settings
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# -------------------------------------------------------------------
# Kafka message schema
# -------------------------------------------------------------------
TRIP_SCHEMA = StructType(
    [
        StructField("trip_id", StringType()),
        StructField("pickup_datetime", StringType()),
        StructField("dropoff_datetime", StringType()),
        StructField("fare_amount", DoubleType()),
        StructField("trip_distance", DoubleType()),
        StructField("passenger_count", DoubleType()),
        StructField("tip_amount", DoubleType()),
        StructField("total_amount", DoubleType()),
        StructField("payment_type", StringType()),
        StructField("vendor_id", StringType()),
        StructField("pickup_longitude", DoubleType()),
        StructField("pickup_latitude", DoubleType()),
        StructField("dropoff_longitude", DoubleType()),
        StructField("dropoff_latitude", DoubleType()),
    ]
)


def _create_spark() -> SparkSession:
    """Build and return a SparkSession configured for Kafka streaming.

    Returns:
        Active SparkSession.
    """
    return (
        SparkSession.builder.appName(settings.spark_app_name)
        .master(settings.spark_master)
        .config("spark.sql.streaming.checkpointLocation", settings.spark_checkpoint_dir)
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def _score_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Map function applied per micro-batch partition to run anomaly scoring.

    Loads the Isolation Forest model once per partition (executor) and scores
    all rows, adding `is_anomaly` (bool) and `anomaly_score` (float) columns.

    Args:
        iterator: Pandas DataFrames for each partition.

    Yields:
        Enriched DataFrames with anomaly columns appended.
    """
    detector = TaxiAnomalyDetector()

    for pdf in iterator:
        if pdf.empty:
            yield pdf
            continue

        from src.utils.schema import TaxiTrip

        def _to_trip(row: pd.Series) -> TaxiTrip:
            return TaxiTrip(
                trip_id=str(row["trip_id"]),
                pickup_datetime=pd.to_datetime(row["pickup_datetime"]),
                dropoff_datetime=pd.to_datetime(row["dropoff_datetime"]),
                fare_amount=float(row["fare_amount"]),
                trip_distance=float(row["trip_distance"]),
                passenger_count=int(row["passenger_count"]),
                tip_amount=float(row["tip_amount"]),
                total_amount=float(row["total_amount"]),
                payment_type=str(row["payment_type"]),
                vendor_id=str(row["vendor_id"]),
                pickup_longitude=float(row["pickup_longitude"]),
                pickup_latitude=float(row["pickup_latitude"]),
                dropoff_longitude=float(row["dropoff_longitude"]),
                dropoff_latitude=float(row["dropoff_latitude"]),
            )

        trips = [_to_trip(row) for _, row in pdf.iterrows()]
        results = detector.predict_batch(trips)

        pdf["is_anomaly"] = [r.is_anomaly for r in results]
        pdf["anomaly_score"] = [r.anomaly_score for r in results]
        yield pdf


def run_streaming_job() -> None:
    """Launch the PySpark Structured Streaming job.

    Reads from Kafka, parses JSON messages, scores each trip for anomalies,
    writes results to Parquet, and forwards to output Kafka topics.
    """
    spark = _create_spark()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        "Starting streaming job",
        extra={
            "input_topic": settings.kafka_input_topic,
            "trigger": settings.spark_trigger_interval,
        },
    )

    # Read raw bytes from Kafka
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_input_topic)
        .option("startingOffsets", settings.kafka_auto_offset_reset)
        .load()
    )

    # Parse JSON payload
    trips_df = (
        raw_stream.select(
            F.from_json(F.col("value").cast("string"), TRIP_SCHEMA).alias("trip")
        )
        .select("trip.*")
        .withColumn(
            "duration_seconds",
            F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime"),
        )
    )

    def process_batch(batch_df, batch_id: int) -> None:
        """Foreach-batch handler: score, log, and sink each micro-batch."""
        count = batch_df.count()
        if count == 0:
            return

        # Score using Isolation Forest via mapInPandas
        scored_df = batch_df.mapInPandas(_score_partition, schema=batch_df.schema.add("is_anomaly", BooleanType()).add("anomaly_score", DoubleType()))

        anomaly_count = scored_df.filter(F.col("is_anomaly")).count()
        logger.info(
            f"Batch #{batch_id} processed",
            extra={
                "total": count,
                "anomalies": anomaly_count,
                "rate_pct": round(100 * anomaly_count / max(count, 1), 1),
            },
        )

        # Persist all trips to Parquet
        (
            scored_df.write.mode("append")
            .partitionBy("payment_type")
            .parquet(settings.parquet_output_path)
        )

        # Forward anomalies to Kafka alert topic
        anomalies = scored_df.filter(F.col("is_anomaly"))
        if anomalies.count() > 0:
            (
                anomalies.selectExpr(
                    "trip_id AS key",
                    "to_json(struct(*)) AS value",
                )
                .write.format("kafka")
                .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
                .option("topic", settings.kafka_output_anomaly_topic)
                .save()
            )

        # Forward normal trips
        normals = scored_df.filter(~F.col("is_anomaly"))
        if normals.count() > 0:
            (
                normals.selectExpr("trip_id AS key", "to_json(struct(*)) AS value")
                .write.format("kafka")
                .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
                .option("topic", settings.kafka_output_normal_topic)
                .save()
            )

    query = (
        trips_df.writeStream.foreachBatch(process_batch)
        .trigger(processingTime=settings.spark_trigger_interval)
        .option("checkpointLocation", settings.spark_checkpoint_dir)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_job()
