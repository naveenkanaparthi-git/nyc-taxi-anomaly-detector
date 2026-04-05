"""Configuration management via environment variables using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_input_topic: str = "taxi.trips"
    kafka_output_normal_topic: str = "taxi.normal"
    kafka_output_anomaly_topic: str = "taxi.anomalies"
    kafka_group_id: str = "taxi-anomaly-detector"
    kafka_auto_offset_reset: str = "latest"

    # Spark
    spark_app_name: str = "NYCTaxiAnomalyDetector"
    spark_master: str = "local[*]"
    spark_trigger_interval: str = "30 seconds"
    spark_checkpoint_dir: str = "data/checkpoints"

    # Model
    model_path: str = "models/isolation_forest.pkl"
    anomaly_contamination: float = 0.05
    model_n_estimators: int = 100
    model_random_state: int = 42

    # Storage
    parquet_output_path: str = "data/output/"
    log_level: str = "INFO"

    # Producer
    producer_rate_per_sec: int = 50
    producer_num_trips: int = 10000


def get_settings() -> Settings:
    """Return a singleton-like settings instance."""
    return Settings()
