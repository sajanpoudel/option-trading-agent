"""
Ingestion Layer Configuration
Environment-driven settings with sensible defaults
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class IngestionSettings(BaseSettings):
    """Configuration for Kafka + Dask ingestion layer"""

    # Feature flags - ingestion is OPTIONAL for graceful degradation
    kafka_enabled: bool = os.getenv('KAFKA_ENABLED', 'false').lower() == 'true'
    dask_enabled: bool = os.getenv('DASK_ENABLED', 'false').lower() == 'true'

    # Kafka Configuration
    kafka_bootstrap_servers: str = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    kafka_consumer_group: str = os.getenv('KAFKA_CONSUMER_GROUP', 'neural-oracle')
    kafka_auto_offset_reset: str = 'latest'
    kafka_enable_auto_commit: bool = True
    kafka_compression_type: str = 'gzip'

    # Kafka Topics - Speed Layer
    topic_market_ticks: str = 'market-ticks'
    topic_options_flow: str = 'options-flow'
    topic_sentiment: str = 'sentiment-events'
    topic_technical: str = 'technical-signals'

    # Dask Configuration - Batch Layer
    dask_scheduler_address: str = os.getenv('DASK_SCHEDULER_ADDRESS', 'tcp://dask-scheduler:8786')
    dask_n_workers: int = int(os.getenv('DASK_N_WORKERS', '2'))
    dask_threads_per_worker: int = 4
    dask_memory_limit: str = '2GB'

    # Stream Buffer Settings
    stream_buffer_size: int = 1000
    stream_flush_interval: int = 5  # seconds

    # Batch Processing Settings
    batch_window_hours: int = 24
    batch_overlap_hours: int = 1

    class Config:
        env_prefix = 'INGESTION_'
        case_sensitive = False


# Global settings instance
ingestion_settings = IngestionSettings()
