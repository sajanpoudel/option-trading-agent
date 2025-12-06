"""
Data Ingestion Layer - Lambda Architecture Implementation
Combines real-time streaming (Kafka) with batch processing (Dask)
"""

from .manager import ingestion_manager

__all__ = ['ingestion_manager']
