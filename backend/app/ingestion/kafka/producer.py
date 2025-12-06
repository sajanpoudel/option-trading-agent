"""
Kafka Producer - Publishing Events to Speed Layer
High-performance async producer with batching and compression
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

try:
    from aiokafka import AIOKafkaProducer
    from aiokafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("aiokafka not installed - Kafka features disabled")

from ..config import ingestion_settings
from .topics import KafkaTopics


class KafkaProducerManager:
    """
    Async Kafka Producer for real-time event streaming
    Optimized for high-throughput with batching and compression
    """

    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_running = False
        self._message_count = 0

    async def start(self):
        """Initialize and start the Kafka producer"""
        if not KAFKA_AVAILABLE or not ingestion_settings.kafka_enabled:
            logger.info("Kafka producer disabled - skipping initialization")
            return

        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=ingestion_settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type=ingestion_settings.kafka_compression_type,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=5,
                enable_idempotence=True,  # Exactly-once semantics
                linger_ms=10,  # Batch window
                batch_size=16384  # 16KB batches
            )

            await self.producer.start()
            self.is_running = True
            logger.info(f"Kafka producer started: {ingestion_settings.kafka_bootstrap_servers}")

        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the producer and flush remaining messages"""
        if self.producer and self.is_running:
            try:
                await self.producer.stop()
                logger.info(f"Kafka producer stopped. Total messages: {self._message_count}")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
            finally:
                self.is_running = False

    async def publish(
        self,
        topic: KafkaTopics,
        event: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """
        Publish an event to a Kafka topic

        Args:
            topic: Target topic (from KafkaTopics enum)
            event: Event data (will be JSON serialized)
            key: Optional partition key (e.g., symbol)

        Returns:
            bool: True if published successfully
        """
        if not self.is_running:
            logger.debug(f"Kafka disabled - event not published to {topic}")
            return False

        try:
            # Use symbol as partition key for consistent routing
            partition_key = key.encode('utf-8') if key else None

            # Send asynchronously
            await self.producer.send_and_wait(
                topic.value,
                value=event,
                key=partition_key
            )

            self._message_count += 1

            if self._message_count % 1000 == 0:
                logger.debug(f"Published {self._message_count} events to Kafka")

            return True

        except KafkaError as e:
            logger.error(f"Kafka error publishing to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing to {topic}: {e}")
            return False

    async def publish_batch(
        self,
        topic: KafkaTopics,
        events: list[Dict[str, Any]],
        keys: Optional[list[str]] = None
    ) -> int:
        """
        Publish multiple events efficiently

        Args:
            topic: Target topic
            events: List of events
            keys: Optional list of partition keys

        Returns:
            int: Number of successfully published events
        """
        if not self.is_running:
            return 0

        success_count = 0
        tasks = []

        for i, event in enumerate(events):
            key = keys[i] if keys and i < len(keys) else None
            task = self.publish(topic, event, key)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

        logger.info(f"Batch published {success_count}/{len(events)} events to {topic}")
        return success_count

    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics"""
        return {
            'is_running': self.is_running,
            'total_messages': self._message_count,
            'kafka_enabled': ingestion_settings.kafka_enabled,
            'kafka_available': KAFKA_AVAILABLE
        }


# Global producer instance
kafka_producer = KafkaProducerManager()
