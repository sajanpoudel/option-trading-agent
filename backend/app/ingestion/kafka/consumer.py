"""
Kafka Consumer - Processing Real-time Events from Speed Layer
Async consumer with configurable message handlers
"""

import json
import asyncio
from typing import Dict, Any, Callable, Optional, Awaitable
from collections import defaultdict
from loguru import logger

try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from ..config import ingestion_settings
from .topics import KafkaTopics


class KafkaConsumerManager:
    """
    Async Kafka Consumer for processing real-time events
    Supports multiple topic subscriptions with custom handlers
    """

    def __init__(self):
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.handlers: Dict[KafkaTopics, Callable] = {}
        self.is_running = False
        self._consumer_tasks = []
        self._stats = defaultdict(int)

    async def start(self):
        """Initialize and start all registered consumers"""
        if not KAFKA_AVAILABLE or not ingestion_settings.kafka_enabled:
            logger.info("Kafka consumer disabled - skipping initialization")
            return

        if not self.handlers:
            logger.warning("No message handlers registered - consumer not started")
            return

        try:
            # Create consumer for each registered handler
            for topic, handler in self.handlers.items():
                await self._start_consumer(topic, handler)

            self.is_running = True
            logger.info(f"Kafka consumers started for topics: {list(self.handlers.keys())}")

        except Exception as e:
            logger.error(f"Failed to start Kafka consumers: {e}")
            self.is_running = False

    async def stop(self):
        """Stop all consumers gracefully"""
        if not self.is_running:
            return

        # Cancel all consumer tasks
        for task in self._consumer_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

        # Stop all consumers
        for consumer in self.consumers.values():
            try:
                await consumer.stop()
            except Exception as e:
                logger.error(f"Error stopping consumer: {e}")

        self.is_running = False
        logger.info(f"Kafka consumers stopped. Stats: {dict(self._stats)}")

    def register_handler(
        self,
        topic: KafkaTopics,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """
        Register a message handler for a topic

        Args:
            topic: Topic to consume from
            handler: Async function to process messages
        """
        self.handlers[topic] = handler
        logger.info(f"Registered handler for topic: {topic}")

    async def _start_consumer(
        self,
        topic: KafkaTopics,
        handler: Callable
    ):
        """Start a consumer for a specific topic"""
        try:
            consumer = AIOKafkaConsumer(
                topic.value,
                bootstrap_servers=ingestion_settings.kafka_bootstrap_servers,
                group_id=ingestion_settings.kafka_consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset=ingestion_settings.kafka_auto_offset_reset,
                enable_auto_commit=ingestion_settings.kafka_enable_auto_commit,
                max_poll_records=500,  # Fetch up to 500 records per poll
                session_timeout_ms=30000
            )

            await consumer.start()
            self.consumers[topic.value] = consumer

            # Start background task to consume messages
            task = asyncio.create_task(
                self._consume_messages(topic, consumer, handler)
            )
            self._consumer_tasks.append(task)

            logger.info(f"Started consumer for topic: {topic}")

        except Exception as e:
            logger.error(f"Failed to start consumer for {topic}: {e}")

    async def _consume_messages(
        self,
        topic: KafkaTopics,
        consumer: AIOKafkaConsumer,
        handler: Callable
    ):
        """Background task to consume and process messages"""
        try:
            async for message in consumer:
                try:
                    # Process message with registered handler
                    await handler(message.value)

                    # Update stats
                    self._stats[topic.value] += 1

                    if self._stats[topic.value] % 1000 == 0:
                        logger.debug(
                            f"Processed {self._stats[topic.value]} messages from {topic}"
                        )

                except Exception as e:
                    logger.error(f"Error processing message from {topic}: {e}")
                    # Continue processing other messages

        except asyncio.CancelledError:
            logger.info(f"Consumer task cancelled for {topic}")
        except Exception as e:
            logger.error(f"Consumer error for {topic}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        return {
            'is_running': self.is_running,
            'active_topics': list(self.consumers.keys()),
            'message_counts': dict(self._stats),
            'kafka_enabled': ingestion_settings.kafka_enabled,
            'kafka_available': KAFKA_AVAILABLE
        }


# Global consumer instance
kafka_consumer = KafkaConsumerManager()
