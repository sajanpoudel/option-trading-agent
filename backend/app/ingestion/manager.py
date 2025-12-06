"""
Ingestion Manager - Unified Interface for Lambda Architecture
Coordinates Kafka (Speed Layer) + Dask (Batch Layer) + Serving Layer
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from .config import ingestion_settings
from .kafka.producer import kafka_producer
from .kafka.consumer import kafka_consumer
from .kafka.topics import KafkaTopics
from .dask.cluster import dask_cluster
from .dask.tasks import BatchTasks
from .streams.market_stream import market_stream
from .streams.options_stream import options_stream
from .streams.sentiment_stream import sentiment_stream


class IngestionManager:
    """
    Unified interface for the Lambda Architecture ingestion layer

    Responsibilities:
    1. Manage Kafka producers/consumers (Speed Layer)
    2. Manage Dask cluster connection (Batch Layer)
    3. Coordinate real-time streams
    4. Provide unified query interface (Serving Layer)
    5. Gracefully degrade if components unavailable
    """

    def __init__(self):
        self.is_running = False
        self._start_time: Optional[datetime] = None
        self._latest_ticks: Dict[str, Dict[str, Any]] = {}  # Buffer for latest ticks
        self._latest_flows: Dict[str, List[Dict[str, Any]]] = {}  # Buffer for options flow
        self._latest_sentiment: Dict[str, Dict[str, Any]] = {}  # Buffer for sentiment

    async def start(self):
        """
        Start the ingestion layer components
        Non-blocking - starts available components only
        """
        self._start_time = datetime.utcnow()
        logger.info("Starting Ingestion Layer...")

        # Start Kafka producer (Speed Layer)
        if ingestion_settings.kafka_enabled:
            await kafka_producer.start()
            logger.info("✓ Kafka producer initialized")
        else:
            logger.info("⊘ Kafka disabled - skipping producer")

        # Connect to Dask cluster (Batch Layer)
        if ingestion_settings.dask_enabled:
            await dask_cluster.connect()
            logger.info("✓ Dask cluster connected")
        else:
            logger.info("⊘ Dask disabled - batch processing will run locally")

        # Register Kafka consumers with message handlers
        if ingestion_settings.kafka_enabled:
            self._register_consumer_handlers()
            await kafka_consumer.start()
            logger.info("✓ Kafka consumers initialized")

        # Start real-time data streams (if Kafka enabled)
        if ingestion_settings.kafka_enabled:
            await self._start_streams()
            logger.info("✓ Data streams started")

        self.is_running = True
        logger.info(
            f"Ingestion Layer started successfully "
            f"(Kafka: {ingestion_settings.kafka_enabled}, "
            f"Dask: {ingestion_settings.dask_enabled})"
        )

    async def stop(self):
        """Stop all ingestion components gracefully"""
        logger.info("Stopping Ingestion Layer...")

        # Stop data streams
        await self._stop_streams()

        # Stop Kafka components
        await kafka_consumer.stop()
        await kafka_producer.stop()

        # Disconnect from Dask
        await dask_cluster.disconnect()

        self.is_running = False
        uptime = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
        logger.info(f"Ingestion Layer stopped. Uptime: {uptime:.1f}s")

    # ===== SPEED LAYER (Kafka) =====

    def _register_consumer_handlers(self):
        """Register message handlers for Kafka topics"""
        # Handle market ticks
        kafka_consumer.register_handler(
            KafkaTopics.MARKET_TICKS,
            self._handle_market_tick
        )

        # Handle options flow
        kafka_consumer.register_handler(
            KafkaTopics.OPTIONS_FLOW,
            self._handle_options_flow
        )

        # Handle sentiment events
        kafka_consumer.register_handler(
            KafkaTopics.SENTIMENT,
            self._handle_sentiment
        )

        logger.info("Registered Kafka consumer handlers")

    async def _handle_market_tick(self, event: Dict[str, Any]):
        """Process market tick events from Kafka"""
        symbol = event.get('symbol')
        if symbol:
            # Store in buffer for serving layer
            self._latest_ticks[symbol] = event
            logger.debug(f"Processed market tick: {symbol} @ ${event.get('price', 0):.2f}")

    async def _handle_options_flow(self, event: Dict[str, Any]):
        """Process options flow events from Kafka"""
        symbol = event.get('symbol')
        if symbol:
            # Store in buffer (keep last 100 flows per symbol)
            if symbol not in self._latest_flows:
                self._latest_flows[symbol] = []
            self._latest_flows[symbol].append(event)
            self._latest_flows[symbol] = self._latest_flows[symbol][-100:]  # Keep last 100
            logger.debug(f"Processed options flow: {symbol} {event.get('option_type')} ${event.get('strike')}")

    async def _handle_sentiment(self, event: Dict[str, Any]):
        """Process sentiment events from Kafka"""
        symbol = event.get('symbol')
        if symbol:
            # Store latest sentiment
            self._latest_sentiment[symbol] = event
            logger.debug(f"Processed sentiment: {symbol} score={event.get('sentiment_score', 0):.2f}")

    async def _start_streams(self):
        """Start real-time data streams"""
        default_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']

        await market_stream.start(default_symbols)
        await options_stream.start(default_symbols)
        await sentiment_stream.start(default_symbols)

        logger.info(f"Started streams for {len(default_symbols)} symbols")

    async def _stop_streams(self):
        """Stop all data streams"""
        await market_stream.stop()
        await options_stream.stop()
        await sentiment_stream.stop()

    # ===== BATCH LAYER (Dask) =====

    async def run_batch_job(
        self,
        job_type: str,
        symbols: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run a batch processing job on Dask cluster

        Args:
            job_type: Type of job ('features', 'flow', 'sentiment')
            symbols: List of symbols to process
            **kwargs: Additional job parameters

        Returns:
            List of job results
        """
        if not ingestion_settings.dask_enabled:
            logger.warning("Dask disabled - batch job running locally")

        logger.info(f"Running batch {job_type} job for {len(symbols)} symbols")

        try:
            results = await dask_cluster.submit_async(
                BatchTasks.batch_process_symbols,
                symbols,
                job_type
            )
            return results or []

        except Exception as e:
            logger.error(f"Batch job failed: {e}")
            return []

    # ===== SERVING LAYER (Unified View) =====

    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest market tick for a symbol from Kafka buffer

        Args:
            symbol: Stock symbol

        Returns:
            Latest tick data or None if not available
        """
        if not kafka_consumer.is_running:
            return None

        # Return from buffer if available
        return self._latest_ticks.get(symbol)

    def is_streaming(self, symbol: str) -> bool:
        """Check if real-time data is available for a symbol"""
        return (
            market_stream.is_streaming and
            symbol in market_stream.subscribed_symbols
        )

    async def subscribe_symbol(self, symbol: str):
        """Add a symbol to real-time streams"""
        if market_stream.is_streaming:
            await market_stream.subscribe(symbol)
        if options_stream.is_streaming:
            await options_stream.subscribe(symbol)
        if sentiment_stream.is_streaming:
            await sentiment_stream.subscribe(symbol)

        logger.info(f"Subscribed {symbol} to all streams")

    # ===== MONITORING =====

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of ingestion layer"""
        return {
            'is_running': self.is_running,
            'uptime_seconds': (
                (datetime.utcnow() - self._start_time).total_seconds()
                if self._start_time else 0
            ),
            'kafka': {
                'enabled': ingestion_settings.kafka_enabled,
                'producer': kafka_producer.get_stats(),
                'consumer': kafka_consumer.get_stats()
            },
            'dask': {
                'enabled': ingestion_settings.dask_enabled,
                'cluster': dask_cluster.get_stats()
            },
            'streams': {
                'market': market_stream.get_stats(),
                'options': options_stream.get_stats(),
                'sentiment': sentiment_stream.get_stats()
            }
        }

    def get_health(self) -> Dict[str, Any]:
        """Health check endpoint data"""
        status = self.get_status()

        return {
            'healthy': self.is_running,
            'components': {
                'kafka_producer': kafka_producer.is_running,
                'kafka_consumer': kafka_consumer.is_running,
                'dask_cluster': dask_cluster.is_connected,
                'market_stream': market_stream.is_streaming,
                'options_stream': options_stream.is_streaming,
                'sentiment_stream': sentiment_stream.is_streaming
            },
            'timestamp': datetime.utcnow().isoformat()
        }


# Global singleton instance
ingestion_manager = IngestionManager()
