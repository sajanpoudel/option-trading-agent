"""
Market Data Stream - Real-time Stock Quotes
Connects to Alpaca WebSocket and publishes to Kafka
"""

import asyncio
from typing import Set, Optional
from datetime import datetime
from loguru import logger

from ..kafka.producer import kafka_producer
from ..kafka.topics import KafkaTopics, MarketTickEvent
from backend.config.settings import settings


class MarketDataStream:
    """
    Real-time market data streaming from Alpaca
    Publishes tick data to Kafka for downstream processing
    """

    def __init__(self):
        self.subscribed_symbols: Set[str] = set()
        self.is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None

    async def start(self, symbols: list[str] = None):
        """
        Start streaming market data

        Args:
            symbols: List of symbols to stream (default: SPY, QQQ, AAPL)
        """
        if symbols:
            self.subscribed_symbols.update(symbols)
        else:
            # Default watchlist
            self.subscribed_symbols.update(['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA'])

        self.is_streaming = True
        self._stream_task = asyncio.create_task(self._stream_loop())

        logger.info(f"Market data stream started for: {self.subscribed_symbols}")

    async def stop(self):
        """Stop the market data stream"""
        self.is_streaming = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        logger.info("Market data stream stopped")

    async def subscribe(self, symbol: str):
        """Add a symbol to the stream"""
        self.subscribed_symbols.add(symbol)
        logger.info(f"Subscribed to {symbol}")

    async def unsubscribe(self, symbol: str):
        """Remove a symbol from the stream"""
        self.subscribed_symbols.discard(symbol)
        logger.info(f"Unsubscribed from {symbol}")

    async def _stream_loop(self):
        """
        Main streaming loop
        In production, this would connect to Alpaca WebSocket
        For demo, simulates tick data
        """
        try:
            while self.is_streaming:
                for symbol in self.subscribed_symbols:
                    # Simulate tick data (in production, this comes from Alpaca WebSocket)
                    tick_event = await self._get_market_tick(symbol)

                    # Publish to Kafka
                    await kafka_producer.publish(
                        topic=KafkaTopics.MARKET_TICKS,
                        event=tick_event,
                        key=symbol
                    )

                # Adjust interval based on market hours
                await asyncio.sleep(1)  # 1 second between ticks

        except asyncio.CancelledError:
            logger.info("Market stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in market stream loop: {e}")

    async def _get_market_tick(self, symbol: str) -> dict:
        """
        Get current market tick for a symbol

        Connects to real market data via yfinance (fallback from Alpaca WebSocket)

        Args:
            symbol: Stock symbol

        Returns:
            Market tick event dictionary with real data
        """
        try:
            # Import here to avoid circular dependency
            from backend.app.services.alpaca import AlpacaMarketDataClient

            # Use AlpacaMarketDataClient which has real data
            alpaca_client = AlpacaMarketDataClient()
            quote = await alpaca_client.get_current_quote(symbol)

            if quote and 'price' in quote:
                return MarketTickEvent.create(
                    symbol=symbol,
                    price=quote.get('price', 0),
                    volume=quote.get('volume', 0),
                    bid=quote.get('bid', quote.get('price', 0) - 0.01),
                    ask=quote.get('ask', quote.get('price', 0) + 0.01),
                    bid_size=quote.get('bid_size', 0),
                    ask_size=quote.get('ask_size', 0)
                )

            # Fallback if no data
            return MarketTickEvent.create(
                symbol=symbol,
                price=0,
                volume=0,
                bid=0,
                ask=0,
                bid_size=0,
                ask_size=0
            )

        except Exception as e:
            logger.error(f"Error getting real market tick for {symbol}: {e}")
            # Return zero data on error
            return MarketTickEvent.create(
                symbol=symbol,
                price=0,
                volume=0,
                bid=0,
                ask=0,
                bid_size=0,
                ask_size=0
            )

    def get_stats(self) -> dict:
        """Get stream statistics"""
        return {
            'is_streaming': self.is_streaming,
            'subscribed_symbols': list(self.subscribed_symbols),
            'symbol_count': len(self.subscribed_symbols)
        }


# Global stream instance
market_stream = MarketDataStream()
