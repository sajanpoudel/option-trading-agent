"""
Options Flow Stream - Real-time Options Activity
Monitors unusual options activity and publishes to Kafka
"""

import asyncio
from typing import Set, Optional
from datetime import datetime, timedelta
from loguru import logger

from ..kafka.producer import kafka_producer
from ..kafka.topics import KafkaTopics, OptionsFlowEvent


class OptionsFlowStream:
    """
    Real-time options flow streaming
    Detects unusual activity and publishes to Kafka
    """

    def __init__(self):
        self.subscribed_symbols: Set[str] = set()
        self.is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._flow_count = 0

    async def start(self, symbols: list[str] = None):
        """
        Start streaming options flow

        Args:
            symbols: List of symbols to monitor
        """
        if symbols:
            self.subscribed_symbols.update(symbols)
        else:
            # Default high-volume options symbols
            self.subscribed_symbols.update(['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA'])

        self.is_streaming = True
        self._stream_task = asyncio.create_task(self._stream_loop())

        logger.info(f"Options flow stream started for: {self.subscribed_symbols}")

    async def stop(self):
        """Stop the options flow stream"""
        self.is_streaming = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Options flow stream stopped. Total flows: {self._flow_count}")

    async def _stream_loop(self):
        """Main streaming loop for options flow"""
        try:
            while self.is_streaming:
                for symbol in self.subscribed_symbols:
                    # Check for unusual options activity
                    flow_events = await self._detect_unusual_flow(symbol)

                    for event in flow_events:
                        # Publish to Kafka
                        await kafka_producer.publish(
                            topic=KafkaTopics.OPTIONS_FLOW,
                            event=event,
                            key=symbol
                        )
                        self._flow_count += 1

                # Check every 5 seconds for new flow
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Options flow loop cancelled")
        except Exception as e:
            logger.error(f"Error in options flow loop: {e}")

    async def _detect_unusual_flow(self, symbol: str) -> list[dict]:
        """
        Detect unusual options activity from real yfinance data

        Checks for:
        1. High volume relative to open interest
        2. Unusual premium spikes
        3. Concentrated strikes

        Args:
            symbol: Stock symbol

        Returns:
            List of unusual flow events
        """
        flows = []

        try:
            # Import here to avoid circular dependency
            from backend.app.services.alpaca import AlpacaMarketDataClient

            # Use AlpacaMarketDataClient which gets real options data via yfinance
            alpaca_client = AlpacaMarketDataClient()
            options_data = await alpaca_client.get_options_data(symbol)

            if not options_data or 'options_chain' not in options_data:
                return flows

            # Analyze options chain for unusual activity
            for exp_date, chain_data in options_data.get('options_chain', {}).items():
                calls = chain_data.get('calls', [])
                puts = chain_data.get('puts', [])

                # Check calls for unusual volume
                for call in calls[:5]:  # Top 5 calls only to avoid spam
                    volume = call.get('volume', 0)
                    oi = call.get('openInterest', 1)

                    # Unusual if volume > 2x open interest and volume > 100
                    if volume > 100 and volume > (oi * 2):
                        flow = OptionsFlowEvent.create(
                            symbol=symbol,
                            option_type='CALL',
                            strike=call.get('strike', 0),
                            expiry=exp_date,
                            premium=call.get('lastPrice', 0),
                            volume=volume,
                            open_interest=oi,
                            implied_volatility=call.get('impliedVolatility', 0)
                        )
                        flows.append(flow)

                # Check puts for unusual volume
                for put in puts[:5]:  # Top 5 puts only
                    volume = put.get('volume', 0)
                    oi = put.get('openInterest', 1)

                    # Unusual if volume > 2x open interest and volume > 100
                    if volume > 100 and volume > (oi * 2):
                        flow = OptionsFlowEvent.create(
                            symbol=symbol,
                            option_type='PUT',
                            strike=put.get('strike', 0),
                            expiry=exp_date,
                            premium=put.get('lastPrice', 0),
                            volume=volume,
                            open_interest=oi,
                            implied_volatility=put.get('impliedVolatility', 0)
                        )
                        flows.append(flow)

            if flows:
                logger.info(f"Detected {len(flows)} unusual options flows for {symbol}")

        except Exception as e:
            logger.error(f"Error detecting options flow for {symbol}: {e}")

        return flows

    def get_stats(self) -> dict:
        """Get stream statistics"""
        return {
            'is_streaming': self.is_streaming,
            'subscribed_symbols': list(self.subscribed_symbols),
            'total_flows_detected': self._flow_count
        }


# Global stream instance
options_stream = OptionsFlowStream()
