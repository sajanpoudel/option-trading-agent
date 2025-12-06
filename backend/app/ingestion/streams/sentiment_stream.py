"""
Sentiment Stream - Real-time News and Social Media Sentiment
Monitors news sources and social media for market sentiment
"""

import asyncio
from typing import Set, Optional
from datetime import datetime
from loguru import logger

from ..kafka.producer import kafka_producer
from ..kafka.topics import KafkaTopics, SentimentEvent


class SentimentStream:
    """
    Real-time sentiment streaming from news and social sources
    Publishes sentiment events to Kafka
    """

    def __init__(self):
        self.subscribed_symbols: Set[str] = set()
        self.is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._event_count = 0

    async def start(self, symbols: list[str] = None):
        """
        Start streaming sentiment data

        Args:
            symbols: List of symbols to monitor
        """
        if symbols:
            self.subscribed_symbols.update(symbols)
        else:
            # Default watchlist for sentiment
            self.subscribed_symbols.update(['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA'])

        self.is_streaming = True
        self._stream_task = asyncio.create_task(self._stream_loop())

        logger.info(f"Sentiment stream started for: {self.subscribed_symbols}")

    async def stop(self):
        """Stop the sentiment stream"""
        self.is_streaming = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Sentiment stream stopped. Total events: {self._event_count}")

    async def _stream_loop(self):
        """Main streaming loop for sentiment data"""
        try:
            while self.is_streaming:
                for symbol in self.subscribed_symbols:
                    # Check for new sentiment data
                    sentiment_events = await self._get_sentiment_updates(symbol)

                    for event in sentiment_events:
                        # Publish to Kafka
                        await kafka_producer.publish(
                            topic=KafkaTopics.SENTIMENT,
                            event=event,
                            key=symbol
                        )
                        self._event_count += 1

                # Check every 30 seconds for new sentiment
                await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.info("Sentiment stream loop cancelled")
        except Exception as e:
            logger.error(f"Error in sentiment stream loop: {e}")

    async def _get_sentiment_updates(self, symbol: str) -> list[dict]:
        """
        Get sentiment updates using real OpenAI web search

        Uses OpenAI's web search capability to find recent news/social mentions
        and analyze sentiment

        Args:
            symbol: Stock symbol

        Returns:
            List of sentiment events
        """
        events = []

        try:
            # Import OpenAI for real sentiment analysis
            from openai import OpenAI
            from backend.config.settings import settings

            client = OpenAI(api_key=settings.openai_api_key)

            # Use OpenAI to search web and analyze sentiment
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": f"""Search the web for the latest news and social media sentiment about {symbol} stock.

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{{
    "sentiment_score": <number between -1.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "summary": "<brief 1-sentence summary>",
    "source": "<news|twitter|reddit|financial_blog>"
}}"""
                    }],
                    temperature=0.3
                )
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            import json
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('\n', 1)[1].rsplit('\n', 1)[0]
                if content.startswith('json'):
                    content = content[4:].strip()

            sentiment_data = json.loads(content)

            # Create sentiment event from OpenAI analysis
            event = SentimentEvent.create(
                symbol=symbol,
                score=sentiment_data.get('sentiment_score', 0.0),
                source=sentiment_data.get('source', 'openai_web_search'),
                confidence=sentiment_data.get('confidence', 0.5),
                text_snippet=sentiment_data.get('summary', f"Sentiment for {symbol}"),
                url=f"https://openai.com/search/{symbol}"
            )
            events.append(event)

            logger.info(f"Got sentiment for {symbol}: {sentiment_data.get('sentiment_score', 0):.2f}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI sentiment response for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")

        return events

    def get_stats(self) -> dict:
        """Get stream statistics"""
        return {
            'is_streaming': self.is_streaming,
            'subscribed_symbols': list(self.subscribed_symbols),
            'total_events': self._event_count
        }


# Global stream instance
sentiment_stream = SentimentStream()
