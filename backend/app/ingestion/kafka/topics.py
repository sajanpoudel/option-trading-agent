"""
Kafka Topic Definitions and Schemas
Centralized topic configuration for maintainability
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime


class KafkaTopics(str, Enum):
    """Topic name definitions"""
    MARKET_TICKS = 'market-ticks'
    OPTIONS_FLOW = 'options-flow'
    SENTIMENT = 'sentiment-events'
    TECHNICAL = 'technical-signals'


@dataclass
class TopicConfig:
    """Topic configuration"""
    name: str
    partitions: int
    replication_factor: int
    retention_ms: int  # How long to keep data
    cleanup_policy: str = 'delete'


class TopicDefinitions:
    """Centralized topic configurations"""

    CONFIGS = {
        KafkaTopics.MARKET_TICKS: TopicConfig(
            name=KafkaTopics.MARKET_TICKS,
            partitions=3,
            replication_factor=1,
            retention_ms=3600000,  # 1 hour - high frequency data
            cleanup_policy='delete'
        ),
        KafkaTopics.OPTIONS_FLOW: TopicConfig(
            name=KafkaTopics.OPTIONS_FLOW,
            partitions=2,
            replication_factor=1,
            retention_ms=86400000,  # 24 hours
            cleanup_policy='delete'
        ),
        KafkaTopics.SENTIMENT: TopicConfig(
            name=KafkaTopics.SENTIMENT,
            partitions=2,
            replication_factor=1,
            retention_ms=86400000,  # 24 hours
            cleanup_policy='delete'
        ),
        KafkaTopics.TECHNICAL: TopicConfig(
            name=KafkaTopics.TECHNICAL,
            partitions=2,
            replication_factor=1,
            retention_ms=7200000,  # 2 hours
            cleanup_policy='delete'
        )
    }

    @classmethod
    def get_config(cls, topic: KafkaTopics) -> TopicConfig:
        """Get configuration for a topic"""
        return cls.CONFIGS[topic]

    @classmethod
    def get_all_topics(cls) -> list[str]:
        """Get all topic names"""
        return [topic.value for topic in KafkaTopics]


# Event Schemas for validation
class MarketTickEvent:
    """Schema for market tick events"""

    @staticmethod
    def create(symbol: str, price: float, volume: int, **kwargs) -> Dict[str, Any]:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'bid': kwargs.get('bid'),
            'ask': kwargs.get('ask'),
            'bid_size': kwargs.get('bid_size'),
            'ask_size': kwargs.get('ask_size')
        }


class OptionsFlowEvent:
    """Schema for options flow events"""

    @staticmethod
    def create(symbol: str, option_type: str, strike: float, expiry: str, **kwargs) -> Dict[str, Any]:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'premium': kwargs.get('premium'),
            'volume': kwargs.get('volume'),
            'open_interest': kwargs.get('open_interest'),
            'implied_volatility': kwargs.get('implied_volatility')
        }


class SentimentEvent:
    """Schema for sentiment events"""

    @staticmethod
    def create(symbol: str, score: float, source: str, **kwargs) -> Dict[str, Any]:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'sentiment_score': score,
            'source': source,
            'confidence': kwargs.get('confidence', 0.5),
            'text_snippet': kwargs.get('text_snippet', ''),
            'url': kwargs.get('url')
        }


class TechnicalSignalEvent:
    """Schema for technical signal events"""

    @staticmethod
    def create(symbol: str, signal_type: str, value: float, **kwargs) -> Dict[str, Any]:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'signal_type': signal_type,
            'value': value,
            'indicator': kwargs.get('indicator', 'custom'),
            'timeframe': kwargs.get('timeframe', '1m'),
            'metadata': kwargs.get('metadata', {})
        }
