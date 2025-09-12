"""
Neural Options Oracle++ Data Pipeline
Real-time market data integration
"""

from .alpaca_client import AlpacaMarketDataClient
from .market_data_manager import MarketDataManager

__all__ = [
    'AlpacaMarketDataClient',
    'MarketDataManager'
]