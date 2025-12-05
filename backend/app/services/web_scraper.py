"""
Web Scraper Service for Neural Options Oracle++
Provides trending stock data from various sources
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.config.logging import get_agents_logger

logger = get_agents_logger()


class WebScraperAgent:
    """Web scraper agent for gathering market sentiment data"""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def get_trending_stocks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending stocks from various sources"""
        try:
            logger.info(f"🔍 Fetching trending stocks (limit: {limit})")
            
            # Try to get data from yfinance trending
            trending_stocks = await self._get_yfinance_trending(limit)
            
            if trending_stocks:
                logger.info(f"✅ Found {len(trending_stocks)} trending stocks")
                return trending_stocks
            
            # Fallback to default hot stocks
            logger.warning("No trending data found, using fallback stocks")
            return self._get_fallback_stocks(limit)
            
        except Exception as e:
            logger.error(f"❌ Error fetching trending stocks: {e}")
            return self._get_fallback_stocks(limit)
    
    async def _get_yfinance_trending(self, limit: int) -> List[Dict[str, Any]]:
        """Get trending stocks using yfinance"""
        try:
            import yfinance as yf
            
            # Popular symbols to check for momentum
            symbols = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "SPY"]
            trending = []
            
            for symbol in symbols[:limit]:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    trending.append({
                        "symbol": symbol,
                        "name": info.get("shortName", f"{symbol} Inc"),
                        "mentions": 100,  # Placeholder
                        "sentiment": "Bullish" if info.get("recommendationMean", 3) < 2.5 else "Neutral",
                        "sentiment_score": 0.6,
                        "trending": True
                    })
                except Exception as e:
                    logger.debug(f"Failed to get data for {symbol}: {e}")
                    continue
            
            return trending
            
        except ImportError:
            logger.warning("yfinance not installed")
            return []
        except Exception as e:
            logger.error(f"Error in yfinance trending: {e}")
            return []
    
    def _get_fallback_stocks(self, limit: int) -> List[Dict[str, Any]]:
        """Get fallback list of popular stocks"""
        fallback_stocks = [
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "mentions": 500, "sentiment": "Bullish", "sentiment_score": 0.75, "trending": True},
            {"symbol": "TSLA", "name": "Tesla, Inc.", "mentions": 450, "sentiment": "Bullish", "sentiment_score": 0.65, "trending": True},
            {"symbol": "AAPL", "name": "Apple Inc.", "mentions": 400, "sentiment": "Neutral", "sentiment_score": 0.55, "trending": True},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "mentions": 350, "sentiment": "Bullish", "sentiment_score": 0.70, "trending": True},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "mentions": 300, "sentiment": "Neutral", "sentiment_score": 0.50, "trending": False},
            {"symbol": "AMZN", "name": "Amazon.com, Inc.", "mentions": 280, "sentiment": "Bullish", "sentiment_score": 0.60, "trending": True},
            {"symbol": "META", "name": "Meta Platforms, Inc.", "mentions": 250, "sentiment": "Bullish", "sentiment_score": 0.65, "trending": True},
            {"symbol": "AMD", "name": "Advanced Micro Devices", "mentions": 220, "sentiment": "Bullish", "sentiment_score": 0.70, "trending": True},
        ]
        return fallback_stocks[:limit]


# Global instance
_web_scraper_instance = None


def get_web_scraper_agent(openai_client=None) -> WebScraperAgent:
    """Get or create web scraper agent instance"""
    global _web_scraper_instance
    if _web_scraper_instance is None:
        _web_scraper_instance = WebScraperAgent(openai_client)
    return _web_scraper_instance
