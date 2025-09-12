"""
Market Data Manager
Centralized market data coordination and caching with OpenAI intelligence
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .alpaca_client import AlpacaMarketDataClient
from .openai_only_orchestrator import OpenAIMarketIntelligence
from config.database import db_manager
from config.logging import get_data_logger
from config.settings import settings

logger = get_data_logger()


class MarketDataManager:
    """Centralized market data management with AI intelligence and caching"""
    
    def __init__(self):
        self.alpaca_client = AlpacaMarketDataClient()
        self.ai_intelligence = OpenAIMarketIntelligence(settings.openai_api_key)
        self.cache_ttl = 300  # 5 minutes cache
        
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all market data for a symbol"""
        
        try:
            logger.info(f"Fetching comprehensive market data for {symbol}")
            
            # Check cache first
            cached_data = await self._get_cached_data(symbol)
            if cached_data:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
            
            # Fetch data in parallel
            tasks = [
                self.alpaca_client.get_current_quote(symbol),
                self.alpaca_client.get_technical_indicators(symbol),
                self.alpaca_client.get_options_data(symbol)
            ]
            
            quote_data, technical_data, options_data = await asyncio.gather(*tasks)
            
            # Combine all data
            comprehensive_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'quote': quote_data,
                'technical': technical_data,
                'options': options_data,
                'market_conditions': await self._get_market_conditions()
            }
            
            # Cache the data
            await self._cache_data(symbol, comprehensive_data)
            
            logger.info(f"Comprehensive data fetched for {symbol}")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive data for {symbol}: {e}")
            return self._get_fallback_comprehensive_data(symbol)
    
    async def get_comprehensive_ai_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive AI-powered analysis with real options data"""
        
        try:
            logger.info(f"Starting comprehensive AI analysis for {symbol}")
            
            # Check cache first for AI analysis
            cached_analysis = await self._get_cached_ai_analysis(symbol)
            if cached_analysis:
                logger.info(f"Using cached AI analysis for {symbol}")
                return cached_analysis
            
            # Get comprehensive AI intelligence
            intelligence = await self.ai_intelligence.get_comprehensive_intelligence(symbol)
            
            # Get basic market data for context
            basic_data = await self.get_comprehensive_data(symbol)
            
            # Combine everything
            comprehensive_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'ai_intelligence': {
                    'options_analysis': intelligence.options_analysis,
                    'news_sentiment': intelligence.news_sentiment,
                    'social_sentiment': intelligence.social_sentiment,
                    'technical_signals': intelligence.technical_signals,
                    'market_outlook': intelligence.market_outlook,
                    'confidence_score': intelligence.confidence_score
                },
                'market_data': basic_data,
                'analysis_type': 'openai_comprehensive'
            }
            
            # Cache the AI analysis
            await self._cache_ai_analysis(symbol, comprehensive_analysis)
            
            logger.info(f"Comprehensive AI analysis completed for {symbol} with {intelligence.confidence_score:.1%} confidence")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive AI analysis for {symbol}: {e}")
            return self._get_fallback_ai_analysis(symbol)
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols"""
        
        try:
            tasks = [self.alpaca_client.get_current_quote(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            quotes = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.warning(f"Quote failed for {symbol}: {result}")
                    quotes[symbol] = {'error': str(result)}
                else:
                    quotes[symbol] = result
            
            return quotes
            
        except Exception as e:
            logger.error(f"Failed to get multiple quotes: {e}")
            return {}
    
    async def _get_market_conditions(self) -> Dict[str, Any]:
        """Get overall market conditions"""
        
        try:
            # Get VIX as volatility indicator
            vix_data = await self.alpaca_client.get_current_quote('^VIX')
            
            # Get SPY for market direction
            spy_data = await self.alpaca_client.get_technical_indicators('SPY')
            
            return {
                'vix': vix_data.get('price', 20.0),
                'market_trend': self._determine_market_trend(spy_data),
                'volatility_regime': self._determine_volatility_regime(vix_data.get('price', 20.0)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Market conditions fetch failed: {e}")
            return {
                'vix': 20.0,
                'market_trend': 'neutral',
                'volatility_regime': 'medium',
                'timestamp': datetime.now().isoformat()
            }
    
    def _determine_market_trend(self, spy_data: Dict) -> str:
        """Determine overall market trend from SPY data"""
        
        try:
            current_price = spy_data.get('current_price', 400)
            ma20 = spy_data.get('ma20', 400)
            ma50 = spy_data.get('ma50', 400)
            
            if current_price > ma20 > ma50:
                return 'bullish'
            elif current_price < ma20 < ma50:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _determine_volatility_regime(self, vix_level: float) -> str:
        """Determine volatility regime based on VIX"""
        
        if vix_level < 15:
            return 'low'
        elif vix_level > 25:
            return 'high'
        else:
            return 'medium'
    
    async def _get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        
        try:
            # Query database cache
            result = await asyncio.create_task(
                asyncio.to_thread(
                    db_manager.client.table('market_data_cache').select('*').eq('symbol', symbol).single().execute
                )
            )
            
            if result.data:
                cache_time = datetime.fromisoformat(result.data['last_updated'].replace('Z', '+00:00'))
                if datetime.now(cache_time.tzinfo) - cache_time < timedelta(seconds=self.cache_ttl):
                    return result.data['price_data']
            
        except Exception as e:
            logger.debug(f"Cache read failed for {symbol}: {e}")
        
        return None
    
    async def _cache_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Cache market data"""
        
        try:
            cache_record = {
                'symbol': symbol,
                'price_data': data,
                'technical_indicators': data.get('technical', {}),
                'options_chain': data.get('options', {}),
                'last_updated': datetime.now().isoformat()
            }
            
            # Upsert to database
            await asyncio.create_task(
                asyncio.to_thread(
                    db_manager.client.table('market_data_cache').upsert(cache_record).execute
                )
            )
            
            logger.debug(f"Data cached for {symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {e}")
    
    async def _get_cached_ai_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached AI analysis"""
        
        try:
            # Query database cache for AI analysis (longer TTL: 15 minutes)
            result = await asyncio.create_task(
                asyncio.to_thread(
                    db_manager.client.table('ai_analysis_cache').select('*').eq('symbol', symbol).single().execute
                )
            )
            
            if result.data:
                cache_time = datetime.fromisoformat(result.data['last_updated'].replace('Z', '+00:00'))
                if datetime.now(cache_time.tzinfo) - cache_time < timedelta(seconds=900):  # 15 minutes
                    return result.data['analysis_data']
            
        except Exception as e:
            logger.debug(f"AI analysis cache read failed for {symbol}: {e}")
        
        return None
    
    async def _cache_ai_analysis(self, symbol: str, analysis: Dict[str, Any]) -> None:
        """Cache AI analysis"""
        
        try:
            cache_record = {
                'symbol': symbol,
                'analysis_data': analysis,
                'confidence_score': analysis['ai_intelligence']['confidence_score'],
                'last_updated': datetime.now().isoformat()
            }
            
            # Upsert to database
            await asyncio.create_task(
                asyncio.to_thread(
                    db_manager.client.table('ai_analysis_cache').upsert(cache_record).execute
                )
            )
            
            logger.debug(f"AI analysis cached for {symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to cache AI analysis for {symbol}: {e}")
    
    def _get_fallback_ai_analysis(self, symbol: str) -> Dict[str, Any]:
        """Fallback AI analysis when intelligence fails"""
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ai_intelligence': {
                'options_analysis': {'flow_sentiment': 'neutral', 'confidence': 0.0},
                'news_sentiment': {'sentiment_score': 0.0, 'confidence': 0.0},
                'social_sentiment': {'overall_sentiment': 0.0, 'confidence': 0.0},
                'technical_signals': {'trend_direction': 'neutral', 'confidence': 0.0},
                'market_outlook': {'price_target_consensus': 0.0, 'confidence': 0.0},
                'confidence_score': 0.0
            },
            'market_data': self._get_fallback_comprehensive_data(symbol),
            'analysis_type': 'fallback_ai',
            'error': 'AI analysis failed - using fallback data'
        }
    
    def _get_fallback_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback comprehensive data"""
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'quote': {
                'symbol': symbol,
                'price': 100.0,
                'source': 'fallback'
            },
            'technical': {
                'current_price': 100.0,
                'rsi': 50.0,
                'macd': 0.0,
                'volatility': 25.0,
                'source': 'fallback'
            },
            'options': {
                'symbol': symbol,
                'put_call_ratio': 1.0,
                'total_volume': 1000,
                'source': 'fallback'
            },
            'market_conditions': {
                'vix': 20.0,
                'market_trend': 'neutral',
                'volatility_regime': 'medium'
            },
            'error': 'Using fallback data'
        }


# Global market data manager instance
market_data_manager = MarketDataManager()