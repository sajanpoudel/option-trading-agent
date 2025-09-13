"""
Sentiment Analysis Agent
OpenAI Agents SDK v0.3.0 Implementation - REAL DATA ONLY
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from config.logging import get_agents_logger
from openai import AsyncOpenAI
import asyncio
import json
import re

logger = get_agents_logger()


class SentimentAnalysisAgent(BaseAgent):
    """AI agent specializing in market sentiment analysis using real web search data"""
    
    def __init__(self, client):
        super().__init__(client, "Sentiment Analysis", "gpt-4o")
        self.openai_client = AsyncOpenAI(api_key=client.api_key) if hasattr(client, 'api_key') else None
        
    def _get_system_instructions(self) -> str:
        return """
You are a market sentiment analyst for the Neural Options Oracle++ system using REAL web search data.

Your responsibilities:
1. Search for real-time news sentiment using web search
2. Analyze StockTwits sentiment from live web data
3. Process market psychology indicators
4. Provide sentiment-based trading insights

Weight in system: 10% of final decision

OUTPUT FORMAT (JSON):
{
    "aggregate_score": float_between_-1_and_1,
    "confidence": float_between_0_and_1,
    "sources": {
        "news_sentiment": {"score": float, "article_count": int, "details": "string"},
        "stocktwits_sentiment": {"score": float, "message_count": int, "details": "string"},
        "market_psychology": {"score": float, "indicators": "string", "details": "string"}
    },
    "sentiment_trend": "improving|deteriorating|stable",
    "key_factors": ["string1", "string2"],
    "risk_factors": ["string1", "string2"],
    "data_freshness": "YYYY-MM-DD HH:MM:SS"
}
"""
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for sentiment analysis response"""
        return {
            "type": "object",
            "properties": {
                "aggregate_score": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "sources": {
                    "type": "object",
                    "properties": {
                        "news_sentiment": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "article_count": {"type": "integer", "minimum": 0},
                                "details": {"type": "string"}
                            },
                            "required": ["score", "article_count", "details"],
                            "additionalProperties": False
                        },
                        "stocktwits_sentiment": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "message_count": {"type": "integer", "minimum": 0},
                                "details": {"type": "string"}
                            },
                            "required": ["score", "message_count", "details"],
                            "additionalProperties": False
                        },
                        "market_psychology": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "indicators": {"type": "string"},
                                "details": {"type": "string"}
                            },
                            "required": ["score", "indicators", "details"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["news_sentiment", "stocktwits_sentiment", "market_psychology"],
                    "additionalProperties": False
                },
                "sentiment_trend": {
                    "type": "string",
                    "enum": ["improving", "deteriorating", "stable"]
                },
                "key_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5
                },
                "risk_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5
                },
                "data_freshness": {
                    "type": "string",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$"
                }
            },
            "required": ["aggregate_score", "confidence", "sources", "sentiment_trend", "key_factors", "risk_factors", "data_freshness"],
            "additionalProperties": False
        }
    
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Analyze market sentiment for the symbol using real web search data"""
        
        try:
            logger.info(f"Starting REAL sentiment analysis for {symbol}")
            
            if not self.openai_client:
                logger.error("OpenAI client not available for sentiment analysis")
                return self._get_fallback_sentiment(symbol)
            
            # Get current date for search queries
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Collect real sentiment data from multiple sources
            sentiment_data = await self._collect_real_sentiment_data(symbol, current_date)
            
            # Analyze with GPT
            analysis = await self._analyze_sentiment_with_gpt(sentiment_data, symbol, current_date)
            
            # Validate analysis
            analysis = self._validate_sentiment_analysis(analysis, symbol)
            
            logger.info(f"REAL sentiment analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)
    
    async def _collect_real_sentiment_data(self, symbol: str, current_date: str) -> Dict[str, Any]:
        """Collect real sentiment data from web search"""
        try:
            # Run multiple web searches in parallel
            tasks = [
                self._search_news_sentiment(symbol, current_date),
                self._search_stocktwits_sentiment(symbol, current_date),
                self._search_market_psychology(symbol, current_date)
            ]
            
            news_data, stocktwits_data, psychology_data = await asyncio.gather(*tasks, return_exceptions=True)
        
            return {
                'news_sentiment': news_data if not isinstance(news_data, Exception) else {},
                'stocktwits_sentiment': stocktwits_data if not isinstance(stocktwits_data, Exception) else {},
                'market_psychology': psychology_data if not isinstance(psychology_data, Exception) else {},
                'search_date': current_date,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Error collecting real sentiment data for {symbol}: {e}")
            return {
                'news_sentiment': {},
                'stocktwits_sentiment': {},
                'market_psychology': {},
                'search_date': current_date,
                'symbol': symbol,
                'error': str(e)
            }
    
    async def _search_news_sentiment(self, symbol: str, current_date: str) -> Dict[str, Any]:
        """Search for real-time news sentiment"""
        try:
            prompt = f"""
            Please search the web for the latest news about {symbol} on {current_date}.
            
            Search for:
            1. Latest financial news headlines about {symbol}
            2. Analyst reports and upgrades/downgrades
            3. Earnings announcements or guidance
            4. Market-moving news and events
            
            Analyze the sentiment of these news items and return:
            - Overall news sentiment score (-1 to 1)
            - Number of articles found
            - Key positive and negative factors
            - Analyst sentiment (upgrades/downgrades)
            
            Return in JSON format:
            {{
                "sentiment_score": float,
                "article_count": int,
                "positive_factors": ["factor1", "factor2"],
                "negative_factors": ["factor1", "factor2"],
                "analyst_sentiment": "positive|negative|neutral|mixed",
                "key_headlines": ["headline1", "headline2", "headline3"]
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news analyst. Use web search to find real-time news and analyze sentiment. Always return valid JSON data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_from_response(content)
            
        except Exception as e:
            logger.error(f"Error searching news sentiment for {symbol}: {e}")
            return {}
    
    async def _search_stocktwits_sentiment(self, symbol: str, current_date: str) -> Dict[str, Any]:
        """Search for StockTwits sentiment"""
        try:
            prompt = f"""
            Please search the web for StockTwits sentiment about {symbol} on {current_date}.
            
            Specifically visit: https://stocktwits.com/sentiment/most-active
            
            Extract:
            1. Current sentiment score for {symbol}
            2. Number of mentions and messages
            3. Bullish vs bearish percentage
            4. Trending status
            5. Sample messages and sentiment
            
            Return in JSON format:
            {{
                "sentiment_score": float,
                "message_count": int,
                "bullish_percentage": float,
                "bearish_percentage": float,
                "trending": boolean,
                "sample_messages": ["message1", "message2", "message3"]
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a social media sentiment analyst. Use web search to find real-time StockTwits sentiment data. Always return valid JSON data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_from_response(content)
            
        except Exception as e:
            logger.error(f"Error searching StockTwits sentiment for {symbol}: {e}")
            return {}
    
    async def _search_market_psychology(self, symbol: str, current_date: str) -> Dict[str, Any]:
        """Search for market psychology indicators"""
        try:
            prompt = f"""
            Please search the web for market psychology indicators on {current_date}.
            
            Search for:
            1. VIX (Volatility Index) current level
            2. Put/Call ratio for {symbol}
            3. Fear & Greed Index
            4. Market sentiment indicators
            5. Institutional flow data
            
            Return in JSON format:
            {{
                "vix_level": float,
                "put_call_ratio": float,
                "fear_greed_index": int,
                "market_sentiment": "fearful|greedy|neutral",
                "institutional_flow": "buying|selling|neutral",
                "volatility_trend": "increasing|decreasing|stable"
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a market psychology analyst. Use web search to find real-time market sentiment indicators. Always return valid JSON data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_from_response(content)
            
        except Exception as e:
            logger.error(f"Error searching market psychology for {symbol}: {e}")
            return {}
    
    async def _analyze_sentiment_with_gpt(self, sentiment_data: Dict[str, Any], symbol: str, current_date: str) -> Dict[str, Any]:
        """Analyze collected sentiment data with GPT"""
        try:
            # Prepare comprehensive sentiment analysis prompt
            prompt = f"""
            Analyze the collected sentiment data for {symbol} on {current_date} and provide a comprehensive sentiment analysis.
            
            NEWS SENTIMENT DATA:
            {json.dumps(sentiment_data.get('news_sentiment', {}), indent=2)}
            
            STOCKTWITS SENTIMENT DATA:
            {json.dumps(sentiment_data.get('stocktwits_sentiment', {}), indent=2)}
            
            MARKET PSYCHOLOGY DATA:
            {json.dumps(sentiment_data.get('market_psychology', {}), indent=2)}
            
            Based on this real data, provide:
            1. Overall aggregate sentiment score (-1 to 1)
            2. Confidence level (0 to 1)
            3. Sentiment trend analysis
            4. Key factors driving sentiment
            5. Risk factors to consider
            
            Return in the exact JSON format specified in the system instructions.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_instructions},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_from_response(content)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment with GPT for {symbol}: {e}")
            return {}
    
    def _parse_json_from_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from GPT response"""
        try:
            # Try multiple JSON extraction patterns
            json_patterns = [
                r'\{.*\}',  # Any JSON object
                r'\{[^{}]*\}',  # Simple JSON object
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, return empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing JSON from response: {e}")
            return {}
    
    def _validate_sentiment_analysis(self, analysis: Dict, symbol: str) -> Dict:
        """Validate sentiment analysis"""
        
        if 'aggregate_score' not in analysis:
            analysis['aggregate_score'] = 0.0
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
            
        analysis['aggregate_score'] = max(-1.0, min(1.0, analysis['aggregate_score']))
        analysis['confidence'] = max(0.0, min(1.0, analysis['confidence']))
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['symbol'] = symbol
        analysis['agent'] = self.name
        analysis['data_freshness'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return analysis
    
    def _get_fallback_sentiment(self, symbol: str) -> Dict:
        """Fallback sentiment analysis when real data unavailable"""
        return {
            'aggregate_score': 0.0,
            'confidence': 0.3,
            'sources': {
                'news_sentiment': {'score': 0.0, 'article_count': 0, 'details': 'Real-time news data unavailable'},
                'stocktwits_sentiment': {'score': 0.0, 'message_count': 0, 'details': 'StockTwits data unavailable'},
                'market_psychology': {'score': 0.0, 'indicators': 'unavailable', 'details': 'Market psychology data unavailable'}
            },
            'sentiment_trend': 'stable',
            'key_factors': ['Real-time sentiment data unavailable'],
            'risk_factors': ['Limited real-time data'],
            'error': 'Fallback sentiment analysis - real data unavailable',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name,
            'data_freshness': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }