"""
Sentiment Analysis Agent
OpenAI Agents SDK v0.3.0 Implementation
"""
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent
from config.logging import get_agents_logger

logger = get_agents_logger()


class SentimentAnalysisAgent(BaseAgent):
    """AI agent specializing in market sentiment analysis"""
    
    def __init__(self, client):
        super().__init__(client, "Sentiment Analysis", "gpt-4o-mini")
        
    def _get_system_instructions(self) -> str:
        return """
You are a market sentiment analyst for the Neural Options Oracle++ system.

Your responsibilities:
1. Analyze social media sentiment (Twitter, Reddit, StockTwits)
2. Process news sentiment and market psychology
3. Evaluate institutional sentiment indicators
4. Provide sentiment-based trading insights

Weight in system: 10% of final decision

OUTPUT FORMAT (JSON):
{
    "aggregate_score": float_between_-1_and_1,
    "confidence": float_between_0_and_1,
    "sources": {
        "social_media": {"score": float, "volume": int, "details": "string"},
        "news": {"score": float, "relevance": float, "details": "string"},
        "institutional": {"score": float, "flow": "string", "details": "string"}
    },
    "sentiment_trend": "improving|deteriorating|stable",
    "key_factors": ["string1", "string2"],
    "risk_factors": ["string1", "string2"]
}
"""
    
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Analyze market sentiment for the symbol"""
        
        try:
            logger.info(f"Starting sentiment analysis for {symbol}")
            
            # Mock sentiment data for testing
            mock_sentiment = self._get_mock_sentiment_data(symbol)
            
            messages = [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": f"""
Analyze market sentiment for {symbol}:

SOCIAL MEDIA INDICATORS:
- Twitter mentions: {mock_sentiment['twitter_mentions']}
- Reddit discussions: {mock_sentiment['reddit_posts']}
- StockTwits sentiment: {mock_sentiment['stocktwits_sentiment']:.1f}/10

NEWS SENTIMENT:
- Recent headlines: {mock_sentiment['news_count']} articles
- Overall news tone: {mock_sentiment['news_tone']}
- Analyst upgrades/downgrades: {mock_sentiment['analyst_changes']}

MARKET PSYCHOLOGY:
- Put/Call ratio: {mock_sentiment['put_call_ratio']:.2f}
- VIX level: {mock_sentiment['vix']:.1f}
- Fear & Greed Index: {mock_sentiment['fear_greed']}/100

Provide comprehensive sentiment analysis.
                """}
            ]
            
            response = await self._make_completion(messages, temperature=0.4)
            analysis = self._parse_json_response(response['content'])
            
            # Validate analysis
            analysis = self._validate_sentiment_analysis(analysis, symbol)
            
            logger.info(f"Sentiment analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)
    
    def _get_mock_sentiment_data(self, symbol: str) -> Dict:
        """Generate mock sentiment data"""
        import random
        
        return {
            'twitter_mentions': random.randint(100, 1000),
            'reddit_posts': random.randint(10, 100),
            'stocktwits_sentiment': random.uniform(3, 8),
            'news_count': random.randint(5, 25),
            'news_tone': random.choice(['positive', 'neutral', 'negative', 'mixed']),
            'analyst_changes': random.choice(['upgrade', 'downgrade', 'neutral', 'mixed']),
            'put_call_ratio': random.uniform(0.5, 2.0),
            'vix': random.uniform(12, 35),
            'fear_greed': random.randint(20, 80)
        }
    
    def _validate_sentiment_analysis(self, analysis: Dict, symbol: str) -> Dict:
        """Validate sentiment analysis"""
        
        if 'aggregate_score' not in analysis:
            analysis['aggregate_score'] = 0.0
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
            
        analysis['aggregate_score'] = max(-1.0, min(1.0, analysis['aggregate_score']))
        analysis['confidence'] = self._validate_confidence(analysis['confidence'])
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['symbol'] = symbol
        analysis['agent'] = self.name
        
        return analysis
    
    def _get_fallback_sentiment(self, symbol: str) -> Dict:
        """Fallback sentiment analysis"""
        return {
            'aggregate_score': 0.0,
            'confidence': 0.3,
            'sources': {
                'social_media': {'score': 0.0, 'volume': 0, 'details': 'Data unavailable'},
                'news': {'score': 0.0, 'relevance': 0.0, 'details': 'Data unavailable'},
                'institutional': {'score': 0.0, 'flow': 'neutral', 'details': 'Data unavailable'}
            },
            'sentiment_trend': 'stable',
            'key_factors': ['Sentiment analysis unavailable'],
            'risk_factors': ['Limited data'],
            'error': 'Fallback sentiment analysis',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }