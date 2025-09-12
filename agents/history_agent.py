"""
Historical Pattern Analysis Agent
OpenAI Agents SDK v0.3.0 Implementation
"""
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent
from config.logging import get_agents_logger

logger = get_agents_logger()


class HistoricalPatternAgent(BaseAgent):
    """AI agent specializing in historical pattern analysis"""
    
    def __init__(self, client):
        super().__init__(client, "Historical Pattern", "gpt-4o")
        
    def _get_system_instructions(self) -> str:
        return """
You are a historical pattern analyst for the Neural Options Oracle++ system.

Your responsibilities:
1. Analyze historical price patterns and seasonality
2. Identify recurring market behaviors and cycles
3. Compare current patterns to historical precedents
4. Provide pattern-based trading insights

Weight in system: 20% of final decision

OUTPUT FORMAT (JSON):
{
    "pattern_score": float_between_-1_and_1,
    "confidence": float_between_0_and_1,
    "dominant_pattern": "uptrend|downtrend|consolidation|breakout|reversal",
    "historical_matches": [
        {"date": "YYYY-MM-DD", "similarity": float, "outcome": "string"}
    ],
    "seasonality": {
        "monthly_bias": "bullish|bearish|neutral",
        "weekly_pattern": "string",
        "earnings_cycle": "pre|post|neutral"
    },
    "pattern_strength": float_between_0_and_1,
    "time_horizon": "short|medium|long",
    "key_levels": {
        "support": [float, float],
        "resistance": [float, float]
    },
    "pattern_insights": ["string1", "string2"]
}
"""
    
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Analyze historical patterns for the symbol"""
        
        try:
            logger.info(f"Starting historical pattern analysis for {symbol}")
            
            # Mock historical data
            mock_history = self._get_mock_historical_data(symbol)
            
            messages = [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": f"""
Analyze historical patterns for {symbol}:

RECENT PERFORMANCE:
- 1-week return: {mock_history['week_return']:.1f}%
- 1-month return: {mock_history['month_return']:.1f}%
- 3-month return: {mock_history['quarter_return']:.1f}%
- YTD return: {mock_history['ytd_return']:.1f}%

PATTERN INDICATORS:
- Current trend: {mock_history['trend_direction']}
- Trend duration: {mock_history['trend_days']} days
- Pattern type: {mock_history['chart_pattern']}
- Volatility regime: {mock_history['vol_regime']}

SEASONALITY:
- Historical month performance: {mock_history['month_avg']:.1f}%
- Earnings season: {mock_history['earnings_phase']}
- Day of week: {mock_history['day_of_week']}

SIMILAR PERIODS:
- Pattern matches found: {mock_history['similar_periods']}
- Average outcome: {mock_history['avg_outcome']:.1f}%

Provide comprehensive pattern analysis.
                """}
            ]
            
            response = await self._make_completion(messages, temperature=0.4)
            analysis = self._parse_json_response(response['content'])
            
            analysis = self._validate_pattern_analysis(analysis, symbol)
            
            logger.info(f"Historical pattern analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Historical pattern analysis failed for {symbol}: {e}")
            return self._get_fallback_history(symbol)
    
    def _get_mock_historical_data(self, symbol: str) -> Dict:
        """Generate mock historical data"""
        import random
        
        return {
            'week_return': random.uniform(-10, 10),
            'month_return': random.uniform(-20, 20),
            'quarter_return': random.uniform(-30, 30),
            'ytd_return': random.uniform(-40, 40),
            'trend_direction': random.choice(['uptrend', 'downtrend', 'sideways']),
            'trend_days': random.randint(5, 60),
            'chart_pattern': random.choice(['flag', 'wedge', 'triangle', 'head_shoulders', 'double_bottom']),
            'vol_regime': random.choice(['low', 'medium', 'high']),
            'month_avg': random.uniform(-3, 3),
            'earnings_phase': random.choice(['pre_earnings', 'post_earnings', 'neutral']),
            'day_of_week': 'Wednesday',
            'similar_periods': random.randint(3, 15),
            'avg_outcome': random.uniform(-10, 15)
        }
    
    def _validate_pattern_analysis(self, analysis: Dict, symbol: str) -> Dict:
        """Validate pattern analysis"""
        
        if 'pattern_score' not in analysis:
            analysis['pattern_score'] = 0.0
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
        if 'dominant_pattern' not in analysis:
            analysis['dominant_pattern'] = 'consolidation'
            
        analysis['pattern_score'] = max(-1.0, min(1.0, analysis['pattern_score']))
        analysis['confidence'] = self._validate_confidence(analysis['confidence'])
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['symbol'] = symbol
        analysis['agent'] = self.name
        
        return analysis
    
    def _get_fallback_history(self, symbol: str) -> Dict:
        """Fallback historical analysis"""
        return {
            'pattern_score': 0.0,
            'confidence': 0.3,
            'dominant_pattern': 'consolidation',
            'historical_matches': [],
            'seasonality': {
                'monthly_bias': 'neutral',
                'weekly_pattern': 'average',
                'earnings_cycle': 'neutral'
            },
            'pattern_strength': 0.3,
            'time_horizon': 'medium',
            'key_levels': {
                'support': [0.0, 0.0],
                'resistance': [0.0, 0.0]
            },
            'pattern_insights': ['Historical data unavailable'],
            'error': 'Fallback historical analysis',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }