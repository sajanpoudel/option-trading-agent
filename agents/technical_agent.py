"""
Technical Analysis Agent
OpenAI Agents SDK v0.3.0 Implementation
"""
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from config.logging import get_agents_logger

logger = get_agents_logger()


class TechnicalAnalysisAgent(BaseAgent):
    """AI agent specializing in technical analysis for options trading"""
    
    def __init__(self, client):
        super().__init__(client, "Technical Analysis", "gpt-4o")
        
    def _get_system_instructions(self) -> str:
        """Get system instructions for technical analysis"""
        return """
You are an expert technical analyst specializing in options trading for the Neural Options Oracle++ system.

Your core responsibilities:
1. Analyze stock price movements, trends, and technical indicators
2. Detect current market scenarios and recommend dynamic weight adjustments
3. Provide actionable insights for options trading strategies
4. Calculate confidence scores based on technical signal strength

DYNAMIC SCENARIO DETECTION & WEIGHTS:
You must identify one of these market scenarios and apply the specified indicator weights:

1. STRONG_UPTREND: Clear upward momentum
   - MA(30%), RSI(15%), BB(10%), MACD(25%), VWAP(20%)

2. STRONG_DOWNTREND: Clear downward momentum  
   - MA(30%), RSI(15%), BB(10%), MACD(25%), VWAP(20%)

3. RANGE_BOUND: Sideways movement, low volatility
   - MA(15%), RSI(25%), BB(30%), MACD(15%), VWAP(15%)

4. BREAKOUT: Price breaking key levels, high volume
   - MA(20%), RSI(15%), BB(30%), MACD(20%), VWAP(15%)

5. POTENTIAL_REVERSAL: Signs of trend change
   - MA(15%), RSI(25%), BB(20%), MACD(30%), VWAP(10%)

6. HIGH_VOLATILITY: Elevated volatility environment
   - Increase BB weight by +10%, reduce others proportionally

KEY TECHNICAL INDICATORS TO ANALYZE:
- Moving Averages (5, 10, 20, 50, 200 day)
- RSI (14-period)
- Bollinger Bands (20, 2)
- MACD (12, 26, 9)
- VWAP (Volume Weighted Average Price)
- Volume analysis
- Support/Resistance levels
- Chart patterns

CRITICAL: You must respond with valid JSON format only. Always return a JSON object with this exact structure:
{
    "scenario": "scenario_name",
    "weighted_score": float_between_-1_and_1,
    "confidence": float_between_0_and_1,
    "indicators": {
        "ma": {"signal": float, "weight": float, "details": "string"},
        "rsi": {"signal": float, "weight": float, "details": "string"},
        "bb": {"signal": float, "weight": float, "details": "string"},
        "macd": {"signal": float, "weight": float, "details": "string"},
        "vwap": {"signal": float, "weight": float, "details": "string"}
    },
    "support_resistance": {
        "support": [float, float],
        "resistance": [float, float]
    },
    "volatility": {
        "current": float,
        "percentile": float,
        "trend": "increasing|decreasing|stable"
    },
    "volume_analysis": {
        "relative_volume": float,
        "volume_trend": "string",
        "volume_score": float
    },
    "key_insights": ["string1", "string2", "string3"],
    "options_strategy_suggestion": "string"
}

IMPORTANT CALCULATION RULES:
- weighted_score = sum of (indicator_signal * indicator_weight) for all indicators
- All signals must be between -1 (strong bearish) and +1 (strong bullish)  
- Confidence reflects how aligned indicators are (high when most agree)
- Adjust weights dynamically based on detected scenario
- Consider volatility environment for strategy suggestions

Remember: You are the primary decision driver with 60% weight in the final system decision.
"""
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for technical analysis response"""
        return {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": ["STRONG_UPTREND", "STRONG_DOWNTREND", "RANGE_BOUND", "BREAKOUT", "POTENTIAL_REVERSAL", "HIGH_VOLATILITY"]
                },
                "weighted_score": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "indicators": {
                    "type": "object",
                    "properties": {
                        "ma": {
                            "type": "object",
                            "properties": {
                                "signal": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "details": {"type": "string"}
                            },
                            "required": ["signal", "weight", "details"],
                            "additionalProperties": False
                        },
                        "rsi": {
                            "type": "object",
                            "properties": {
                                "signal": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "details": {"type": "string"}
                            },
                            "required": ["signal", "weight", "details"],
                            "additionalProperties": False
                        },
                        "bb": {
                            "type": "object",
                            "properties": {
                                "signal": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "details": {"type": "string"}
                            },
                            "required": ["signal", "weight", "details"],
                            "additionalProperties": False
                        },
                        "macd": {
                            "type": "object",
                            "properties": {
                                "signal": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "details": {"type": "string"}
                            },
                            "required": ["signal", "weight", "details"],
                            "additionalProperties": False
                        },
                        "vwap": {
                            "type": "object",
                            "properties": {
                                "signal": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "details": {"type": "string"}
                            },
                            "required": ["signal", "weight", "details"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["ma", "rsi", "bb", "macd", "vwap"],
                    "additionalProperties": False
                },
                "support_resistance": {
                    "type": "object",
                    "properties": {
                        "support": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "resistance": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },
                    "required": ["support", "resistance"],
                    "additionalProperties": False
                },
                "volatility": {
                    "type": "object",
                    "properties": {
                        "current": {"type": "number", "minimum": 0.0},
                        "percentile": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        "trend": {"type": "string", "enum": ["increasing", "decreasing", "stable"]}
                    },
                    "required": ["current", "percentile", "trend"],
                    "additionalProperties": False
                },
                "volume_analysis": {
                    "type": "object",
                    "properties": {
                        "relative_volume": {"type": "number", "minimum": 0.0},
                        "volume_trend": {"type": "string"},
                        "volume_score": {"type": "number", "minimum": -1.0, "maximum": 1.0}
                    },
                    "required": ["relative_volume", "volume_trend", "volume_score"],
                    "additionalProperties": False
                },
                "key_insights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5
                },
                "options_strategy_suggestion": {"type": "string"}
            },
            "required": ["scenario", "weighted_score", "confidence", "indicators", "support_resistance", "volatility", "volume_analysis", "key_insights", "options_strategy_suggestion"],
            "additionalProperties": False
        }

    async def analyze(self, symbol: str, timeframe: str = "1d", **kwargs) -> Dict[str, Any]:
        """Analyze technical indicators for the given symbol"""
        
        try:
            logger.info(f"Starting technical analysis for {symbol}")
            
            # Get real market data
            from src.data.market_data_manager import market_data_manager
            market_data = await market_data_manager.get_comprehensive_data(symbol)
            
            # Extract technical indicators
            tech_data = market_data.get('technical', {})
            quote_data = market_data.get('quote', {})
            market_conditions = market_data.get('market_conditions', {})
            
            # Prepare the analysis prompt with real market data
            messages = [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": f"""
Analyze the technical indicators for {symbol} using the following REAL MARKET DATA:

PRICE DATA:
- Current Price: ${tech_data.get('current_price', 0):.2f}
- Daily Change: {tech_data.get('change_percent', 0):.2f}%
- Volume: {tech_data.get('current_volume', 0):,.0f} (Avg: {tech_data.get('avg_volume', 0):,.0f})
- Volume Ratio: {tech_data.get('volume_ratio', 1.0):.2f}x

TECHNICAL INDICATORS:
- RSI(14): {tech_data.get('rsi', 50):.1f}
- MACD: {tech_data.get('macd', 0):.3f} (Signal: {tech_data.get('macd_signal', 0):.3f})
- MACD Histogram: {tech_data.get('macd_histogram', 0):.3f}
- BB Position: {tech_data.get('bb_position', 0.5):.2f} (0=lower, 0.5=middle, 1=upper)
- VWAP: ${tech_data.get('vwap', 0):.2f}

MOVING AVERAGES:
- MA5: ${tech_data.get('ma5', 0):.2f}
- MA20: ${tech_data.get('ma20', 0):.2f}  
- MA50: ${tech_data.get('ma50', 0):.2f}
- MA200: ${tech_data.get('ma200', 0):.2f}

BOLLINGER BANDS:
- Upper: ${tech_data.get('bb_upper', 0):.2f}
- Middle: ${tech_data.get('bb_middle', 0):.2f}
- Lower: ${tech_data.get('bb_lower', 0):.2f}

VOLATILITY & MARKET CONDITIONS:
- 30-day HV: {tech_data.get('volatility', 25):.1f}%
- Market VIX: {market_conditions.get('vix', 20):.1f}
- Market Trend: {market_conditions.get('market_trend', 'neutral')}
- Volatility Regime: {market_conditions.get('volatility_regime', 'medium')}

SUPPORT/RESISTANCE:
- Resistance: ${tech_data.get('resistance', 0):.2f}
- Support: ${tech_data.get('support', 0):.2f}

Data Source: {tech_data.get('source', 'unknown')}

Please provide a comprehensive technical analysis with scenario detection and weighted scoring using this REAL market data.
                """}
            ]
            
            # Get analysis from GPT-4 with structured outputs
            response = await self._make_completion(
                messages, 
                temperature=0.3,
                response_schema=self._get_response_schema()
            )
            
            # Parse the response
            analysis = self._parse_json_response(response['content'])
            
            # Validate and enhance the analysis
            analysis = self._validate_analysis(analysis, symbol, tech_data)
            
            logger.info(f"Technical analysis completed for {symbol}: {analysis.get('scenario', 'unknown')} scenario")
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return self._get_fallback_analysis(symbol)
    
    def _get_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock market data for testing"""
        
        # In production, this would fetch real data from Alpaca API
        import random
        
        base_price = 150.0  # Mock base price
        
        return {
            'symbol': symbol,
            'current_price': base_price + random.uniform(-10, 10),
            'change_percent': random.uniform(-5, 5),
            'volume': random.randint(1000000, 5000000),
            'avg_volume': random.randint(1000000, 3000000),
            'rsi': random.uniform(20, 80),
            'macd': random.uniform(-2, 2),
            'macd_signal': random.uniform(-2, 2),
            'bb_position': random.uniform(0, 1),
            'vwap': base_price + random.uniform(-5, 5),
            'ma5': base_price + random.uniform(-3, 3),
            'ma20': base_price + random.uniform(-8, 8),
            'ma50': base_price + random.uniform(-15, 15),
            'ma200': base_price + random.uniform(-30, 30),
            'volatility': random.uniform(15, 45),
            'vix': random.uniform(12, 35)
        }
    
    def _validate_analysis(self, analysis: Dict, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Validate and enhance the analysis response"""
        
        # Ensure required fields exist
        if 'scenario' not in analysis:
            analysis['scenario'] = 'range_bound'
        
        if 'weighted_score' not in analysis:
            analysis['weighted_score'] = 0.0
        
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
        
        # Validate numeric ranges
        analysis['weighted_score'] = max(-1.0, min(1.0, analysis['weighted_score']))
        analysis['confidence'] = self._validate_confidence(analysis['confidence'])
        
        # Add metadata
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['symbol'] = symbol
        analysis['agent'] = self.name
        analysis['market_data_snapshot'] = market_data
        
        # Ensure indicators structure exists
        if 'indicators' not in analysis:
            analysis['indicators'] = {
                'ma': {'signal': 0.0, 'weight': 0.3, 'details': 'Analysis failed'},
                'rsi': {'signal': 0.0, 'weight': 0.15, 'details': 'Analysis failed'},
                'bb': {'signal': 0.0, 'weight': 0.1, 'details': 'Analysis failed'},
                'macd': {'signal': 0.0, 'weight': 0.25, 'details': 'Analysis failed'},
                'vwap': {'signal': 0.0, 'weight': 0.2, 'details': 'Analysis failed'}
            }
        
        return analysis
    
    def _get_fallback_analysis(self, symbol: str) -> Dict[str, Any]:
        """Provide fallback analysis when main analysis fails"""
        
        return {
            'scenario': 'range_bound',
            'weighted_score': 0.0,
            'confidence': 0.3,
            'indicators': {
                'ma': {'signal': 0.0, 'weight': 0.15, 'details': 'Analysis unavailable'},
                'rsi': {'signal': 0.0, 'weight': 0.25, 'details': 'Analysis unavailable'},
                'bb': {'signal': 0.0, 'weight': 0.3, 'details': 'Analysis unavailable'},
                'macd': {'signal': 0.0, 'weight': 0.15, 'details': 'Analysis unavailable'},
                'vwap': {'signal': 0.0, 'weight': 0.15, 'details': 'Analysis unavailable'}
            },
            'support_resistance': {
                'support': [0.0, 0.0],
                'resistance': [0.0, 0.0]
            },
            'volatility': {
                'current': 20.0,
                'percentile': 50.0,
                'trend': 'stable'
            },
            'volume_analysis': {
                'relative_volume': 1.0,
                'volume_trend': 'average',
                'volume_score': 0.0
            },
            'key_insights': ['Technical analysis temporarily unavailable'],
            'options_strategy_suggestion': 'Wait for better signal',
            'error': 'Fallback analysis due to system error',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }