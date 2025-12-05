"""
Options Flow Analysis Agent
OpenAI Agents SDK v0.3.0 Implementation
"""
from typing import Dict, Any, List
from datetime import datetime
from backend.app.agents.base import BaseAgent
from backend.config.logging import get_agents_logger

logger = get_agents_logger()


class OptionsFlowAgent(BaseAgent):
    """AI agent specializing in options flow analysis"""
    
    def __init__(self, client):
        super().__init__(client, "Options Flow", "gpt-4o-mini")  # Fallback to OpenAI for now
        
    def _get_system_instructions(self) -> str:
        return """
You are an options flow analyst for the Neural Options Oracle++ system.

Your responsibilities:
1. Analyze options volume and open interest patterns
2. Detect unusual options activity and large block trades
3. Calculate gamma exposure and dealer positioning
4. Provide flow-based trading insights

Weight in system: 10% of final decision

OUTPUT FORMAT (JSON):
{
    "flow_score": float_between_-1_and_1,
    "confidence": float_between_0_and_1,
    "unusual_activity": boolean,
    "metrics": {
        "put_call_ratio": float,
        "call_volume": int,
        "put_volume": int,
        "total_volume": int,
        "avg_volume_ratio": float
    },
    "gamma_exposure": {
        "net_gamma": float,
        "gamma_level": "low|medium|high",
        "dealer_positioning": "long|short|neutral"
    },
    "large_trades": [
        {"type": "call|put", "strike": float, "volume": int, "premium": float}
    ],
    "flow_sentiment": "bullish|bearish|neutral",
    "key_insights": ["string1", "string2"]
}
"""
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for options flow response"""
        return {
            "type": "object",
            "properties": {
                "flow_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "unusual_activity": {"type": "boolean"},
                "metrics": {
                    "type": "object",
                    "properties": {
                        "put_call_ratio": {"type": "number", "minimum": 0.0},
                        "call_volume": {"type": "integer", "minimum": 0},
                        "put_volume": {"type": "integer", "minimum": 0},
                        "total_volume": {"type": "integer", "minimum": 0},
                        "avg_volume_ratio": {"type": "number", "minimum": 0.0}
                    },
                    "required": ["put_call_ratio", "call_volume", "put_volume", "total_volume", "avg_volume_ratio"],
                    "additionalProperties": False
                },
                "gamma_exposure": {
                    "type": "object",
                    "properties": {
                        "net_gamma": {"type": "number"},
                        "gamma_level": {"type": "string", "enum": ["low", "medium", "high"]},
                        "dealer_positioning": {"type": "string", "enum": ["long", "short", "neutral"]}
                    },
                    "required": ["net_gamma", "gamma_level", "dealer_positioning"],
                    "additionalProperties": False
                },
                "large_trades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["call", "put"]},
                            "strike": {"type": "number", "minimum": 0.0},
                            "volume": {"type": "integer", "minimum": 0},
                            "premium": {"type": "number", "minimum": 0.0}
                        },
                        "required": ["type", "strike", "volume", "premium"],
                        "additionalProperties": False
                    },
                    "maxItems": 10
                },
                "flow_sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                "key_insights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5
                }
            },
            "required": ["flow_score", "confidence", "unusual_activity", "metrics", "gamma_exposure", "large_trades", "flow_sentiment", "key_insights"],
            "additionalProperties": False
        }
    
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Analyze options flow for the symbol"""
        
        try:
            logger.info(f"Starting options flow analysis for {symbol}")
            
            # Get real options data
            from backend.app.services.market_data import market_data_manager
            market_data = await market_data_manager.get_comprehensive_data(symbol)
            options_data = market_data.get('options', {})
            
            messages = [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": f"""
Analyze options flow for {symbol} using REAL OPTIONS DATA:

VOLUME DATA:
- Call Volume: {options_data.get('total_call_volume', 0):,}
- Put Volume: {options_data.get('total_put_volume', 0):,}  
- Put/Call Ratio: {options_data.get('put_call_ratio', 1.0):.2f}
- Total Options: Calls: {options_data.get('call_count', 0)}, Puts: {options_data.get('put_count', 0)}

CURRENT MARKET:
- Stock Price: ${options_data.get('current_price', 0):.2f}
- Next Expiration: {options_data.get('expiration', 'N/A')}
- Data Source: {options_data.get('source', 'unknown')}

AT-THE-MONEY OPTIONS:
- ATM Call Options: {len(options_data.get('atm_calls', []))} contracts
- ATM Put Options: {len(options_data.get('atm_puts', []))} contracts

KEY METRICS:
- Volume Analysis: {'High' if options_data.get('total_call_volume', 0) + options_data.get('total_put_volume', 0) > 10000 else 'Normal'} volume
- Flow Bias: {'Bullish' if options_data.get('put_call_ratio', 1.0) < 0.8 else 'Bearish' if options_data.get('put_call_ratio', 1.0) > 1.2 else 'Neutral'}

ATM CALL DETAILS:
{self._format_options_list(options_data.get('atm_calls', [])[:3])}

ATM PUT DETAILS:
{self._format_options_list(options_data.get('atm_puts', [])[:3])}

Provide comprehensive flow analysis with this REAL options data.
                """}
            ]
            
            # Note: In production, this would use Gemini API
            # For now, using OpenAI as fallback
            response = await self._make_completion(
                messages, 
                temperature=0.3,
                response_schema=self._get_response_schema()
            )
            analysis = self._parse_json_response(response['content'])
            
            analysis = self._validate_flow_analysis(analysis, symbol)
            
            logger.info(f"Options flow analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Options flow analysis failed for {symbol}: {e}")
            return self._get_fallback_flow(symbol)
    
    def _format_options_list(self, options_list: List[Dict]) -> str:
        """Format options list for display"""
        if not options_list:
            return "No options data available"
        
        formatted = []
        for opt in options_list:
            strike = opt.get('strike', 0)
            volume = opt.get('volume', 0)
            oi = opt.get('openInterest', 0)
            iv = opt.get('impliedVolatility', 0)
            formatted.append(f"Strike ${strike:.0f}: Vol={volume}, OI={oi}, IV={iv:.1%}")
        
        return "\n".join(formatted) if formatted else "No detailed options data"
    
    def _validate_flow_analysis(self, analysis: Dict, symbol: str) -> Dict:
        """Validate flow analysis"""
        
        if 'flow_score' not in analysis:
            analysis['flow_score'] = 0.0
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
        if 'unusual_activity' not in analysis:
            analysis['unusual_activity'] = False
            
        analysis['flow_score'] = max(-1.0, min(1.0, analysis['flow_score']))
        analysis['confidence'] = self._validate_confidence(analysis['confidence'])
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['symbol'] = symbol
        analysis['agent'] = self.name
        
        return analysis
    
    def _get_fallback_flow(self, symbol: str) -> Dict:
        """Fallback flow analysis"""
        return {
            'flow_score': 0.0,
            'confidence': 0.3,
            'unusual_activity': False,
            'metrics': {
                'put_call_ratio': 1.0,
                'call_volume': 0,
                'put_volume': 0,
                'total_volume': 0,
                'avg_volume_ratio': 1.0
            },
            'gamma_exposure': {
                'net_gamma': 0.0,
                'gamma_level': 'low',
                'dealer_positioning': 'neutral'
            },
            'large_trades': [],
            'flow_sentiment': 'neutral',
            'key_insights': ['Options flow data unavailable'],
            'error': 'Fallback flow analysis',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }