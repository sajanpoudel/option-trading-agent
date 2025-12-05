"""
Risk Management Agent
OpenAI Agents SDK v0.3.0 Implementation
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from backend.app.agents.base import BaseAgent
from backend.config.logging import get_agents_logger

logger = get_agents_logger()


class RiskManagementAgent(BaseAgent):
    """AI agent specializing in risk management and strike selection"""
    
    def __init__(self, client):
        super().__init__(client, "Risk Management", "gpt-4o")
        
    def _get_system_instructions(self) -> str:
        return """
You are a risk management specialist for the Neural Options Oracle++ system.

Your responsibilities:
1. Assess portfolio and position-level risk
2. Recommend appropriate option strikes based on user risk profile
3. Calculate position sizing and risk metrics
4. Provide risk-adjusted recommendations

Risk Profiles:
- CONSERVATIVE: Delta 0.15-0.35, Max loss 2% of account
- MODERATE: Delta 0.25-0.55, Max loss 5% of account  
- AGGRESSIVE: Delta 0.45-0.85, Max loss 10% of account

OUTPUT FORMAT (JSON):
{
    "risk_assessment": {
        "overall_risk": "low|medium|high",
        "risk_score": float_between_0_and_1,
        "key_risks": ["string1", "string2"]
    },
    "position_sizing": {
        "recommended_contracts": int,
        "max_loss_dollar": float,
        "max_loss_percent": float,
        "risk_reward_ratio": float
    },
    "strike_recommendations": [
        {
            "strike": float,
            "option_type": "call|put",
            "expiration": "YYYY-MM-DD",
            "delta": float,
            "probability_profit": float,
            "max_loss": float,
            "max_gain": float,
            "risk_level": "low|medium|high"
        }
    ],
    "risk_mitigation": {
        "stop_loss": float,
        "take_profit": float,
        "time_decay_warning": boolean,
        "volatility_risk": "low|medium|high"
    },
    "portfolio_impact": {
        "correlation_risk": float,
        "concentration_risk": float,
        "diversification_score": float
    }
}
"""
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for risk management response"""
        return {
            "type": "object",
            "properties": {
                "risk_assessment": {
                    "type": "object",
                    "properties": {
                        "overall_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                        "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "key_risks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        }
                    },
                    "required": ["overall_risk", "risk_score", "key_risks"],
                    "additionalProperties": False
                },
                "position_sizing": {
                    "type": "object",
                    "properties": {
                        "recommended_contracts": {"type": "integer", "minimum": 1},
                        "max_loss_dollar": {"type": "number", "minimum": 0.0},
                        "max_loss_percent": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        "risk_reward_ratio": {"type": "number", "minimum": 0.0}
                    },
                    "required": ["recommended_contracts", "max_loss_dollar", "max_loss_percent", "risk_reward_ratio"],
                    "additionalProperties": False
                },
                "strike_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "strike": {"type": "number", "minimum": 0.0},
                            "option_type": {"type": "string", "enum": ["call", "put"]},
                            "expiration": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
                            "delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                            "probability_profit": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "max_loss": {"type": "number", "minimum": 0.0},
                            "max_gain": {"type": "number", "minimum": 0.0},
                            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                        },
                        "required": ["strike", "option_type", "expiration", "delta", "probability_profit", "max_loss", "max_gain", "risk_level"],
                        "additionalProperties": False
                    },
                    "minItems": 1,
                    "maxItems": 5
                },
                "risk_mitigation": {
                    "type": "object",
                    "properties": {
                        "stop_loss": {"type": "number", "minimum": 0.0},
                        "take_profit": {"type": "number", "minimum": 0.0},
                        "time_decay_warning": {"type": "boolean"},
                        "volatility_risk": {"type": "string", "enum": ["low", "medium", "high"]}
                    },
                    "required": ["stop_loss", "take_profit", "time_decay_warning", "volatility_risk"],
                    "additionalProperties": False
                },
                "portfolio_impact": {
                    "type": "object",
                    "properties": {
                        "correlation_risk": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "concentration_risk": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "diversification_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["correlation_risk", "concentration_risk", "diversification_score"],
                    "additionalProperties": False
                }
            },
            "required": ["risk_assessment", "position_sizing", "strike_recommendations", "risk_mitigation", "portfolio_impact"],
            "additionalProperties": False
        }
    
    async def recommend_strikes(
        self, 
        signal: Dict[str, Any], 
        user_risk_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recommend option strikes based on signal and risk profile"""
        
        try:
            logger.info(f"Generating strike recommendations for {signal.get('direction', 'UNKNOWN')}")
            
            # Mock current market data
            mock_data = self._get_mock_options_data(signal.get('symbol', 'UNKNOWN'))
            risk_level = user_risk_profile.get('risk_level', 'moderate')
            
            messages = [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": f"""
Generate risk-appropriate strike recommendations:

TRADING SIGNAL:
- Direction: {signal.get('direction', 'HOLD')}
- Confidence: {signal.get('confidence', 0.5):.2f}
- Strategy Type: {signal.get('strategy_type', 'neutral')}

USER RISK PROFILE:
- Risk Level: {risk_level}
- Account Size: ${user_risk_profile.get('account_size', 100000)}
- Max Position Size: {user_risk_profile.get('max_position_percent', 5)}%

CURRENT MARKET DATA:
- Current Price: ${mock_data['current_price']:.2f}
- IV Rank: {mock_data['iv_rank']:.0f}%
- Days to Earnings: {mock_data['days_to_earnings']}

AVAILABLE STRIKES:
{self._format_options_chain(mock_data['options_chain'])}

Recommend 3-5 appropriate strikes with full risk analysis.
                """}
            ]
            
            response = await self._make_completion(
                messages, 
                temperature=0.3,
                response_schema=self._get_response_schema()
            )
            analysis = self._parse_json_response(response['content'])
            
            # Extract just the strike recommendations
            recommendations = analysis.get('strike_recommendations', [])
            
            # Validate recommendations
            recommendations = self._validate_strike_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} strike recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Strike recommendation failed: {e}")
            return self._get_fallback_strikes(signal, user_risk_profile)
    
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """General risk analysis (not used in main flow but required by base class)"""
        
        return {
            'risk_score': 0.5,
            'confidence': 0.7,
            'risk_level': 'medium',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }
    
    def _get_mock_options_data(self, symbol: str) -> Dict:
        """Generate mock options data"""
        import random
        
        current_price = 150.0 + random.uniform(-20, 20)
        
        # Generate mock options chain
        options_chain = []
        for i in range(-5, 6):  # 11 strikes around current price
            strike = current_price + (i * 5)  # $5 intervals
            
            call_data = {
                'strike': strike,
                'type': 'call',
                'delta': max(0.05, min(0.95, 0.5 + (i * 0.1))),
                'premium': max(0.5, abs(i * 2) + random.uniform(1, 5)),
                'iv': random.uniform(0.2, 0.6)
            }
            
            put_data = {
                'strike': strike,
                'type': 'put', 
                'delta': max(-0.95, min(-0.05, -0.5 - (i * 0.1))),
                'premium': max(0.5, abs(i * 2) + random.uniform(1, 5)),
                'iv': random.uniform(0.2, 0.6)
            }
            
            options_chain.extend([call_data, put_data])
        
        return {
            'current_price': current_price,
            'iv_rank': random.uniform(10, 90),
            'days_to_earnings': random.randint(5, 45),
            'options_chain': options_chain
        }
    
    def _format_options_chain(self, chain: List[Dict]) -> str:
        """Format options chain for prompt"""
        
        formatted = []
        for opt in chain[:10]:  # Limit to first 10 for brevity
            formatted.append(
                f"{opt['strike']:.0f} {opt['type']}: "
                f"${opt['premium']:.2f} (δ={opt['delta']:.2f}, IV={opt['iv']:.1%})"
            )
        
        return "\n".join(formatted)
    
    def _validate_strike_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Validate strike recommendations"""
        
        validated = []
        for rec in recommendations:
            if isinstance(rec, dict):
                # Ensure required fields with safe float conversion
                def safe_float(value, default):
                    try:
                        if isinstance(value, str) and value.lower() in ['unlimited', 'infinite', 'inf']:
                            return 10000.0  # Use large number for unlimited
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                validated_rec = {
                    'strike': safe_float(rec.get('strike'), 150.0),
                    'option_type': rec.get('option_type', 'call'),
                    'expiration': rec.get('expiration', (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')),
                    'delta': max(-1.0, min(1.0, safe_float(rec.get('delta'), 0.5))),
                    'probability_profit': max(0.0, min(1.0, safe_float(rec.get('probability_profit'), 0.5))),
                    'max_loss': abs(safe_float(rec.get('max_loss'), 500.0)),
                    'max_gain': abs(safe_float(rec.get('max_gain'), 1000.0)),
                    'risk_level': rec.get('risk_level', 'medium'),
                    'premium': abs(safe_float(rec.get('premium'), 5.0)),
                    'contracts': max(1, int(rec.get('contracts', 1)))
                }
                validated.append(validated_rec)
        
        return validated[:5]  # Limit to 5 recommendations
    
    def _get_fallback_strikes(self, signal: Dict, user_profile: Dict) -> List[Dict]:
        """Fallback strike recommendations"""
        
        direction = signal.get('direction', 'HOLD')
        risk_level = user_profile.get('risk_level', 'moderate')
        
        # Simple fallback based on direction
        if direction in ['BUY', 'STRONG_BUY']:
            option_type = 'call'
            delta_range = (0.45, 0.65) if risk_level == 'aggressive' else (0.25, 0.45)
        elif direction in ['SELL', 'STRONG_SELL']:
            option_type = 'put'
            delta_range = (-0.65, -0.45) if risk_level == 'aggressive' else (-0.45, -0.25)
        else:
            option_type = 'call'
            delta_range = (0.35, 0.55)
        
        return [
            {
                'strike': 150.0,
                'option_type': option_type,
                'expiration': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'delta': (delta_range[0] + delta_range[1]) / 2,
                'probability_profit': 0.5,
                'max_loss': 500.0,
                'max_gain': 1000.0,
                'risk_level': risk_level,
                'premium': 5.0,
                'contracts': 1,
                'note': 'Fallback recommendation'
            }
        ]