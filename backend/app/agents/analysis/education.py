"""
Education Agent
OpenAI Agents SDK v0.3.0 Implementation
"""
from typing import Dict, Any
from datetime import datetime
from backend.app.agents.base import BaseAgent
from backend.config.logging import get_agents_logger

logger = get_agents_logger()


class EducationAgent(BaseAgent):
    """AI agent specializing in educational content generation"""
    
    def __init__(self, client):
        super().__init__(client, "Education", "gpt-4o-mini")
        
    def _get_system_instructions(self) -> str:
        return """
You are an educational content specialist for the Neural Options Oracle++ system.

Your responsibilities:
1. Generate personalized educational explanations for trading decisions
2. Create interactive learning content based on user actions
3. Explain complex options concepts in simple terms
4. Provide contextual learning opportunities

OUTPUT FORMAT (JSON):
{
    "explanation": {
        "why_this_signal": "string",
        "market_context": "string", 
        "risk_factors": ["string1", "string2"],
        "learning_points": ["string1", "string2"]
    },
    "educational_content": {
        "concept": "string",
        "simple_explanation": "string",
        "example": "string",
        "quiz_question": {
            "question": "string",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A|B|C|D",
            "explanation": "string"
        }
    },
    "next_steps": {
        "recommended_reading": ["string1", "string2"],
        "practice_exercises": ["string1", "string2"],
        "skill_level": "beginner|intermediate|advanced"
    },
    "confidence_building": {
        "success_probability": "string",
        "similar_examples": ["string1", "string2"],
        "what_to_watch": ["string1", "string2"]
    }
}
"""
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for education response"""
        return {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "object",
                    "properties": {
                        "why_this_signal": {"type": "string"},
                        "market_context": {"type": "string"},
                        "risk_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        },
                        "learning_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        }
                    },
                    "required": ["why_this_signal", "market_context", "risk_factors", "learning_points"],
                    "additionalProperties": False
                },
                "educational_content": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "simple_explanation": {"type": "string"},
                        "example": {"type": "string"},
                        "quiz_question": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 4,
                                    "maxItems": 4
                                },
                                "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                                "explanation": {"type": "string"}
                            },
                            "required": ["question", "options", "correct_answer", "explanation"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["concept", "simple_explanation", "example", "quiz_question"],
                    "additionalProperties": False
                },
                "next_steps": {
                    "type": "object",
                    "properties": {
                        "recommended_reading": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        },
                        "practice_exercises": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        },
                        "skill_level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]}
                    },
                    "required": ["recommended_reading", "practice_exercises", "skill_level"],
                    "additionalProperties": False
                },
                "confidence_building": {
                    "type": "object",
                    "properties": {
                        "success_probability": {"type": "string"},
                        "similar_examples": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        },
                        "what_to_watch": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        }
                    },
                    "required": ["success_probability", "similar_examples", "what_to_watch"],
                    "additionalProperties": False
                }
            },
            "required": ["explanation", "educational_content", "next_steps", "confidence_building"],
            "additionalProperties": False
        }
    
    async def generate_explanation(
        self, 
        symbol: str, 
        signal: Dict[str, Any], 
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate educational explanation for trading decision"""
        
        try:
            logger.info(f"Generating educational content for {symbol} signal")
            
            messages = [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": f"""
Generate educational content explaining this trading decision:

TRADING SIGNAL:
- Symbol: {symbol}
- Direction: {signal.get('direction', 'HOLD')}
- Confidence: {signal.get('confidence', 0.5):.2f}
- Strategy: {signal.get('strategy_type', 'neutral')}
- Reasoning: {signal.get('reasoning', 'Analysis completed')}

AGENT ANALYSIS SUMMARY:
- Technical: {self._summarize_technical(agent_results.get('technical', {}))}
- Sentiment: {self._summarize_sentiment(agent_results.get('sentiment', {}))}
- Flow: {self._summarize_flow(agent_results.get('flow', {}))}
- Historical: {self._summarize_history(agent_results.get('history', {}))}

Create educational content that explains WHY this decision was made and what the user can learn from it.
                """}
            ]
            
            response = await self._make_completion(
                messages, 
                temperature=0.6,
                response_schema=self._get_response_schema()
            )
            explanation = self._parse_json_response(response['content'])
            
            # Validate and enhance
            explanation = self._validate_explanation(explanation, symbol, signal)
            
            logger.info(f"Educational content generated for {symbol}")
            return explanation
            
        except Exception as e:
            logger.error(f"Educational content generation failed for {symbol}: {e}")
            return self._get_fallback_explanation(symbol, signal)
    
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """General analysis method (required by base class)"""
        return {
            'educational_readiness': True,
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }
    
    def _summarize_technical(self, technical_data: Dict) -> str:
        """Summarize technical analysis for education"""
        scenario = technical_data.get('scenario', 'unknown')
        score = technical_data.get('weighted_score', 0)
        return f"{scenario} scenario with {score:.2f} technical score"
    
    def _summarize_sentiment(self, sentiment_data: Dict) -> str:
        """Summarize sentiment analysis for education"""
        score = sentiment_data.get('aggregate_score', 0)
        trend = sentiment_data.get('sentiment_trend', 'stable')
        return f"{score:.2f} sentiment score, {trend} trend"
    
    def _summarize_flow(self, flow_data: Dict) -> str:
        """Summarize flow analysis for education"""
        unusual = flow_data.get('unusual_activity', False)
        sentiment = flow_data.get('flow_sentiment', 'neutral')
        return f"{sentiment} flow sentiment, unusual activity: {unusual}"
    
    def _summarize_history(self, history_data: Dict) -> str:
        """Summarize historical analysis for education"""
        pattern = history_data.get('dominant_pattern', 'unknown')
        score = history_data.get('pattern_score', 0)
        return f"{pattern} pattern with {score:.2f} strength"
    
    def _validate_explanation(self, explanation: Dict, symbol: str, signal: Dict) -> Dict:
        """Validate educational explanation"""
        
        # Ensure basic structure
        if 'explanation' not in explanation:
            explanation['explanation'] = {
                'why_this_signal': f"Analysis suggests {signal.get('direction', 'HOLD')} for {symbol}",
                'market_context': 'Based on comprehensive multi-agent analysis',
                'risk_factors': ['Market volatility', 'Timing risk'],
                'learning_points': ['Always consider multiple indicators', 'Risk management is crucial']
            }
        
        if 'educational_content' not in explanation:
            explanation['educational_content'] = {
                'concept': 'Options Trading Signals',
                'simple_explanation': 'Trading signals help identify potential opportunities based on market analysis.',
                'example': f'A {signal.get("direction", "HOLD")} signal means the analysis suggests this directional bias.',
                'quiz_question': {
                    'question': 'What should you always consider before making a trade?',
                    'options': ['A) Risk management', 'B) Only the signal direction', 'C) Market news only', 'D) Your emotions'],
                    'correct_answer': 'A',
                    'explanation': 'Risk management should always be the primary consideration in trading.'
                }
            }
        
        # Add metadata
        explanation['timestamp'] = datetime.now().isoformat()
        explanation['symbol'] = symbol
        explanation['agent'] = self.name
        
        return explanation
    
    def _get_fallback_explanation(self, symbol: str, signal: Dict) -> Dict:
        """Fallback educational explanation"""
        
        direction = signal.get('direction', 'HOLD')
        
        return {
            'explanation': {
                'why_this_signal': f'The analysis resulted in a {direction} signal for {symbol} based on multiple factors.',
                'market_context': 'The decision combines technical indicators, sentiment analysis, options flow, and historical patterns.',
                'risk_factors': ['Market volatility can change rapidly', 'Options have time decay risk', 'All trades involve potential losses'],
                'learning_points': ['Diversification is important', 'Never risk more than you can afford to lose', 'Always have an exit strategy']
            },
            'educational_content': {
                'concept': 'Multi-Agent Trading Analysis',
                'simple_explanation': 'Our system uses multiple AI agents to analyze different aspects of the market before making recommendations.',
                'example': 'Like having a team of experts each specializing in different areas of market analysis.',
                'quiz_question': {
                    'question': 'Why is it better to use multiple analysis methods?',
                    'options': ['A) More comprehensive view', 'B) Faster results', 'C) Lower costs', 'D) Guaranteed profits'],
                    'correct_answer': 'A',
                    'explanation': 'Multiple analysis methods provide a more comprehensive and reliable view of market conditions.'
                }
            },
            'next_steps': {
                'recommended_reading': ['Options Trading Basics', 'Risk Management Fundamentals'],
                'practice_exercises': ['Paper trading with small positions', 'Track your prediction accuracy'],
                'skill_level': 'beginner'
            },
            'confidence_building': {
                'success_probability': f'{signal.get("confidence", 0.5)*100:.0f}% confidence based on analysis',
                'similar_examples': ['Similar patterns have occurred in the past', 'Historical success rate varies by market conditions'],
                'what_to_watch': ['Price action at key levels', 'Changes in volume', 'News and earnings announcements']
            },
            'error': 'Fallback educational content',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'agent': self.name
        }