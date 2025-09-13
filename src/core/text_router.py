"""
Intelligent Text Router for Neural Options Oracle++
Uses OpenAI function calling to intelligently route user text to appropriate agents
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from openai import AsyncOpenAI
from config.logging import get_agents_logger
from config.settings import settings

logger = get_agents_logger()


class IntentType(Enum):
    """Types of user intents"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    OPTIONS_FLOW = "options_flow"
    HISTORICAL_ANALYSIS = "historical_analysis"
    TRADING_SIGNALS = "trading_signals"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    RISK_ASSESSMENT = "risk_assessment"
    EDUCATION = "education"
    CHAT_QUERY = "chat_query"
    UNKNOWN = "unknown"


@dataclass
class RoutedRequest:
    """Structured request after text routing"""
    intent: IntentType
    symbol: Optional[str]
    timeframe: Optional[str]
    parameters: Dict[str, Any]
    confidence: float
    original_text: str
    formatted_query: str
    api_endpoint: str
    agent_methods: List[str]


class TextRouter:
    """Intelligent text router using OpenAI function calling for agent selection"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        
        # Define available agents and their capabilities
        self.available_agents = {
            "technical_analysis": {
                "name": "Technical Analysis Agent",
                "description": "Analyzes technical indicators, charts, patterns, and price action",
                "capabilities": ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Support/Resistance", "Chart Patterns", "Trend Analysis"],
                "endpoint": "/api/v1/technical/{symbol}",
                "methods": ["technical"]
            },
            "sentiment_analysis": {
                "name": "Sentiment Analysis Agent", 
                "description": "Analyzes market sentiment from news, social media, and market psychology",
                "capabilities": ["News Sentiment", "Social Media Analysis", "Market Psychology", "Sentiment Trends"],
                "endpoint": "/api/v1/agents/{symbol}",
                "methods": ["sentiment"]
            },
            "options_flow": {
                "name": "Options Flow Agent",
                "description": "Analyzes options volume, flow, and unusual activity",
                "capabilities": ["Options Volume", "Put/Call Ratios", "Unusual Activity", "Greeks Analysis", "Flow Patterns"],
                "endpoint": "/api/v1/agents/{symbol}",
                "methods": ["flow"]
            },
            "historical_analysis": {
                "name": "Historical Pattern Agent",
                "description": "Analyzes historical patterns, seasonality, and performance",
                "capabilities": ["Historical Patterns", "Seasonality", "Earnings Cycles", "Performance Analysis"],
                "endpoint": "/api/v1/agents/{symbol}",
                "methods": ["history"]
            },
            "trading_signals": {
                "name": "Trading Signals Agent",
                "description": "Generates buy/sell/hold signals with risk management",
                "capabilities": ["Signal Generation", "Risk Assessment", "Position Sizing", "Entry/Exit Points"],
                "endpoint": "/api/v1/signals/{symbol}",
                "methods": ["technical", "sentiment", "flow", "history", "decision_engine"]
            },
            "portfolio_management": {
                "name": "Portfolio Management Agent",
                "description": "Manages portfolio positions, allocation, and performance",
                "capabilities": ["Portfolio Analysis", "Position Management", "Allocation", "Performance Tracking"],
                "endpoint": "/api/v1/portfolio/",
                "methods": ["portfolio"]
            },
            "risk_assessment": {
                "name": "Risk Assessment Agent",
                "description": "Assesses risk levels and provides risk management strategies",
                "capabilities": ["Risk Analysis", "Hedging Strategies", "Risk Metrics", "Portfolio Risk"],
                "endpoint": "/api/v1/analysis/{symbol}",
                "methods": ["risk"]
            },
            "education": {
                "name": "Education Agent",
                "description": "Provides educational content and explanations",
                "capabilities": ["Concept Explanations", "Strategy Education", "Market Education", "Tutorials"],
                "endpoint": "/api/v1/education/",
                "methods": ["education"]
            }
        }
        
        # Define function schema for OpenAI
        self.function_schema = {
            "name": "route_user_query",
            "description": "Route user query to appropriate agents and extract parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_intent": {
                        "type": "string",
                        "enum": list(self.available_agents.keys()) + ["chat_query"],
                        "description": "Primary intent of the user query"
                    },
                    "secondary_intents": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(self.available_agents.keys())},
                        "description": "Secondary intents that might be relevant"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol if mentioned (e.g., AAPL, TSLA, NVDA)"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1d", "1w", "1m", "3m", "1y", "5y"],
                        "description": "Timeframe for analysis if specified"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in the routing decision (0-1)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Additional parameters extracted from the query",
                        "properties": {
                            "risk_level": {
                                "type": "string",
                                "enum": ["conservative", "moderate", "aggressive"],
                                "description": "Risk level if mentioned"
                            },
                            "position_size": {
                                "type": "string",
                                "description": "Position size if mentioned"
                            },
                            "indicators": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific technical indicators mentioned"
                            },
                            "option_type": {
                                "type": "string",
                                "enum": ["call", "put"],
                                "description": "Option type if mentioned"
                            },
                            "strike_price": {
                                "type": "number",
                                "description": "Strike price if mentioned"
                            },
                            "educational_topic": {
                                "type": "string",
                                "description": "Educational topic if asking for explanation"
                            }
                        }
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this routing was chosen"
                    }
                },
                "required": ["primary_intent", "confidence", "reasoning"]
            }
        }
    
    async def route_text(self, user_text: str, context: Dict[str, Any] = None) -> RoutedRequest:
        """
        Main routing method using OpenAI function calling to transform user text into structured request
        
        Args:
            user_text: Raw user input text
            context: Additional context (selected stock, user preferences, etc.)
            
        Returns:
            RoutedRequest: Structured request with intent, parameters, and routing info
        """
        logger.info(f"🔄 Routing text with OpenAI: '{user_text}'")
        
        if not self.openai_client:
            logger.warning("OpenAI client not available, using fallback routing")
            return self._fallback_route(user_text, context)
        
        try:
            # Prepare system message with agent information
            system_message = self._create_system_message()
            
            # Prepare user message with context
            user_message = self._create_user_message(user_text, context)
            
            # Call OpenAI with function calling
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                tools=[{"type": "function", "function": self.function_schema}],
                tool_choice={"type": "function", "function": {"name": "route_user_query"}},
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract function call result
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"OpenAI routing result: {function_args}")
            
            # Convert to RoutedRequest
            return self._convert_to_routed_request(user_text, function_args, context)
            
        except Exception as e:
            logger.error(f"OpenAI routing failed: {e}, using fallback")
            return self._fallback_route(user_text, context)
    
    def _create_system_message(self) -> str:
        """Create system message for OpenAI routing"""
        agent_descriptions = []
        for agent_id, agent_info in self.available_agents.items():
            capabilities = ", ".join(agent_info["capabilities"])
            agent_descriptions.append(f"- {agent_id}: {agent_info['description']} (Capabilities: {capabilities})")
        
        return f"""You are an intelligent routing system for the Neural Options Oracle++ trading platform.

Available Agents:
{chr(10).join(agent_descriptions)}

Your task is to analyze user queries and route them to the most appropriate agent(s). Consider:

1. Primary intent - what is the user primarily asking for?
2. Secondary intents - are there additional relevant agents?
3. Stock symbol extraction - extract any mentioned stock symbols
4. Timeframe - extract any mentioned time periods
5. Parameters - extract specific parameters like risk level, indicators, etc.

Examples:
- "Analyze TSLA" → technical_analysis for TSLA
- "What's the sentiment for NVDA?" → sentiment_analysis for NVDA  
- "Should I buy AAPL?" → trading_signals for AAPL
- "Explain RSI" → education with educational_topic: RSI
- "Options flow for MSFT" → options_flow for MSFT
- "Portfolio analysis" → portfolio_management
- "How risky is TSLA?" → risk_assessment for TSLA

Be precise and confident in your routing decisions."""
    
    def _create_user_message(self, user_text: str, context: Dict[str, Any] = None) -> str:
        """Create user message with context"""
        message = f"User query: '{user_text}'"
        
        if context:
            if context.get('selectedStock'):
                message += f"\nContext: Currently selected stock is {context['selectedStock']}"
            if context.get('user_id'):
                message += f"\nUser ID: {context['user_id']}"
            if context.get('session_id'):
                message += f"\nSession ID: {context['session_id']}"
        
        return message
    
    def _convert_to_routed_request(self, user_text: str, function_args: Dict[str, Any], context: Dict[str, Any] = None) -> RoutedRequest:
        """Convert OpenAI function call result to RoutedRequest"""
        
        # Get primary intent
        primary_intent_str = function_args.get('primary_intent', 'chat_query')
        try:
            primary_intent = IntentType(primary_intent_str)
        except ValueError:
            primary_intent = IntentType.CHAT_QUERY
        
        # Get symbol (from function args or context)
        symbol = function_args.get('symbol')
        if not symbol and context and context.get('selectedStock'):
            symbol = context['selectedStock']
        
        # Get timeframe
        timeframe = function_args.get('timeframe')
        
        # Get parameters
        parameters = function_args.get('parameters', {})
        
        # Get confidence
        confidence = function_args.get('confidence', 0.8)
        
        # Format query for agent
        formatted_query = self._format_query_for_agent(user_text, primary_intent, symbol, parameters)
        
        # Determine API endpoint and agent methods
        api_endpoint, agent_methods = self._determine_routing(primary_intent, symbol)
        
        return RoutedRequest(
            intent=primary_intent,
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters,
            confidence=confidence,
            original_text=user_text,
            formatted_query=formatted_query,
            api_endpoint=api_endpoint,
            agent_methods=agent_methods
        )
    
    def _fallback_route(self, user_text: str, context: Dict[str, Any] = None) -> RoutedRequest:
        """Fallback routing when OpenAI is not available"""
        logger.info("Using fallback routing")
        
        # Simple pattern matching fallback
        text_lower = user_text.lower()
        
        # Determine intent
        if any(word in text_lower for word in ['technical', 'chart', 'rsi', 'macd', 'indicators']):
            intent = IntentType.TECHNICAL_ANALYSIS
        elif any(word in text_lower for word in ['sentiment', 'news', 'social', 'feeling']):
            intent = IntentType.SENTIMENT_ANALYSIS
        elif any(word in text_lower for word in ['options', 'calls', 'puts', 'flow']):
            intent = IntentType.OPTIONS_FLOW
        elif any(word in text_lower for word in ['buy', 'sell', 'trade', 'signal']):
            intent = IntentType.TRADING_SIGNALS
        elif any(word in text_lower for word in ['explain', 'what is', 'how does', 'teach']):
            intent = IntentType.EDUCATION
        elif any(word in text_lower for word in ['portfolio', 'positions', 'holdings']):
            intent = IntentType.PORTFOLIO_MANAGEMENT
        elif any(word in text_lower for word in ['risk', 'safe', 'conservative']):
            intent = IntentType.RISK_ASSESSMENT
        else:
            intent = IntentType.CHAT_QUERY
        
        # Extract symbol
        symbol = self._extract_symbol_fallback(user_text, context)
        
        # Format query
        formatted_query = self._format_query_for_agent(user_text, intent, symbol, {})
        
        # Determine routing
        api_endpoint, agent_methods = self._determine_routing(intent, symbol)
        
        return RoutedRequest(
            intent=intent,
            symbol=symbol,
            timeframe=None,
            parameters={},
            confidence=0.6,  # Lower confidence for fallback
            original_text=user_text,
            formatted_query=formatted_query,
            api_endpoint=api_endpoint,
            agent_methods=agent_methods
        )
    
    def _extract_symbol_fallback(self, text: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Fallback symbol extraction"""
        # Check context first
        if context and context.get('selectedStock'):
            return context['selectedStock']
        
        # Simple pattern matching
        import re
        symbol_pattern = r'\b([A-Z]{2,5})\b'
        matches = re.findall(symbol_pattern, text.upper())
        
        excluded = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'USE', 'WAY', 'WHY', 'AIR', 'BAD', 'BIG', 'BOX', 'CAR', 'CAT', 'CUT', 'DOG', 'EAR', 'EYE', 'FAR', 'FUN', 'GOT', 'HOT', 'JOB', 'LOT', 'MAN', 'OWN', 'PUT', 'RUN', 'SIT', 'SUN', 'TOO', 'TOP', 'WIN', 'YES', 'YET'}
        
        for match in matches:
            if match not in excluded:
                return match
        
        return None
    
    def _extract_symbol(self, text: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Extract stock symbol from text"""
        # Check context first
        if context and context.get('selectedStock'):
            return context['selectedStock']
        
        # Extract from text
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                symbol = match if isinstance(match, str) else match[0]
                if symbol not in self.excluded_words and len(symbol) >= 2:
                    return symbol
        
        return None
    
    def _classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """Classify user intent with confidence score"""
        text_lower = text.lower()
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches += 1
                    score += 1.0
            
            # Normalize score
            scores[intent] = min(score / len(patterns), 1.0) if patterns else 0.0
        
        # Find best match
        best_intent = max(scores.items(), key=lambda x: x[1])
        
        # If no clear intent, default to chat query
        if best_intent[1] < 0.3:
            return IntentType.CHAT_QUERY, 0.5
        
        return best_intent[0], best_intent[1]
    
    def _extract_timeframe(self, text: str) -> Optional[str]:
        """Extract timeframe from text"""
        text_lower = text.lower()
        
        for timeframe, patterns in self.timeframe_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return timeframe
        
        return None
    
    def _extract_parameters(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """Extract specific parameters based on intent"""
        parameters = {}
        text_lower = text.lower()
        
        if intent == IntentType.TRADING_SIGNALS:
            # Extract position size, risk level, etc.
            position_match = re.search(r'\b(\d+)\s*(shares?|contracts?|%)\b', text_lower)
            if position_match:
                parameters['position_size'] = position_match.group(1)
            
            risk_match = re.search(r'\b(conservative|moderate|aggressive)\b', text_lower)
            if risk_match:
                parameters['risk_level'] = risk_match.group(1)
        
        elif intent == IntentType.TECHNICAL_ANALYSIS:
            # Extract specific indicators
            indicators = []
            if re.search(r'\brsi\b', text_lower):
                indicators.append('rsi')
            if re.search(r'\bmacd\b', text_lower):
                indicators.append('macd')
            if re.search(r'\bbollinger\b', text_lower):
                indicators.append('bollinger_bands')
            if re.search(r'\bmoving average\b', text_lower):
                indicators.append('moving_averages')
            
            if indicators:
                parameters['indicators'] = indicators
        
        elif intent == IntentType.OPTIONS_FLOW:
            # Extract options-specific parameters
            if re.search(r'\bcalls?\b', text_lower):
                parameters['option_type'] = 'call'
            elif re.search(r'\bputs?\b', text_lower):
                parameters['option_type'] = 'put'
            
            strike_match = re.search(r'\bstrike\s*(\d+)\b', text_lower)
            if strike_match:
                parameters['strike_price'] = float(strike_match.group(1))
        
        return parameters
    
    def _format_query_for_agent(self, text: str, intent: IntentType, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query specifically for the target agent"""
        
        if intent == IntentType.TECHNICAL_ANALYSIS:
            return self._format_technical_query(text, symbol, parameters)
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            return self._format_sentiment_query(text, symbol, parameters)
        elif intent == IntentType.OPTIONS_FLOW:
            return self._format_options_query(text, symbol, parameters)
        elif intent == IntentType.HISTORICAL_ANALYSIS:
            return self._format_historical_query(text, symbol, parameters)
        elif intent == IntentType.TRADING_SIGNALS:
            return self._format_trading_query(text, symbol, parameters)
        elif intent == IntentType.PORTFOLIO_MANAGEMENT:
            return self._format_portfolio_query(text, symbol, parameters)
        elif intent == IntentType.RISK_ASSESSMENT:
            return self._format_risk_query(text, symbol, parameters)
        elif intent == IntentType.EDUCATION:
            return self._format_education_query(text, symbol, parameters)
        else:
            return text  # Return original for chat queries
    
    def _format_technical_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for technical analysis agent"""
        base_query = f"Technical analysis for {symbol}" if symbol else "Technical analysis"
        
        if 'indicators' in parameters:
            indicators = ', '.join(parameters['indicators'])
            base_query += f" focusing on {indicators}"
        
        return base_query
    
    def _format_sentiment_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for sentiment analysis agent"""
        base_query = f"Sentiment analysis for {symbol}" if symbol else "Market sentiment analysis"
        
        # Add specific sentiment sources if mentioned
        if 'news' in text.lower():
            base_query += " including news sentiment"
        if 'social' in text.lower() or 'twitter' in text.lower() or 'reddit' in text.lower():
            base_query += " including social media sentiment"
        
        return base_query
    
    def _format_options_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for options flow agent"""
        base_query = f"Options flow analysis for {symbol}" if symbol else "Options flow analysis"
        
        if 'option_type' in parameters:
            base_query += f" focusing on {parameters['option_type']} options"
        
        if 'strike_price' in parameters:
            base_query += f" around ${parameters['strike_price']} strike"
        
        return base_query
    
    def _format_historical_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for historical analysis agent"""
        base_query = f"Historical pattern analysis for {symbol}" if symbol else "Historical analysis"
        
        if 'earnings' in text.lower():
            base_query += " including earnings patterns"
        if 'seasonal' in text.lower():
            base_query += " including seasonality"
        
        return base_query
    
    def _format_trading_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for trading signals agent"""
        base_query = f"Trading signals for {symbol}" if symbol else "Trading signals"
        
        if 'risk_level' in parameters:
            base_query += f" with {parameters['risk_level']} risk profile"
        
        return base_query
    
    def _format_portfolio_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for portfolio management"""
        if 'portfolio' in text.lower() or 'positions' in text.lower():
            return "Portfolio analysis and management"
        elif symbol:
            return f"Portfolio position analysis for {symbol}"
        else:
            return "Portfolio overview"
    
    def _format_risk_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for risk assessment"""
        base_query = f"Risk assessment for {symbol}" if symbol else "Risk analysis"
        
        if 'hedge' in text.lower():
            base_query += " including hedging strategies"
        
        return base_query
    
    def _format_education_query(self, text: str, symbol: Optional[str], parameters: Dict[str, Any]) -> str:
        """Format query for education agent"""
        # Extract the educational topic
        if 'explain' in text.lower():
            topic = text.replace('explain', '').replace('what is', '').strip()
            return f"Educational explanation: {topic}"
        elif 'how to' in text.lower():
            topic = text.replace('how to', '').strip()
            return f"Educational guide: {topic}"
        else:
            return f"Educational content: {text}"
    
    def _determine_routing(self, intent: IntentType, symbol: Optional[str]) -> Tuple[str, List[str]]:
        """Determine API endpoint and agent methods based on intent"""
        
        routing_map = {
            IntentType.TECHNICAL_ANALYSIS: (
                f"/api/v1/technical/{symbol}" if symbol else "/api/v1/technical/AAPL",
                ["technical"]
            ),
            IntentType.SENTIMENT_ANALYSIS: (
                f"/api/v1/agents/{symbol}" if symbol else "/api/v1/agents/AAPL",
                ["sentiment"]
            ),
            IntentType.OPTIONS_FLOW: (
                f"/api/v1/agents/{symbol}" if symbol else "/api/v1/agents/AAPL",
                ["flow"]
            ),
            IntentType.HISTORICAL_ANALYSIS: (
                f"/api/v1/agents/{symbol}" if symbol else "/api/v1/agents/AAPL",
                ["history"]
            ),
            IntentType.TRADING_SIGNALS: (
                f"/api/v1/signals/{symbol}" if symbol else "/api/v1/signals/AAPL",
                ["technical", "sentiment", "flow", "history", "decision_engine"]
            ),
            IntentType.PORTFOLIO_MANAGEMENT: (
                "/api/v1/portfolio/",
                ["portfolio"]
            ),
            IntentType.RISK_ASSESSMENT: (
                f"/api/v1/analysis/{symbol}" if symbol else "/api/v1/analysis/AAPL",
                ["risk"]
            ),
            IntentType.EDUCATION: (
                "/api/v1/education/",
                ["education"]
            ),
            IntentType.CHAT_QUERY: (
                "/api/v1/chat/",
                ["technical", "sentiment", "flow", "history", "education"]
            )
        }
        
        return routing_map.get(intent, ("/api/v1/chat/", ["technical"]))
    
    def get_agent_parameters(self, routed_request: RoutedRequest) -> Dict[str, Any]:
        """Get formatted parameters for specific agent calls"""
        params = {
            'symbol': routed_request.symbol or 'AAPL',
            'timeframe': routed_request.timeframe or '1d',
            'query': routed_request.formatted_query,
            'confidence': routed_request.confidence
        }
        
        # Add intent-specific parameters
        params.update(routed_request.parameters)
        
        return params


# Global router instance
text_router = TextRouter()
