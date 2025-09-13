"""
Chat Router API - Intelligent text routing for user queries
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from src.core.ai_intent_router import route_with_ai
from src.api.intelligent_orchestrator import IntelligentOrchestrator
from agents.orchestrator import OptionsOracleOrchestrator
from config.logging import get_agents_logger

logger = get_agents_logger()

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Router"])


class ChatMessage(BaseModel):
    """Chat message model"""
    message: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    intent: str
    symbol: Optional[str]
    confidence: float
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    agents_triggered: Optional[List[str]] = None
    timestamp: str




@router.post("/message", response_model=ChatResponse)
async def send_chat_message(message_data: ChatMessage):
    """
    Send chat message and get intelligent response using AI-powered routing with tool calling
    """
    try:
        logger.info(f"💬 Processing chat message: '{message_data.message}'")
        
        # Use AI Intent Router with tool calling
        context = {
            "user_id": message_data.user_id,
            "session_id": message_data.session_id,
            **(message_data.context or {})
        }
        
        # Route and process with AI
        result = await route_with_ai(message_data.message, context)
        
        # Extract info from AI result
        intent = result.get("intent", "GENERAL_CHAT")
        confidence = result.get("confidence", 0.5)
        response_text = result.get("response", "I'm here to help!")
        tools_called = result.get("tools_called", [])
        
        # Determine symbol from tools called
        symbol = None
        data = {}
        
        # Check if any analysis was performed
        for tool_result in result.get("tool_results", []):
            if tool_result.get("tool") == "analyze_stock":
                symbol = tool_result.get("symbol")
                if "analysis_result" in tool_result:
                    data["analysis_result"] = tool_result["analysis_result"]
                    data["stock_data"] = _extract_stock_data(tool_result["analysis_result"])
        
        # Generate suggestions based on intent and tools used
        suggestions = _generate_ai_suggestions(intent, symbol, tools_called)
        
        logger.info(f"🎯 AI Intent: {intent} ({confidence:.2f}) -> Tools: {tools_called}")
        
        # Map tools to agent names for frontend display
        agent_mapping = {
            "analyze_stock": ["technical", "sentiment", "flow", "history", "risk"],
            "buy_option": ["buy", "technical", "risk"],
            "buy_multiple_options": ["buy", "technical", "sentiment", "flow", "risk"],
            "explain_concept": ["education"],
            "get_market_trends": ["sentiment", "flow"],
            "portfolio_analysis": ["risk"],
            "generate_quiz": ["education"],
            "casual_response": []
        }
        
        agents_triggered = []
        for tool in tools_called:
            agents_triggered.extend(agent_mapping.get(tool, []))
        
        # Remove duplicates while preserving order
        agents_triggered = list(dict.fromkeys(agents_triggered))
        
        return ChatResponse(
            response=response_text,
            intent=intent,
            symbol=symbol,
            confidence=confidence,
            data=data,
            suggestions=suggestions,
            agents_triggered=agents_triggered,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ AI Chat routing error: {e}")
        # Fallback response
        return ChatResponse(
            response="I apologize, but I encountered an issue processing your request. Please try asking about a specific stock symbol or trading concept, and I'll do my best to help!",
            intent="ERROR",
            symbol=None,
            confidence=0.0,
            data={},
            suggestions=[
                "Analyze AAPL stock",
                "What is delta in options?", 
                "Show trending stocks",
                "Portfolio overview"
            ],
            agents_triggered=[],
            timestamp=datetime.now().isoformat()
        )


@router.post("/route")
async def route_user_message_detailed(message_data: ChatMessage):
    """
    Analyze user message intent and provide detailed routing information
    """
    try:
        logger.info(f"🔄 Analyzing message routing: '{message_data.message}'")
        
        # Get routing plan
        routing_plan = await route_user_message(message_data.message)
        
        return {
            "routing_plan": routing_plan,
            "processing_time": datetime.now().isoformat(),
            "message_analysis": {
                "original_message": message_data.message,
                "detected_intent": routing_plan['intent'],
                "confidence_score": routing_plan['confidence'],
                "agents_required": routing_plan['agents_to_call'],
                "extracted_entities": routing_plan['extracted_data']
            },
            "recommendations": {
                "should_process": len(routing_plan['agents_to_call']) > 0 or routing_plan['intent'] != 'GENERAL_CHAT',
                "response_strategy": routing_plan['response_type'],
                "suggested_followups": _generate_intent_suggestions(
                    routing_plan['intent'], 
                    routing_plan['extracted_data']['symbols']
                )
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Message routing analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Routing analysis failed: {str(e)}")


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(message_data: ChatMessage):
    """
    Send chat message and get intelligent response with intent-based routing
    """
    try:
        logger.info(f"💬 Processing chat message: '{message_data.message}'")
        
        # Step 1: Detect intent and get routing plan
        routing_plan = await route_user_message(message_data.message)
        
        intent = routing_plan["intent"]
        confidence = routing_plan["confidence"]
        agents_to_call = routing_plan["agents_to_call"]
        extracted_symbols = routing_plan["extracted_data"]["symbols"]
        
        logger.info(f"🎯 Intent: {intent} ({confidence:.2f}) -> Agents: {agents_to_call}")
        
        # Step 2: Handle different intent types
        if intent == "GENERAL_CHAT":
            # Generate casual response for non-trading conversation
            response_text = await generate_casual_response(message_data.message)
            
            return ChatResponse(
                response=response_text,
                intent=intent,
                symbol=None,
                confidence=confidence,
                data={},
                suggestions=[
                    "Analyze AAPL",
                    "Show me trending stocks",
                    "What are options?",
                    "Portfolio overview"
                ],
                timestamp=datetime.now().isoformat()
            )
        
        elif not agents_to_call:
            # No agents needed - direct educational response
            if intent == "OPTIONS_EDUCATION":
                response_text = "I can help explain options concepts! Try asking about specific topics like 'What is delta?' or 'How do puts work?'"
            else:
                response_text = "I'd be happy to help! Could you be more specific about what you'd like to analyze?"
            
            return ChatResponse(
                response=response_text,
                intent=intent,
                symbol=extracted_symbols[0] if extracted_symbols else None,
                confidence=confidence,
                data={},
                suggestions=_generate_intent_suggestions(intent, extracted_symbols),
                timestamp=datetime.now().isoformat()
            )
        
        # Step 3: For trading-related intents, call appropriate agents
        else:
            # Use orchestrator for multi-agent analysis
            orchestrator = OptionsOracleOrchestrator()
            
            # Ensure orchestrator is initialized
            if not orchestrator.initialized:
                await orchestrator.initialize()
            
            # Determine the stock symbol to analyze
            symbol = None
            if extracted_symbols:
                symbol = extracted_symbols[0]
            elif intent == "MARKET_TRENDS":
                # For market trends, we'll analyze popular stocks
                symbol = "AAPL"  # Default for demonstration
            
            if symbol and intent == "STOCK_ANALYSIS":
                # Full stock analysis
                user_risk_profile = {"risk_tolerance": "moderate", "experience": "beginner"}
                analysis_result = await orchestrator.analyze_stock(symbol, user_risk_profile)
                
                response_text = _format_stock_analysis_response(symbol, analysis_result)
                
                return ChatResponse(
                    response=response_text,
                    intent=intent,
                    symbol=symbol,
                    confidence=confidence,
                    data={
                        "analysis_result": analysis_result,
                        "stock_data": _extract_stock_data(analysis_result)
                    },
                    suggestions=[
                        f"Technical analysis for {symbol}",
                        f"Options strategies for {symbol}",
                        f"Risk assessment for {symbol}",
                        "Portfolio impact analysis"
                    ],
                    timestamp=datetime.now().isoformat()
                )
            
            elif intent == "OPTIONS_EDUCATION":
                # Educational content with context
                if symbol:
                    response_text = f"Let me explain options concepts in the context of {symbol}. What specifically would you like to learn about?"
                else:
                    response_text = "I can help you learn about options trading! What concept would you like me to explain?"
                
                return ChatResponse(
                    response=response_text,
                    intent=intent,
                    symbol=symbol,
                    confidence=confidence,
                    data={},
                    suggestions=[
                        "What is delta?",
                        "How do calls work?",
                        "Explain put options",
                        "Options Greeks overview"
                    ],
                    timestamp=datetime.now().isoformat()
                )
            
            else:
                # Fallback for other intents
                response_text = f"I understand you're interested in {intent.lower().replace('_', ' ')}. Let me help you with that!"
                
                return ChatResponse(
                    response=response_text,
                    intent=intent,
                    symbol=symbol,
                    confidence=confidence,
                    data={},
                    suggestions=_generate_intent_suggestions(intent, extracted_symbols),
                    timestamp=datetime.now().isoformat()
                )
        
    except Exception as e:
        logger.error(f"❌ Chat message error: {e}")
        # Fallback response
        return ChatResponse(
            response="I'm sorry, I encountered an issue processing your request. Please try asking about a specific stock symbol or options concept.",
            intent="GENERAL_CHAT",
            symbol=None,
            confidence=0.0,
            data={},
            suggestions=["Analyze AAPL", "What are options?", "Portfolio overview"],
            timestamp=datetime.now().isoformat()
        )


@router.get("/intent/{text}")
async def analyze_intent(text: str):
    """
    Analyze intent of text without processing
    """
    try:
        routing_plan = await route_user_message(text)
        
        return {
            'intent': routing_plan['intent'],
            'confidence': routing_plan['confidence'],
            'agents_to_call': routing_plan['agents_to_call'],
            'extracted_symbols': routing_plan['extracted_data']['symbols'],
            'extracted_keywords': routing_plan['extracted_data']['keywords'],
            'response_type': routing_plan['response_type'],
            'timestamp': routing_plan['timestamp']
        }
        
    except Exception as e:
        logger.error(f"❌ Intent analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Intent analysis failed: {str(e)}")


# Helper functions for AI-powered routing

def _generate_ai_suggestions(intent: str, symbol: Optional[str], tools_called: List[str]) -> List[str]:
    """Generate contextual suggestions based on AI analysis"""
    base_suggestions = []
    
    if "analyze_stock" in tools_called and symbol:
        base_suggestions.extend([
            f"Technical analysis for {symbol}",
            f"Options strategies for {symbol}",
            f"Risk assessment for {symbol}",
            f"Compare {symbol} to sector"
        ])
    elif intent == "OPTIONS_EDUCATION":
        base_suggestions.extend([
            "What is delta?",
            "Explain call options",
            "How do puts work?",
            "Options Greeks overview"
        ])
    elif intent == "MARKET_TRENDS":
        base_suggestions.extend([
            "Sector performance",
            "Top gainers today",
            "Options flow analysis",
            "Market sentiment overview"
        ])
    else:
        base_suggestions.extend([
            "Analyze AAPL stock",
            "What are options?",
            "Show trending stocks", 
            "Portfolio overview"
        ])
    
    return base_suggestions[:4]  # Return max 4 suggestions

def _extract_stock_data(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stock data for frontend"""
    return {
        "symbol": analysis_result.get('symbol'),
        "signal": analysis_result.get('signal', {}),
        "confidence": analysis_result.get('confidence', 0),
        "scenario": analysis_result.get('market_scenario'),
        "timestamp": analysis_result.get('timestamp')
    }
















def _generate_intent_suggestions(intent: str, symbols: List[str]) -> List[str]:
    """Generate follow-up suggestions based on intent"""
    symbol = symbols[0] if symbols else "AAPL"
    
    suggestions_map = {
        "STOCK_ANALYSIS": [
            f"Technical analysis for {symbol}",
            f"Options strategies for {symbol}",
            f"Sentiment analysis for {symbol}",
            f"Risk assessment for {symbol}"
        ],
        "OPTIONS_EDUCATION": [
            "What is delta?",
            "Explain call options",
            "How do puts work?",
            "Options Greeks overview"
        ],
        "PORTFOLIO_MANAGEMENT": [
            "Show my positions",
            "Portfolio performance",
            "Risk analysis",
            "Rebalancing suggestions"
        ],
        "MARKET_TRENDS": [
            "Trending stocks today",
            "Market sentiment overview",
            "Sector performance",
            "Options flow analysis"
        ],
        "QUIZ_LEARNING": [
            "Options basics quiz",
            "Trading strategies test",
            "Risk management questions",
            "Market analysis practice"
        ]
    }
    
    return suggestions_map.get(intent, [
        f"Analyze {symbol}",
        "Show trending stocks",
        "Options education",
        "Portfolio overview"
    ])[:4]

def _format_stock_analysis_response(symbol: str, analysis_result: Dict[str, Any]) -> str:
    """Format comprehensive stock analysis response"""
    try:
        signal = analysis_result.get('signal', {})
        confidence = analysis_result.get('confidence', 0)
        scenario = analysis_result.get('market_scenario', 'unknown')
        
        response = f"🔍 **Analysis for {symbol}**\n\n"
        response += f"**Signal**: {signal.get('direction', 'HOLD')}\n"
        response += f"**Confidence**: {confidence:.1%}\n"
        response += f"**Market Scenario**: {scenario.title()}\n\n"
        
        # Add key insights
        educational_content = analysis_result.get('educational_content', {})
        if educational_content:
            explanation = educational_content.get('explanation', {})
            why_signal = explanation.get('why_this_signal', '')
            if why_signal:
                response += f"**Why this signal**: {why_signal}\n\n"
        
        # Add agent summary
        agent_results = analysis_result.get('agent_results', {})
        active_agents = [name for name, result in agent_results.items() if result.get('confidence', 0) > 0]
        if active_agents:
            response += f"**Analysis based on**: {', '.join(active_agents)} agents\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting analysis response: {e}")
        return f"Analysis completed for {symbol}. See detailed results in the dashboard."

def _extract_stock_data(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stock data for frontend"""
    return {
        "symbol": analysis_result.get('symbol'),
        "signal": analysis_result.get('signal', {}),
        "confidence": analysis_result.get('confidence', 0),
        "scenario": analysis_result.get('market_scenario'),
        "timestamp": analysis_result.get('timestamp')
    }
