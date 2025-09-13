"""
Chat Router API - Intelligent text routing for user queries
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from src.core.ai_intent_router import route_with_ai
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


# Duplicate route removed - using AI Intent Router above


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
















# Helper functions moved to AI Intent Router
