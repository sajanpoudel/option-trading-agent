"""
Chat Router API - Intelligent text routing for user queries
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from src.core.text_router import text_router, RoutedRequest, IntentType
from src.api.intelligent_orchestrator import IntelligentOrchestrator
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
    timestamp: str


class RoutedResponse(BaseModel):
    """Response from routed request"""
    routed_request: Dict[str, Any]
    agent_response: Dict[str, Any]
    formatted_response: str


@router.post("/route", response_model=RoutedResponse)
async def route_user_message(message_data: ChatMessage):
    """
    Route user message to appropriate agents and return structured response
    """
    try:
        logger.info(f"🔄 Routing user message: '{message_data.message}'")
        
        # Route the text
        routed_request = await text_router.route_text(
            message_data.message, 
            message_data.context or {}
        )
        
        # Process with appropriate agents
        orchestrator = IntelligentOrchestrator()
        
        # Prepare context for orchestrator
        context = {
            'selectedStock': routed_request.symbol,
            'user_id': message_data.user_id,
            'session_id': message_data.session_id,
            **(message_data.context or {})
        }
        
        # Get agent parameters
        agent_params = text_router.get_agent_parameters(routed_request)
        
        # Process with orchestrator
        agent_response = await orchestrator.process_user_query(
            routed_request.formatted_query,
            context
        )
        
        # Format response based on intent
        formatted_response = _format_response_for_intent(
            routed_request, 
            agent_response
        )
        
        # Convert routed request to dict for response
        routed_dict = {
            'intent': routed_request.intent.value,
            'symbol': routed_request.symbol,
            'timeframe': routed_request.timeframe,
            'parameters': routed_request.parameters,
            'confidence': routed_request.confidence,
            'original_text': routed_request.original_text,
            'formatted_query': routed_request.formatted_query,
            'api_endpoint': routed_request.api_endpoint,
            'agent_methods': routed_request.agent_methods
        }
        
        return RoutedResponse(
            routed_request=routed_dict,
            agent_response=agent_response,
            formatted_response=formatted_response
        )
        
    except Exception as e:
        logger.error(f"❌ Chat routing error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat routing failed: {str(e)}")


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(message_data: ChatMessage):
    """
    Send chat message and get intelligent response
    """
    try:
        logger.info(f"💬 Processing chat message: '{message_data.message}'")
        
        # Route the message
        routed_request = await text_router.route_text(
            message_data.message,
            message_data.context or {}
        )
        
        # Process with orchestrator
        orchestrator = IntelligentOrchestrator()
        
        context = {
            'selectedStock': routed_request.symbol,
            'user_id': message_data.user_id,
            'session_id': message_data.session_id,
            **(message_data.context or {})
        }
        
        # Get response from orchestrator
        response = await orchestrator.process_user_query(
            routed_request.formatted_query,
            context
        )
        
        # Format the response text
        response_text = _format_chat_response(routed_request, response)
        
        # Generate suggestions
        suggestions = _generate_suggestions(routed_request, response)
        
        return ChatResponse(
            response=response_text,
            intent=routed_request.intent.value,
            symbol=routed_request.symbol,
            confidence=routed_request.confidence,
            data=response.get('frontend_data', {}),
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Chat message error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/intent/{text}")
async def analyze_intent(text: str):
    """
    Analyze intent of text without processing
    """
    try:
        routed_request = await text_router.route_text(text)
        
        return {
            'intent': routed_request.intent.value,
            'symbol': routed_request.symbol,
            'confidence': routed_request.confidence,
            'timeframe': routed_request.timeframe,
            'parameters': routed_request.parameters,
            'suggested_endpoint': routed_request.api_endpoint,
            'agent_methods': routed_request.agent_methods
        }
        
    except Exception as e:
        logger.error(f"❌ Intent analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Intent analysis failed: {str(e)}")


def _format_response_for_intent(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format response based on intent type"""
    
    if routed_request.intent == IntentType.TECHNICAL_ANALYSIS:
        return _format_technical_response(routed_request, agent_response)
    elif routed_request.intent == IntentType.SENTIMENT_ANALYSIS:
        return _format_sentiment_response(routed_request, agent_response)
    elif routed_request.intent == IntentType.OPTIONS_FLOW:
        return _format_options_response(routed_request, agent_response)
    elif routed_request.intent == IntentType.TRADING_SIGNALS:
        return _format_trading_response(routed_request, agent_response)
    elif routed_request.intent == IntentType.EDUCATION:
        return _format_education_response(routed_request, agent_response)
    else:
        return _format_general_response(routed_request, agent_response)


def _format_technical_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format technical analysis response"""
    symbol = routed_request.symbol or "the stock"
    
    # Get technical data
    frontend_data = agent_response.get('frontend_data', {})
    technical_indicators = frontend_data.get('technical_indicators', {})
    
    response = f"📊 **Technical Analysis for {symbol}**\n\n"
    
    if technical_indicators:
        # RSI
        rsi = technical_indicators.get('rsi', {})
        if rsi:
            rsi_value = rsi.get('value', 50)
            rsi_signal = rsi.get('signal', 'neutral')
            response += f"**RSI**: {rsi_value:.1f} ({rsi_signal})\n"
        
        # MACD
        macd = technical_indicators.get('macd', {})
        if macd:
            macd_value = macd.get('macd', 0)
            signal_value = macd.get('signal', 0)
            response += f"**MACD**: {macd_value:.3f} (Signal: {signal_value:.3f})\n"
        
        # Moving Averages
        ma = technical_indicators.get('moving_averages', {})
        if ma:
            ma20 = ma.get('ma20', 0)
            ma50 = ma.get('ma50', 0)
            response += f"**Moving Averages**: MA20: ${ma20:.2f}, MA50: ${ma50:.2f}\n"
    
    # Add AI analysis
    ai_response = agent_response.get('ai_response', '')
    if ai_response:
        response += f"\n**AI Analysis**: {ai_response}"
    
    return response


def _format_sentiment_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format sentiment analysis response"""
    symbol = routed_request.symbol or "the market"
    
    response = f"💭 **Sentiment Analysis for {symbol}**\n\n"
    
    # Get sentiment data
    frontend_data = agent_response.get('frontend_data', {})
    agent_analysis = frontend_data.get('agent_analysis', [])
    
    for agent in agent_analysis:
        if agent.get('name') == 'Sentiment':
            sentiment_score = agent.get('score', 50)
            sentiment_scenario = agent.get('scenario', 'Neutral')
            response += f"**Overall Sentiment**: {sentiment_scenario} ({sentiment_score}%)\n"
            break
    
    # Add AI analysis
    ai_response = agent_response.get('ai_response', '')
    if ai_response:
        response += f"\n**AI Analysis**: {ai_response}"
    
    return response


def _format_options_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format options flow response"""
    symbol = routed_request.symbol or "the stock"
    
    response = f"⚡ **Options Flow Analysis for {symbol}**\n\n"
    
    # Get flow data
    frontend_data = agent_response.get('frontend_data', {})
    agent_analysis = frontend_data.get('agent_analysis', [])
    
    for agent in agent_analysis:
        if agent.get('name') == 'Flow':
            flow_score = agent.get('score', 50)
            flow_scenario = agent.get('scenario', 'Normal')
            response += f"**Options Flow**: {flow_scenario} ({flow_score}%)\n"
            break
    
    # Add AI analysis
    ai_response = agent_response.get('ai_response', '')
    if ai_response:
        response += f"\n**AI Analysis**: {ai_response}"
    
    return response


def _format_trading_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format trading signals response"""
    symbol = routed_request.symbol or "the stock"
    
    response = f"🎯 **Trading Signals for {symbol}**\n\n"
    
    # Get trading signals
    frontend_data = agent_response.get('frontend_data', {})
    trading_signals = frontend_data.get('trading_signals', [])
    
    if trading_signals:
        signal = trading_signals[0]  # Get first signal
        direction = signal.get('direction', 'HOLD')
        confidence = signal.get('confidence', 0)
        reasoning = signal.get('reasoning', 'No reasoning provided')
        
        response += f"**Signal**: {direction}\n"
        response += f"**Confidence**: {confidence}%\n"
        response += f"**Reasoning**: {reasoning}\n"
    else:
        response += "No trading signals available at this time.\n"
    
    # Add AI analysis
    ai_response = agent_response.get('ai_response', '')
    if ai_response:
        response += f"\n**AI Analysis**: {ai_response}"
    
    return response


def _format_education_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format education response"""
    response = "📚 **Educational Content**\n\n"
    
    # Get educational content
    frontend_data = agent_response.get('frontend_data', {})
    educational_content = frontend_data.get('educational_content', {})
    
    if educational_content:
        content = educational_content.get('content', '')
        response += content
    else:
        # Fallback to AI response
        ai_response = agent_response.get('ai_response', '')
        response += ai_response if ai_response else "Educational content not available."
    
    return response


def _format_general_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format general response"""
    ai_response = agent_response.get('ai_response', '')
    
    if ai_response:
        return ai_response
    else:
        return f"I've analyzed your request about {routed_request.symbol or 'the market'}. Here's what I found: [Analysis results would be displayed here]"


def _format_chat_response(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> str:
    """Format response for chat interface"""
    return _format_response_for_intent(routed_request, agent_response)


def _generate_suggestions(routed_request: RoutedRequest, agent_response: Dict[str, Any]) -> List[str]:
    """Generate follow-up suggestions based on intent and response"""
    symbol = routed_request.symbol or "AAPL"
    suggestions = []
    
    if routed_request.intent == IntentType.TECHNICAL_ANALYSIS:
        suggestions = [
            f"What's the sentiment for {symbol}?",
            f"Show me options flow for {symbol}",
            f"Generate trading signals for {symbol}",
            f"Historical analysis of {symbol}"
        ]
    elif routed_request.intent == IntentType.SENTIMENT_ANALYSIS:
        suggestions = [
            f"Technical analysis for {symbol}",
            f"Options flow for {symbol}",
            f"Trading signals for {symbol}",
            f"Risk assessment for {symbol}"
        ]
    elif routed_request.intent == IntentType.TRADING_SIGNALS:
        suggestions = [
            f"Portfolio analysis",
            f"Risk management for {symbol}",
            f"Educational content on options trading",
            f"Historical performance of {symbol}"
        ]
    else:
        suggestions = [
            f"Analyze {symbol}",
            f"Trading signals for {symbol}",
            f"Portfolio overview",
            "Educational content"
        ]
    
    return suggestions[:4]  # Return max 4 suggestions
