"""Chat Router API - Intelligent text routing for user queries
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.core.intent_router import route_with_ai
from backend.config.logging import get_agents_logger

logger = get_agents_logger()

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Router"])


class ChatMessage(BaseModel):
    """Chat message model"""

    message: str
    context: dict[str, Any] | None = None
    user_id: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response model"""

    response: str
    intent: str
    symbol: str | None
    confidence: float
    data: dict[str, Any] | None = None
    suggestions: list[str] | None = None
    agents_triggered: list[str] | None = None
    interactive_elements: dict[str, Any] | None = None
    timestamp: str


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(message_data: ChatMessage):
    """Send chat message and get intelligent response using AI-powered routing with tool calling
    """
    try:
        logger.info(f"💬 Processing chat message: '{message_data.message}'")

        # Use AI Intent Router with tool calling
        context = {
            "user_id": message_data.user_id,
            "session_id": message_data.session_id,
            **(message_data.context or {}),
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

            elif tool_result.get("tool") == "buy_option":
                symbol = tool_result.get("symbol")
                if tool_result.get("recommendations"):
                    # Extract the best recommendation for interactive display
                    recommendations = tool_result["recommendations"]
                    best_rec = recommendations[0] if recommendations else None

                    if best_rec:
                        data["buy_recommendation"] = best_rec
                        data["option_analysis"] = {
                            "symbol": symbol,
                            "budget": tool_result.get("budget"),
                            "current_price": tool_result.get("current_price"),
                            "recommendations": recommendations,
                        }

                        # Create interactive trading response
                        option_details = best_rec["option_details"]
                        option_type = option_details["type"].upper()
                        strike = option_details["strike_price"]
                        expiration = option_details["expiration_date"]
                        contracts = option_details["contracts"]
                        total_cost = option_details["total_estimated_cost"]

                        # Override the AI-generated response with interactive options display
                        response_text = f"""🎯 **Options Recommendation for {symbol}**

**Best Option**: {option_type} ${strike} (Exp: {expiration})
- **Contracts**: {contracts}
- **Total Cost**: ${total_cost:.2f}
- **Budget Utilization**: {best_rec["risk_metrics"]["budget_utilization"]}%
- **Risk Level**: {best_rec["risk_metrics"]["risk_level"]}
- **Potential Return**: {best_rec["risk_metrics"]["potential_return"]}

**Strategy**: {best_rec["reasoning"]}

Would you like to execute this trade?"""

                        # Add interactive elements for trade confirmation
                        data["interactive_elements"] = {
                            "tradeActions": [
                                {
                                    "symbol": symbol,
                                    "type": "buy",
                                    "recommendation": best_rec,
                                    "requiresConfirmation": True,
                                },
                            ],
                        }

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
            "casual_response": [],
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
            interactive_elements=data.get("interactive_elements"),
            timestamp=datetime.now().isoformat(),
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
                "Portfolio overview",
            ],
            agents_triggered=[],
            timestamp=datetime.now().isoformat(),
        )


# Removed unused /route endpoint - functionality replaced by AI Intent Router


@router.get("/intent/{text}")
async def analyze_intent(text: str):
    """Return the intent the AI router detects for a piece of text
    """
    try:
        result = await route_with_ai(text)

        return {
            "intent": result.get("intent", "GENERAL_CHAT"),
            "confidence": result.get("confidence", 0.5),
            "tools_called": result.get("tools_called", []),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
        }

    except Exception as e:
        logger.error(f"❌ Intent analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Intent analysis failed: {e!s}")


# Helper functions for AI-powered routing


def _generate_ai_suggestions(intent: str, symbol: str | None, tools_called: list[str]) -> list[str]:
    """Generate contextual suggestions based on AI analysis"""
    base_suggestions = []

    if "analyze_stock" in tools_called and symbol:
        base_suggestions.extend(
            [
                f"Technical analysis for {symbol}",
                f"Options strategies for {symbol}",
                f"Risk assessment for {symbol}",
                f"Compare {symbol} to sector",
            ],
        )
    elif intent == "OPTIONS_EDUCATION":
        base_suggestions.extend(
            [
                "What is delta?",
                "Explain call options",
                "How do puts work?",
                "Options Greeks overview",
            ],
        )
    elif intent == "MARKET_TRENDS":
        base_suggestions.extend(
            [
                "Sector performance",
                "Top gainers today",
                "Options flow analysis",
                "Market sentiment overview",
            ],
        )
    else:
        base_suggestions.extend(
            [
                "Analyze AAPL stock",
                "What are options?",
                "Show trending stocks",
                "Portfolio overview",
            ],
        )

    return base_suggestions[:4]  # Return max 4 suggestions


def _extract_stock_data(analysis_result: dict[str, Any]) -> dict[str, Any]:
    """Extract stock data for frontend"""
    return {
        "symbol": analysis_result.get("symbol"),
        "signal": analysis_result.get("signal", {}),
        "confidence": analysis_result.get("confidence", 0),
        "scenario": analysis_result.get("market_scenario"),
        "timestamp": analysis_result.get("timestamp"),
    }


# Helper functions moved to AI Intent Router
