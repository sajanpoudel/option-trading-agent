"""
Intelligent Intent Router
Routes user messages to appropriate agents based on detected intent
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI
from config.settings import settings
from config.logging import get_api_logger

logger = get_api_logger()


class IntentRouter:
    """Router that detects user intent and routes to appropriate agents"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # Define intent categories and their descriptions
        self.intent_categories = {
            "STOCK_ANALYSIS": {
                "description": "User wants analysis of a specific stock symbol",
                "keywords": ["analyze", "stock", "price", "chart", "technical", "buy", "sell"],
                "agents": ["technical", "sentiment", "flow", "history", "risk"],
                "examples": ["Analyze AAPL", "What do you think about TSLA?", "Should I buy NVDA?"]
            },
            "OPTIONS_EDUCATION": {
                "description": "User wants to learn about options trading concepts",
                "keywords": ["learn", "explain", "what is", "how does", "education", "tutorial", "greeks"],
                "agents": ["education"],
                "examples": ["What is delta?", "Explain options", "How do puts work?"]
            },
            "PORTFOLIO_MANAGEMENT": {
                "description": "User wants to manage their portfolio or positions",
                "keywords": ["portfolio", "positions", "pnl", "profit", "loss", "balance"],
                "agents": ["risk"],
                "examples": ["Show my portfolio", "What are my positions?", "How much did I make?"]
            },
            "MARKET_TRENDS": {
                "description": "User wants information about market trends and hot stocks",
                "keywords": ["trending", "hot stocks", "market", "popular", "active"],
                "agents": ["sentiment", "flow"],
                "examples": ["What's trending?", "Show me hot stocks", "Market overview"]
            },
            "GENERAL_CHAT": {
                "description": "General conversation, greetings, or non-trading related queries",
                "keywords": ["hello", "hi", "how are you", "thanks", "bye", "weather"],
                "agents": [],
                "examples": ["Hello", "How are you?", "Thank you", "Good morning"]
            },
            "QUIZ_LEARNING": {
                "description": "User wants to take quizzes or test knowledge",
                "keywords": ["quiz", "test", "question", "practice", "learn"],
                "agents": ["education"],
                "examples": ["Give me a quiz", "Test my knowledge", "Practice questions"]
            }
        }
        
        logger.info("Intent Router initialized with 6 intent categories")
    
    async def detect_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Detect the intent of the user message using OpenAI
        Returns intent classification with confidence and routing info
        """
        try:
            logger.info(f"Detecting intent for message: '{user_message[:50]}...'")
            
            # First try simple keyword matching for speed
            keyword_intent = self._detect_intent_by_keywords(user_message)
            if keyword_intent["confidence"] > 0.8:
                logger.info(f"High confidence keyword match: {keyword_intent['intent']}")
                return keyword_intent
            
            # Use OpenAI for more complex intent detection
            ai_intent = await self._detect_intent_with_ai(user_message)
            
            # Combine results, preferring AI if confidence is high
            if ai_intent["confidence"] > keyword_intent["confidence"]:
                logger.info(f"AI intent detection: {ai_intent['intent']} ({ai_intent['confidence']:.2f})")
                return ai_intent
            else:
                logger.info(f"Keyword intent detection: {keyword_intent['intent']} ({keyword_intent['confidence']:.2f})")
                return keyword_intent
                
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return self._get_fallback_intent()
    
    def _detect_intent_by_keywords(self, message: str) -> Dict[str, Any]:
        """Fast keyword-based intent detection"""
        message_lower = message.lower()
        
        # Check for stock symbols (1-5 uppercase letters)
        import re
        stock_symbols = re.findall(r'\b[A-Z]{1,5}\b', message)
        if stock_symbols:
            return {
                "intent": "STOCK_ANALYSIS",
                "confidence": 0.9,
                "extracted_symbols": stock_symbols,
                "method": "keyword_symbol"
            }
        
        # Score each intent category
        intent_scores = {}
        for intent, config in self.intent_categories.items():
            score = 0
            keywords_found = []
            
            for keyword in config["keywords"]:
                if keyword in message_lower:
                    score += 1
                    keywords_found.append(keyword)
            
            if score > 0:
                # Normalize score by number of keywords
                normalized_score = min(score / len(config["keywords"]), 1.0)
                intent_scores[intent] = {
                    "score": normalized_score,
                    "keywords_found": keywords_found
                }
        
        if not intent_scores:
            return {"intent": "GENERAL_CHAT", "confidence": 0.5, "method": "keyword_fallback"}
        
        # Get highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "intent": best_intent[0],
            "confidence": best_intent[1]["score"],
            "keywords_found": best_intent[1]["keywords_found"],
            "method": "keyword_matching"
        }
    
    async def _detect_intent_with_ai(self, message: str) -> Dict[str, Any]:
        """Use OpenAI to detect intent for complex messages"""
        try:
            # Create intent classification prompt
            intent_descriptions = "\n".join([
                f"- {intent}: {config['description']}"
                for intent, config in self.intent_categories.items()
            ])
            
            prompt = f"""
Classify the user's intent from this message: "{message}"

Available intent categories:
{intent_descriptions}

Instructions:
1. Choose the MOST LIKELY intent category
2. Provide confidence score (0.0-1.0)
3. Extract any stock symbols mentioned (if applicable)
4. If the message is just a greeting or casual conversation, classify as GENERAL_CHAT

Respond in JSON format:
{{
    "intent": "INTENT_NAME",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "extracted_symbols": ["SYMBOL1", "SYMBOL2"] or null,
    "suggested_response_type": "analysis|education|portfolio|trends|casual"
}}
"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate the intent exists
            if result.get("intent") not in self.intent_categories:
                result["intent"] = "GENERAL_CHAT"
                result["confidence"] = 0.5
            
            result["method"] = "ai_classification"
            return result
            
        except Exception as e:
            logger.error(f"AI intent detection failed: {e}")
            return {"intent": "GENERAL_CHAT", "confidence": 0.5, "method": "ai_fallback"}
    
    def get_routing_plan(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate routing plan based on detected intent
        """
        intent = intent_result.get("intent", "GENERAL_CHAT")
        config = self.intent_categories.get(intent, self.intent_categories["GENERAL_CHAT"])
        
        routing_plan = {
            "intent": intent,
            "confidence": intent_result.get("confidence", 0.5),
            "agents_to_call": config["agents"],
            "response_type": intent_result.get("suggested_response_type", "casual"),
            "extracted_data": {
                "symbols": intent_result.get("extracted_symbols", []),
                "keywords": intent_result.get("keywords_found", [])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Routing plan: {intent} -> {config['agents']} agents")
        return routing_plan
    
    def _get_fallback_intent(self) -> Dict[str, Any]:
        """Fallback intent when detection fails"""
        return {
            "intent": "GENERAL_CHAT",
            "confidence": 0.3,
            "method": "fallback",
            "error": "Intent detection failed"
        }
    
    async def generate_casual_response(self, message: str) -> str:
        """Generate a friendly response for casual conversation"""
        try:
            prompt = f"""
You are a helpful assistant for the Neural Options Oracle++ trading platform.
The user said: "{message}"

This appears to be casual conversation, not a trading-related request.
Provide a brief, friendly response that:
1. Acknowledges their message appropriately
2. Gently guides them toward trading-related features if appropriate
3. Keeps it concise (1-2 sentences)

Examples:
- For greetings: "Hello! I'm here to help with your options trading analysis. What stock would you like me to analyze?"
- For thanks: "You're welcome! Feel free to ask about any stocks or options concepts you'd like to explore."
- For casual questions: "I'm doing well, thanks! I'm specialized in options trading analysis. Is there a particular stock you'd like me to look at?"
"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate casual response: {e}")
            return "Hello! I'm here to help with options trading analysis. What can I assist you with today?"


# Global instance
intent_router = IntentRouter()

async def route_user_message(message: str) -> Dict[str, Any]:
    """
    Main function to route user messages based on intent
    """
    # Detect intent
    intent_result = await intent_router.detect_intent(message)
    
    # Get routing plan
    routing_plan = intent_router.get_routing_plan(intent_result)
    
    return routing_plan

async def generate_casual_response(message: str) -> str:
    """Generate response for casual conversation"""
    return await intent_router.generate_casual_response(message)