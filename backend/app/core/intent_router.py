"""
AI-Powered Intent Router with Tool Calling
Uses OpenAI's tool calling to intelligently route and process user requests
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI
from backend.config.settings import settings
from backend.config.logging import get_api_logger

logger = get_api_logger()


class AIIntentRouter:
    """AI-powered router that uses OpenAI tool calling for intelligent request routing"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # Define available tools/functions
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_stock",
                    "description": "Perform comprehensive stock analysis including technical, sentiment, and options flow",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol to analyze (e.g., AAPL, TSLA)"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["full", "technical", "sentiment", "options", "risk"],
                                "description": "Type of analysis to perform"
                            },
                            "time_horizon": {
                                "type": "string",
                                "enum": ["short", "medium", "long"],
                                "description": "Investment time horizon"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "explain_concept",
                    "description": "Explain trading or options concepts in educational format",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "Trading concept to explain (e.g., delta, put options, call options)"
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Explanation difficulty level"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for the explanation"
                            }
                        },
                        "required": ["concept"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_trends",
                    "description": "Get trending stocks and market overview",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sector": {
                                "type": "string",
                                "description": "Specific sector to focus on (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of trending stocks to return",
                                "default": 10
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "portfolio_analysis",
                    "description": "Analyze user's portfolio performance and positions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "enum": ["performance", "risk", "rebalancing", "summary"],
                                "description": "Type of portfolio analysis"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_quiz",
                    "description": "Create educational quizzes on trading topics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Quiz topic (e.g., options_basics, technical_analysis)"
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Quiz difficulty level"
                            },
                            "question_count": {
                                "type": "integer",
                                "description": "Number of questions to generate",
                                "default": 5
                            }
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "casual_response",
                    "description": "Generate friendly response for casual conversation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The casual message to respond to"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context about the conversation"
                            }
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "buy_option",
                    "description": "Analyze and buy a single option for a specific stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol to buy options for (e.g., AAPL, TSLA)"
                            },
                            "budget": {
                                "type": "number",
                                "description": "Budget available for option purchase (default: 500 if not specified)"
                            },
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["conservative", "moderate", "aggressive"],
                                "description": "User's risk tolerance level"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "buy_multiple_options",
                    "description": "Analyze hot stocks and create optimized options portfolio within budget",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "budget": {
                                "type": "number",
                                "description": "Total budget for options portfolio"
                            },
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["conservative", "moderate", "aggressive"],
                                "description": "User's risk tolerance level"
                            },
                            "diversification": {
                                "type": "string",
                                "enum": ["low", "moderate", "high"],
                                "description": "Desired diversification level"
                            }
                        },
                        "required": ["budget"]
                    }
                }
            }
        ]
        
        logger.info("AI Intent Router initialized with tool calling capabilities")
    
    async def route_and_process(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use OpenAI to determine intent and call appropriate tools
        Returns formatted response ready for user display
        """
        try:
            logger.info(f"AI routing message: '{user_message[:50]}...'")
            
            # Step 1: Let OpenAI decide what tools to call
            system_prompt = """
You are an intelligent assistant for the Neural Options Oracle++ trading platform.

Your job is to understand what the user wants and call the appropriate tools to help them.

Available capabilities:
1. **Stock Analysis** - Analyze any stock symbol with technical indicators, sentiment, options flow
2. **Options Buying** - Buy single options for specific stocks with budget analysis
3. **Multi-Options Buying** - Find and buy best options from hot stocks within budget
4. **Education** - Explain trading concepts, options Greeks, strategies in simple terms  
5. **Market Trends** - Show trending stocks and market overview
6. **Portfolio** - Analyze portfolio performance and risk
7. **Quizzes** - Generate educational quizzes to test knowledge
8. **Casual Chat** - Handle greetings and general conversation

Instructions:
- For stock analysis requests, extract the symbol and call analyze_stock
- For buying options with budget (e.g., "buy AAPL option with $500"), call buy_option
- For buying best options from hot stocks (e.g., "find best options with $500 budget"), call buy_multiple_options
- For questions about trading concepts, call explain_concept  
- For casual greetings/conversation, call casual_response
- For portfolio requests, call portfolio_analysis
- For quiz requests, call generate_quiz
- For market trends, call get_market_trends

IMPORTANT BUY REQUEST HANDLING:
- If user wants to buy a specific option (e.g., "buy TSLA strike 425"), call buy_option with symbol and infer budget as $500 if not specified
- If user asks for "most profitable option" or "best option with $X budget", call buy_multiple_options
- Always assume a default budget of $500 for option purchases if not explicitly mentioned
- For buy requests, ALWAYS extract: symbol, budget (default $500), risk_tolerance (default "moderate")

Always choose the most appropriate tool(s) for the user's request.
You can call multiple tools if needed.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Make OpenAI call with tool calling
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.1
            )
            
            # Process the response
            message = response.choices[0].message
            
            # Check if OpenAI wants to call tools
            if message.tool_calls:
                # Step 2: Execute the tool calls
                tool_results = []
                for tool_call in message.tool_calls:
                    result = await self._execute_tool_call(tool_call)
                    tool_results.append(result)
                
                # Step 3: Let OpenAI format the final response
                final_response = await self._format_final_response(
                    user_message, message, tool_results
                )
                
                # Extract symbol from tool results if available
                symbol = None
                for result in tool_results:
                    if result.get("symbol"):
                        symbol = result["symbol"]
                        break
                
                response_data = {
                    "response": final_response,
                    "intent": self._determine_intent_from_tools(message.tool_calls),
                    "tools_called": [tc.function.name for tc in message.tool_calls],
                    "tool_results": tool_results,  # Include tool results for frontend
                    "confidence": 0.9,  # High confidence when using tools
                    "formatted": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add symbol if found
                if symbol:
                    response_data["symbol"] = symbol
                    
                return response_data
            
            else:
                # No tools needed - direct response
                return {
                    "response": message.content,
                    "intent": "GENERAL_CHAT", 
                    "tools_called": [],
                    "confidence": 0.7,
                    "formatted": True,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"AI routing failed: {e}")
            return await self._fallback_response(user_message)
    
    async def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a single tool call and return results"""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        logger.info(f"Executing tool: {function_name} with args: {arguments}")
        
        try:
            if function_name == "analyze_stock":
                return await self._analyze_stock(arguments)
            elif function_name == "explain_concept":
                return await self._explain_concept(arguments)
            elif function_name == "get_market_trends":
                return await self._get_market_trends(arguments)
            elif function_name == "portfolio_analysis":
                return await self._portfolio_analysis(arguments)
            elif function_name == "generate_quiz":
                return await self._generate_quiz(arguments)
            elif function_name == "casual_response":
                return await self._casual_response(arguments)
            elif function_name == "buy_option":
                return await self._buy_option(arguments)
            elif function_name == "buy_multiple_options":
                return await self._buy_multiple_options(arguments)
            else:
                return {"error": f"Unknown tool: {function_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution failed for {function_name}: {e}")
            return {"error": str(e), "tool": function_name}
    
    async def _analyze_stock(self, args: Dict) -> Dict[str, Any]:
        """Execute stock analysis"""
        try:
            from backend.app.agents.orchestrator import OptionsOracleOrchestrator
            
            symbol = args["symbol"].upper()
            analysis_type = args.get("analysis_type", "full")
            
            # Initialize orchestrator
            orchestrator = OptionsOracleOrchestrator()
            if not orchestrator.initialized:
                await orchestrator.initialize()
            
            # Run analysis
            user_risk_profile = {"risk_tolerance": "moderate", "experience": "beginner"}
            result = await orchestrator.analyze_stock(symbol, user_risk_profile, analysis_type)
            
            return {
                "tool": "analyze_stock",
                "symbol": symbol,
                "analysis_result": result,
                "success": True
            }
            
        except Exception as e:
            return {"tool": "analyze_stock", "error": str(e), "success": False}
    
    async def _explain_concept(self, args: Dict) -> Dict[str, Any]:
        """Execute concept explanation"""
        try:
            from backend.app.api.routes.education import explain_concept as explain_api
            
            concept = args["concept"]
            context = args.get("context", {})
            
            # Call education API
            explanation = await explain_api(concept, context, session={})
            
            return {
                "tool": "explain_concept",
                "concept": concept,
                "explanation": explanation,
                "success": True
            }
            
        except Exception as e:
            return {"tool": "explain_concept", "error": str(e), "success": False}
    
    async def _get_market_trends(self, args: Dict) -> Dict[str, Any]:
        """Get market trends"""
        try:
            # This would integrate with your hot stocks API
            from backend.app.api.main import app  # Get trending stocks
            
            return {
                "tool": "get_market_trends", 
                "trends": "Market trends data would be here",
                "success": True
            }
            
        except Exception as e:
            return {"tool": "get_market_trends", "error": str(e), "success": False}
    
    async def _portfolio_analysis(self, args: Dict) -> Dict[str, Any]:
        """Analyze portfolio"""
        try:
            # This would integrate with portfolio API
            return {
                "tool": "portfolio_analysis",
                "analysis": "Portfolio analysis would be here", 
                "success": True
            }
            
        except Exception as e:
            return {"tool": "portfolio_analysis", "error": str(e), "success": False}
    
    async def _generate_quiz(self, args: Dict) -> Dict[str, Any]:
        """Generate educational quiz"""
        try:
            from backend.app.api.routes.education import generate_quiz as quiz_api
            from backend.app.api.routes.education import QuizRequest
            
            topic = args["topic"]
            difficulty = args.get("difficulty", "beginner")
            count = args.get("question_count", 5)
            
            request = QuizRequest(topic=topic, difficulty=difficulty, question_count=count)
            quiz = await quiz_api(request, session={})
            
            return {
                "tool": "generate_quiz",
                "quiz": quiz,
                "success": True
            }
            
        except Exception as e:
            return {"tool": "generate_quiz", "error": str(e), "success": False}
    
    async def _casual_response(self, args: Dict) -> Dict[str, Any]:
        """Generate casual response"""
        message = args["message"]
        
        casual_responses = {
            "hello": "Hello! I'm here to help with your options trading and stock analysis. What would you like to explore today?",
            "how are you": "I'm doing great, thanks for asking! Ready to help you analyze stocks and learn about options trading.",
            "thanks": "You're very welcome! Feel free to ask about any stocks or trading concepts.",
            "bye": "Goodbye! Come back anytime you need help with trading analysis or have questions about options."
        }
        
        # Simple matching for common phrases
        for key, response in casual_responses.items():
            if key in message.lower():
                return {
                    "tool": "casual_response",
                    "response": response,
                    "success": True
                }
        
        return {
            "tool": "casual_response", 
            "response": "I'm here to help with options trading and stock analysis. What can I assist you with?",
            "success": True
        }
    
    async def _buy_option(self, args: Dict) -> Dict[str, Any]:
        """Execute single option purchase analysis"""
        try:
            from backend.app.agents.trading.buy import analyze_option_buy
            
            symbol = args["symbol"].upper()
            budget = float(args.get("budget", 500))  # Default to $500 if not specified
            risk_tolerance = args.get("risk_tolerance", "moderate")
            
            preferences = {
                "risk_tolerance": risk_tolerance,
                "strategy": "growth",
                "time_horizon": "short"
            }
            
            analysis = await analyze_option_buy(symbol, budget, preferences)
            
            return {
                "tool": "buy_option",
                "symbol": symbol,
                "budget": budget,
                "analysis": analysis,
                "requires_confirmation": True,
                "success": True
            }
            
        except Exception as e:
            return {"tool": "buy_option", "error": str(e), "success": False}
    
    async def _buy_multiple_options(self, args: Dict) -> Dict[str, Any]:
        """Execute multi-options portfolio analysis"""
        try:
            from backend.app.agents.trading.multi_options import analyze_multi_options_buy
            
            budget = float(args["budget"])
            risk_tolerance = args.get("risk_tolerance", "moderate")
            diversification = args.get("diversification", "moderate")
            
            preferences = {
                "risk_tolerance": risk_tolerance,
                "diversification": diversification,
                "strategy": "growth"
            }
            
            portfolio = await analyze_multi_options_buy(budget, preferences)
            
            return {
                "tool": "buy_multiple_options",
                "budget": budget,
                "portfolio": portfolio,
                "requires_confirmation": True,
                "success": True
            }
            
        except Exception as e:
            return {"tool": "buy_multiple_options", "error": str(e), "success": False}
    
    async def _format_final_response(
        self, 
        user_message: str, 
        ai_message, 
        tool_results: List[Dict]
    ) -> str:
        """Let OpenAI format the final human-readable response"""
        try:
            # Create context for final formatting
            context = {
                "user_message": user_message,
                "tool_results": tool_results,
                "tools_used": [result.get("tool", "unknown") for result in tool_results]
            }
            
            format_prompt = f"""
Based on the user's request and the tool results, provide a clear, helpful response in markdown format.

User asked: "{user_message}"

Tool results: {json.dumps(tool_results, indent=2)}

Instructions:
1. Write in a friendly, conversational tone
2. Use markdown formatting for readability (headers, bullet points, code blocks, etc.)
3. If stock analysis was performed, summarize key findings
4. If explaining concepts, make it easy to understand
5. Include relevant data and insights from tool results
6. End with helpful suggestions for next steps

Make the response human-readable and engaging!
"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": format_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return self._create_fallback_formatted_response(tool_results)
    
    def _create_fallback_formatted_response(self, tool_results: List[Dict]) -> str:
        """Create a basic formatted response if AI formatting fails"""
        response = "## Analysis Complete\n\n"
        
        for result in tool_results:
            if result.get("success"):
                tool = result.get("tool", "analysis")
                response += f"✅ **{tool.replace('_', ' ').title()}** completed successfully\n\n"
                
                if "analysis_result" in result:
                    analysis = result["analysis_result"]
                    signal = analysis.get("signal", {})
                    response += f"- **Signal**: {signal.get('direction', 'HOLD')}\n"
                    response += f"- **Confidence**: {analysis.get('confidence', 0):.1%}\n\n"
            else:
                response += f"❌ **{result.get('tool', 'Tool')}** encountered an error\n\n"
        
        return response
    
    def _determine_intent_from_tools(self, tool_calls) -> str:
        """Determine intent based on tools called"""
        if not tool_calls:
            return "GENERAL_CHAT"
        
        tool_names = [tc.function.name for tc in tool_calls]
        
        if "analyze_stock" in tool_names:
            return "STOCK_ANALYSIS"
        elif "buy_option" in tool_names:
            return "OPTIONS_BUYING"
        elif "buy_multiple_options" in tool_names:
            return "PORTFOLIO_BUYING"
        elif "explain_concept" in tool_names:
            return "OPTIONS_EDUCATION"
        elif "get_market_trends" in tool_names:
            return "MARKET_TRENDS"
        elif "portfolio_analysis" in tool_names:
            return "PORTFOLIO_MANAGEMENT"
        elif "generate_quiz" in tool_names:
            return "QUIZ_LEARNING"
        else:
            return "GENERAL_CHAT"
    
    async def _fallback_response(self, user_message: str) -> Dict[str, Any]:
        """Fallback response when AI routing fails"""
        return {
            "response": f"I encountered an issue processing your request: '{user_message}'. Please try asking about a specific stock symbol or trading concept.",
            "intent": "ERROR",
            "tools_called": [],
            "confidence": 0.0,
            "formatted": True,
            "timestamp": datetime.now().isoformat(),
            "error": True
        }


# Global instance
ai_intent_router = AIIntentRouter()

async def route_with_ai(message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to route messages using AI with tool calling
    """
    return await ai_intent_router.route_and_process(message, context or {})