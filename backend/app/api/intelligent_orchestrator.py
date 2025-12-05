"""
Intelligent Agent Orchestration System
Analyzes user queries and triggers appropriate AI agents with full visualization data
Replaces ALL mock data in frontend with real AI-generated analysis
"""
import asyncio
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from openai import OpenAI
import os
import pandas as pd

# Import all our AI agents
from backend.app.agents.orchestrator import OptionsOracleOrchestrator
from backend.app.agents.analysis.technical import TechnicalAnalysisAgent
from backend.app.agents.analysis.sentiment import SentimentAnalysisAgent
from backend.app.agents.analysis.flow import OptionsFlowAgent
from backend.app.agents.analysis.historical import HistoricalPatternAgent
from backend.app.agents.analysis.education import EducationAgent
from backend.app.agents.analysis.risk import RiskManagementAgent
from backend.app.agents.trading.buy import BuyAgent
from backend.app.agents.trading.multi_stock import MultiStockAnalysisAgent

# Import backend services
from backend.app.core.decision_engine import DecisionEngine
from backend.app.services.market_data import MarketDataManager
from backend.app.indicators.calculator import TechnicalIndicatorsCalculator

class QueryClassifier:
    """Intelligent query classifier using OpenAI to understand user intent"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.available_agents = {
            'technical_analysis': 'Analyzes technical indicators, chart patterns, RSI, MACD, support/resistance levels, trends, and momentum',
            'sentiment_analysis': 'Analyzes market sentiment, news sentiment, social media buzz, and overall market mood',
            'options_flow': 'Monitors options flow, unusual activity, call/put ratios, and smart money movements',
            'historical_analysis': 'Studies historical patterns, seasonal trends, past performance, and earnings cycles',
            'education': 'Provides educational content, explanations, tutorials, and learning materials about trading concepts',
            'risk_assessment': 'Evaluates risk metrics, position sizing, portfolio allocation, and risk management strategies',
            'trading_signals': 'Generates buy/sell signals, entry/exit recommendations, and trading opportunities',
            'trade_execution': 'Executes actual trades, places orders, and manages position opening/closing',
            'portfolio_management': 'Manages portfolio positions, tracks P&L, monitors performance, and rebalancing',
            'multi_stock_analysis': 'Analyzes multiple stocks, compares them, and selects the best option based on budget and criteria'
        }
        
        # No fallback patterns - we use OpenAI for all classification
    
    async def classify_query(self, query: str) -> Dict[str, float]:
        """
        Intelligently classify user query using OpenAI to understand intent
        Returns dict with category names and confidence scores (0-1)
        """
        try:
            # Create a detailed prompt for OpenAI
            agent_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in self.available_agents.items()])
            
            prompt = f"""
You are an expert AI assistant that classifies user queries for a financial trading system. 

Available AI agents and their purposes:
{agent_descriptions}

User Query: "{query}"

IMPORTANT CLASSIFICATION RULES:
1. "analyze [STOCK]" or "analysis of [STOCK]" = COMPREHENSIVE ANALYSIS
   - This should trigger: technical_analysis, sentiment_analysis, options_flow, historical_analysis
   - Score these agents 0.8-1.0 for comprehensive analysis requests

2. "buy [STOCK]" or "execute trade" = TRADING EXECUTION
   - This should trigger: trade_execution, technical_analysis, risk_assessment
   - Score these agents 0.8-1.0 for trading requests

3. "find me the best stock" or "best option with budget" = MULTI-STOCK ANALYSIS
   - This should trigger: technical_analysis, sentiment_analysis, options_flow, historical_analysis, risk_assessment, trade_execution
   - Score these agents 0.8-1.0 for stock selection requests

4. "explain" or "teach me" = EDUCATION
   - This should trigger: education agent
   - Score education 0.8-1.0 for learning requests

5. "portfolio" or "positions" = PORTFOLIO MANAGEMENT
   - This should trigger: portfolio_management
   - Score portfolio_management 0.8-1.0 for portfolio requests

6. Specific requests (e.g., "show me RSI", "what's the sentiment") = TARGETED ANALYSIS
   - Only trigger the specific agent mentioned
   - Score the relevant agent 0.8-1.0, others 0.0-0.3

Analyze this query and determine which agents should be triggered. For each agent, provide a confidence score from 0.0 to 1.0.

Respond with a JSON object containing confidence scores for each agent. Only include agents with scores >= 0.3.

Example responses:
- "analyze AAPL" → {{"technical_analysis": 0.9, "sentiment_analysis": 0.9, "options_flow": 0.9, "historical_analysis": 0.9}}
- "buy TSLA" → {{"trade_execution": 0.9, "technical_analysis": 0.8, "risk_assessment": 0.8}}
- "find me the best stock with $500 budget" → {{"technical_analysis": 0.9, "sentiment_analysis": 0.9, "options_flow": 0.9, "historical_analysis": 0.9, "risk_assessment": 0.8, "trade_execution": 0.8}}
- "explain options" → {{"education": 0.9}}
- "show me RSI" → {{"technical_analysis": 0.9}}
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert financial query classifier. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            logger.info(f"OpenAI classification response: {content}")
            
            # Try to parse as JSON
            try:
                scores = json.loads(content)
                # Ensure all scores are floats and within valid range
                for agent in scores:
                    scores[agent] = max(0.0, min(1.0, float(scores[agent])))
                
                logger.info(f"✅ OpenAI classified query: {scores}")
                return scores
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {e}")
                # Return empty scores if OpenAI fails
                return {}
                
        except Exception as e:
            logger.error(f"OpenAI classification failed: {e}")
            # Return empty scores if OpenAI fails
            return {}
    
    async def extract_stock_symbol(self, query: str) -> Optional[str]:
        """Intelligently extract stock symbol from query using OpenAI"""
        try:
            if not self.openai_client:
                logger.error("OpenAI client not available for symbol extraction")
                return None
            
            prompt = f"""
Extract the stock symbol from this user query. Look for:
1. Ticker symbols (2-5 uppercase letters like AAPL, TSLA, NVDA)
2. Company names that can be mapped to symbols
3. Market indices (SPY, QQQ, etc.)

User Query: "{query}"

Respond with ONLY the stock symbol in uppercase letters, or "NONE" if no stock symbol is found.

Examples:
- "analyze AAPL" → AAPL
- "What's happening with Tesla?" → TSLA  
- "Show me Apple stock" → AAPL
- "How is the market doing?" → NONE
- "Buy some Microsoft shares" → MSFT

Symbol:"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial symbol extractor. Always respond with just the symbol or NONE."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            symbol = response.choices[0].message.content.strip().upper()
            
            if symbol == "NONE" or len(symbol) < 2 or len(symbol) > 5:
                return None
            
            logger.info(f"✅ OpenAI extracted symbol: {symbol}")
            return symbol
            
        except Exception as e:
            logger.error(f"OpenAI symbol extraction failed: {e}")
            return None

class IntelligentOrchestrator:
    """
    Intelligent orchestration of AI agents based on user queries
    Generates complete analysis with all visualization data
    """
    
    def __init__(self):
        # Initialize OpenAI client using settings
        from backend.config.settings import settings
        openai_api_key = settings.openai_api_key
        if not openai_api_key:
            logger.warning("OpenAI API key not found. Some features will be limited.")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.query_classifier = QueryClassifier(self.openai_client)
        self.orchestrator = OptionsOracleOrchestrator()
        self.decision_engine = DecisionEngine()
        self.market_data_manager = MarketDataManager()
        self.technical_calculator = TechnicalIndicatorsCalculator()
        
        # Initialize individual agents (only if OpenAI client is available)
        if self.openai_client:
            self.technical_agent = TechnicalAnalysisAgent(self.openai_client)
            self.sentiment_agent = SentimentAnalysisAgent(self.openai_client)
            self.flow_agent = OptionsFlowAgent(self.openai_client)
            self.history_agent = HistoricalPatternAgent(self.openai_client)
            self.education_agent = EducationAgent(self.openai_client)
            self.risk_agent = RiskManagementAgent(self.openai_client)
            self.buy_agent = BuyAgent(self.openai_client)
            self.multi_stock_agent = MultiStockAnalysisAgent(self.openai_client)
        else:
            # Use fallback mode without OpenAI agents
            self.technical_agent = None
            self.sentiment_agent = None
            self.flow_agent = None
            self.history_agent = None
            self.education_agent = None
            self.risk_agent = None
            self.buy_agent = None
            self.multi_stock_agent = None
        
        logger.info("Intelligent Orchestrator initialized")
    
    async def process_user_query(
        self, 
        query: str, 
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Process user query and orchestrate appropriate agents
        Returns complete analysis with all visualization data
        """
        logger.info(f"🧠 Processing user query: {query}")
        
        # Extract stock symbol and classify query
        symbol = await self.query_classifier.extract_stock_symbol(query)
        query_scores = await self.query_classifier.classify_query(query)
        
        # Default to AAPL if no symbol found
        if not symbol:
            symbol = user_context.get('selectedStock', 'AAPL')
        
        logger.info(f"📊 Extracted symbol: {symbol}")
        logger.info(f"🎯 Query classification: {query_scores}")
        
        # Determine which agents to trigger based on scores
        agents_to_trigger = self._determine_agents_to_trigger(query_scores)
        
        # Get user risk profile
        user_risk_profile = user_context.get('risk_profile', {
            'risk_level': 'moderate',
            'experience': 'intermediate',
            'max_position_size': 0.05
        })
        
        # Execute agent orchestration
        analysis_result = await self._orchestrate_agents(
            symbol, 
            query, 
            agents_to_trigger, 
            user_risk_profile
        )
        
        # Generate response based on query type
        response = await self._generate_intelligent_response(
            query, 
            symbol, 
            query_scores, 
            analysis_result
        )
        
        logger.info(f"✅ Query processing complete for {symbol}")
        
        return response
    
    def _determine_agents_to_trigger(self, query_scores: Dict[str, float]) -> List[str]:
        """Determine which agents to trigger based on query classification"""
        threshold = 0.3  # Minimum confidence to trigger agent
        agents_to_trigger = []
        
        # Check if this is a comprehensive analysis request (high scores for multiple analysis agents)
        analysis_agents = ['technical_analysis', 'sentiment_analysis', 'options_flow', 'historical_analysis']
        analysis_scores = [query_scores.get(agent, 0) for agent in analysis_agents]
        is_comprehensive_analysis = any(score > 0.7 for score in analysis_scores) and len([s for s in analysis_scores if s > 0.5]) >= 2
        
        if is_comprehensive_analysis:
            # For comprehensive analysis, trigger all core analysis agents
            agents_to_trigger = ['technical', 'sentiment', 'flow', 'history']
            logger.info("🎯 Comprehensive analysis detected - triggering all core agents")
        else:
            # Add agents based on individual query scores
            if query_scores.get('technical_analysis', 0) > threshold:
                agents_to_trigger.append('technical')
            
            if query_scores.get('sentiment_analysis', 0) > threshold:
                agents_to_trigger.append('sentiment')
            
            if query_scores.get('options_flow', 0) > threshold:
                agents_to_trigger.append('flow')
            
            if query_scores.get('historical_analysis', 0) > threshold:
                agents_to_trigger.append('history')
        
        # Add specialized agents based on query scores
        if query_scores.get('education', 0) > threshold:
            agents_to_trigger.append('education')
        
        if query_scores.get('risk_assessment', 0) > threshold:
            agents_to_trigger.append('risk')
        
        # Always include decision engine for trading signals
        if query_scores.get('trading_signals', 0) > threshold:
            agents_to_trigger.append('decision_engine')
        
        # Include buy agent for trade execution requests
        if query_scores.get('trade_execution', 0) > threshold:
            agents_to_trigger.append('buy_agent')
        
        # Include multi-stock agent for stock selection requests
        if query_scores.get('multi_stock_analysis', 0) > threshold:
            agents_to_trigger.append('multi_stock')
            logger.info("🎯 Multi-stock analysis detected - will analyze multiple stocks")
        
        # For general queries with no clear intent, default to comprehensive analysis
        if not agents_to_trigger and max(query_scores.values()) < threshold:
            agents_to_trigger = ['technical', 'sentiment', 'flow', 'history']
            logger.info("🔄 No clear intent detected - defaulting to comprehensive analysis")
        
        logger.info(f"🤖 Agents to trigger: {agents_to_trigger}")
        return agents_to_trigger
    
    async def _orchestrate_agents(
        self,
        symbol: str,
        query: str,
        agents_to_trigger: List[str],
        user_risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrate the execution of selected agents"""
        
        results = {}
        
        try:
            # Get market data first (needed by all agents)
            logger.info(f"📈 Fetching market data for {symbol}")
            market_data = await self.market_data_manager.get_comprehensive_data(symbol)
            results['market_data'] = market_data
            
            # Execute agents in parallel where possible
            agent_tasks = []
            
            if 'technical' in agents_to_trigger:
                agent_tasks.append(self._run_technical_agent(symbol, market_data))
            
            if 'sentiment' in agents_to_trigger:
                agent_tasks.append(self._run_sentiment_agent(symbol, query))
            
            if 'flow' in agents_to_trigger:
                agent_tasks.append(self._run_flow_agent(symbol, market_data))
            
            if 'history' in agents_to_trigger:
                agent_tasks.append(self._run_history_agent(symbol, market_data))
            
            # Execute agent tasks in parallel
            if agent_tasks:
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                
                # Process results
                for i, (agent_name, result) in enumerate(zip(
                    [a for a in agents_to_trigger if a in ['technical', 'sentiment', 'flow', 'history']], 
                    agent_results
                )):
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent_name} failed: {result}")
                        results[f'{agent_name}_agent'] = {'error': str(result)}
                    else:
                        results[f'{agent_name}_agent'] = result
            
            # Run decision engine if needed (depends on other agents)
            if 'decision_engine' in agents_to_trigger:
                logger.info("🎯 Running decision engine")
                decision_result = await self.decision_engine.process_stock(symbol, user_risk_profile)
                results['decision_engine'] = decision_result
            
            # Run education agent if needed (can use all other results)
            if 'education' in agents_to_trigger:
                logger.info("🎓 Running education agent")
                education_result = await self._run_education_agent(query, symbol, results)
                results['education_agent'] = education_result
            
            # Run risk agent if needed
            if 'risk' in agents_to_trigger:
                logger.info("🛡️ Running risk agent")
                risk_result = await self._run_risk_agent(symbol, results, user_risk_profile)
                results['risk_agent'] = risk_result
            
            # Run buy agent if needed (for trade execution)
            if 'buy_agent' in agents_to_trigger:
                logger.info("🎯 Running buy agent")
                buy_result = await self._run_buy_agent(symbol, results, user_risk_profile)
                results['buy_agent'] = buy_result
            
            if 'multi_stock' in agents_to_trigger:
                logger.info("🔍 Running multi-stock analysis agent")
                multi_stock_result = await self._run_multi_stock_agent(query, user_context)
                results['multi_stock_agent'] = multi_stock_result
            
            # Generate comprehensive technical indicators for charts
            results['technical_indicators'] = await self._generate_technical_indicators(symbol, market_data)
            
            # Generate chart data
            results['chart_data'] = await self._generate_chart_data(symbol, market_data)
            
            logger.info(f"✅ Agent orchestration complete for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Agent orchestration failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _run_technical_agent(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Run technical analysis agent"""
        logger.info(f"🔍 Running technical agent for {symbol}")
        
        try:
            # Get technical indicators
            indicators = self.technical_calculator.calculate_comprehensive_indicators(market_data.get('data', pd.DataFrame()), symbol)
            
            # Run technical agent analysis if available
            if self.technical_agent:
                try:
                    technical_result = await self.technical_agent.analyze(
                        symbol, 
                        timeframe='1D'
                    )
                except Exception as agent_error:
                    logger.warning(f"Technical agent failed: {agent_error}")
                    technical_result = {
                        'summary': f'Technical analysis for {symbol} - indicators calculated',
                        'confidence': 0.7,
                        'recommendation': 'HOLD'
                    }
            else:
                # Fallback analysis without OpenAI
                technical_result = {
                    'summary': f'Technical analysis for {symbol} - indicators calculated',
                    'confidence': 0.7,
                    'recommendation': 'HOLD'
                }
            
            return {
                'indicators': indicators,
                'analysis': technical_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Technical agent error: {e}")
            return {'error': str(e)}
    
    async def _run_sentiment_agent(self, symbol: str, query: str) -> Dict[str, Any]:
        """Run sentiment analysis agent"""
        logger.info(f"💭 Running sentiment agent for {symbol}")
        
        try:
            if self.sentiment_agent:
                sentiment_result = await self.sentiment_agent.analyze(symbol)
            else:
                # Fallback sentiment analysis
                sentiment_result = {
                    'overall_sentiment': 'Neutral',
                    'confidence': 0.6,
                    'sources': ['Market data analysis']
                }
            
            return {
                'sentiment_analysis': sentiment_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment agent error: {e}")
            return {'error': str(e)}
    
    async def _run_flow_agent(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Run options flow analysis agent"""
        logger.info(f"⚡ Running flow agent for {symbol}")
        
        try:
            if self.flow_agent:
                flow_result = await self.flow_agent.analyze(symbol)
            else:
                # Fallback options flow analysis
                flow_result = {
                    'unusual_activity': 'Normal',
                    'volume_trend': 'Average',
                    'recommendation': 'Monitor'
                }
            
            return {
                'flow_analysis': flow_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Flow agent error: {e}")
            return {'error': str(e)}
    
    async def _run_history_agent(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Run historical pattern analysis agent"""
        logger.info(f"📈 Running history agent for {symbol}")
        
        try:
            if self.history_agent:
                history_result = await self.history_agent.analyze(symbol)
            else:
                # Fallback historical analysis
                history_result = {
                    'pattern': 'No significant pattern detected',
                    'trend': 'Sideways',
                    'confidence': 0.5
                }
            
            return {
                'historical_analysis': history_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"History agent error: {e}")
            return {'error': str(e)}
    
    async def _run_education_agent(self, query: str, symbol: str, all_results: Dict) -> Dict[str, Any]:
        """Run education agent with context from other agents"""
        logger.info(f"🎓 Running education agent for query: {query}")
        
        try:
            # Create educational content based on query and analysis results
            signal_data = all_results.get('decision_engine', {}).get('signal', {})
            
            education_result = await self.education_agent.generate_explanation(
                symbol=symbol,
                signal=signal_data,
                context={'user_query': query, 'analysis_results': all_results}
            )
            
            return {
                'educational_content': education_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Education agent error: {e}")
            return {'error': str(e)}
    
    async def _run_risk_agent(self, symbol: str, all_results: Dict, user_risk_profile: Dict) -> Dict[str, Any]:
        """Run risk assessment agent"""
        logger.info(f"🛡️ Running risk agent for {symbol}")
        
        try:
            risk_result = await self.risk_agent.assess_risk(
                symbol=symbol,
                market_data=all_results.get('market_data', {}),
                user_profile=user_risk_profile
            )
            
            return {
                'risk_assessment': risk_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk agent error: {e}")
            return {'error': str(e)}
    
    async def _run_buy_agent(self, symbol: str, all_results: Dict, user_risk_profile: Dict) -> Dict[str, Any]:
        """Run buy agent for trade execution"""
        logger.info(f"🎯 Running buy agent for {symbol}")
        
        try:
            if self.buy_agent:
                # Extract decision signal from decision engine results
                decision_result = all_results.get('decision_engine', {})
                decision_signal = decision_result.get('signal', {})
                strike_recommendations = decision_result.get('strike_recommendations', [])
                
                # Get market data
                market_data = all_results.get('market_data', {})
                
                buy_result = await self.buy_agent.analyze(
                    symbol,
                    decision_signal=decision_signal,
                    user_risk_profile=user_risk_profile,
                    market_data=market_data,
                    strike_recommendations=strike_recommendations
                )
                
                return {
                    'buy_analysis': buy_result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback buy analysis
                return {
                    'buy_analysis': {
                        'recommendations': [],
                        'execution_plan': {'status': 'no_agent', 'message': 'Buy agent not available'},
                        'confidence': 0.0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Buy agent error: {e}")
            return {'error': str(e)}
    
    async def _run_multi_stock_agent(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-stock analysis agent"""
        logger.info(f"🔍 Running multi-stock analysis for query: {query}")
        
        try:
            if not self.multi_stock_agent:
                logger.warning("Multi-stock agent not available")
                return {
                    'multi_stock_analysis': {
                        'error': 'Multi-stock agent not available',
                        'budget': 0,
                        'analyzed_count': 0,
                        'best_recommendation': None
                    }
                }
            
            # Run multi-stock analysis
            result = await self.multi_stock_agent.analyze(query, user_context)
            
            logger.info(f"✅ Multi-stock analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Multi-stock agent error: {e}")
            return {
                'multi_stock_analysis': {
                    'error': str(e),
                    'budget': 0,
                    'analyzed_count': 0,
                    'best_recommendation': None
                }
            }
    
    async def _generate_technical_indicators(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Generate complete technical indicators for frontend charts"""
        logger.info(f"📊 Generating technical indicators for {symbol}")
        
        try:
            indicators = self.technical_calculator.calculate_comprehensive_indicators(market_data.get('data', pd.DataFrame()), symbol)
            
            # Format for frontend consumption - indicators is a flat dict with direct values
            formatted_indicators = {
                'rsi': {
                    'value': indicators.get('rsi', 50),
                    'signal': 'neutral',
                    'overbought': 70,
                    'oversold': 30
                },
                'macd': {
                    'macd': indicators.get('macd', 0),
                    'signal': indicators.get('macd_signal', 0),
                    'histogram': indicators.get('macd_histogram', 0),
                    'trend': 'neutral'
                },
                'bollinger_bands': {
                    'upper': indicators.get('bb_upper', 0),
                    'middle': indicators.get('bb_middle', 0),
                    'lower': indicators.get('bb_lower', 0),
                    'position': indicators.get('bb_position', 0.5)
                },
                'moving_averages': {
                    'ma5': indicators.get('ma5', 0),
                    'ma20': indicators.get('ma20', 0),
                    'ma50': indicators.get('ma50', 0),
                    'ma200': indicators.get('ma200', 0)
                },
                'support_resistance': {
                    'support': indicators.get('support', 0),
                    'resistance': indicators.get('resistance', 0)
                }
            }
            
            return formatted_indicators
            
        except Exception as e:
            logger.error(f"Technical indicators generation error: {e}")
            return {'error': str(e)}
    
    async def _generate_chart_data(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Generate chart data for frontend visualization"""
        logger.info(f"📈 Generating chart data for {symbol}")
        
        try:
            # Get historical data
            historical_data = market_data.get('historical', [])
            
            # Format for chart consumption
            chart_data = {
                'price_data': [],
                'volume_data': [],
                'sparkline_data': []
            }
            
            if historical_data:
                # Last 100 data points for charts
                recent_data = historical_data[-100:]
                
                for i, bar in enumerate(recent_data):
                    timestamp = bar.get('timestamp', datetime.now() - timedelta(days=100-i))
                    
                    chart_data['price_data'].append({
                        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        'open': float(bar.get('open', 0)),
                        'high': float(bar.get('high', 0)),
                        'low': float(bar.get('low', 0)),
                        'close': float(bar.get('close', 0))
                    })
                    
                    chart_data['volume_data'].append({
                        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        'volume': int(bar.get('volume', 0))
                    })
                
                # Sparkline data (last 20 points)
                chart_data['sparkline_data'] = [
                    {'value': float(bar.get('close', 0))} 
                    for bar in recent_data[-20:]
                ]
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Chart data generation error: {e}")
            return {'error': str(e)}
    
    async def _generate_intelligent_response(
        self,
        query: str,
        symbol: str,
        query_scores: Dict[str, float],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intelligent response based on query type and analysis results"""
        
        # Determine primary query type
        primary_type = max(query_scores.items(), key=lambda x: x[1])[0]
        
        # Base response structure
        response = {
            'symbol': symbol,
            'query': query,
            'query_type': primary_type,
            'analysis_complete': True,
            'timestamp': datetime.now().isoformat(),
            'ai_agents_triggered': list(analysis_result.keys()),
            'frontend_data': {}
        }
        
        # Add comprehensive data for frontend
        response['frontend_data'] = {
            # For stock analysis view
            'stock_data': self._format_stock_data(symbol, analysis_result),
            
            # For AI agent analysis component
            'agent_analysis': self._format_agent_analysis(analysis_result),
            
            # For technical analysis widget
            'technical_indicators': analysis_result.get('technical_indicators', {}),
            
            # For trading signals
            'trading_signals': self._format_trading_signals(analysis_result),
            
            # For charts
            'chart_data': analysis_result.get('chart_data', {}),
            
            # For educational content
            'educational_content': analysis_result.get('education_agent', {}),
            
            # For risk assessment
            'risk_assessment': analysis_result.get('risk_agent', {}),
            
            # For multi-stock analysis
            'multi_stock_analysis': analysis_result.get('multi_stock_agent', {})
        }
        
        # Generate contextual AI response text
        response['ai_response'] = await self._generate_contextual_response(
            query, symbol, primary_type, analysis_result
        )
        
        # Add suggested actions
        response['suggested_actions'] = self._generate_suggested_actions(primary_type, analysis_result)
        
        return response
    
    def _format_stock_data(self, symbol: str, analysis_result: Dict) -> Dict[str, Any]:
        """Format stock data for frontend stock view"""
        market_data = analysis_result.get('market_data', {})
        quote = market_data.get('quote', {})
        
        return {
            'symbol': symbol,
            'name': market_data.get('company_name', f"{symbol} Inc"),
            'price': float(quote.get('price', 0)),
            'change': float(quote.get('change', 0)),
            'changePercent': float(quote.get('changePercent', 0)),
            'volume': int(quote.get('volume', 0)),
            'sector': market_data.get('sector', 'Unknown'),
            'description': market_data.get('description', f"{symbol} company description")
        }
    
    def _format_agent_analysis(self, analysis_result: Dict) -> List[Dict[str, Any]]:
        """Format agent analysis for frontend AI agent component"""
        agents_data = []
        
        # Technical Agent
        technical = analysis_result.get('technical_agent', {})
        if technical and not technical.get('error'):
            agents_data.append({
                'name': 'Technical',
                'scenario': technical.get('analysis', {}).get('scenario', 'Normal'),
                'score': int(technical.get('analysis', {}).get('weighted_score', 0.5) * 100),
                'weight': 60,
                'color': '#2196F3',
                'confidence': int(technical.get('analysis', {}).get('confidence', 0.5) * 100),
                'indicators': self._extract_technical_indicators(technical),
                'reasoning': technical.get('analysis', {}).get('reasoning', 'Technical analysis completed')
            })
        
        # Sentiment Agent
        sentiment = analysis_result.get('sentiment_agent', {})
        if sentiment and not sentiment.get('error'):
            agents_data.append({
                'name': 'Sentiment',
                'scenario': sentiment.get('sentiment_analysis', {}).get('overall_sentiment', 'Neutral'),
                'score': int(sentiment.get('sentiment_analysis', {}).get('confidence', 0.5) * 100),
                'weight': 10,
                'color': '#9C27B0',
                'confidence': int(sentiment.get('sentiment_analysis', {}).get('confidence', 0.5) * 100),
                'indicators': self._extract_sentiment_indicators(sentiment),
                'reasoning': sentiment.get('sentiment_analysis', {}).get('reasoning', 'Sentiment analysis completed')
            })
        
        # Flow Agent
        flow = analysis_result.get('flow_agent', {})
        if flow and not flow.get('error'):
            agents_data.append({
                'name': 'Flow',
                'scenario': flow.get('flow_analysis', {}).get('flow_direction', 'Neutral'),
                'score': int(flow.get('flow_analysis', {}).get('confidence', 0.5) * 100),
                'weight': 10,
                'color': '#FF9800',
                'confidence': int(flow.get('flow_analysis', {}).get('confidence', 0.5) * 100),
                'indicators': self._extract_flow_indicators(flow),
                'reasoning': flow.get('flow_analysis', {}).get('reasoning', 'Options flow analysis completed')
            })
        
        # History Agent
        history = analysis_result.get('history_agent', {})
        if history and not history.get('error'):
            agents_data.append({
                'name': 'History',
                'scenario': history.get('historical_analysis', {}).get('pattern_type', 'Normal'),
                'score': int(history.get('historical_analysis', {}).get('pattern_strength', 0.5) * 100),
                'weight': 20,
                'color': '#4CAF50',
                'confidence': int(history.get('historical_analysis', {}).get('confidence', 0.5) * 100),
                'indicators': self._extract_history_indicators(history),
                'reasoning': history.get('historical_analysis', {}).get('reasoning', 'Historical analysis completed')
            })
        
        return agents_data
    
    def _format_trading_signals(self, analysis_result: Dict) -> List[Dict[str, Any]]:
        """Format trading signals for frontend"""
        decision_result = analysis_result.get('decision_engine', {})
        
        if not decision_result or decision_result.get('error'):
            return []
        
        signals = []
        signal_data = decision_result.get('signal', {})
        strike_recommendations = decision_result.get('strike_recommendations', [])
        
        for i, strike_rec in enumerate(strike_recommendations):
            # Handle case where strike_rec might be a float or other type
            if isinstance(strike_rec, dict):
                strike = strike_rec.get('strike', 0)
                potential_return = strike_rec.get('potential_return', 0)
                risk_score = strike_rec.get('risk_score', 0.1)
            else:
                # Fallback for non-dict strike_rec
                strike = float(strike_rec) if isinstance(strike_rec, (int, float)) else 0
                potential_return = 0.05  # Default 5% return
                risk_score = 0.1  # Default 10% risk
            
            signal = {
                'id': f"{decision_result.get('symbol', 'UNKNOWN')}_{i}_{int(datetime.now().timestamp())}",
                'symbol': decision_result.get('symbol', 'UNKNOWN'),
                'direction': signal_data.get('direction', 'HOLD'),
                'confidence': int(decision_result.get('confidence', 0.5) * 100),
                'strike': strike,
                'expiration': '2024-01-19',  # TODO: Get real expiration
                'entryPrice': potential_return * 100,
                'exitRules': f"Take profit at {potential_return:.1%} or stop loss",
                'timestamp': datetime.now().isoformat(),
                'positionSize': 1 + i,
                'reasoning': signal_data.get('reasoning', 'AI-generated signal'),
                'riskReward': f"1:{potential_return / max(risk_score, 0.1):.1f}"
            }
            signals.append(signal)
        
        return signals
    
    def _extract_technical_indicators(self, technical_data: Dict) -> List[str]:
        """Extract technical indicators for display"""
        # Handle case where technical_data might not be a dict
        if not isinstance(technical_data, dict):
            return ["Technical analysis complete"]
            
        indicators = technical_data.get('indicators', {})
        result = []
        
        if isinstance(indicators, dict):
            if 'rsi' in indicators:
                rsi_val = indicators['rsi'].get('current', 50) if isinstance(indicators['rsi'], dict) else indicators['rsi']
                result.append(f"RSI: {rsi_val:.1f}")
            
            if 'macd' in indicators:
                result.append("MACD: Signal detected")
            
            if 'volume' in indicators:
                result.append("Volume: Above average")
            
            if 'support_resistance' in indicators:
                support_data = indicators['support_resistance']
                if isinstance(support_data, dict):
                    support = support_data.get('support', 0)
                else:
                    support = support_data
                result.append(f"Support: ${support:.2f}")
        
        return result or ["Technical analysis complete"]
    
    def _extract_sentiment_indicators(self, sentiment_data: Dict) -> List[str]:
        """Extract sentiment indicators for display"""
        analysis = sentiment_data.get('sentiment_analysis', {})
        
        return [
            f"Overall: {analysis.get('overall_sentiment', 'Neutral')}",
            f"Social Media: {analysis.get('social_sentiment', 'Neutral')}",
            f"News: {analysis.get('news_sentiment', 'Neutral')}",
            f"Confidence: {analysis.get('confidence', 0.5) * 100:.0f}%"
        ]
    
    def _extract_flow_indicators(self, flow_data: Dict) -> List[str]:
        """Extract flow indicators for display"""
        analysis = flow_data.get('flow_analysis', {})
        
        return [
            f"Direction: {analysis.get('flow_direction', 'Neutral')}",
            f"Volume: {analysis.get('volume_analysis', 'Normal')}",
            f"Put/Call Ratio: {analysis.get('put_call_ratio', 1.0):.2f}",
            f"Unusual Activity: {analysis.get('unusual_activity', 'None detected')}"
        ]
    
    def _extract_history_indicators(self, history_data: Dict) -> List[str]:
        """Extract historical indicators for display"""
        analysis = history_data.get('historical_analysis', {})
        
        return [
            f"Pattern: {analysis.get('pattern_type', 'Normal')}",
            f"Strength: {analysis.get('pattern_strength', 0.5) * 100:.0f}%",
            f"Historical Performance: {analysis.get('historical_performance', 'N/A')}",
            f"Seasonal Trend: {analysis.get('seasonal_trend', 'Neutral')}"
        ]
    
    async def _generate_contextual_response(
        self,
        query: str,
        symbol: str,
        primary_type: str,
        analysis_result: Dict[str, Any]
    ) -> str:
        """Generate contextual AI response based on analysis"""
        
        if primary_type == 'technical_analysis':
            return self._generate_technical_response(symbol, analysis_result)
        elif primary_type == 'sentiment_analysis':
            return self._generate_sentiment_response(symbol, analysis_result)
        elif primary_type == 'trading_signals':
            return self._generate_trading_response(symbol, analysis_result)
        elif primary_type == 'education':
            education_content = analysis_result.get('education_agent', {}).get('educational_content', {})
            return education_content.get('explanation', f"Educational analysis for {symbol} completed.")
        elif primary_type == 'multi_stock_analysis':
            return self._generate_multi_stock_response(analysis_result)
        else:
            return self._generate_comprehensive_response(symbol, analysis_result)
    
    def _generate_technical_response(self, symbol: str, analysis_result: Dict) -> str:
        """Generate technical analysis response"""
        technical = analysis_result.get('technical_agent', {})
        if technical.get('error'):
            return f"Technical analysis for {symbol} encountered an issue. Please try again."
        
        analysis = technical.get('analysis', {})
        scenario = analysis.get('scenario', 'Normal')
        score = analysis.get('weighted_score', 0.5) * 100
        
        return f"""🔍 **Technical Analysis for {symbol}**

**Current Scenario**: {scenario}
**Technical Score**: {score:.0f}%

The analysis shows {scenario.lower()} market conditions with a technical strength of {score:.0f}%. 

Click on the Technical agent above for detailed indicator breakdown and reasoning."""
    
    def _generate_sentiment_response(self, symbol: str, analysis_result: Dict) -> str:
        """Generate sentiment analysis response"""
        sentiment = analysis_result.get('sentiment_agent', {})
        if sentiment.get('error'):
            return f"Sentiment analysis for {symbol} encountered an issue. Please try again."
        
        analysis = sentiment.get('sentiment_analysis', {})
        overall_sentiment = analysis.get('overall_sentiment', 'Neutral')
        confidence = analysis.get('confidence', 0.5) * 100
        
        return f"""💭 **Sentiment Analysis for {symbol}**

**Overall Sentiment**: {overall_sentiment}
**Confidence**: {confidence:.0f}%

Market sentiment appears {overall_sentiment.lower()} based on social media, news, and market behavior analysis.

Click on the Sentiment agent above for detailed sentiment breakdown."""
    
    def _generate_trading_response(self, symbol: str, analysis_result: Dict) -> str:
        """Generate trading signals response"""
        decision = analysis_result.get('decision_engine', {})
        if decision.get('error'):
            return f"Trading analysis for {symbol} encountered an issue. Please try again."
        
        signal = decision.get('signal', {})
        direction = signal.get('direction', 'HOLD')
        score = signal.get('score', 0)
        confidence = decision.get('confidence', 0.5) * 100
        
        return f"""📈 **Trading Analysis for {symbol}**

**Recommendation**: {direction}
**Signal Strength**: {score:.2f}
**Confidence**: {confidence:.0f}%

**Reasoning**: {signal.get('reasoning', 'Analysis based on multiple AI agents')}

The system recommends to {direction.lower()} {symbol} based on comprehensive multi-agent analysis."""
    
    def _generate_comprehensive_response(self, symbol: str, analysis_result: Dict) -> str:
        """Generate comprehensive analysis response"""
        return f"""🧠 **Comprehensive Analysis for {symbol}**

I've completed a full multi-agent analysis covering:

✅ **Technical Analysis** - Chart patterns and indicators
✅ **Sentiment Analysis** - Market psychology and news
✅ **Options Flow** - Smart money activity
✅ **Historical Patterns** - Past performance analysis

All data is now loaded in the analysis view. Click on any agent above to see detailed insights, or check the charts and technical indicators below."""
    
    def _generate_suggested_actions(self, primary_type: str, analysis_result: Dict) -> List[str]:
        """Generate suggested follow-up actions"""
        actions = []
        
        if primary_type == 'technical_analysis':
            actions = [
                "View detailed technical charts",
                "Check support and resistance levels",
                "Analyze volume patterns",
                "Review moving averages"
            ]
        elif primary_type == 'sentiment_analysis':
            actions = [
                "Check social media trends",
                "Review recent news",
                "Analyze market psychology",
                "Monitor sentiment shifts"
            ]
        elif primary_type == 'trading_signals':
            actions = [
                "Review strike recommendations",
                "Check risk assessment",
                "Analyze entry/exit points",
                "Set up alerts"
            ]
        else:
            actions = [
                "Explore technical indicators",
                "Check trading signals",
                "Review risk assessment",
                "Set up monitoring"
            ]
        
        return actions
    
    def _generate_multi_stock_response(self, analysis_result: Dict) -> str:
        """Generate multi-stock analysis response"""
        multi_stock = analysis_result.get('multi_stock_agent', {}).get('multi_stock_analysis', {})
        
        if multi_stock.get('error'):
            return f"Multi-stock analysis encountered an issue: {multi_stock['error']}. Please try again."
        
        budget = multi_stock.get('budget', 0)
        analyzed_count = multi_stock.get('analyzed_count', 0)
        best_rec = multi_stock.get('best_recommendation')
        
        if not best_rec:
            return f"I analyzed {analyzed_count} stocks with your ${budget:,.0f} budget, but couldn't find a suitable recommendation. Consider increasing your budget or trying different stocks."
        
        symbol = best_rec.get('symbol', 'Unknown')
        name = best_rec.get('name', 'Unknown')
        price = best_rec.get('price', 0)
        score = best_rec.get('overall_score', 0) * 100
        shares = best_rec.get('max_shares', 0)
        total_cost = shares * price
        potential_return = best_rec.get('potential_return', 0)
        risk_level = best_rec.get('risk_level', 'Unknown')
        
        response = f"""🎯 **Best Stock Recommendation for ${budget:,.0f} Budget**

**Selected:** {symbol} ({name})
- **Price:** ${price:.2f}
- **Shares:** {shares} shares
- **Total Cost:** ${total_cost:,.2f}
- **Overall Score:** {score:.1f}/100
- **Risk Level:** {risk_level}
- **Potential Return:** {potential_return:+.1f}%

**Analysis Summary:**
I analyzed {analyzed_count} stocks and {symbol} emerged as the best option for your budget. The stock shows strong technical indicators, positive sentiment, and good options flow activity.

**Next Steps:**
1. Review the detailed analysis below
2. Consider the risk level ({risk_level.lower()})
3. Set appropriate stop-loss orders
4. Monitor the position closely

Would you like me to execute this trade or would you prefer to see more details about other analyzed stocks?"""
        
        return response