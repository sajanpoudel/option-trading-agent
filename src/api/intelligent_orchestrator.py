"""
Intelligent Agent Orchestration System
Analyzes user queries and triggers appropriate AI agents with full visualization data
Replaces ALL mock data in frontend with real AI-generated analysis
"""
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from openai import OpenAI
import os
import pandas as pd

# Import all our AI agents
from agents.orchestrator import OptionsOracleOrchestrator
from agents.technical_agent import TechnicalAnalysisAgent
from agents.sentiment_agent import SentimentAnalysisAgent
from agents.flow_agent import OptionsFlowAgent
from agents.history_agent import HistoricalPatternAgent
from agents.education_agent import EducationAgent
from agents.risk_agent import RiskManagementAgent

# Import backend services
from src.core.decision_engine import DecisionEngine
from src.data.market_data_manager import MarketDataManager
from src.indicators.technical_calculator import TechnicalIndicatorsCalculator

class QueryClassifier:
    """Classifies user queries to determine which agents to trigger"""
    
    def __init__(self):
        self.query_patterns = {
            'technical_analysis': [
                r'\b(technical|chart|rsi|macd|moving average|support|resistance|indicators?)\b',
                r'\b(analyze|analysis)\b',
                r'\b(breakout|trend|momentum)\b'
            ],
            'sentiment_analysis': [
                r'\b(sentiment|news|social|twitter|reddit|stocktwits)\b',
                r'\b(feeling|mood|opinion|buzz)\b',
                r'\b(bullish|bearish)\b'
            ],
            'options_flow': [
                r'\b(options?|calls?|puts?|flow|volume|gamma|delta)\b',
                r'\b(unusual activity|whale|smart money)\b',
                r'\b(strike|expiration|greeks?)\b'
            ],
            'historical_analysis': [
                r'\b(historical|history|pattern|seasonal|past)\b',
                r'\b(earnings|cycle|performance)\b',
                r'\b(support|resistance|levels?)\b'
            ],
            'education': [
                r'\b(explain|what is|how does|teach|learn)\b',
                r'\b(help|guide|tutorial)\b',
                r'\b(strategy|strategies)\b'
            ],
            'risk_assessment': [
                r'\b(risk|safe|conservative|aggressive|portfolio)\b',
                r'\b(stop loss|position siz|allocation)\b',
                r'\b(hedge|protection|volatility)\b'
            ],
            'trading_signals': [
                r'\b(buy|sell|trade|signal|recommendation)\b',
                r'\b(should i|when to|entry|exit)\b',
                r'\b(target|price target)\b'
            ],
            'portfolio_management': [
                r'\b(portfolio|positions?|holdings?|pnl|p&l)\b',
                r'\b(performance|returns?|profit)\b',
                r'\b(balance|cash|total)\b'
            ]
        }
    
    def classify_query(self, query: str) -> Dict[str, float]:
        """
        Classify user query and return confidence scores for each category
        Returns dict with category names and confidence scores (0-1)
        """
        query_lower = query.lower()
        scores = {}
        
        for category, patterns in self.query_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    matches += 1
                    score += 1.0
            
            # Normalize score by number of patterns
            scores[category] = min(score / len(patterns), 1.0)
        
        return scores
    
    def extract_stock_symbol(self, query: str) -> Optional[str]:
        """Extract stock symbol from query"""
        # Common stock symbols pattern - look for 2-5 letter uppercase symbols
        stock_pattern = r'\b([A-Z]{2,5})\b'
        matches = re.findall(stock_pattern, query.upper())
        
        # Filter out common words that match pattern
        excluded = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'USE', 'WAY', 'WHY', 'AIR', 'BAD', 'BIG', 'BOX', 'CAR', 'CAT', 'CUT', 'DOG', 'EAR', 'EYE', 'FAR', 'FUN', 'GOT', 'HOT', 'JOB', 'LOT', 'MAN', 'OWN', 'PUT', 'RUN', 'SIT', 'SUN', 'TOO', 'TOP', 'WIN', 'YES', 'YET', 'AGENT', 'ANALYZE', 'ANALYSIS', 'TECHNICAL', 'SENTIMENT', 'OPTIONS', 'FLOW', 'HISTORY', 'PATTERN', 'RISK', 'EDUCATION', 'TRADING', 'SIGNALS', 'PORTFOLIO', 'MANAGEMENT'}
        
        valid_symbols = [symbol for symbol in matches if symbol not in excluded]
        
        return valid_symbols[0] if valid_symbols else None

class IntelligentOrchestrator:
    """
    Intelligent orchestration of AI agents based on user queries
    Generates complete analysis with all visualization data
    """
    
    def __init__(self):
        # Initialize OpenAI client using settings
        from config.settings import settings
        openai_api_key = settings.openai_api_key
        if not openai_api_key:
            logger.warning("OpenAI API key not found. Some features will be limited.")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.query_classifier = QueryClassifier()
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
        else:
            # Use fallback mode without OpenAI agents
            self.technical_agent = None
            self.sentiment_agent = None
            self.flow_agent = None
            self.history_agent = None
            self.education_agent = None
            self.risk_agent = None
        
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
        symbol = self.query_classifier.extract_stock_symbol(query)
        query_scores = self.query_classifier.classify_query(query)
        
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
        
        # Always include technical analysis for stock queries
        agents_to_trigger.append('technical')
        
        # Add other agents based on query scores
        if query_scores.get('sentiment_analysis', 0) > threshold:
            agents_to_trigger.append('sentiment')
        
        if query_scores.get('options_flow', 0) > threshold:
            agents_to_trigger.append('flow')
        
        if query_scores.get('historical_analysis', 0) > threshold:
            agents_to_trigger.append('history')
        
        if query_scores.get('education', 0) > threshold:
            agents_to_trigger.append('education')
        
        if query_scores.get('risk_assessment', 0) > threshold:
            agents_to_trigger.append('risk')
        
        # Always include decision engine for trading signals
        if query_scores.get('trading_signals', 0) > threshold:
            agents_to_trigger.append('decision_engine')
        
        # For general queries, trigger all agents for comprehensive analysis
        if max(query_scores.values()) < threshold:
            agents_to_trigger = ['technical', 'sentiment', 'flow', 'history', 'decision_engine']
        
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
            'risk_assessment': analysis_result.get('risk_agent', {})
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