"""
Neural Options Oracle++ Master Orchestrator
OpenAI Agents SDK v0.3.0 Implementation
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from openai import OpenAI
from config.settings import settings
from config.logging import get_agents_logger
from config.database import db_manager

logger = get_agents_logger()


class OptionsOracleOrchestrator:
    """Master orchestrator for AI-driven options trading analysis"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.agents = {}
        self.initialized = False
        
        # Agent weights - these will be dynamically adjusted based on market scenario
        self.base_weights = {
            'technical': 0.60,    # Technical analysis weight
            'sentiment': 0.10,    # Sentiment analysis weight  
            'flow': 0.10,         # Options flow weight
            'history': 0.20       # Historical patterns weight
        }
        
        logger.info("Options Oracle Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all AI agents"""
        try:
            logger.info("🧠 Initializing AI Agent System...")
            
            # Import agents (lazy loading to avoid circular imports)
            from .technical_agent import TechnicalAnalysisAgent
            from .sentiment_agent import SentimentAnalysisAgent
            from .flow_agent import OptionsFlowAgent
            from .history_agent import HistoricalPatternAgent
            from .risk_agent import RiskManagementAgent
            from .education_agent import EducationAgent
            from .buy_agent import BuyAgent
            
            # Initialize agents
            self.agents = {
                'technical': TechnicalAnalysisAgent(self.client),
                'sentiment': SentimentAnalysisAgent(self.client),
                'flow': OptionsFlowAgent(self.client),
                'history': HistoricalPatternAgent(self.client),
                'risk': RiskManagementAgent(self.client),
                'education': EducationAgent(self.client),
                'buy': BuyAgent(self.client)
            }
            
            # Initialize each agent
            for name, agent in self.agents.items():
                try:
                    await agent.initialize()
                    logger.info(f"✅ {name.title()} Agent initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {name} agent: {e}")
                    
            self.initialized = True
            logger.info("🎉 AI Agent System fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            return False
    
    async def analyze_stock(
        self, 
        symbol: str, 
        user_risk_profile: Dict,
        analysis_type: str = "full"
    ) -> Dict[str, Any]:
        """Complete stock analysis using all agents"""
        
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info(f"🔍 Starting comprehensive analysis for {symbol}")
            start_time = datetime.now()
            
            # Step 1: Gather agent analyses in parallel
            agent_tasks = []
            for agent_name in ['technical', 'sentiment', 'flow', 'history']:
                if agent_name in self.agents:
                    task = asyncio.create_task(
                        self.agents[agent_name].analyze(symbol),
                        name=f"{agent_name}_analysis"
                    )
                    agent_tasks.append((agent_name, task))
            
            # Wait for all analyses to complete
            agent_results = {}
            for agent_name, task in agent_tasks:
                try:
                    result = await task
                    agent_results[agent_name] = result
                    logger.info(f"✅ {agent_name.title()} analysis completed")
                except Exception as e:
                    logger.error(f"❌ {agent_name} analysis failed: {e}")
                    agent_results[agent_name] = {"error": str(e), "confidence": 0.0}
            
            # Step 2: Detect market scenario and adjust weights
            scenario = await self._detect_market_scenario(symbol, agent_results)
            adjusted_weights = self._adjust_weights_for_scenario(scenario)
            
            # Step 3: Calculate weighted decision
            decision_score = self._calculate_weighted_decision(agent_results, adjusted_weights)
            
            # Step 4: Generate trading signal
            signal = await self._generate_trading_signal(
                symbol, decision_score, agent_results, scenario
            )
            
            # Step 5: Get strike recommendations from risk agent
            strike_recommendations = await self.agents['risk'].recommend_strikes(
                signal, user_risk_profile
            )
            
            # Step 6: Generate educational content
            educational_content = await self.agents['education'].generate_explanation(
                symbol, signal, agent_results
            )
            
            # Calculate analysis duration
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Compile final analysis
            final_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'analysis_time_seconds': analysis_time,
                'market_scenario': scenario,
                'agent_weights': adjusted_weights,
                'agent_results': agent_results,
                'decision_score': decision_score,
                'signal': signal,
                'strike_recommendations': strike_recommendations,
                'educational_content': educational_content,
                'confidence': self._calculate_overall_confidence(agent_results),
                'risk_profile': user_risk_profile
            }
            
            # Save analysis to database
            await self._save_analysis_to_db(final_analysis)
            
            logger.info(f"🎉 Analysis complete for {symbol}: {signal['direction']} signal with {final_analysis['confidence']:.2f} confidence")
            return final_analysis
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            raise
    
    async def execute_buy_request(
        self, 
        symbol: str, 
        user_risk_profile: Dict,
        user_query: str = None
    ) -> Dict[str, Any]:
        """Execute buy request using buy agent"""
        
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info(f"🎯 Processing buy request for {symbol}")
            
            # First get comprehensive analysis
            analysis_result = await self.analyze_stock(symbol, user_risk_profile)
            
            # Extract decision signal and strike recommendations
            decision_signal = analysis_result.get('signal', {})
            strike_recommendations = analysis_result.get('strike_recommendations', [])
            
            # Get market data
            from src.data.market_data_manager import MarketDataManager
            market_data_manager = MarketDataManager()
            market_data = await market_data_manager.get_comprehensive_data(symbol)
            
            # Run buy agent analysis
            buy_agent = self.agents.get('buy')
            if not buy_agent:
                raise Exception("Buy agent not available")
            
            buy_result = await buy_agent.analyze(
                symbol,
                decision_signal=decision_signal,
                user_risk_profile=user_risk_profile,
                market_data=market_data,
                strike_recommendations=strike_recommendations
            )
            
            # Combine analysis with buy recommendations
            result = {
                'symbol': symbol,
                'analysis': analysis_result,
                'buy_recommendations': buy_result,
                'execution_ready': len(buy_result.get('recommendations', [])) > 0,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Buy request processed for {symbol}: {len(buy_result.get('recommendations', []))} recommendations")
            return result
            
        except Exception as e:
            logger.error(f"Buy request failed for {symbol}: {e}")
            raise
    
    async def _detect_market_scenario(
        self, 
        symbol: str, 
        agent_results: Dict
    ) -> str:
        """Detect current market scenario for dynamic weight adjustment"""
        
        try:
            tech_data = agent_results.get('technical', {})
            sentiment_data = agent_results.get('sentiment', {})
            flow_data = agent_results.get('flow', {})
            
            # Check technical indicators for scenario detection
            if 'scenario' in tech_data:
                return tech_data['scenario']
            
            # Fallback scenario detection based on available data
            if tech_data.get('volatility', {}).get('current', 0) > 30:
                return 'high_volatility'
            elif sentiment_data.get('aggregate_score', 0) > 0.7:
                return 'strong_uptrend'
            elif sentiment_data.get('aggregate_score', 0) < -0.7:
                return 'strong_downtrend'
            else:
                return 'range_bound'
                
        except Exception as e:
            logger.warning(f"Scenario detection failed: {e}")
            return 'range_bound'  # Default scenario
    
    def _adjust_weights_for_scenario(self, scenario: str) -> Dict[str, float]:
        """Dynamically adjust agent weights based on market scenario"""
        
        weights = self.base_weights.copy()
        
        scenario_adjustments = {
            'high_volatility': {
                'technical': 0.10, 'sentiment': -0.05, 'flow': -0.05, 'history': 0.00
            },
            'low_volatility': {
                'technical': -0.10, 'sentiment': 0.05, 'flow': 0.05, 'history': 0.00
            },
            'strong_uptrend': {
                'technical': 0.05, 'sentiment': 0.05, 'flow': 0.00, 'history': -0.10
            },
            'strong_downtrend': {
                'technical': 0.05, 'sentiment': 0.05, 'flow': 0.00, 'history': -0.10
            },
            'breakout': {
                'technical': 0.15, 'sentiment': -0.05, 'flow': 0.00, 'history': -0.10
            },
            'range_bound': {
                'technical': -0.05, 'sentiment': 0.00, 'flow': 0.05, 'history': 0.00
            }
        }
        
        if scenario in scenario_adjustments:
            adjustments = scenario_adjustments[scenario]
            for agent, adjustment in adjustments.items():
                weights[agent] += adjustment
        
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        logger.info(f"Adjusted weights for {scenario}: {weights}")
        return weights
    
    def _calculate_weighted_decision(
        self, 
        agent_results: Dict, 
        weights: Dict[str, float]
    ) -> float:
        """Calculate final weighted decision score"""
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for agent_name, weight in weights.items():
            if agent_name in agent_results:
                agent_data = agent_results[agent_name]
                
                # Extract score based on agent type
                if agent_name == 'technical':
                    score = agent_data.get('weighted_score', 0.0)
                elif agent_name == 'sentiment':
                    score = agent_data.get('aggregate_score', 0.0)
                elif agent_name == 'flow':
                    score = agent_data.get('flow_score', 0.0)
                elif agent_name == 'history':
                    score = agent_data.get('pattern_score', 0.0)
                else:
                    score = 0.0
                
                # Ensure score is in [-1, 1] range
                score = max(-1.0, min(1.0, score))
                
                weighted_score += score * weight
                total_weight += weight
        
        # Normalize if total weight is not 1.0
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        
        return weighted_score
    
    async def _generate_trading_signal(
        self, 
        symbol: str, 
        decision_score: float,
        agent_results: Dict,
        scenario: str
    ) -> Dict[str, Any]:
        """Generate trading signal from decision score"""
        
        # Signal thresholds
        thresholds = {
            'strong_buy': 0.6,
            'buy': 0.3,
            'sell': -0.3,
            'strong_sell': -0.6
        }
        
        # Determine direction
        if decision_score >= thresholds['strong_buy']:
            direction = 'STRONG_BUY'
            strategy_type = 'aggressive_bullish'
        elif decision_score >= thresholds['buy']:
            direction = 'BUY'
            strategy_type = 'moderate_bullish'
        elif decision_score <= thresholds['strong_sell']:
            direction = 'STRONG_SELL'
            strategy_type = 'aggressive_bearish'
        elif decision_score <= thresholds['sell']:
            direction = 'SELL'
            strategy_type = 'moderate_bearish'
        else:
            direction = 'HOLD'
            strategy_type = 'neutral'
        
        # Determine strength
        abs_score = abs(decision_score)
        if abs_score >= 0.7:
            strength = 'strong'
        elif abs_score >= 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        signal = {
            'direction': direction,
            'strength': strength,
            'decision_score': decision_score,
            'strategy_type': strategy_type,
            'market_scenario': scenario,
            'confidence': abs_score,
            'reasoning': self._generate_signal_reasoning(agent_results, decision_score)
        }
        
        return signal
    
    def _generate_signal_reasoning(
        self, 
        agent_results: Dict, 
        decision_score: float
    ) -> str:
        """Generate human-readable reasoning for the signal"""
        
        reasoning_parts = []
        
        # Technical reasoning
        if 'technical' in agent_results:
            tech = agent_results['technical']
            if tech.get('weighted_score', 0) > 0.3:
                reasoning_parts.append("Strong bullish technical indicators")
            elif tech.get('weighted_score', 0) < -0.3:
                reasoning_parts.append("Bearish technical indicators")
        
        # Sentiment reasoning
        if 'sentiment' in agent_results:
            sent = agent_results['sentiment']
            if sent.get('aggregate_score', 0) > 0.5:
                reasoning_parts.append("Positive market sentiment")
            elif sent.get('aggregate_score', 0) < -0.5:
                reasoning_parts.append("Negative market sentiment")
        
        # Flow reasoning
        if 'flow' in agent_results:
            flow = agent_results['flow']
            if flow.get('unusual_activity', False):
                reasoning_parts.append("Unusual options activity detected")
        
        if not reasoning_parts:
            reasoning_parts.append("Mixed signals from market analysis")
        
        return "; ".join(reasoning_parts)
    
    def _calculate_overall_confidence(self, agent_results: Dict) -> float:
        """Calculate overall confidence score"""
        
        confidences = []
        for agent_name, result in agent_results.items():
            if 'confidence' in result and isinstance(result['confidence'], (int, float)):
                confidences.append(result['confidence'])
        
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.5  # Default confidence
    
    async def _save_analysis_to_db(self, analysis: Dict) -> None:
        """Save analysis results to database"""
        
        try:
            signal_data = {
                'symbol': analysis['symbol'],
                'signal_type': 'hybrid',
                'direction': analysis['signal']['direction'],
                'strength': analysis['signal']['strength'],
                'confidence_score': analysis['confidence'],
                'market_scenario': analysis['market_scenario'],
                'agent_weights': analysis['agent_weights'],
                'technical_analysis': analysis['agent_results'].get('technical', {}),
                'sentiment_analysis': analysis['agent_results'].get('sentiment', {}),
                'flow_analysis': analysis['agent_results'].get('flow', {}),
                'historical_analysis': analysis['agent_results'].get('history', {}),
                'strike_recommendations': analysis['strike_recommendations'],
                'educational_content': analysis['educational_content']
            }
            
            await db_manager.save_trading_signal(signal_data)
            logger.info(f"Analysis saved to database for {analysis['symbol']}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        status = {
            'initialized': self.initialized,
            'agents': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, agent in self.agents.items():
            try:
                agent_status = await agent.get_status()
                status['agents'][name] = agent_status
            except Exception as e:
                status['agents'][name] = {'error': str(e), 'healthy': False}
        
        return status


# Global orchestrator instance
orchestrator = OptionsOracleOrchestrator()