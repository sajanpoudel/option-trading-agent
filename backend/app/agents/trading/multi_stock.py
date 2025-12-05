"""
Multi-Stock Analysis Agent for Neural Options Oracle++
Analyzes multiple stocks, compares them, and selects the best option based on budget and criteria
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from openai import OpenAI
from backend.config.logging import get_agents_logger
from backend.config.settings import settings

from backend.app.agents.base import BaseAgent

logger = get_agents_logger()


@dataclass
class StockAnalysis:
    """Individual stock analysis result"""
    symbol: str
    name: str
    price: float
    technical_score: float
    sentiment_score: float
    options_score: float
    historical_score: float
    risk_score: float
    overall_score: float
    recommendation: str
    reasoning: str
    budget_fit: bool
    max_shares: int
    potential_return: float
    risk_level: str


@dataclass
class MultiStockResult:
    """Multi-stock analysis result"""
    budget: float
    analyzed_stocks: List[StockAnalysis]
    best_recommendation: Optional[StockAnalysis]
    execution_plan: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    timestamp: datetime


class MultiStockAnalysisAgent(BaseAgent):
    """Agent for analyzing multiple stocks and selecting the best option"""
    
    def __init__(self, openai_client: OpenAI):
        super().__init__(
            client=openai_client,
            name="Multi-Stock Analysis Agent",
            model="gpt-4o"
        )
        self.openai_client = openai_client
    
    def _get_system_instructions(self) -> str:
        """Get system instructions for the multi-stock analysis agent"""
        return """
You are a Multi-Stock Analysis Agent for the Neural Options Oracle++ trading system.

Your role is to:
1. Analyze multiple stocks from trending/hot stocks data
2. Compare stocks based on technical, sentiment, options flow, and historical analysis
3. Select the best stock recommendation based on user's budget and criteria
4. Provide detailed execution plans and risk assessments

Key capabilities:
- Multi-stock screening and comparison
- Budget-based stock selection
- Real-time analysis using multiple specialized agents
- Risk assessment and portfolio optimization
- Execution planning with position sizing

Always provide comprehensive analysis with clear reasoning for your recommendations.
"""
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze multiple stocks and select the best option based on budget
        """
        logger.info(f"🔍 Starting multi-stock analysis for query: {query}")
        
        try:
            # Extract budget and criteria from query
            budget_info = self._extract_budget_and_criteria(query)
            
            # Get list of stocks to analyze
            stocks_to_analyze = await self._get_stocks_to_analyze(query, context)
            
            # Check if we have stocks to analyze
            if not stocks_to_analyze:
                logger.error("No stocks available for analysis - hot stocks API may be down")
                return {
                    'multi_stock_analysis': {
                        'error': 'No stocks available for analysis. Hot stocks API may be unavailable.',
                        'budget': budget_info['budget'],
                        'analyzed_count': 0,
                        'best_recommendation': None,
                        'execution_plan': {},
                        'risk_assessment': {},
                        'timestamp': datetime.now().isoformat()
                    }
                }
            
            # Analyze each stock
            stock_analyses = []
            for symbol in stocks_to_analyze:
                try:
                    analysis = await self._analyze_individual_stock(symbol, budget_info)
                    if analysis:
                        stock_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {e}")
                    continue
            
            # Rank and select best option
            best_recommendation = self._select_best_stock(stock_analyses, budget_info)
            
            # Create execution plan
            execution_plan = self._create_execution_plan(best_recommendation, budget_info)
            
            # Risk assessment
            risk_assessment = self._assess_portfolio_risk(stock_analyses, best_recommendation)
            
            result = MultiStockResult(
                budget=budget_info['budget'],
                analyzed_stocks=stock_analyses,
                best_recommendation=best_recommendation,
                execution_plan=execution_plan,
                risk_assessment=risk_assessment,
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ Multi-stock analysis complete. Best recommendation: {best_recommendation.symbol if best_recommendation else 'None'}")
            
            return {
                'multi_stock_analysis': {
                    'budget': result.budget,
                    'analyzed_count': len(result.analyzed_stocks),
                    'best_recommendation': {
                        'symbol': result.best_recommendation.symbol,
                        'name': result.best_recommendation.name,
                        'price': result.best_recommendation.price,
                        'overall_score': result.best_recommendation.overall_score,
                        'recommendation': result.best_recommendation.recommendation,
                        'reasoning': result.best_recommendation.reasoning,
                        'max_shares': result.best_recommendation.max_shares,
                        'potential_return': result.best_recommendation.potential_return,
                        'risk_level': result.best_recommendation.risk_level
                    } if result.best_recommendation else None,
                    'execution_plan': result.execution_plan,
                    'risk_assessment': result.risk_assessment,
                    'all_analyses': [
                        {
                            'symbol': stock.symbol,
                            'name': stock.name,
                            'price': stock.price,
                            'overall_score': stock.overall_score,
                            'recommendation': stock.recommendation,
                            'budget_fit': stock.budget_fit,
                            'max_shares': stock.max_shares,
                            'potential_return': stock.potential_return,
                            'risk_level': stock.risk_level
                        } for stock in result.analyzed_stocks
                    ],
                    'timestamp': result.timestamp.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-stock analysis failed: {e}")
            return {
                'multi_stock_analysis': {
                    'error': str(e),
                    'budget': 0,
                    'analyzed_count': 0,
                    'best_recommendation': None,
                    'execution_plan': {},
                    'risk_assessment': {},
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _extract_budget_and_criteria(self, query: str) -> Dict[str, Any]:
        """Extract budget and investment criteria from query"""
        try:
            # Use OpenAI to extract budget and criteria
            prompt = f"""
Extract budget and investment criteria from this query: "{query}"

Look for:
1. Budget amount (e.g., $500, 500 dollars, 500 budget)
2. Investment type (stocks, options, ETFs)
3. Risk tolerance (conservative, moderate, aggressive)
4. Time horizon (short-term, long-term)
5. Any specific sectors or preferences

Respond with JSON format:
{{
    "budget": 500,
    "investment_type": "stocks",
    "risk_tolerance": "moderate",
    "time_horizon": "short-term",
    "sectors": [],
    "preferences": []
}}

If no budget is specified, default to $1000.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial query parser. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # Ensure budget is a number
            if isinstance(result.get('budget'), str):
                result['budget'] = float(result['budget'].replace('$', '').replace(',', ''))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract budget and criteria: {e}")
            return {
                'budget': 1000,
                'investment_type': 'stocks',
                'risk_tolerance': 'moderate',
                'time_horizon': 'short-term',
                'sectors': [],
                'preferences': []
            }
    
    async def _get_stocks_to_analyze(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Get stocks to analyze from hot stocks API and OpenAI selection"""
        try:
            # First, get hot stocks from our API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/api/v1/stocks/hot-stocks') as response:
                    if response.status == 200:
                        hot_stocks_data = await response.json()
                        hot_stocks = hot_stocks_data.get('stocks', [])
                        
                        # Extract symbols from hot stocks
                        hot_symbols = [stock.get('symbol') for stock in hot_stocks if stock.get('symbol')]
                        
                        if hot_symbols:
                            logger.info(f"Retrieved {len(hot_symbols)} hot stocks: {hot_symbols[:10]}")
                            
                            # Use OpenAI to select best stocks from hot stocks based on query
                            selected_stocks = await self._select_stocks_from_hot_list(query, hot_symbols)
                            return selected_stocks[:12]  # Limit to 12 stocks
            
            # If hot stocks API fails, return empty list - no fallback
            logger.error("Hot stocks API failed - no stocks to analyze")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get stocks to analyze: {e}")
            return []
    
    async def _select_stocks_from_hot_list(self, query: str, hot_symbols: List[str]) -> List[str]:
        """Use OpenAI to select best stocks from hot stocks list based on query"""
        try:
            prompt = f"""
Based on this query: "{query}"

Select 8-12 relevant stocks to analyze from this hot stocks list:
{', '.join(hot_symbols)}

Consider:
1. Popular and liquid stocks
2. Different sectors for diversification
3. Mix of large cap, growth, and value stocks
4. Any specific mentions in the query
5. Budget constraints if mentioned

Respond with a JSON array of stock symbols:
["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ", "AMD", "META", "GOOGL", "AMZN"]
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a stock selection expert. Always respond with valid JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            stocks = json.loads(response.choices[0].message.content.strip())
            return stocks[:12]  # Limit to 12 stocks
            
        except Exception as e:
            logger.error(f"Failed to select stocks from hot list: {e}")
            return []  # No fallback - return empty list
    
    async def _analyze_individual_stock(self, symbol: str, budget_info: Dict[str, Any]) -> Optional[StockAnalysis]:
        """Analyze an individual stock using real APIs and agents"""
        try:
            # Get real market data from our market data manager
            from backend.app.services.market_data import MarketDataManager
            market_data_manager = MarketDataManager()
            
            # Get comprehensive market data
            market_data = await market_data_manager.get_comprehensive_data(symbol)
            
            if not market_data or not market_data.get('current_price'):
                logger.error(f"No market data available for {symbol}")
                return None
            
            price = market_data['current_price']
            name = market_data.get('company_name', f'{symbol} Inc.')
            
            # Calculate max shares that fit budget
            max_shares = int(budget_info['budget'] / price)
            budget_fit = max_shares > 0
            
            # Use our real agents to analyze the stock
            analysis_scores = await self._get_real_analysis_scores(symbol, market_data)
            
            if not analysis_scores:
                logger.error(f"Failed to get analysis scores for {symbol}")
                return None
            
            # Calculate overall score
            overall_score = (
                analysis_scores['technical_score'] * 0.3 + 
                analysis_scores['sentiment_score'] * 0.2 + 
                analysis_scores['options_score'] * 0.2 + 
                analysis_scores['historical_score'] * 0.2 + 
                analysis_scores['risk_score'] * 0.1
            )
            
            # Determine recommendation
            if overall_score >= 0.8:
                recommendation = "STRONG BUY"
            elif overall_score >= 0.7:
                recommendation = "BUY"
            elif overall_score >= 0.6:
                recommendation = "HOLD"
            elif overall_score >= 0.5:
                recommendation = "WEAK HOLD"
            else:
                recommendation = "SELL"
            
            # Risk level
            risk_score = analysis_scores['risk_score']
            if risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            # Calculate potential return based on technical indicators
            potential_return = self._calculate_potential_return(market_data, analysis_scores)
            
            reasoning = f"Technical analysis shows {analysis_scores['technical_score']:.1%} score, sentiment is {analysis_scores['sentiment_score']:.1%}, options flow indicates {analysis_scores['options_score']:.1%} activity, and historical performance is {analysis_scores['historical_score']:.1%}. Risk level is {risk_level.lower()}."
            
            return StockAnalysis(
                symbol=symbol,
                name=name,
                price=price,
                technical_score=analysis_scores['technical_score'],
                sentiment_score=analysis_scores['sentiment_score'],
                options_score=analysis_scores['options_score'],
                historical_score=analysis_scores['historical_score'],
                risk_score=analysis_scores['risk_score'],
                overall_score=overall_score,
                recommendation=recommendation,
                reasoning=reasoning,
                budget_fit=budget_fit,
                max_shares=max_shares,
                potential_return=potential_return,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze individual stock {symbol}: {e}")
            return None
    
    async def _get_real_analysis_scores(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get real analysis scores using our existing agents"""
        try:
            # Import our agents
            from backend.app.agents.analysis.technical import TechnicalAnalysisAgent
            from backend.app.agents.analysis.sentiment import SentimentAnalysisAgent
            from backend.app.agents.analysis.flow import OptionsFlowAgent
            from backend.app.agents.analysis.historical import HistoricalPatternAgent
            from backend.app.agents.analysis.risk import RiskManagementAgent
            
            # Initialize agents
            technical_agent = TechnicalAnalysisAgent(self.openai_client)
            sentiment_agent = SentimentAnalysisAgent(self.openai_client)
            flow_agent = OptionsFlowAgent(self.openai_client)
            history_agent = HistoricalPatternAgent(self.openai_client)
            risk_agent = RiskManagementAgent(self.openai_client)
            
            # Run all agents in parallel
            tasks = [
                self._run_technical_analysis(technical_agent, symbol, market_data),
                self._run_sentiment_analysis(sentiment_agent, symbol, market_data),
                self._run_options_flow_analysis(flow_agent, symbol, market_data),
                self._run_historical_analysis(history_agent, symbol, market_data),
                self._run_risk_analysis(risk_agent, symbol, market_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extract scores from results
            technical_score = results[0] if not isinstance(results[0], Exception) else 0.5
            sentiment_score = results[1] if not isinstance(results[1], Exception) else 0.5
            options_score = results[2] if not isinstance(results[2], Exception) else 0.5
            historical_score = results[3] if not isinstance(results[3], Exception) else 0.5
            risk_score = results[4] if not isinstance(results[4], Exception) else 0.5
            
            return {
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'options_score': options_score,
                'historical_score': historical_score,
                'risk_score': risk_score
            }
            
        except Exception as e:
            logger.error(f"Failed to get real analysis scores for {symbol}: {e}")
            return None
    
    async def _run_technical_analysis(self, agent, symbol: str, market_data: Dict[str, Any]) -> float:
        """Run technical analysis and extract score"""
        try:
            result = await agent.analyze(f"technical analysis for {symbol}", {'symbol': symbol})
            analysis = result.get('analysis', {})
            score = analysis.get('weighted_score', 0.5)
            return float(score)
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return 0.5
    
    async def _run_sentiment_analysis(self, agent, symbol: str, market_data: Dict[str, Any]) -> float:
        """Run sentiment analysis and extract score"""
        try:
            result = await agent.analyze(f"sentiment analysis for {symbol}", {'symbol': symbol})
            sentiment = result.get('sentiment', {})
            score = sentiment.get('overall_sentiment_score', 0.5)
            return float(score)
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return 0.5
    
    async def _run_options_flow_analysis(self, agent, symbol: str, market_data: Dict[str, Any]) -> float:
        """Run options flow analysis and extract score"""
        try:
            result = await agent.analyze(f"options flow analysis for {symbol}", {'symbol': symbol})
            flow = result.get('options_flow', {})
            score = flow.get('flow_score', 0.5)
            return float(score)
        except Exception as e:
            logger.error(f"Options flow analysis failed for {symbol}: {e}")
            return 0.5
    
    async def _run_historical_analysis(self, agent, symbol: str, market_data: Dict[str, Any]) -> float:
        """Run historical analysis and extract score"""
        try:
            result = await agent.analyze(f"historical analysis for {symbol}", {'symbol': symbol})
            history = result.get('historical_analysis', {})
            score = history.get('pattern_score', 0.5)
            return float(score)
        except Exception as e:
            logger.error(f"Historical analysis failed for {symbol}: {e}")
            return 0.5
    
    async def _run_risk_analysis(self, agent, symbol: str, market_data: Dict[str, Any]) -> float:
        """Run risk analysis and extract score"""
        try:
            result = await agent.analyze(f"risk analysis for {symbol}", {'symbol': symbol})
            risk = result.get('risk_assessment', {})
            score = risk.get('risk_score', 0.5)
            return float(score)
        except Exception as e:
            logger.error(f"Risk analysis failed for {symbol}: {e}")
            return 0.5
    
    def _calculate_potential_return(self, market_data: Dict[str, Any], analysis_scores: Dict[str, float]) -> float:
        """Calculate potential return based on technical indicators and analysis"""
        try:
            # Get technical indicators from market data
            technical_indicators = market_data.get('technical_indicators', {})
            
            # Calculate potential return based on RSI, MACD, and other indicators
            rsi = technical_indicators.get('rsi', 50)
            macd = technical_indicators.get('macd', 0)
            
            # RSI-based return potential
            if rsi < 30:  # Oversold
                rsi_return = 0.15  # 15% potential upside
            elif rsi > 70:  # Overbought
                rsi_return = -0.10  # 10% potential downside
            else:
                rsi_return = 0.05  # 5% neutral potential
            
            # MACD-based return potential
            if macd > 0:  # Bullish
                macd_return = 0.10
            else:  # Bearish
                macd_return = -0.05
            
            # Overall score influence
            overall_score = sum(analysis_scores.values()) / len(analysis_scores)
            score_return = (overall_score - 0.5) * 0.20  # -10% to +10%
            
            # Combine all factors
            potential_return = (rsi_return + macd_return + score_return) * 100  # Convert to percentage
            
            return max(-20, min(20, potential_return))  # Cap between -20% and +20%
            
        except Exception as e:
            logger.error(f"Failed to calculate potential return: {e}")
            return 0.0
    
    def _select_best_stock(self, analyses: List[StockAnalysis], budget_info: Dict[str, Any]) -> Optional[StockAnalysis]:
        """Select the best stock based on analysis results"""
        if not analyses:
            return None
        
        # Filter stocks that fit budget
        budget_fit_analyses = [a for a in analyses if a.budget_fit]
        
        if not budget_fit_analyses:
            # If no stocks fit budget, return the one with highest score
            return max(analyses, key=lambda x: x.overall_score)
        
        # Sort by overall score (descending)
        budget_fit_analyses.sort(key=lambda x: x.overall_score, reverse=True)
        
        return budget_fit_analyses[0]
    
    def _create_execution_plan(self, recommendation: Optional[StockAnalysis], budget_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan for the best recommendation"""
        if not recommendation:
            return {
                'action': 'no_recommendation',
                'reason': 'No suitable stocks found within budget',
                'next_steps': ['Increase budget', 'Consider different stocks', 'Wait for better opportunities']
            }
        
        return {
            'action': 'buy',
            'symbol': recommendation.symbol,
            'name': recommendation.name,
            'shares': recommendation.max_shares,
            'price': recommendation.price,
            'total_cost': recommendation.max_shares * recommendation.price,
            'remaining_budget': budget_info['budget'] - (recommendation.max_shares * recommendation.price),
            'reasoning': recommendation.reasoning,
            'risk_level': recommendation.risk_level,
            'potential_return': recommendation.potential_return,
            'next_steps': [
                'Review the analysis',
                'Set stop-loss order',
                'Monitor position',
                'Consider taking profits at target'
            ]
        }
    
    def _assess_portfolio_risk(self, analyses: List[StockAnalysis], recommendation: Optional[StockAnalysis]) -> Dict[str, Any]:
        """Assess portfolio risk for the recommendation"""
        if not recommendation:
            return {'risk_level': 'UNKNOWN', 'diversification': 'N/A', 'recommendations': []}
        
        # Calculate portfolio metrics
        avg_risk = sum(a.risk_score for a in analyses) / len(analyses) if analyses else 0
        high_risk_count = len([a for a in analyses if a.risk_score > 0.6])
        
        risk_level = "HIGH" if recommendation.risk_score > 0.6 else "MODERATE" if recommendation.risk_score > 0.4 else "LOW"
        
        recommendations = []
        if recommendation.risk_score > 0.6:
            recommendations.append("Consider position sizing to limit risk")
            recommendations.append("Set tight stop-loss orders")
        if high_risk_count > len(analyses) * 0.5:
            recommendations.append("Consider diversifying across different sectors")
        
        return {
            'risk_level': risk_level,
            'diversification': 'Single stock - consider diversification',
            'average_risk': avg_risk,
            'high_risk_stocks': high_risk_count,
            'recommendations': recommendations
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get multi-stock analysis agent status"""
        base_status = await super().get_status()
        base_status.update({
            'analysis_capability': 'multi_stock_screening_from_hot_stocks',
            'data_source': 'hot_stocks_api',
            'last_check': datetime.now().isoformat()
        })
        return base_status
