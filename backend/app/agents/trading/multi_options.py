"""
Multi-Options Buy Agent
Analyzes hot stocks and creates optimized options portfolio within budget
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI

from backend.app.agents.trading.buy import BuyAgent as OptionsBuyAgent
from backend.config.settings import settings
from backend.config.logging import get_agents_logger

logger = get_agents_logger()


class MultiOptionsBuyAgent:
    """Agent for analyzing hot stocks and creating optimized options portfolio"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.single_buy_agent = OptionsBuyAgent(self.openai_client)
        
    async def analyze_best_options_portfolio(
        self, 
        total_budget: float, 
        user_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze hot stocks and create optimized options portfolio within budget
        """
        try:
            logger.info(f"🔥 Analyzing best options portfolio with ${total_budget} budget")
            
            # Get current hot stocks by making API call
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/api/v1/stocks/hot-stocks') as response:
                    hot_stocks_data = await response.json()
            
            hot_stocks = hot_stocks_data.get('stocks', [])
            
            if not hot_stocks:
                return {"error": "No hot stocks data available"}
            
            # Analyze top performing stocks
            top_stocks = sorted(
                hot_stocks, 
                key=lambda x: (x.get('aiScore', 0) + abs(x.get('changePercent', 0))), 
                reverse=True
            )[:8]  # Analyze top 8 stocks
            
            logger.info(f"Analyzing top stocks: {[s['symbol'] for s in top_stocks]}")
            
            # Get options opportunities for each stock
            stock_opportunities = []
            for stock in top_stocks:
                symbol = stock['symbol']
                
                # Allocate portion of budget for individual analysis
                individual_budget = min(total_budget * 0.5, total_budget / 3)  # Max 50% or 1/3 of budget per stock
                
                try:
                    opportunity = await self.single_buy_agent.analyze_option_opportunity(
                        symbol, 
                        individual_budget, 
                        user_preferences or {}
                    )
                    
                    if 'error' not in opportunity:
                        opportunity['stock_data'] = stock
                        stock_opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {symbol}: {e}")
                    continue
            
            if not stock_opportunities:
                return {"error": "No valid options opportunities found in hot stocks"}
            
            # Use AI to create optimized portfolio
            portfolio_analysis = await self._create_optimized_portfolio(
                stock_opportunities, 
                total_budget, 
                user_preferences or {}
            )
            
            return portfolio_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing options portfolio: {e}")
            return {"error": str(e)}
    
    async def _create_optimized_portfolio(
        self, 
        opportunities: List[Dict[str, Any]], 
        total_budget: float,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to create optimized options portfolio"""
        
        # Prepare data for AI analysis
        opportunities_summary = []
        for opp in opportunities:
            best_option = opp.get('best_option', {})
            stock_data = opp.get('stock_data', {})
            
            opportunities_summary.append({
                "symbol": opp.get('symbol'),
                "current_price": opp.get('current_price'),
                "ai_score": stock_data.get('aiScore', 0),
                "change_percent": stock_data.get('changePercent', 0),
                "volume": stock_data.get('volume', 0),
                "best_option": {
                    "strike": best_option.get('strike'),
                    "type": best_option.get('type'),
                    "premium": best_option.get('premium'),
                    "cost": best_option.get('cost'),
                    "expiration": best_option.get('expiration')
                },
                "confidence": opp.get('confidence', 0),
                "profit_potential": opp.get('profit_potential'),
                "max_loss": opp.get('max_loss')
            })
        
        prompt = f"""
        Create an optimized options portfolio from these hot stock opportunities:
        
        Total Budget: ${total_budget}
        Available Opportunities: {json.dumps(opportunities_summary, indent=2)}
        
        User Preferences:
        - Risk tolerance: {user_preferences.get('risk_tolerance', 'moderate')}
        - Diversification: {user_preferences.get('diversification', 'moderate')}
        - Strategy: {user_preferences.get('strategy', 'growth')}
        
        Select the best combination of options to maximize profit potential while managing risk.
        Consider:
        1. Diversification across different stocks
        2. Risk-reward balance
        3. Correlation between positions
        4. Budget allocation efficiency
        
        Provide analysis in this JSON format:
        {{
            "recommended_portfolio": [
                {{
                    "symbol": "AAPL",
                    "option": {{
                        "strike": 150.0,
                        "type": "call",
                        "premium": 2.50,
                        "cost": 250,
                        "expiration": "2024-01-19"
                    }},
                    "quantity": 1,
                    "allocation_percent": 25.0,
                    "reasoning": "Why this position"
                }}
            ],
            "total_cost": 800,
            "remaining_budget": 200,
            "expected_return": "20-35%",
            "risk_assessment": "Medium",
            "diversification_score": 8.5,
            "strategy_explanation": "Portfolio strategy overview",
            "risk_factors": ["Risk 1", "Risk 2"],
            "profit_scenarios": {{
                "best_case": "+45%",
                "likely_case": "+20%",
                "worst_case": "-100%"
            }}
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0]
            
            portfolio = json.loads(ai_response)
            
            # Add metadata
            portfolio.update({
                "analysis_timestamp": datetime.now().isoformat(),
                "ai_optimized": True,
                "total_opportunities_analyzed": len(opportunities),
                "requires_confirmation": True
            })
            
            return portfolio
            
        except Exception as e:
            logger.error(f"AI portfolio optimization failed: {e}")
            
            # Fallback: Simple diversified approach
            recommended = []
            budget_used = 0
            budget_per_position = total_budget / min(3, len(opportunities))
            
            for opp in opportunities[:3]:  # Take top 3
                best_option = opp.get('best_option', {})
                cost = best_option.get('cost', 0)
                
                if budget_used + cost <= total_budget:
                    recommended.append({
                        "symbol": opp.get('symbol'),
                        "option": best_option,
                        "quantity": 1,
                        "allocation_percent": (cost / total_budget) * 100,
                        "reasoning": "Fallback selection based on ranking"
                    })
                    budget_used += cost
            
            return {
                "recommended_portfolio": recommended,
                "total_cost": budget_used,
                "remaining_budget": total_budget - budget_used,
                "strategy_explanation": "Fallback diversified portfolio",
                "ai_optimized": False,
                "error": "AI optimization failed, using fallback strategy"
            }
    
    async def execute_portfolio_purchase(
        self, 
        portfolio: Dict[str, Any], 
        confirmed: bool = False
    ) -> Dict[str, Any]:
        """Execute the entire options portfolio if confirmed"""
        
        if not confirmed:
            return {"error": "User confirmation required before execution"}
        
        try:
            recommended_portfolio = portfolio.get('recommended_portfolio', [])
            
            if not recommended_portfolio:
                return {"error": "No portfolio to execute"}
            
            logger.info(f"🚀 Executing options portfolio with {len(recommended_portfolio)} positions")
            
            execution_results = []
            total_executed_cost = 0
            
            for position in recommended_portfolio:
                try:
                    symbol = position['symbol']
                    option_details = position['option']
                    quantity = position.get('quantity', 1)
                    
                    # Execute individual position
                    # In real implementation, this would call Alpaca trading API
                    execution = {
                        "symbol": symbol,
                        "option_symbol": f"{symbol}_{option_details.get('expiration', '')}_{option_details.get('type', 'C').upper()}{option_details.get('strike', '')}",
                        "quantity": quantity,
                        "premium_paid": option_details.get('premium'),
                        "total_cost": option_details.get('cost') * quantity,
                        "execution_time": datetime.now().isoformat(),
                        "order_id": f"OPT_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "status": "executed"
                    }
                    
                    execution_results.append(execution)
                    total_executed_cost += execution['total_cost']
                    
                    logger.info(f"✅ Executed {symbol} option: {execution}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute {position.get('symbol', 'unknown')}: {e}")
                    execution_results.append({
                        "symbol": position.get('symbol', 'unknown'),
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Portfolio execution summary
            successful_executions = [r for r in execution_results if r.get('status') == 'executed']
            failed_executions = [r for r in execution_results if r.get('status') == 'failed']
            
            return {
                "portfolio_status": "executed" if successful_executions else "failed",
                "total_positions": len(recommended_portfolio),
                "successful_executions": len(successful_executions),
                "failed_executions": len(failed_executions),
                "total_cost": total_executed_cost,
                "execution_details": execution_results,
                "execution_time": datetime.now().isoformat(),
                "confirmation_message": f"Portfolio execution completed: {len(successful_executions)}/{len(recommended_portfolio)} positions successful"
            }
            
        except Exception as e:
            logger.error(f"Failed to execute portfolio: {e}")
            return {"error": f"Portfolio execution failed: {str(e)}", "status": "failed"}


# Global instance
multi_options_agent = MultiOptionsBuyAgent()

async def analyze_multi_options_buy(budget: float, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main function to analyze multi-options portfolio"""
    if preferences is None:
        preferences = {
            "risk_tolerance": "moderate", 
            "diversification": "moderate", 
            "strategy": "growth"
        }
    
    return await multi_options_agent.analyze_best_options_portfolio(budget, preferences)

async def execute_multi_options_buy(portfolio: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
    """Main function to execute options portfolio"""
    return await multi_options_agent.execute_portfolio_purchase(portfolio, confirmed)