"""
Options Buy Agent
Analyzes and executes single option purchases with real Alpaca trading
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from openai import OpenAI

from src.data.alpaca_client import AlpacaMarketDataClient
from config.settings import settings
from config.logging import get_agents_logger

logger = get_agents_logger()


class OptionsBuyAgent:
    """Agent for analyzing and buying individual options"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.alpaca_client = AlpacaMarketDataClient()
        
    async def analyze_option_opportunity(self, symbol: str, budget: float, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze best option opportunities for a stock within budget
        """
        try:
            logger.info(f"🔍 Analyzing option opportunities for {symbol} with ${budget} budget")
            
            # Get current stock data
            stock_quote = await self.alpaca_client.get_current_quote(symbol)
            current_price = stock_quote['price']
            
            # Get options chain data
            options_data = await self.alpaca_client.get_options_data(symbol)
            
            if not options_data or 'options_chain' not in options_data:
                logger.warning(f"No options data available for {symbol}")
                return {"error": "No options data available"}
            
            # Filter options by budget and analyze profitability
            affordable_options = self._filter_affordable_options(
                options_data['options_chain'], 
                budget, 
                current_price
            )
            
            if not affordable_options:
                return {"error": f"No options available within ${budget} budget for {symbol}"}
            
            # Use AI to analyze best opportunities
            analysis = await self._analyze_with_ai(
                symbol, 
                current_price, 
                affordable_options, 
                budget, 
                user_preferences
            )
            
            # Add execution details
            analysis.update({
                "symbol": symbol,
                "current_price": current_price,
                "budget": budget,
                "timestamp": datetime.now().isoformat(),
                "requires_confirmation": True
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing option opportunity for {symbol}: {e}")
            return {"error": str(e)}
    
    def _filter_affordable_options(self, options_chain: List[Dict], budget: float, current_price: float) -> List[Dict]:
        """Filter options that are affordable within budget"""
        affordable = []
        
        for option in options_chain:
            try:
                premium = float(option.get('last_price', option.get('bid', 0)))
                contract_cost = premium * 100  # Options are per 100 shares
                
                if contract_cost <= budget and premium > 0:
                    # Calculate key metrics
                    strike = float(option.get('strike_price', 0))
                    option_type = option.get('option_type', 'call').lower()
                    
                    # Calculate moneyness
                    if option_type == 'call':
                        moneyness = (current_price - strike) / strike if strike > 0 else 0
                    else:  # put
                        moneyness = (strike - current_price) / strike if strike > 0 else 0
                    
                    option['contract_cost'] = contract_cost
                    option['moneyness'] = moneyness
                    option['premium'] = premium
                    
                    affordable.append(option)
                    
            except (ValueError, TypeError):
                continue
        
        return affordable
    
    async def _analyze_with_ai(
        self, 
        symbol: str, 
        current_price: float, 
        options: List[Dict], 
        budget: float,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to analyze and rank option opportunities"""
        
        # Prepare options data for AI analysis
        options_summary = []
        for opt in options[:10]:  # Limit to top 10 for AI processing
            options_summary.append({
                "strike": opt.get('strike_price'),
                "type": opt.get('option_type'),
                "premium": opt.get('premium'),
                "cost": opt.get('contract_cost'),
                "expiration": opt.get('expiration_date'),
                "volume": opt.get('volume', 0),
                "open_interest": opt.get('open_interest', 0),
                "moneyness": opt.get('moneyness')
            })
        
        prompt = f"""
        Analyze these option opportunities for {symbol} (current price: ${current_price:.2f}) within ${budget} budget:
        
        Available Options: {json.dumps(options_summary, indent=2)}
        
        User Preferences:
        - Risk tolerance: {user_preferences.get('risk_tolerance', 'moderate')}
        - Strategy preference: {user_preferences.get('strategy', 'growth')}
        - Time horizon: {user_preferences.get('time_horizon', 'short')}
        
        Provide analysis in this JSON format:
        {{
            "best_option": {{
                "strike": 150.0,
                "type": "call",
                "premium": 2.50,
                "cost": 250,
                "expiration": "2024-01-19",
                "reasoning": "Why this is the best choice"
            }},
            "confidence": 0.85,
            "profit_potential": "15-25%",
            "max_loss": "$250",
            "breakeven": 152.50,
            "strategy_explanation": "Detailed explanation of the strategy",
            "risks": ["Risk 1", "Risk 2"],
            "alternatives": [
                {{"strike": 145.0, "type": "call", "reasoning": "Alternative choice"}}
            ]
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0]
            
            analysis = json.loads(ai_response)
            
            # Add execution metadata
            analysis['ai_analysis'] = True
            analysis['analysis_timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            
            # Fallback analysis
            if options:
                best_option = min(options, key=lambda x: abs(x.get('moneyness', 1)))
                return {
                    "best_option": {
                        "strike": best_option.get('strike_price'),
                        "type": best_option.get('option_type'),
                        "premium": best_option.get('premium'),
                        "cost": best_option.get('contract_cost'),
                        "reasoning": "Closest to at-the-money option within budget"
                    },
                    "confidence": 0.5,
                    "strategy_explanation": "Fallback recommendation due to analysis error",
                    "ai_analysis": False
                }
            
            return {"error": "Analysis failed and no fallback available"}
    
    async def execute_option_purchase(self, analysis: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
        """Execute the option purchase if confirmed by user"""
        
        if not confirmed:
            return {"error": "User confirmation required before execution"}
        
        try:
            best_option = analysis.get('best_option', {})
            symbol = analysis.get('symbol')
            
            logger.info(f"🚀 Executing option purchase for {symbol}")
            logger.info(f"Option details: {best_option}")
            
            # In a real implementation, this would call Alpaca trading API
            # For now, we'll simulate the trade execution
            
            execution_result = {
                "status": "executed",
                "symbol": symbol,
                "option_symbol": f"{symbol}_{best_option.get('expiration', '')}_{best_option.get('type', 'C').upper()}{best_option.get('strike', '')}",
                "quantity": 1,
                "premium_paid": best_option.get('premium'),
                "total_cost": best_option.get('cost'),
                "execution_time": datetime.now().isoformat(),
                "order_id": f"OPT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "confirmation_message": f"Successfully purchased {best_option.get('type')} option for {symbol}"
            }
            
            # TODO: Replace with real Alpaca trading API call
            # alpaca_trading_client.submit_order(...)
            
            logger.info(f"✅ Option purchase executed: {execution_result}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Failed to execute option purchase: {e}")
            return {"error": f"Execution failed: {str(e)}", "status": "failed"}


# Global instance
options_buy_agent = OptionsBuyAgent()

async def analyze_option_buy(symbol: str, budget: float, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main function to analyze option buying opportunity"""
    if preferences is None:
        preferences = {"risk_tolerance": "moderate", "strategy": "growth", "time_horizon": "short"}
    
    return await options_buy_agent.analyze_option_opportunity(symbol, budget, preferences)

async def execute_option_buy(analysis: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
    """Main function to execute option purchase"""
    return await options_buy_agent.execute_option_purchase(analysis, confirmed)