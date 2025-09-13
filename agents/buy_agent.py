"""
Buy Agent for Neural Options Oracle++
Intelligent trade execution based on AI analysis and decision engine signals
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from openai import OpenAI
from config.logging import get_agents_logger
from config.settings import settings

from .base_agent import BaseAgent

logger = get_agents_logger()


@dataclass
class TradeExecution:
    """Trade execution result"""
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    total_value: float
    order_type: str
    option_details: Optional[Dict[str, Any]] = None
    execution_time: datetime = None
    trade_id: str = ""
    status: str = "pending"  # pending, filled, rejected
    reasoning: str = ""


@dataclass
class PositionRecommendation:
    """Position recommendation from buy agent"""
    symbol: str
    action: str
    quantity: int
    option_type: Optional[str] = None  # 'call' or 'put'
    strike_price: Optional[float] = None
    expiration_date: Optional[str] = None
    entry_price: Optional[float] = None
    confidence: float = 0.0
    risk_score: float = 0.0
    potential_return: float = 0.0
    max_loss: float = 0.0
    reasoning: str = ""


class BuyAgent(BaseAgent):
    """Intelligent buy agent for paper trading execution"""
    
    def __init__(self, client: OpenAI):
        super().__init__(client, "Buy Agent", "gpt-4o")
        self.alpaca_client = None
        self.options_api = None
        self._initialize_trading_clients()
    
    def _initialize_trading_clients(self):
        """Initialize trading API clients"""
        try:
            # Initialize Alpaca client for paper trading
            from alpaca.trading.client import TradingClient
            self.alpaca_client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=True  # Paper trading only
            )
            logger.info("Alpaca trading client initialized for paper trading")
            
            # Initialize OptionsProfitCalculator API
            from src.data.openai_only_orchestrator import OptionsProfitCalculatorAPI
            self.options_api = OptionsProfitCalculatorAPI()
            logger.info("OptionsProfitCalculator API initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading clients: {e}")
            self.alpaca_client = None
            self.options_api = None
    
    def _get_system_instructions(self) -> str:
        """Get system instructions for the buy agent"""
        return """You are an intelligent buy agent for options trading. Your role is to:

1. **Analyze Trading Signals**: Review decision engine signals and determine optimal trade execution
2. **Risk Management**: Ensure all trades comply with user risk profile and position sizing rules
3. **Options Selection**: Choose optimal strikes and expirations based on market conditions
4. **Execution Planning**: Create detailed trade execution plans with entry/exit strategies
5. **Portfolio Integration**: Consider existing positions and portfolio balance

**Key Responsibilities**:
- Convert AI signals into actionable trades
- Select appropriate option contracts (calls/puts, strikes, expirations)
- Calculate position sizes based on risk tolerance
- Provide clear reasoning for all trade decisions
- Ensure compliance with paper trading rules

**Risk Management Rules**:
- Never risk more than user's max position size percentage
- Always provide stop-loss and take-profit levels
- Consider portfolio delta and overall risk exposure
- Validate trade feasibility before execution

**Response Format**: Always return structured JSON with trade recommendations, risk metrics, and detailed reasoning."""

    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Main analysis method - processes trading signals and generates buy recommendations"""
        try:
            logger.info(f"🎯 Buy agent analyzing {symbol} for trade execution")
            
            # Extract context from kwargs
            decision_signal = kwargs.get('decision_signal', {})
            user_risk_profile = kwargs.get('user_risk_profile', {})
            market_data = kwargs.get('market_data', {})
            strike_recommendations = kwargs.get('strike_recommendations', [])
            
            # Generate buy recommendations
            recommendations = await self._generate_buy_recommendations(
                symbol, decision_signal, user_risk_profile, market_data, strike_recommendations
            )
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(
                symbol, recommendations, user_risk_profile
            )
            
            return {
                'symbol': symbol,
                'recommendations': [rec.__dict__ for rec in recommendations],
                'execution_plan': execution_plan,
                'risk_assessment': self._assess_trade_risk(recommendations, user_risk_profile),
                'confidence': self._calculate_execution_confidence(recommendations),
                'timestamp': datetime.now().isoformat(),
                'agent_type': 'buy_agent'
            }
            
        except Exception as e:
            logger.error(f"Buy agent analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'recommendations': [],
                'execution_plan': {},
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'agent_type': 'buy_agent'
            }
    
    async def _generate_buy_recommendations(
        self,
        symbol: str,
        decision_signal: Dict[str, Any],
        user_risk_profile: Dict[str, Any],
        market_data: Dict[str, Any],
        strike_recommendations: List[Dict[str, Any]]
    ) -> List[PositionRecommendation]:
        """Generate buy recommendations based on decision signal"""
        
        recommendations = []
        
        try:
            # Extract signal information
            direction = decision_signal.get('direction', 'HOLD')
            confidence = decision_signal.get('confidence', 0.5)
            strategy_type = decision_signal.get('strategy_type', 'neutral')
            
            # Skip if signal is HOLD or confidence too low
            if direction == 'HOLD' or confidence < 0.3:
                logger.info(f"Signal for {symbol} is {direction} with {confidence:.2f} confidence - no buy recommendation")
                return recommendations
            
            # Get current market price
            current_price = market_data.get('quote', {}).get('price', 0)
            if current_price <= 0:
                logger.warning(f"No valid price data for {symbol}")
                return recommendations
            
            # Determine position size based on risk profile
            max_position_size = user_risk_profile.get('max_position_size', 0.05)  # 5% default
            account_balance = user_risk_profile.get('account_balance', 100000)  # $100k default
            max_trade_value = account_balance * max_position_size
            
            # Generate recommendations based on signal direction
            if direction in ['BUY', 'STRONG_BUY']:
                recommendations.extend(await self._generate_bullish_recommendations(
                    symbol, current_price, max_trade_value, strike_recommendations, confidence
                ))
            elif direction in ['SELL', 'STRONG_SELL']:
                recommendations.extend(await self._generate_bearish_recommendations(
                    symbol, current_price, max_trade_value, strike_recommendations, confidence
                ))
            
            # Filter and rank recommendations
            recommendations = self._filter_and_rank_recommendations(recommendations, user_risk_profile)
            
            logger.info(f"Generated {len(recommendations)} buy recommendations for {symbol}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate buy recommendations for {symbol}: {e}")
            return []
    
    async def _generate_bullish_recommendations(
        self,
        symbol: str,
        current_price: float,
        max_trade_value: float,
        strike_recommendations: List[Dict[str, Any]],
        confidence: float
    ) -> List[PositionRecommendation]:
        """Generate bullish (call) recommendations"""
        
        recommendations = []
        
        try:
            # Get options data for the symbol
            options_data = await self._get_options_data(symbol)
            if not options_data:
                logger.warning(f"No options data available for {symbol}")
                return recommendations
            
            # Use strike recommendations if available, otherwise generate our own
            if strike_recommendations:
                for strike_rec in strike_recommendations[:3]:  # Top 3 recommendations
                    if isinstance(strike_rec, dict) and strike_rec.get('option_type') == 'call':
                        rec = PositionRecommendation(
                            symbol=symbol,
                            action='buy',
                            quantity=1,  # Will be calculated based on risk
                            option_type='call',
                            strike_price=strike_rec.get('strike', current_price * 1.02),
                            expiration_date=strike_rec.get('expiration', self._get_next_expiration()),
                            entry_price=strike_rec.get('entry_price', 5.0),
                            confidence=confidence,
                            risk_score=strike_rec.get('risk_score', 0.3),
                            potential_return=strike_rec.get('potential_return', 0.15),
                            max_loss=strike_rec.get('max_loss', 0.05),
                            reasoning=f"AI signal: {direction} with {confidence:.1%} confidence"
                        )
                        recommendations.append(rec)
            else:
                # Generate default call recommendations
                strikes = [
                    current_price * 1.01,  # Slightly OTM
                    current_price * 1.02,  # OTM
                    current_price * 0.99   # Slightly ITM
                ]
                
                for i, strike in enumerate(strikes):
                    rec = PositionRecommendation(
                        symbol=symbol,
                        action='buy',
                        quantity=1,
                        option_type='call',
                        strike_price=strike,
                        expiration_date=self._get_next_expiration(),
                        entry_price=5.0 + i * 2.0,  # Estimated option price
                        confidence=confidence * (0.8 + i * 0.1),
                        risk_score=0.2 + i * 0.1,
                        potential_return=0.15 + i * 0.05,
                        max_loss=0.05,
                        reasoning=f"Bullish call recommendation based on {confidence:.1%} confidence signal"
                    )
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate bullish recommendations: {e}")
            return []
    
    async def _generate_bearish_recommendations(
        self,
        symbol: str,
        current_price: float,
        max_trade_value: float,
        strike_recommendations: List[Dict[str, Any]],
        confidence: float
    ) -> List[PositionRecommendation]:
        """Generate bearish (put) recommendations"""
        
        recommendations = []
        
        try:
            # Use strike recommendations if available
            if strike_recommendations:
                for strike_rec in strike_recommendations[:3]:
                    if isinstance(strike_rec, dict) and strike_rec.get('option_type') == 'put':
                        rec = PositionRecommendation(
                            symbol=symbol,
                            action='buy',
                            quantity=1,
                            option_type='put',
                            strike_price=strike_rec.get('strike', current_price * 0.98),
                            expiration_date=strike_rec.get('expiration', self._get_next_expiration()),
                            entry_price=strike_rec.get('entry_price', 4.0),
                            confidence=confidence,
                            risk_score=strike_rec.get('risk_score', 0.3),
                            potential_return=strike_rec.get('potential_return', 0.15),
                            max_loss=strike_rec.get('max_loss', 0.05),
                            reasoning=f"AI signal: {direction} with {confidence:.1%} confidence"
                        )
                        recommendations.append(rec)
            else:
                # Generate default put recommendations
                strikes = [
                    current_price * 0.99,  # Slightly OTM
                    current_price * 0.98,  # OTM
                    current_price * 1.01   # Slightly ITM
                ]
                
                for i, strike in enumerate(strikes):
                    rec = PositionRecommendation(
                        symbol=symbol,
                        action='buy',
                        quantity=1,
                        option_type='put',
                        strike_price=strike,
                        expiration_date=self._get_next_expiration(),
                        entry_price=4.0 + i * 2.0,
                        confidence=confidence * (0.8 + i * 0.1),
                        risk_score=0.2 + i * 0.1,
                        potential_return=0.15 + i * 0.05,
                        max_loss=0.05,
                        reasoning=f"Bearish put recommendation based on {confidence:.1%} confidence signal"
                    )
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate bearish recommendations: {e}")
            return []
    
    async def _get_options_data(self, symbol: str) -> Dict[str, Any]:
        """Get options data for the symbol"""
        try:
            if self.options_api:
                async with self.options_api as api:
                    return await api.get_comprehensive_options_data(symbol)
            return {}
        except Exception as e:
            logger.error(f"Failed to get options data for {symbol}: {e}")
            return {}
    
    def _get_next_expiration(self) -> str:
        """Get next Friday expiration date"""
        today = datetime.now()
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_friday = today + timedelta(days=days_ahead)
        return next_friday.strftime('%Y-%m-%d')
    
    def _filter_and_rank_recommendations(
        self,
        recommendations: List[PositionRecommendation],
        user_risk_profile: Dict[str, Any]
    ) -> List[PositionRecommendation]:
        """Filter and rank recommendations based on risk profile"""
        
        # Filter by risk tolerance
        max_risk = user_risk_profile.get('max_risk_score', 0.5)
        filtered = [rec for rec in recommendations if rec.risk_score <= max_risk]
        
        # Sort by risk-adjusted return (potential_return / risk_score)
        filtered.sort(key=lambda x: x.potential_return / max(x.risk_score, 0.1), reverse=True)
        
        # Return top 3 recommendations
        return filtered[:3]
    
    async def _create_execution_plan(
        self,
        symbol: str,
        recommendations: List[PositionRecommendation],
        user_risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed execution plan for recommendations"""
        
        if not recommendations:
            return {'status': 'no_recommendations', 'message': 'No suitable trades found'}
        
        # Select best recommendation
        best_rec = recommendations[0]
        
        # Calculate position size
        account_balance = user_risk_profile.get('account_balance', 100000)
        max_position_size = user_risk_profile.get('max_position_size', 0.05)
        max_trade_value = account_balance * max_position_size
        
        # Calculate quantity based on option price and max trade value
        if best_rec.entry_price and best_rec.entry_price > 0:
            max_contracts = int(max_trade_value / (best_rec.entry_price * 100))  # Options are 100 shares per contract
            quantity = min(max_contracts, 10)  # Cap at 10 contracts
        else:
            quantity = 1
        
        execution_plan = {
            'symbol': symbol,
            'recommended_trade': {
                'action': best_rec.action,
                'option_type': best_rec.option_type,
                'strike_price': best_rec.strike_price,
                'expiration_date': best_rec.expiration_date,
                'quantity': quantity,
                'estimated_price': best_rec.entry_price,
                'total_cost': quantity * best_rec.entry_price * 100,
                'confidence': best_rec.confidence,
                'risk_score': best_rec.risk_score,
                'potential_return': best_rec.potential_return,
                'max_loss': best_rec.max_loss,
                'reasoning': best_rec.reasoning
            },
            'risk_management': {
                'stop_loss': best_rec.entry_price * 0.5,  # 50% stop loss
                'take_profit': best_rec.entry_price * 2.0,  # 100% profit target
                'max_position_size_pct': max_position_size,
                'portfolio_impact': (quantity * best_rec.entry_price * 100) / account_balance
            },
            'execution_timing': {
                'recommended_entry': 'market_open',  # or 'immediate'
                'time_sensitivity': 'medium',
                'market_conditions': 'favorable'
            },
            'alternatives': [
                {
                    'rank': i + 2,
                    'option_type': rec.option_type,
                    'strike_price': rec.strike_price,
                    'confidence': rec.confidence,
                    'reasoning': rec.reasoning
                }
                for i, rec in enumerate(recommendations[1:3])
            ]
        }
        
        return execution_plan
    
    def _assess_trade_risk(
        self,
        recommendations: List[PositionRecommendation],
        user_risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall risk of the trade recommendations"""
        
        if not recommendations:
            return {'risk_level': 'none', 'message': 'No recommendations to assess'}
        
        # Calculate portfolio risk metrics
        total_risk_score = sum(rec.risk_score for rec in recommendations) / len(recommendations)
        max_potential_loss = max(rec.max_loss for rec in recommendations)
        avg_confidence = sum(rec.confidence for rec in recommendations) / len(recommendations)
        
        # Determine risk level
        if total_risk_score < 0.3 and max_potential_loss < 0.05:
            risk_level = 'low'
        elif total_risk_score < 0.6 and max_potential_loss < 0.10:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'average_risk_score': total_risk_score,
            'max_potential_loss': max_potential_loss,
            'average_confidence': avg_confidence,
            'recommendation_count': len(recommendations),
            'risk_tolerance_match': total_risk_score <= user_risk_profile.get('max_risk_score', 0.5),
            'warnings': self._generate_risk_warnings(recommendations, user_risk_profile)
        }
    
    def _generate_risk_warnings(
        self,
        recommendations: List[PositionRecommendation],
        user_risk_profile: Dict[str, Any]
    ) -> List[str]:
        """Generate risk warnings for the recommendations"""
        
        warnings = []
        
        # Check risk score
        high_risk_recs = [rec for rec in recommendations if rec.risk_score > 0.6]
        if high_risk_recs:
            warnings.append(f"{len(high_risk_recs)} high-risk recommendations detected")
        
        # Check confidence
        low_confidence_recs = [rec for rec in recommendations if rec.confidence < 0.4]
        if low_confidence_recs:
            warnings.append(f"{len(low_confidence_recs)} low-confidence recommendations")
        
        # Check potential loss
        high_loss_recs = [rec for rec in recommendations if rec.max_loss > 0.10]
        if high_loss_recs:
            warnings.append(f"{len(high_loss_recs)} recommendations with >10% max loss")
        
        return warnings
    
    def _calculate_execution_confidence(self, recommendations: List[PositionRecommendation]) -> float:
        """Calculate overall confidence in the execution plan"""
        
        if not recommendations:
            return 0.0
        
        # Weight by recommendation quality
        total_confidence = 0.0
        total_weight = 0.0
        
        for i, rec in enumerate(recommendations):
            weight = 1.0 / (i + 1)  # Higher weight for better recommendations
            total_confidence += rec.confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    async def execute_trade(
        self,
        symbol: str,
        recommendation: PositionRecommendation,
        user_risk_profile: Dict[str, Any]
    ) -> TradeExecution:
        """Execute a trade based on recommendation"""
        
        try:
            logger.info(f"🎯 Executing trade: {recommendation.action} {recommendation.quantity} {symbol} {recommendation.option_type}")
            
            # Validate trade before execution
            validation_result = await self._validate_trade(recommendation, user_risk_profile)
            if not validation_result['valid']:
                return TradeExecution(
                    symbol=symbol,
                    action=recommendation.action,
                    quantity=recommendation.quantity,
                    price=0.0,
                    total_value=0.0,
                    order_type='market',
                    status='rejected',
                    reasoning=f"Trade validation failed: {validation_result['reason']}"
                )
            
            # Execute the trade (paper trading)
            execution = await self._execute_paper_trade(symbol, recommendation)
            
            logger.info(f"✅ Trade executed: {execution.trade_id} - {execution.status}")
            return execution
            
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
            return TradeExecution(
                symbol=symbol,
                action=recommendation.action,
                quantity=recommendation.quantity,
                price=0.0,
                total_value=0.0,
                order_type='market',
                status='rejected',
                reasoning=f"Execution failed: {str(e)}"
            )
    
    async def _validate_trade(
        self,
        recommendation: PositionRecommendation,
        user_risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trade before execution"""
        
        # Check risk limits
        if recommendation.risk_score > user_risk_profile.get('max_risk_score', 0.5):
            return {'valid': False, 'reason': 'Risk score exceeds user limits'}
        
        # Check confidence
        if recommendation.confidence < 0.3:
            return {'valid': False, 'reason': 'Confidence too low for execution'}
        
        # Check position size
        max_position_size = user_risk_profile.get('max_position_size', 0.05)
        account_balance = user_risk_profile.get('account_balance', 100000)
        trade_value = recommendation.quantity * recommendation.entry_price * 100
        
        if trade_value > account_balance * max_position_size:
            return {'valid': False, 'reason': 'Position size exceeds limits'}
        
        return {'valid': True, 'reason': 'Trade validated successfully'}
    
    async def _execute_paper_trade(
        self,
        symbol: str,
        recommendation: PositionRecommendation
    ) -> TradeExecution:
        """Execute paper trade using Alpaca API"""
        
        try:
            # For paper trading, we'll simulate the execution
            # In a real implementation, this would use the Alpaca API
            
            execution_price = recommendation.entry_price
            total_value = recommendation.quantity * execution_price * 100
            
            execution = TradeExecution(
                symbol=symbol,
                action=recommendation.action,
                quantity=recommendation.quantity,
                price=execution_price,
                total_value=total_value,
                order_type='market',
                option_details={
                    'type': recommendation.option_type,
                    'strike': recommendation.strike_price,
                    'expiration': recommendation.expiration_date
                },
                execution_time=datetime.now(),
                trade_id=f"paper_{symbol}_{int(datetime.now().timestamp())}",
                status='filled',
                reasoning=recommendation.reasoning
            )
            
            # Log the trade execution
            logger.info(f"Paper trade executed: {execution.trade_id}")
            
            return execution
            
        except Exception as e:
            logger.error(f"Paper trade execution failed: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get buy agent status"""
        base_status = await super().get_status()
        base_status.update({
            'alpaca_connected': self.alpaca_client is not None,
            'options_api_available': self.options_api is not None,
            'trading_mode': 'paper_trading',
            'last_check': datetime.now().isoformat()
        })
        return base_status
