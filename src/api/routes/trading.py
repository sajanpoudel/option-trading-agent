"""
Neural Options Oracle++ Trading API Routes
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import time

from config.database import db_manager
from config.logging import get_api_logger
from src.api.dependencies import get_current_session, rate_limiter

logger = get_api_logger()
router = APIRouter()


# Request/Response Models
class PaperTradeRequest(BaseModel):
    """Paper trade execution request"""
    symbol: str = Field(..., description="Stock symbol")
    action: str = Field(..., description="buy or sell")
    quantity: int = Field(..., description="Number of shares/contracts")
    order_type: str = Field("market", description="market, limit, stop")
    price: Optional[float] = Field(None, description="Price for limit orders")
    option_details: Optional[Dict[str, Any]] = Field(None, description="Option contract details")


class BuyAnalysisRequest(BaseModel):
    """Buy analysis request"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    user_query: Optional[str] = Field(None, description="User's trading query")
    risk_profile: Optional[Dict[str, Any]] = Field(None, description="User risk profile")


class TradeExecutionRequest(BaseModel):
    """Trade execution request"""
    symbol: str = Field(..., description="Stock symbol")
    recommendation_id: str = Field(..., description="ID of the recommendation to execute")
    quantity: Optional[int] = Field(None, description="Override quantity")
    user_risk_profile: Dict[str, Any] = Field(..., description="User risk profile")


class TradeResponse(BaseModel):
    """Trade execution response"""
    trade_id: str
    status: str
    symbol: str
    quantity: int
    price: float
    total_value: float
    timestamp: float
    estimated_fill_time: Optional[float] = None


class PositionResponse(BaseModel):
    """Position information response"""
    position_id: str
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    position_type: str
    status: str


# Trading endpoints
@router.get("/")
async def trading_info() -> Dict[str, Any]:
    """Get trading API information"""
    
    return {
        "name": "Trading API",
        "version": "1.0.0",
        "description": "Paper trading execution and position management",
        "mode": "paper_trading",
        "endpoints": {
            "execute": "/execute",
            "positions": "/positions",
            "orders": "/orders",
            "portfolio": "/portfolio"
        },
        "features": [
            "Paper trading execution",
            "Real-time position tracking",
            "Options trading support",
            "Portfolio management",
            "Risk management"
        ]
    }


@router.post("/execute")
@rate_limiter(max_requests=20, time_window=60)
async def execute_paper_trade(
    trade_request: PaperTradeRequest,
    session: Dict = Depends(get_current_session)
) -> TradeResponse:
    """Execute paper trade"""
    
    symbol = trade_request.symbol.upper()
    logger.info(f"Executing paper trade: {trade_request.action} {trade_request.quantity} {symbol}")
    
    try:
        # Validate trade request
        if trade_request.action not in ["buy", "sell"]:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'buy' or 'sell'")
        
        if trade_request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        # Mock trade execution - will be replaced with Alpaca integration
        mock_price = 155.25  # This would come from real market data
        
        if trade_request.order_type == "limit" and trade_request.price:
            execution_price = trade_request.price
        else:
            execution_price = mock_price
        
        # Calculate trade value
        if trade_request.option_details:
            # Options trade (multiply by 100 for contract size)
            total_value = execution_price * trade_request.quantity * 100
            position_type = "option"
        else:
            # Stock trade
            total_value = execution_price * trade_request.quantity
            position_type = "stock"
        
        # Create position in database
        position_data = {
            "symbol": symbol,
            "position_type": position_type,
            "option_type": trade_request.option_details.get("type") if trade_request.option_details else None,
            "strike_price": trade_request.option_details.get("strike") if trade_request.option_details else None,
            "expiration_date": trade_request.option_details.get("expiration") if trade_request.option_details else None,
            "quantity": trade_request.quantity if trade_request.action == "buy" else -trade_request.quantity,
            "entry_price": execution_price,
            "current_price": execution_price,
            "entry_order_id": f"paper_{symbol}_{int(time.time())}"
        }
        
        position_id = await db_manager.create_position(position_data)
        
        trade_response = {
            "trade_id": position_id or f"trade_{symbol}_{int(time.time())}",
            "status": "filled",
            "symbol": symbol,
            "quantity": trade_request.quantity,
            "price": execution_price,
            "total_value": total_value,
            "timestamp": time.time()
        }
        
        logger.info(f"Paper trade executed: {symbol} @ ${execution_price}")
        
        return TradeResponse(**trade_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trade execution failed: {str(e)}")


@router.get("/positions")
async def get_positions(
    symbol: Optional[str] = None,
    status: str = "open",
    session: Dict = Depends(get_current_session)
) -> List[PositionResponse]:
    """Get current positions"""
    
    try:
        positions = await db_manager.get_open_positions()
        
        # Filter by symbol if provided
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol.upper()]
        
        # Convert to response format
        position_responses = []
        for pos in positions:
            response = PositionResponse(
                position_id=pos["id"],
                symbol=pos["symbol"],
                quantity=pos["quantity"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                unrealized_pnl=pos["unrealized_pnl"] or 0,
                unrealized_pnl_percent=pos["unrealized_pnl_percent"] or 0,
                position_type=pos["position_type"],
                status=pos["status"]
            )
            position_responses.append(response)
        
        return position_responses
        
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")


@router.get("/positions/{position_id}")
async def get_position_details(
    position_id: str,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get detailed position information"""
    
    try:
        # Mock position details - in production would query database
        return {
            "position_id": position_id,
            "symbol": "AAPL",
            "quantity": 1,
            "entry_price": 155.25,
            "current_price": 157.30,
            "unrealized_pnl": 205.0,
            "unrealized_pnl_percent": 1.32,
            "position_type": "option",
            "option_details": {
                "type": "call",
                "strike": 155.0,
                "expiration": "2024-03-15",
                "delta": 0.65,
                "gamma": 0.05,
                "theta": -0.12,
                "vega": 0.18
            },
            "entry_date": time.time() - 3600,
            "status": "open"
        }
        
    except Exception as e:
        logger.error(f"Failed to get position details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get position details")


@router.post("/positions/{position_id}/close")
@rate_limiter(max_requests=20, time_window=60)
async def close_position(
    position_id: str,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Close a position"""
    
    try:
        # Mock position close - will be replaced with actual implementation
        logger.info(f"Closing position: {position_id}")
        
        return {
            "position_id": position_id,
            "status": "closed",
            "close_price": 157.30,
            "realized_pnl": 205.0,
            "close_timestamp": time.time(),
            "message": "Position closed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        raise HTTPException(status_code=500, detail="Failed to close position")


@router.get("/orders")
async def get_orders(
    symbol: Optional[str] = None,
    status: str = "all",
    limit: int = 20,
    session: Dict = Depends(get_current_session)
) -> List[Dict[str, Any]]:
    """Get order history"""
    
    try:
        # Mock order history
        orders = [
            {
                "order_id": f"order_{i}",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1,
                "order_type": "market",
                "status": "filled",
                "fill_price": 155.25,
                "submitted_at": time.time() - (i * 3600),
                "filled_at": time.time() - (i * 3600) + 60
            }
            for i in range(min(limit, 5))
        ]
        
        return orders
        
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        raise HTTPException(status_code=500, detail="Failed to get orders")


@router.get("/portfolio/summary")
async def get_portfolio_summary(
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get portfolio summary"""
    
    try:
        # Get portfolio data from database
        analytics = await db_manager.get_system_analytics()
        portfolio_summary = analytics.get("portfolio_summary", {})
        
        # Add mock data if empty
        if not portfolio_summary:
            portfolio_summary = {
                "total_positions": 3,
                "total_value": 50000.0,
                "total_unrealized_pnl": 1250.0,
                "avg_return_percent": 2.5,
                "portfolio_delta": 125.0,
                "portfolio_gamma": 8.5,
                "portfolio_theta": -45.0,
                "portfolio_vega": 180.0,
                "calculated_at": time.time()
            }
        
        return portfolio_summary
        
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio summary")


@router.get("/portfolio/performance")
async def get_portfolio_performance(
    period: str = "1m",
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get portfolio performance metrics"""
    
    try:
        # Mock performance data
        performance = {
            "period": period,
            "total_return": 2500.0,
            "total_return_percent": 5.0,
            "win_rate": 68.5,
            "profit_factor": 1.85,
            "max_drawdown": -1200.0,
            "sharpe_ratio": 1.25,
            "trades": {
                "total": 15,
                "winning": 10,
                "losing": 5,
                "avg_win": 450.0,
                "avg_loss": -180.0
            },
            "calculated_at": time.time()
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get portfolio performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio performance")


@router.post("/analyze-buy")
@rate_limiter(max_requests=10, time_window=60)
async def analyze_buy_opportunity(
    request: BuyAnalysisRequest,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Analyze buy opportunity using AI agents"""
    
    symbol = request.symbol.upper()
    logger.info(f"🎯 Analyzing buy opportunity for {symbol}")
    
    try:
        # Import the intelligent orchestrator
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        
        orchestrator = IntelligentOrchestrator()
        
        # Set up user context
        user_context = {
            'selectedStock': symbol,
            'risk_profile': request.risk_profile or {
                'risk_level': 'moderate',
                'experience': 'intermediate',
                'max_position_size': 0.05,
                'account_balance': 100000
            }
        }
        
        # Process the buy request
        user_query = request.user_query or f"analyze and buy {symbol}"
        result = await orchestrator.process_user_query(user_query, user_context)
        
        # Extract buy recommendations
        buy_agent_result = result.get('frontend_data', {}).get('buy_agent', {})
        buy_analysis = buy_agent_result.get('buy_analysis', {})
        
        return {
            'symbol': symbol,
            'analysis_complete': True,
            'buy_recommendations': buy_analysis.get('recommendations', []),
            'execution_plan': buy_analysis.get('execution_plan', {}),
            'risk_assessment': buy_analysis.get('risk_assessment', {}),
            'confidence': buy_analysis.get('confidence', 0.0),
            'timestamp': time.time(),
            'user_query': user_query
        }
        
    except Exception as e:
        logger.error(f"Buy analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Buy analysis failed: {str(e)}")


@router.post("/execute-recommendation")
@rate_limiter(max_requests=5, time_window=60)
async def execute_trade_recommendation(
    request: TradeExecutionRequest,
    session: Dict = Depends(get_current_session)
) -> TradeResponse:
    """Execute a trade based on AI recommendation"""
    
    symbol = request.symbol.upper()
    logger.info(f"🎯 Executing trade recommendation for {symbol}")
    
    try:
        # Import the buy agent
        from agents.buy_agent import BuyAgent, PositionRecommendation
        from openai import OpenAI
        from config.settings import settings
        
        # Initialize buy agent
        openai_client = OpenAI(api_key=settings.openai_api_key)
        buy_agent = BuyAgent(openai_client)
        
        # For now, create a mock recommendation based on the request
        # In a real implementation, you would fetch the actual recommendation by ID
        recommendation = PositionRecommendation(
            symbol=symbol,
            action='buy',
            quantity=request.quantity or 1,
            option_type='call',
            strike_price=150.0,  # Mock strike
            expiration_date='2024-01-19',  # Mock expiration
            entry_price=5.0,  # Mock entry price
            confidence=0.8,
            risk_score=0.3,
            potential_return=0.15,
            max_loss=0.05,
            reasoning="AI-generated recommendation"
        )
        
        # Execute the trade
        execution = await buy_agent.execute_trade(symbol, recommendation, request.user_risk_profile)
        
        # Create trade response
        trade_response = TradeResponse(
            trade_id=execution.trade_id,
            status=execution.status,
            symbol=symbol,
            quantity=execution.quantity,
            price=execution.price,
            total_value=execution.total_value,
            timestamp=time.time(),
            estimated_fill_time=time.time() + 60  # 1 minute estimated fill
        )
        
        logger.info(f"✅ Trade executed: {execution.trade_id} - {execution.status}")
        return trade_response
        
    except Exception as e:
        logger.error(f"Trade execution failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Trade execution failed: {str(e)}")


@router.get("/buy-recommendations/{symbol}")
async def get_buy_recommendations(
    symbol: str,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get buy recommendations for a symbol"""
    
    symbol = symbol.upper()
    logger.info(f"📊 Getting buy recommendations for {symbol}")
    
    try:
        # Import the intelligent orchestrator
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        
        orchestrator = IntelligentOrchestrator()
        
        # Set up user context
        user_context = {
            'selectedStock': symbol,
            'risk_profile': {
                'risk_level': 'moderate',
                'experience': 'intermediate',
                'max_position_size': 0.05,
                'account_balance': 100000
            }
        }
        
        # Process buy analysis
        result = await orchestrator.process_user_query(f"buy {symbol}", user_context)
        
        # Extract buy recommendations
        buy_agent_result = result.get('frontend_data', {}).get('buy_agent', {})
        buy_analysis = buy_agent_result.get('buy_analysis', {})
        
        return {
            'symbol': symbol,
            'recommendations': buy_analysis.get('recommendations', []),
            'execution_plan': buy_analysis.get('execution_plan', {}),
            'risk_assessment': buy_analysis.get('risk_assessment', {}),
            'confidence': buy_analysis.get('confidence', 0.0),
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get buy recommendations for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get buy recommendations")