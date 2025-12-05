"""
Neural Options Oracle++ Portfolio API Routes
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import time

from backend.config.database import db_manager
from backend.config.logging import get_api_logger
from backend.app.api.dependencies import get_current_session

logger = get_api_logger()
router = APIRouter()


# Response Models
class PortfolioSummary(BaseModel):
    """Portfolio summary response"""
    total_value: float
    total_positions: int
    unrealized_pnl: float
    unrealized_pnl_percent: float
    cash_balance: float
    buying_power: float
    day_change: float
    day_change_percent: float


class PortfolioGreeks(BaseModel):
    """Portfolio Greeks response"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    net_exposure: float


class RiskMetrics(BaseModel):
    """Risk metrics response"""
    var_95: float  # Value at Risk 95%
    max_loss_scenario: float
    beta: float
    correlation_spy: float
    concentration_risk: Dict[str, float]


# Portfolio endpoints
@router.get("/")
async def portfolio_info() -> Dict[str, Any]:
    """Get portfolio API information"""
    
    return {
        "name": "Portfolio API",
        "version": "1.0.0",
        "description": "Portfolio management and risk analytics",
        "endpoints": {
            "summary": "/summary",
            "positions": "/positions",
            "greeks": "/greeks",
            "risk": "/risk",
            "performance": "/performance",
            "allocation": "/allocation"
        },
        "features": [
            "Real-time portfolio tracking",
            "Greeks aggregation",
            "Risk analytics",
            "Performance attribution",
            "Asset allocation analysis"
        ]
    }


@router.get("/summary")
async def get_portfolio_summary(
    session: Dict = Depends(get_current_session)
) -> PortfolioSummary:
    """Get portfolio summary"""
    
    try:
        # Get real portfolio data from database
        analytics = await db_manager.get_system_analytics()
        portfolio_summary_list = analytics.get("portfolio_summary", [])
        
        # Extract first item if list is not empty, otherwise use empty dict
        portfolio_data = portfolio_summary_list[0] if portfolio_summary_list else {}
        
        # Mock data if database is empty
        if not portfolio_data or not portfolio_data.get("total_value"):
            portfolio_data = {
                "total_value": 52500.0,
                "total_positions": 5,
                "total_unrealized_pnl": 2500.0,
                "avg_return_percent": 5.0,
                "cash_balance": 47500.0,
                "buying_power": 95000.0
            }
        
        # Calculate additional metrics
        unrealized_pnl_percent = (portfolio_data.get("total_unrealized_pnl", 0) / 50000.0) * 100
        day_change = 750.0  # Mock daily change
        day_change_percent = 1.5
        
        return PortfolioSummary(
            total_value=portfolio_data.get("total_value", 0),
            total_positions=portfolio_data.get("total_positions", 0),
            unrealized_pnl=portfolio_data.get("total_unrealized_pnl", 0),
            unrealized_pnl_percent=unrealized_pnl_percent,
            cash_balance=portfolio_data.get("cash_balance", 50000.0),
            buying_power=portfolio_data.get("buying_power", 100000.0),
            day_change=day_change,
            day_change_percent=day_change_percent
        )
        
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio summary")


@router.get("/positions")
async def get_portfolio_positions(
    include_closed: bool = False,
    session: Dict = Depends(get_current_session)
) -> List[Dict[str, Any]]:
    """Get all portfolio positions"""
    
    try:
        if include_closed:
            # Would need to implement get_all_positions method
            positions = await db_manager.get_open_positions()
        else:
            positions = await db_manager.get_open_positions()
        
        # Mock positions if none in database
        if not positions:
            positions = [
                {
                    "id": "pos_1",
                    "symbol": "AAPL",
                    "quantity": 1,
                    "position_type": "option",
                    "option_type": "call",
                    "strike_price": 150.0,
                    "expiration_date": "2024-03-15",
                    "entry_price": 2.50,
                    "current_price": 3.25,
                    "unrealized_pnl": 75.0,
                    "unrealized_pnl_percent": 30.0,
                    "delta": 0.65,
                    "gamma": 0.05,
                    "theta": -0.12,
                    "vega": 0.18,
                    "status": "open",
                    "entry_date": time.time() - 86400
                },
                {
                    "id": "pos_2",
                    "symbol": "MSFT",
                    "quantity": 100,
                    "position_type": "stock",
                    "entry_price": 380.00,
                    "current_price": 385.50,
                    "unrealized_pnl": 550.0,
                    "unrealized_pnl_percent": 1.45,
                    "status": "open",
                    "entry_date": time.time() - 172800
                }
            ]
        
        return positions
        
    except Exception as e:
        logger.error(f"Failed to get portfolio positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio positions")


@router.get("/greeks")
async def get_portfolio_greeks(
    session: Dict = Depends(get_current_session)
) -> PortfolioGreeks:
    """Get aggregated portfolio Greeks"""
    
    try:
        # Get positions with Greeks
        positions = await db_manager.get_open_positions()
        
        # Calculate total Greeks
        total_delta = sum(pos.get("delta", 0) * pos.get("quantity", 0) for pos in positions)
        total_gamma = sum(pos.get("gamma", 0) * pos.get("quantity", 0) for pos in positions)
        total_theta = sum(pos.get("theta", 0) * pos.get("quantity", 0) for pos in positions)
        total_vega = sum(pos.get("vega", 0) * pos.get("quantity", 0) for pos in positions)
        total_rho = sum(pos.get("rho", 0) * pos.get("quantity", 0) for pos in positions)
        
        # Mock Greeks if no positions
        if not positions:
            total_delta = 125.5
            total_gamma = 8.2
            total_theta = -45.8
            total_vega = 180.3
            total_rho = 12.7
        
        # Calculate net exposure (delta-adjusted notional)
        net_exposure = total_delta * 100  # Approximate dollar exposure
        
        return PortfolioGreeks(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            total_rho=total_rho,
            net_exposure=net_exposure
        )
        
    except Exception as e:
        logger.error(f"Failed to get portfolio Greeks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio Greeks")


@router.get("/risk")
async def get_risk_metrics(
    session: Dict = Depends(get_current_session)
) -> RiskMetrics:
    """Get portfolio risk metrics"""
    
    try:
        # Mock risk calculations - in production would use historical data
        risk_metrics = RiskMetrics(
            var_95=-2500.0,  # 95% VaR
            max_loss_scenario=-8500.0,  # Maximum loss scenario
            beta=1.25,  # Portfolio beta vs market
            correlation_spy=0.85,  # Correlation with SPY
            concentration_risk={
                "AAPL": 0.35,  # 35% concentration in AAPL
                "MSFT": 0.25,  # 25% in MSFT
                "tech_sector": 0.70  # 70% in tech
            }
        )
        
        return risk_metrics
        
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get risk metrics")


@router.get("/performance")
async def get_portfolio_performance(
    period: str = "1M",
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get portfolio performance analytics"""
    
    try:
        # Mock performance data
        performance_data = {
            "period": period,
            "total_return": 2500.0,
            "total_return_percent": 5.0,
            "annualized_return": 15.5,
            "volatility": 18.2,
            "sharpe_ratio": 0.85,
            "max_drawdown": -8.5,
            "win_rate": 65.5,
            "profit_factor": 1.45,
            "calmar_ratio": 1.83,
            "daily_returns": [
                {"date": "2024-01-01", "return": 0.5},
                {"date": "2024-01-02", "return": -0.2},
                {"date": "2024-01-03", "return": 1.1}
            ],
            "monthly_returns": {
                "2024-01": 2.5,
                "2023-12": 1.8,
                "2023-11": -0.5
            },
            "benchmark_comparison": {
                "portfolio_return": 5.0,
                "spy_return": 3.2,
                "alpha": 1.8,
                "beta": 1.25
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance data")


@router.get("/allocation")
async def get_asset_allocation(
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get portfolio asset allocation"""
    
    try:
        # Calculate allocation from positions
        positions = await db_manager.get_open_positions()
        
        # Mock allocation data
        allocation_data = {
            "by_asset_class": {
                "stocks": 45.0,
                "options": 35.0,
                "cash": 20.0
            },
            "by_sector": {
                "technology": 70.0,
                "healthcare": 15.0,
                "financials": 10.0,
                "consumer": 5.0
            },
            "by_symbol": {
                "AAPL": 35.0,
                "MSFT": 25.0,
                "GOOGL": 15.0,
                "NVDA": 10.0,
                "others": 15.0
            },
            "by_strategy": {
                "long_calls": 20.0,
                "covered_calls": 15.0,
                "long_stock": 45.0,
                "cash": 20.0
            },
            "concentration_metrics": {
                "max_position_size": 35.0,
                "top_3_concentration": 75.0,
                "number_of_positions": len(positions) or 5,
                "diversification_ratio": 0.65
            }
        }
        
        return allocation_data
        
    except Exception as e:
        logger.error(f"Failed to get asset allocation: {e}")
        raise HTTPException(status_code=500, detail="Failed to get asset allocation")


@router.get("/alerts")
async def get_portfolio_alerts(
    session: Dict = Depends(get_current_session)
) -> List[Dict[str, Any]]:
    """Get portfolio alerts and notifications"""
    
    try:
        # Mock alerts
        alerts = [
            {
                "id": "alert_1",
                "type": "risk",
                "severity": "medium",
                "title": "High Concentration Risk",
                "message": "70% of portfolio is concentrated in technology sector",
                "timestamp": time.time() - 3600,
                "action_required": False
            },
            {
                "id": "alert_2",
                "type": "expiration",
                "severity": "high",
                "title": "Option Expiring Soon",
                "message": "AAPL $150 Call expires in 3 days",
                "timestamp": time.time() - 1800,
                "action_required": True
            },
            {
                "id": "alert_3",
                "type": "performance",
                "severity": "low",
                "title": "Strong Performance",
                "message": "Portfolio up 5.2% this month, outperforming benchmark",
                "timestamp": time.time() - 86400,
                "action_required": False
            }
        ]
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get portfolio alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio alerts")


@router.post("/rebalance")
async def suggest_rebalancing(
    target_allocation: Dict[str, float],
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Suggest portfolio rebalancing actions"""
    
    try:
        # Mock rebalancing suggestions
        suggestions = {
            "current_allocation": {
                "AAPL": 35.0,
                "MSFT": 25.0,
                "cash": 40.0
            },
            "target_allocation": target_allocation,
            "rebalancing_actions": [
                {
                    "symbol": "AAPL",
                    "action": "reduce",
                    "current_percent": 35.0,
                    "target_percent": target_allocation.get("AAPL", 30.0),
                    "shares_to_trade": -25,
                    "dollar_amount": -3750.0
                },
                {
                    "symbol": "NVDA",
                    "action": "add",
                    "current_percent": 0.0,
                    "target_percent": target_allocation.get("NVDA", 10.0),
                    "shares_to_trade": 15,
                    "dollar_amount": 5000.0
                }
            ],
            "estimated_costs": {
                "commission": 0.0,  # Paper trading
                "market_impact": 25.0,
                "total_cost": 25.0
            },
            "benefits": {
                "improved_diversification": True,
                "reduced_concentration_risk": True,
                "expected_return_impact": 0.2
            }
        }
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to generate rebalancing suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate rebalancing suggestions")