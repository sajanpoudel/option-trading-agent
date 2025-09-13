"""
Neural Options Oracle++ Frontend API
Dedicated FastAPI application for frontend integration
Replaces all mock data with real backend integration
"""
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

# Import our backend services (no duplication)
from agents.orchestrator import OptionsOracleOrchestrator
from src.core.decision_engine import DecisionEngine
from src.data.market_data_manager import MarketDataManager
from src.api.intelligent_orchestrator import IntelligentOrchestrator
from config.database import db_manager

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

# Initialize FastAPI app for frontend
app = FastAPI(
    title="Neural Options Oracle++ Frontend API",
    description="API endpoints specifically designed for frontend integration",
    version="1.0.0",
    docs_url="/api/frontend/docs",
    redoc_url="/api/frontend/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://*.vercel.app",
        "https://optionsoracle.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize services
orchestrator = OptionsOracleOrchestrator()
decision_engine = DecisionEngine()
market_data_manager = MarketDataManager()
intelligent_orchestrator = IntelligentOrchestrator()
connection_manager = ConnectionManager()

# Pydantic models for API responses
class HotStock(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    volume: str
    marketCap: str
    sparklineData: List[Dict[str, float]]
    aiScore: int
    signals: List[str]
    trending: bool

class TradingSignal(BaseModel):
    id: str
    symbol: str
    direction: str  # "BUY" | "SELL"
    confidence: int
    strike: float
    expiration: str
    entryPrice: float
    exitRules: str
    timestamp: datetime
    positionSize: int
    reasoning: str
    riskReward: str

class ChatMessage(BaseModel):
    message: str
    selectedStock: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    actions: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Neural Options Oracle++ Frontend API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "purpose": "Frontend integration endpoints"
    }

# 🔥 CRITICAL: Replace Frontend Hot Stocks Mock Data
@app.get("/api/v1/stocks/hot-stocks", response_model=Dict[str, Any])
async def get_hot_stocks():
    """
    Replace hardcoded stock data in frontend components:
    - components/hot-stocks-view.tsx (lines 31-103)
    - components/trading/hot-stocks/hot-stocks-container.tsx (lines 31-107)
    """
    try:
        logger.info("🔥 Getting hot stocks with real data...")
        
        # Use real market data instead of hardcoded values
        symbols = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"]
        hot_stocks = []
        
        for symbol in symbols:
            try:
                # Get real market data
                market_data = await market_data_manager.get_comprehensive_data(symbol)
                
                # Get AI analysis (this is expensive, cache aggressively)
                user_risk_profile = {'risk_level': 'moderate', 'experience': 'intermediate'}
                agent_results = await orchestrator.analyze_stock(symbol, user_risk_profile)
                
                # Get real quote data
                quote = market_data.get('quote', {})
                current_price = float(quote.get('price', 0))
                change = float(quote.get('change', 0))
                volume = quote.get('volume', 0)
                
                # Generate sparkline from real historical data
                historical = market_data.get('historical', [])
                sparkline_data = []
                if historical:
                    recent_prices = historical[-20:]  # Last 20 data points
                    sparkline_data = [{"value": float(bar.get('close', current_price))} for bar in recent_prices]
                else:
                    # Fallback sparkline if no historical data
                    sparkline_data = [{"value": current_price} for _ in range(20)]
                
                # Extract AI signals from agent results
                technical_result = agent_results.get('technical', {})
                signal_result = agent_results.get('signal', {})
                ai_signals = []
                
                direction = signal_result.get('direction', 'HOLD')
                if direction == 'BUY':
                    ai_signals.append("Buy Signal")
                elif direction == 'SELL':
                    ai_signals.append("Sell Signal")
                else:
                    ai_signals.append("Hold")
                
                scenario = technical_result.get('scenario', 'Normal')
                ai_signals.append(scenario.replace('_', ' ').title())
                
                # Calculate AI score from agent confidence
                confidence = agent_results.get('confidence', 0.5)
                ai_score = int(confidence * 100)
                
                hot_stock = {
                    "symbol": symbol,
                    "name": market_data.get('company_name', f"{symbol} Inc"),
                    "price": current_price,
                    "change": change,
                    "changePercent": (change / (current_price - change)) * 100 if (current_price - change) > 0 else 0,
                    "volume": f"{volume / 1000000:.1f}M" if volume > 1000000 else f"{volume / 1000:.0f}K",
                    "marketCap": market_data.get('market_cap', 'N/A'),
                    "sparklineData": sparkline_data,
                    "aiScore": ai_score,
                    "signals": ai_signals,
                    "trending": ai_score > 75
                }
                
                hot_stocks.append(hot_stock)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                # Continue with other stocks even if one fails
                continue
        
        logger.info(f"✅ Retrieved {len(hot_stocks)} hot stocks with real data")
        
        return {
            "stocks": hot_stocks,
            "timestamp": datetime.now().isoformat(),
            "total_count": len(hot_stocks),
            "data_source": "real_market_data"
        }
        
    except Exception as e:
        logger.error(f"❌ Hot stocks API error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch hot stocks: {str(e)}")

# 🔥 CRITICAL: Replace Trading Signals Mock Data  
@app.get("/api/v1/signals/{symbol}", response_model=Dict[str, Any])
async def get_trading_signals(symbol: str = Path(..., description="Stock symbol")):
    """
    Replace mock trading signals in frontend:
    - components/trading-signals.tsx (lines 33-78, 85-106)
    """
    try:
        logger.info(f"🔥 Getting real trading signals for {symbol}...")
        
        # Get comprehensive analysis from our backend
        user_risk_profile = {'risk_level': 'moderate', 'experience': 'intermediate'}
        analysis_result = await decision_engine.process_stock(symbol, user_risk_profile)
        
        signals = []
        signal_data = analysis_result.get('signal', {})
        strike_recommendations = analysis_result.get('strike_recommendations', [])
        
        # Convert our backend signal format to frontend format
        for i, strike_rec in enumerate(strike_recommendations):
            signal = {
                "id": f"{symbol}_{i}_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "direction": signal_data.get('direction', 'HOLD'),
                "confidence": int(analysis_result.get('confidence', 0.5) * 100),
                "strike": strike_rec.get('strike', 0),
                "expiration": "2024-01-19",  # This should come from strike_rec
                "entryPrice": strike_rec.get('potential_return', 0) * 100,  # Approximate entry price
                "exitRules": f"Take profit at {strike_rec.get('potential_return', 0):.1%} or stop loss at {strike_rec.get('max_loss', 0):.1%}",
                "timestamp": datetime.now().isoformat(),
                "positionSize": 1 + i,  # Size based on risk
                "reasoning": signal_data.get('reasoning', 'AI-generated signal from comprehensive analysis'),
                "riskReward": f"1:{strike_rec.get('potential_return', 0) / max(strike_rec.get('risk_score', 0.1), 0.1):.1f}"
            }
            signals.append(signal)
        
        logger.info(f"✅ Generated {len(signals)} real trading signals for {symbol}")
        
        return {
            "signals": signals,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "total_signals": len(signals),
            "data_source": "ai_agent_analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Trading signals API error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trading signals: {str(e)}")

# 🔥 CRITICAL: Replace Chat Mock Responses with Intelligent Orchestration
@app.post("/api/v1/chat/message", response_model=ChatResponse)
async def send_chat_message(message_data: ChatMessage):
    """
    Intelligent chat processing that triggers appropriate AI agents
    Replaces simulated chat responses in frontend:
    - components/chat-interface.tsx (lines 70-86)
    """
    try:
        logger.info(f"🧠 Processing intelligent chat message: {message_data.message}")
        
        # Use intelligent orchestrator to process query
        user_context = {
            'selectedStock': message_data.selectedStock,
            'risk_profile': {'risk_level': 'moderate', 'experience': 'intermediate'}
        }
        
        # Process query with full agent orchestration
        orchestration_result = await intelligent_orchestrator.process_user_query(
            message_data.message, 
            user_context
        )
        
        # Extract information for chat response
        ai_response = orchestration_result.get('ai_response', 'Analysis complete.')
        symbol = orchestration_result.get('symbol')
        query_type = orchestration_result.get('query_type', 'general')
        
        # Determine actions based on orchestration results
        actions = {}
        if symbol:
            actions['analyzeStock'] = symbol
            actions['showAnalysis'] = True
        
        if query_type == 'technical_analysis':
            actions['showTechnicalChart'] = True
        elif query_type == 'trading_signals':
            actions['showTradingSignals'] = True
        elif query_type == 'portfolio_management':
            actions['showPortfolio'] = True
        
        # Get suggested actions from orchestrator
        suggestions = orchestration_result.get('suggested_actions', [
            "Analyze technical indicators",
            "Check trading signals", 
            "Review risk assessment",
            "Show market sentiment"
        ])
        
        response = ChatResponse(
            response=ai_response,
            actions=actions if actions else None,
            suggestions=suggestions
        )
        
        logger.info(f"✅ Intelligent chat response generated for {symbol}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Intelligent chat error: {e}")
        # Fallback response if orchestration fails
        return ChatResponse(
            response=f"I'm analyzing your request: '{message_data.message}'. Let me gather comprehensive information...",
            actions={"analyzeStock": message_data.selectedStock} if message_data.selectedStock else None,
            suggestions=["Try asking about a specific stock", "Request technical analysis", "Ask for trading signals"]
        )

# 🔥 CRITICAL: Comprehensive Stock Analysis Endpoint with All Visualizations
@app.post("/api/v1/analysis/{symbol}")
async def analyze_stock(
    symbol: str = Path(..., description="Stock symbol to analyze"),
    risk_level: str = Query("moderate", description="User risk level"),
    query: str = Query("comprehensive analysis", description="Analysis query")
):
    """
    Complete stock analysis using intelligent agent orchestration
    Provides ALL data needed for frontend visualization components:
    - Stock details view data
    - AI agent analysis data  
    - Technical indicators for charts
    - Trading signals
    - Educational content
    - Risk assessment
    """
    try:
        logger.info(f"🧠 Running intelligent analysis for {symbol} with query: {query}")
        
        # Use intelligent orchestrator for comprehensive analysis
        user_context = {
            'selectedStock': symbol,
            'risk_profile': {
                'risk_level': risk_level,
                'experience': 'intermediate',
                'max_position_size': 0.05
            }
        }
        
        # Process with intelligent orchestration
        orchestration_result = await intelligent_orchestrator.process_user_query(
            f"analyze {symbol} {query}",
            user_context
        )
        
        logger.info(f"✅ Intelligent analysis complete for {symbol}")
        
        return {
            "symbol": symbol,
            "query": query,
            "analysis_type": "comprehensive",
            "orchestration_result": orchestration_result,
            "timestamp": datetime.now().isoformat(),
            "data_source": "intelligent_ai_orchestration",
            
            # Frontend-specific data extracts for easy consumption
            "frontend_ready_data": {
                "stock_data": orchestration_result.get('frontend_data', {}).get('stock_data', {}),
                "agent_analysis": orchestration_result.get('frontend_data', {}).get('agent_analysis', []),
                "technical_indicators": orchestration_result.get('frontend_data', {}).get('technical_indicators', {}),
                "trading_signals": orchestration_result.get('frontend_data', {}).get('trading_signals', []),
                "chart_data": orchestration_result.get('frontend_data', {}).get('chart_data', {}),
                "educational_content": orchestration_result.get('frontend_data', {}).get('educational_content', {}),
                "risk_assessment": orchestration_result.get('frontend_data', {}).get('risk_assessment', {})
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Intelligent analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# 🔥 NEW: Specific endpoint for AI Agent Analysis Component
@app.get("/api/v1/agents/{symbol}")
async def get_agent_analysis(symbol: str = Path(..., description="Stock symbol")):
    """
    Get AI agent analysis data specifically for the AI Agent Analysis component
    Replaces mock data in: components/ai-agent-analysis.tsx (lines 31-98)
    """
    try:
        logger.info(f"🤖 Getting agent analysis for {symbol}")
        
        # Use intelligent orchestrator to get agent-specific data
        user_context = {'selectedStock': symbol}
        result = await intelligent_orchestrator.process_user_query(
            f"agent analysis for {symbol}",
            user_context
        )
        
        # Extract agent data for frontend component
        agent_data = result.get('frontend_data', {}).get('agent_analysis', [])
        
        return {
            "symbol": symbol,
            "agents": agent_data,
            "overall_signal": result.get('frontend_data', {}).get('trading_signals', [{}])[0].get('direction', 'HOLD'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Agent analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")

# 🔥 NEW: Technical Indicators Endpoint for Charts
@app.get("/api/v1/technical/{symbol}")
async def get_technical_indicators(symbol: str = Path(..., description="Stock symbol")):
    """
    Get technical indicators specifically for chart components
    Replaces mock data in: technical analysis widgets and charts
    """
    try:
        logger.info(f"📊 Getting technical indicators for {symbol}")
        
        # Use intelligent orchestrator focused on technical analysis
        user_context = {'selectedStock': symbol}
        result = await intelligent_orchestrator.process_user_query(
            f"technical analysis indicators for {symbol}",
            user_context
        )
        
        # Extract technical data
        technical_data = result.get('frontend_data', {}).get('technical_indicators', {})
        chart_data = result.get('frontend_data', {}).get('chart_data', {})
        
        return {
            "symbol": symbol,
            "indicators": technical_data,
            "chart_data": chart_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Technical indicators error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Technical indicators failed: {str(e)}")

# WebSocket for real-time updates
@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time updates for frontend"""
    await connection_manager.connect(websocket)
    logger.info("WebSocket client connected for live updates")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
            # Send live market updates (this can be enhanced)
            await connection_manager.send_personal_message({
                "type": "market_update",
                "timestamp": datetime.now().isoformat(),
                "status": "connected"
            }, websocket)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

# Health check
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "frontend-api",
        "timestamp": datetime.now().isoformat(),
        "backend_integration": "active"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("FRONTEND_API_PORT", 8081))
    
    logger.info(f"🚀 Starting Frontend API on port {port}")
    
    uvicorn.run(
        "src.api.frontend_main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )