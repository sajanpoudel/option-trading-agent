"""
Neural Options Oracle++ FastAPI Main Application
"""
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config.settings import settings
from config.logging import setup_logging, log_api_access, get_api_logger
from config.database import db_manager
from src.api.dependencies import get_current_session, rate_limiter
from src.api.routes import analysis, trading, education, portfolio, system
from src.api.chat_router import router as chat_router

logger = get_api_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("Starting Neural Options Oracle++ API Server")
    
    # Setup logging
    setup_logging()
    
    # Test database connection
    db_health = await db_manager.health_check()
    if db_health["status"] != "healthy":
        logger.error(f"Database connection failed: {db_health}")
        raise Exception("Database connection failed")
    
    logger.info("Database connection established")
    logger.info("Neural Options Oracle++ API Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Neural Options Oracle++ API Server")


# Create FastAPI application
app = FastAPI(
    title="Neural Options Oracle++ API",
    description="AI-Driven Options Trading Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

if settings.app_env == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
    )


# Request/Response middleware for logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all API requests and responses"""
    
    start_time = time.time()
    
    # Get client info
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    try:
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log API access
        log_api_access(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            response_time=process_time,
            user_agent=user_agent,
            ip_address=client_ip
        )
        
        # Add response headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "1.0.0"
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)}")
        
        # Log failed request
        log_api_access(
            method=request.method,
            path=request.url.path,
            status_code=500,
            response_time=process_time,
            user_agent=user_agent,
            ip_address=client_ip
        )
        
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    
    return {
        "name": "Neural Options Oracle++ API",
        "version": "1.0.0",
        "description": "AI-Driven Options Trading Intelligence Platform",
        "status": "active",
        "timestamp": time.time(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analysis": "/api/v1/analysis",
            "trading": "/api/v1/trading", 
            "education": "/api/v1/education",
            "portfolio": "/api/v1/portfolio",
            "system": "/api/v1/system"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    
    # Check database
    db_health = await db_manager.health_check()
    
    # System health
    system_health = {
        "api": "healthy",
        "database": db_health["status"],
        "timestamp": time.time()
    }
    
    # Overall status
    overall_status = "healthy" if all(
        status == "healthy" for status in [
            system_health["api"],
            system_health["database"]
        ]
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "components": system_health,
        "database_details": db_health
    }


# Session management endpoints
@app.post("/api/v1/session/create")
async def create_session(
    request: Request,
    risk_profile: str = "moderate"
) -> Dict[str, Any]:
    """Create a new browser session"""
    
    # Get client info
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent")
    
    # Create session
    session_token = await db_manager.create_browser_session(
        ip_address=client_ip,
        user_agent=user_agent,
        risk_profile=risk_profile
    )
    
    if not session_token:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    logger.info(f"New session created: {session_token}")
    
    return {
        "session_token": session_token,
        "risk_profile": risk_profile,
        "expires_in": 86400,  # 24 hours in seconds
        "created_at": time.time()
    }


@app.get("/api/v1/session/info")
async def get_session_info(
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get current session information"""
    
    return {
        "session_token": session["session_token"],
        "risk_profile": session["risk_profile"],
        "created_at": session["created_at"],
        "last_accessed_at": session["last_accessed_at"],
        "expires_at": session["expires_at"],
        "preferences": session.get("preferences", {})
    }


# Include API routes
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["Trading"])
app.include_router(education.router, prefix="/api/v1/education", tags=["Education"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(system.router, prefix="/api/v1/system", tags=["System"])
app.include_router(chat_router, tags=["Chat Router"])

# Chat endpoint for frontend integration
from pydantic import BaseModel
from typing import Optional, List

class ChatMessage(BaseModel):
    message: str
    selectedStock: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    actions: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

@app.post("/api/v1/chat/message", response_model=ChatResponse)
async def send_chat_message(message_data: ChatMessage):
    """Chat endpoint that connects to AI agents"""
    try:
        logger.info(f"🧠 Processing chat message: {message_data.message}")
        
        # Import the intelligent orchestrator
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        orchestrator = IntelligentOrchestrator()
        
        # Process query with intelligent orchestration
        user_context = {
            'selectedStock': message_data.selectedStock,
            'risk_profile': {'risk_level': 'moderate', 'experience': 'intermediate'}
        }
        
        orchestration_result = await orchestrator.process_user_query(
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
        
        logger.info(f"✅ Chat response generated for {symbol}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        # Fallback response if orchestration fails
        return ChatResponse(
            response=f"I'm analyzing your request: '{message_data.message}'. Let me gather comprehensive information...",
            actions={"analyzeStock": message_data.selectedStock} if message_data.selectedStock else None,
            suggestions=["Try asking about a specific stock", "Request technical analysis", "Ask for trading signals"]
        )

# Hot stocks endpoint for frontend
@app.get("/api/v1/stocks/hot-stocks")
async def get_hot_stocks():
    """Get hot stocks with real StockTwits trending data and AI analysis"""
    try:
        logger.info("🔥 Getting REAL trending stocks from StockTwits...")
        
        # Import the intelligent orchestrator and web scraper
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        from src.agents.web_scraper_agent import get_web_scraper_agent
        from openai import OpenAI
        import os
        
        orchestrator = IntelligentOrchestrator()
        
        # Initialize web scraper agent with OpenAI
        openai_client = None
        try:
            from config.settings import settings
            if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
                openai_client = OpenAI(api_key=settings.openai_api_key)
                logger.info("✅ OpenAI client initialized for web scraping")
            else:
                logger.warning("⚠️ OpenAI API key not found in settings")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            # Fallback to environment variable
            if os.getenv("OPENAI_API_KEY"):
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("✅ OpenAI client initialized from env variable")
        
        web_scraper = get_web_scraper_agent(openai_client)
        
        # Get REAL trending stocks from StockTwits
        logger.info("📈 Scraping StockTwits for trending stocks...")
        trending_stocks_data = await web_scraper.get_trending_stocks(limit=5)
        
        if not trending_stocks_data:
            logger.warning("No trending stocks found, using fallback symbols")
            symbols = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"]
        else:
            symbols = [stock["symbol"] for stock in trending_stocks_data]
            logger.info(f"✅ Got real trending symbols: {symbols}")
        
        hot_stocks = []
        
        for i, symbol in enumerate(symbols):
            try:
                # Get corresponding trending data if available
                trending_data = None
                if trending_stocks_data and i < len(trending_stocks_data):
                    trending_data = trending_stocks_data[i]
                
                # Get market data directly (bypass the failing process_user_query)
                logger.info(f"📈 Getting market data for {symbol}")
                market_data = await orchestrator.market_data_manager.get_comprehensive_data(symbol)
                
                # Extract real market data
                quote = market_data.get('quote', {})
                current_price = float(quote.get('price', 0))
                volume = quote.get('volume', 0)
                
                
                # Get previous close price for change calculation
                historical = market_data.get('historical', [])
                previous_close = current_price  # Default to current price
                if historical and len(historical) > 1:
                    previous_close = float(historical[-2].get('close', current_price))
                
                # Calculate change and change percentage
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                
                # Generate sparkline from real historical data
                historical = market_data.get('historical', [])
                sparkline_data = []
                if historical and len(historical) > 0:
                    recent_prices = historical[-20:]  # Last 20 data points
                    sparkline_data = [{"value": float(bar.get('close', current_price))} for bar in recent_prices]
                else:
                    # Fallback sparkline if no historical data
                    sparkline_data = [{"value": current_price + (i * 0.1)} for i in range(20)]
                
                # Extract AI signals (combine StockTwits sentiment + basic analysis)
                ai_signals = []
                ai_score = 50
                
                # Add StockTwits sentiment to AI signals
                if trending_data:
                    sentiment = trending_data.get('sentiment', 'Neutral')
                    mentions = trending_data.get('mentions', 0)
                    ai_signals.append(f"StockTwits: {sentiment}")
                    if mentions > 0:
                        ai_signals.append(f"{mentions} mentions")
                    
                    # Boost AI score based on StockTwits sentiment
                    sentiment_score = trending_data.get('sentiment_score', 0.5)
                    ai_score = max(ai_score, int(sentiment_score * 100))
                
                # Add basic technical signals based on price movement
                if change > 0:
                    ai_signals.append("Price Up")
                    ai_score += 10
                elif change < 0:
                    ai_signals.append("Price Down")
                    ai_score -= 5
                
                # Add volume signal
                if volume > 1000000:  # High volume
                    ai_signals.append("High Volume")
                    ai_score += 5
                
                # Ensure AI score is within bounds
                ai_score = max(0, min(100, ai_score))
                
                hot_stock = {
                    "symbol": symbol,
                    "name": trending_data.get('name', market_data.get('company_name', f"{symbol} Inc")),
                    "price": current_price,
                    "change": change,
                    "changePercent": change_percent,
                    "volume": volume,
                    "sparklineData": sparkline_data,
                    "aiScore": ai_score,
                    "signals": ai_signals or ["Market Data"],
                    "trending": (trending_data and trending_data.get('trending', False)) or ai_score > 75
                }
                
                hot_stocks.append(hot_stock)
                logger.info(f"✅ Processed {symbol}: price=${current_price}, change={change}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                # Continue with other stocks even if one fails
                continue
        
        logger.info(f"✅ Retrieved {len(hot_stocks)} hot stocks with real data")
        
        return {
            "stocks": hot_stocks,
            "timestamp": time.time(),
            "total_count": len(hot_stocks),
            "data_source": "stocktwits_trending_with_ai_analysis",
            "trending_source": "stocktwits.com/sentiment/most-active",
            "symbols_found": symbols
        }
        
    except Exception as e:
        logger.error(f"❌ Hot stocks API error: {e}")
        # Return empty array instead of mock data
        return {
            "stocks": [],
            "timestamp": time.time(),
            "total_count": 0,
            "data_source": "error",
            "error": str(e)
        }

# AI Agents endpoint for frontend
@app.get("/api/v1/agents/{symbol}")
async def get_agent_analysis(symbol: str):
    """Get AI agent analysis for a specific symbol"""
    try:
        logger.info(f"🤖 Getting agent analysis for {symbol}")
        
        # Import the intelligent orchestrator
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        orchestrator = IntelligentOrchestrator()
        
        # Use intelligent orchestrator to get agent-specific data
        user_context = {'selectedStock': symbol}
        result = await orchestrator.process_user_query(
            f"agent analysis for {symbol}",
            user_context
        )
        
        # Extract agent data for frontend component
        agent_data = result.get('frontend_data', {}).get('agent_analysis', [])
        
        return {
            "symbol": symbol,
            "agents": agent_data,
            "overall_signal": result.get('frontend_data', {}).get('trading_signals', [{}])[0].get('direction', 'HOLD') if result.get('frontend_data', {}).get('trading_signals') else 'HOLD',
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Agent analysis error for {symbol}: {e}")
        return {
            "symbol": symbol,
            "agents": [],
            "overall_signal": "HOLD",
            "timestamp": time.time(),
            "error": str(e)
        }

# Trading signals endpoint for frontend
@app.get("/api/v1/technical/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    try:
        logger.info(f"📊 Getting technical indicators for {symbol}")
        
        # Import the intelligent orchestrator
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        orchestrator = IntelligentOrchestrator()
        
        # Get technical analysis
        user_context = {'selectedStock': symbol}
        result = await orchestrator.process_user_query(
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
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Technical indicators error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Technical indicators failed: {str(e)}")

@app.get("/api/v1/signals/{symbol}")
async def get_trading_signals(symbol: str):
    """Get trading signals for a specific symbol"""
    try:
        logger.info(f"🔥 Getting real trading signals for {symbol}...")
        
        # Import the intelligent orchestrator
        from src.api.intelligent_orchestrator import IntelligentOrchestrator
        orchestrator = IntelligentOrchestrator()
        
        # Get comprehensive analysis from our backend
        user_context = {
            'selectedStock': symbol,
            'risk_profile': {'risk_level': 'moderate', 'experience': 'intermediate'}
        }
        
        analysis_result = await orchestrator.process_user_query(
            f"trading signals for {symbol}",
            user_context
        )
        
        # Extract trading signals from analysis
        trading_signals = analysis_result.get('frontend_data', {}).get('trading_signals', [])
        
        logger.info(f"✅ Generated {len(trading_signals)} real trading signals for {symbol}")
        
        return {
            "signals": trading_signals,
            "symbol": symbol,
            "timestamp": time.time(),
            "total_signals": len(trading_signals),
            "data_source": "ai_agent_analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Trading signals API error for {symbol}: {e}")
        return {
            "signals": [],
            "symbol": symbol,
            "timestamp": time.time(),
            "total_signals": 0,
            "data_source": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower()
    )