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


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower()
    )