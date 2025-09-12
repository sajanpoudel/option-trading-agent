"""
Neural Options Oracle++ System API Routes
"""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
import time
import psutil
import asyncio

from config.database import db_manager
from config.logging import get_api_logger
from config.settings import settings
from src.api.dependencies import get_current_session

logger = get_api_logger()
router = APIRouter()


@router.get("/")
async def system_info() -> Dict[str, Any]:
    """Get system API information"""
    
    return {
        "name": "System API",
        "version": "1.0.0",
        "description": "System monitoring, configuration, and analytics",
        "endpoints": {
            "status": "/status",
            "health": "/health",
            "analytics": "/analytics",
            "config": "/config",
            "metrics": "/metrics"
        },
        "features": [
            "System health monitoring",
            "Performance metrics",
            "Configuration management",
            "Usage analytics",
            "Error tracking"
        ]
    }


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    
    try:
        # System health checks
        db_health = await db_manager.health_check()
        
        # Get system resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get network info if available
        try:
            network = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network.bytes_sent,
                "bytes_received": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_received": network.packets_recv
            }
        except:
            network_stats = {}
        
        # Application-specific status
        app_status = {
            "environment": settings.app_env,
            "debug_mode": settings.app_debug,
            "log_level": settings.log_level,
            "cors_origins": settings.cors_origins
        }
        
        return {
            "system": {
                "status": "healthy",
                "uptime_seconds": time.time(),  # Mock uptime
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 2)
                },
                "network": network_stats
            },
            "database": db_health,
            "application": app_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.get("/health")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check for all system components"""
    
    try:
        health_checks = {}
        
        # Database health
        db_health = await db_manager.health_check()
        health_checks["database"] = {
            "status": db_health["status"],
            "response_time_ms": 50,  # Mock response time
            "connection_pool": "healthy"
        }
        
        # API health
        health_checks["api"] = {
            "status": "healthy",
            "response_time_ms": 10,
            "active_connections": 5
        }
        
        # External services health (mock)
        health_checks["external_services"] = {
            "openai": {"status": "healthy", "last_check": time.time()},
            "alpaca": {"status": "healthy", "last_check": time.time()},
            "jigsawstack": {"status": "healthy", "last_check": time.time()}
        }
        
        # Overall health
        all_healthy = all(
            check.get("status") == "healthy" 
            for check in [health_checks["database"], health_checks["api"]]
        )
        
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "components": health_checks,
            "timestamp": time.time(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/analytics")
async def get_system_analytics(
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get comprehensive system analytics"""
    
    try:
        # Get analytics from database
        analytics = await db_manager.get_system_analytics()
        
        # Add API usage statistics (mock)
        api_stats = {
            "total_requests": 1250,
            "requests_per_minute": 25,
            "avg_response_time_ms": 150,
            "error_rate_percent": 0.5,
            "endpoints": {
                "/api/v1/analysis/analyze": {"count": 450, "avg_time": 2500},
                "/api/v1/trading/execute": {"count": 125, "avg_time": 800},
                "/api/v1/education/content": {"count": 300, "avg_time": 200},
                "/api/v1/portfolio/summary": {"count": 375, "avg_time": 100}
            }
        }
        
        # User activity (using sessions instead of users)
        user_stats = {
            "active_sessions": analytics.get("active_sessions", 0),
            "peak_concurrent_sessions": 15,
            "avg_session_duration_minutes": 45,
            "total_analyses_today": analytics.get("signals_last_24h", 0)
        }
        
        return {
            "database_analytics": analytics,
            "api_statistics": api_stats,
            "session_statistics": user_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system analytics")


@router.get("/config")
async def get_system_config(
    key: str = None,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get system configuration"""
    
    try:
        if key:
            # Get specific config value
            value = await db_manager.get_system_config(key)
            return {
                "key": key,
                "value": value,
                "found": value is not None
            }
        
        # Get all public configuration
        public_config = {
            "analysis_timeout_seconds": settings.analysis_timeout_seconds,
            "max_concurrent_analysis": settings.max_concurrent_analysis,
            "default_risk_profile": settings.default_risk_profile,
            "paper_trading_balance": settings.paper_trading_balance,
            "rate_limits": {
                "per_minute": settings.rate_limit_per_minute,
                "burst": settings.rate_limit_burst
            },
            "cors_origins": settings.cors_origins,
            "environment": settings.app_env
        }
        
        return public_config
        
    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system config")


@router.post("/config")
async def update_system_config(
    key: str,
    value: Any,
    description: str = None,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Update system configuration"""
    
    try:
        # In a real system, you'd want admin authentication here
        success = await db_manager.set_system_config(key, value, description)
        
        if success:
            return {
                "success": True,
                "key": key,
                "value": value,
                "message": "Configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except Exception as e:
        logger.error(f"Failed to update system config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update system config")


@router.get("/metrics")
async def get_system_metrics(
    metric_type: str = "all",
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get detailed system metrics"""
    
    try:
        metrics = {}
        
        if metric_type in ["all", "performance"]:
            metrics["performance"] = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        
        if metric_type in ["all", "business"]:
            analytics = await db_manager.get_system_analytics()
            metrics["business"] = {
                "total_analyses_today": analytics.get("signals_last_24h", 0),
                "active_positions": analytics.get("portfolio_summary", {}).get("total_positions", 0),
                "total_portfolio_value": analytics.get("portfolio_summary", {}).get("total_value", 0),
                "session_count": analytics.get("active_sessions", 0)
            }
        
        if metric_type in ["all", "errors"]:
            # Mock error metrics - in production would come from logging system
            metrics["errors"] = {
                "total_errors_today": 5,
                "error_rate_percent": 0.5,
                "critical_errors": 0,
                "most_common_errors": [
                    {"error": "API timeout", "count": 3},
                    {"error": "Invalid symbol", "count": 2}
                ]
            }
        
        return {
            "metrics": metrics,
            "metric_type": metric_type,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")


@router.get("/logs")
async def get_system_logs(
    level: str = "INFO",
    limit: int = 100,
    component: str = None,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Get system logs (mock implementation)"""
    
    try:
        # Mock log entries - in production would read from log files
        mock_logs = [
            {
                "timestamp": time.time() - 300,
                "level": "INFO",
                "component": "api",
                "message": "Analysis completed for AAPL",
                "context": {"symbol": "AAPL", "duration": "2.5s"}
            },
            {
                "timestamp": time.time() - 600,
                "level": "WARNING", 
                "component": "database",
                "message": "Slow query detected",
                "context": {"query_time": "1.2s"}
            },
            {
                "timestamp": time.time() - 900,
                "level": "ERROR",
                "component": "agents",
                "message": "OpenAI API rate limit exceeded",
                "context": {"retry_after": "60s"}
            }
        ]
        
        # Filter by level
        if level != "ALL":
            mock_logs = [log for log in mock_logs if log["level"] == level]
        
        # Filter by component
        if component:
            mock_logs = [log for log in mock_logs if log["component"] == component]
        
        # Apply limit
        mock_logs = mock_logs[:limit]
        
        return {
            "logs": mock_logs,
            "total_count": len(mock_logs),
            "filters": {
                "level": level,
                "component": component,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs")


@router.post("/maintenance")
async def trigger_maintenance_task(
    task_type: str,
    session: Dict = Depends(get_current_session)
) -> Dict[str, Any]:
    """Trigger system maintenance tasks"""
    
    try:
        if task_type == "cleanup":
            # Mock cleanup task
            result = {
                "task": "cleanup",
                "deleted_records": 150,
                "freed_space_mb": 25,
                "duration_seconds": 5.2
            }
        elif task_type == "backup":
            # Mock backup task  
            result = {
                "task": "backup",
                "backup_size_mb": 125,
                "backup_location": "s3://backups/neural-oracle/",
                "duration_seconds": 30.5
            }
        else:
            raise HTTPException(status_code=400, detail="Unknown maintenance task type")
        
        logger.info(f"Maintenance task completed: {task_type}")
        
        return {
            "success": True,
            "task_type": task_type,
            "result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Maintenance task failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute maintenance task")