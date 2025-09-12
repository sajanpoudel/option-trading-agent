"""
Neural Options Oracle++ API Dependencies
"""
import time
from typing import Dict, Optional, Any
from fastapi import HTTPException, Header, Request
from functools import wraps
import asyncio
from collections import defaultdict

from config.database import db_manager
from config.logging import get_api_logger

logger = get_api_logger()

# Rate limiting storage (in production, use Redis)
request_counts = defaultdict(lambda: defaultdict(int))
request_timestamps = defaultdict(lambda: defaultdict(list))


async def get_current_session(
    x_session_token: Optional[str] = Header(None),
    session_token: Optional[str] = None
) -> Dict[str, Any]:
    """Get current browser session (dependency)"""
    
    token = x_session_token or session_token
    
    if not token:
        # For demo purposes, create a temporary session
        logger.info("No session token provided, creating temporary session")
        return {
            "session_token": "temp",
            "risk_profile": "moderate",
            "created_at": time.time(),
            "last_accessed_at": time.time(),
            "expires_at": time.time() + 86400,
            "preferences": {}
        }
    
    # Get session from database
    session = await db_manager.get_session(token)
    
    if not session:
        raise HTTPException(
            status_code=401, 
            detail="Invalid or expired session token"
        )
    
    # Update session activity
    await db_manager.update_session_activity(token)
    
    return session


def rate_limiter(max_requests: int = 60, time_window: int = 60):
    """Rate limiting decorator"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request object
            request = None
            for arg in args:
                if hasattr(arg, 'client'):
                    request = arg
                    break
            
            if not request:
                return await func(*args, **kwargs)
            
            # Get client IP
            client_ip = request.client.host if request.client else "unknown"
            current_time = time.time()
            
            # Clean old timestamps
            request_timestamps[client_ip] = [
                ts for ts in request_timestamps[client_ip]
                if current_time - ts < time_window
            ]
            
            # Check rate limit
            if len(request_timestamps[client_ip]) >= max_requests:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            # Add current request
            request_timestamps[client_ip].append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class SessionManager:
    """Session management utilities"""
    
    @staticmethod
    async def validate_session(session_token: str) -> bool:
        """Validate session token"""
        session = await db_manager.get_session(session_token)
        return session is not None
    
    @staticmethod
    async def get_session_risk_profile(session_token: str) -> str:
        """Get session risk profile"""
        session = await db_manager.get_session(session_token)
        return session.get("risk_profile", "moderate") if session else "moderate"
    
    @staticmethod
    async def update_session_preferences(
        session_token: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        """Update session preferences"""
        try:
            # This would require an update method in db_manager
            # For now, just return True
            return True
        except Exception as e:
            logger.error(f"Failed to update session preferences: {e}")
            return False


# Global session manager
session_manager = SessionManager()