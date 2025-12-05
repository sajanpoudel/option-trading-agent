"""
Neural Options Oracle++ Backend
Main entry point for the FastAPI application
"""
import uvicorn
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting Neural Options Oracle++ Backend Server")
    
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
