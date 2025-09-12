#!/usr/bin/env python3
"""
Neural Options Oracle++ Main Application Entry Point
"""
import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from config.settings import settings
from config.logging import setup_logging, get_core_logger
import uvicorn

logger = get_core_logger()


def create_logs_directory():
    """Create logs directory if it doesn't exist"""
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def check_environment():
    """Check that required environment variables are set"""
    
    required_env_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY", 
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please copy .env.example to .env and configure your API keys")
        return False
    
    return True


def main():
    """Main application entry point"""
    
    # Create logs directory
    create_logs_directory()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("🧠 NEURAL OPTIONS ORACLE++ STARTING")
    logger.info("=" * 60)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        sys.exit(1)
    
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug Mode: {settings.app_debug}")
    logger.info(f"Host: {settings.app_host}")
    logger.info(f"Port: {settings.app_port}")
    logger.info(f"Log Level: {settings.log_level}")
    
    # Start the server
    logger.info("Starting FastAPI server...")
    
    try:
        uvicorn.run(
            app,
            host=settings.app_host,
            port=settings.app_port,
            reload=settings.app_debug,
            log_level=settings.log_level.lower(),
            access_log=True,
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
    finally:
        logger.info("Neural Options Oracle++ shutting down")


if __name__ == "__main__":
    main()