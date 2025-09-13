#!/usr/bin/env python3
"""
Neural Options Oracle++ Main Application Entry Point
Real Data Sources Only - No Mock Data
"""
import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.api.intelligent_orchestrator import IntelligentOrchestrator
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
    """Check that required environment variables are set for real data sources"""
    
    # Core required variables
    required_env_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY"
    ]
    
    # Optional but recommended for enhanced functionality
    optional_env_vars = [
        "OPENAI_API_KEY",  # For web search agents
        "JIGSAWSTACK_API_KEY",  # For advanced scraping
        "ALPACA_API_KEY",  # For real trading data
        "ALPACA_SECRET_KEY"  # For real trading data
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required settings using the settings object
    if not settings.supabase_url or not settings.supabase_service_key:
        missing_required.extend(["SUPABASE_URL", "SUPABASE_SERVICE_KEY"])
    
    # Check optional settings
    if not settings.openai_api_key:
        missing_optional.append("OPENAI_API_KEY")
    if not settings.jigsawstack_api_key:
        missing_optional.append("JIGSAWSTACK_API_KEY")
    if not settings.alpaca_api_key:
        missing_optional.append("ALPACA_API_KEY")
    if not settings.alpaca_secret_key:
        missing_optional.append("ALPACA_SECRET_KEY")
    
    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        logger.error("Please copy .env.example to .env and configure your API keys")
        return False
    
    if missing_optional:
        logger.warning(f"Missing optional environment variables (reduced functionality): {missing_optional}")
        logger.warning("Consider adding these for enhanced real-time data collection")
    
    return True


async def initialize_real_data_sources():
    """Initialize real data sources orchestrator"""
    try:
        logger.info("Initializing Real Data Sources Orchestrator...")
        
        # Get API keys from environment
        openai_key = os.getenv("OPENAI_API_KEY")
        jigsawstack_key = os.getenv("JIGSAWSTACK_API_KEY")
        
        # Initialize orchestrator with real data sources only
        orchestrator = IntelligentOrchestrator()
        
        logger.info("✅ Intelligent Orchestrator initialized")
        logger.info("📊 AI Agents Available:")
        logger.info("  - Technical Analysis Agent (Real market data)")
        logger.info("  - Sentiment Analysis Agent (Social media & news)")
        logger.info("  - Options Flow Agent (Flow & volume analysis)")
        logger.info("  - Historical Patterns Agent (Pattern recognition)")
        logger.info("  - Education Agent (Interactive learning)")
        logger.info("  - Risk Management Agent (Portfolio protection)")
        if openai_key:
            logger.info("  - OpenAI GPT Models Enabled")
        if jigsawstack_key:
            logger.info("  - JigsawStack Enhanced Features Enabled")
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Failed to initialize real data sources: {e}")
        return None

def main():
    """Main application entry point with real data sources only"""
    
    # Create logs directory
    create_logs_directory()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("🧠 NEURAL OPTIONS ORACLE++ STARTING")
    logger.info("📊 REAL DATA SOURCES ONLY - NO MOCK DATA")
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
    
    # Initialize real data sources
    try:
        orchestrator = asyncio.run(initialize_real_data_sources())
        if not orchestrator:
            logger.error("Failed to initialize real data sources. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing real data sources: {e}")
        sys.exit(1)
    
    # Start the server
    logger.info("Starting FastAPI server...")
    
    try:
        uvicorn.run(
            "src.api.main:app",
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