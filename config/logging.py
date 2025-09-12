"""
Neural Options Oracle++ Logging Configuration
"""
import sys
from typing import Dict, Any
from loguru import logger
from config.settings import settings


def setup_logging() -> None:
    """Setup application logging"""
    
    # Remove default logger
    logger.remove()
    
    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File logging format
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler for all logs
    logger.add(
        "logs/neural_oracle.log",
        format=file_format,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        backtrace=True,
        diagnose=True
    )
    
    # Add separate file handler for errors
    logger.add(
        "logs/neural_oracle_errors.log",
        format=file_format,
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        backtrace=True,
        diagnose=True
    )
    
    # Add API access log handler
    logger.add(
        "logs/api_access.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        level="INFO",
        rotation="50 MB",
        retention="14 days",
        filter=lambda record: "API_ACCESS" in record["extra"]
    )
    
    logger.info("Logging configuration initialized")


def get_logger(name: str):
    """Get logger instance with context"""
    return logger.bind(name=name)


class LoggingContexts:
    """Predefined logging contexts"""
    
    API = "neural_oracle.api"
    AGENTS = "neural_oracle.agents" 
    DATABASE = "neural_oracle.database"
    TRADING = "neural_oracle.trading"
    ML_MODELS = "neural_oracle.ml"
    DATA_PIPELINE = "neural_oracle.data"
    EDUCATION = "neural_oracle.education"
    CORE = "neural_oracle.core"


# Convenience functions for different contexts
def get_api_logger():
    return get_logger(LoggingContexts.API)

def get_agents_logger():
    return get_logger(LoggingContexts.AGENTS)

def get_database_logger():
    return get_logger(LoggingContexts.DATABASE)

def get_trading_logger():
    return get_logger(LoggingContexts.TRADING)

def get_ml_logger():
    return get_logger(LoggingContexts.ML_MODELS)

def get_data_logger():
    return get_logger(LoggingContexts.DATA_PIPELINE)

def get_education_logger():
    return get_logger(LoggingContexts.EDUCATION)

def get_core_logger():
    return get_logger(LoggingContexts.CORE)


def log_api_access(
    method: str, 
    path: str, 
    status_code: int, 
    response_time: float,
    user_agent: str = None,
    ip_address: str = None
) -> None:
    """Log API access"""
    logger.bind(API_ACCESS=True).info(
        f"{method} {path} {status_code} {response_time:.3f}s "
        f"UA: {user_agent} IP: {ip_address}"
    )