"""
Neural Options Oracle++ Configuration Settings
"""
import os
from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseModel):
    """Database configuration"""
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str


class AIModelSettings(BaseModel):
    """AI model configuration"""
    openai_api_key: str
    gemini_api_key: str
    jigsawstack_api_key: str


class TradingSettings(BaseModel):
    """Trading API configuration"""
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"


class ExternalDataSettings(BaseModel):
    """External data API configuration"""
    stocktwits_access_token: Optional[str] = None
    news_api_key: Optional[str] = None


class AppSettings(BaseModel):
    """Application settings"""
    env: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # System configuration
    analysis_timeout_seconds: int = 30
    max_concurrent_analysis: int = 10
    default_risk_profile: str = "moderate"
    paper_trading_balance: float = 100000.00


class Settings(BaseSettings):
    """Main application settings"""
    
    # Database
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    
    # AI Models
    openai_api_key: str
    gemini_api_key: str
    jigsawstack_api_key: str
    
    # Trading
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # External Data
    stocktwits_access_token: Optional[str] = None
    news_api_key: Optional[str] = None
    
    # Application
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # System configuration
    analysis_timeout_seconds: int = 30
    max_concurrent_analysis: int = 10
    default_risk_profile: str = "moderate"
    paper_trading_balance: float = 100000.00
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def database(self) -> DatabaseSettings:
        """Get database settings"""
        return DatabaseSettings(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.supabase_anon_key,
            supabase_service_key=self.supabase_service_key
        )
    
    @property
    def ai_models(self) -> AIModelSettings:
        """Get AI model settings"""
        return AIModelSettings(
            openai_api_key=self.openai_api_key,
            gemini_api_key=self.gemini_api_key,
            jigsawstack_api_key=self.jigsawstack_api_key
        )
    
    @property
    def trading(self) -> TradingSettings:
        """Get trading settings"""
        return TradingSettings(
            alpaca_api_key=self.alpaca_api_key,
            alpaca_secret_key=self.alpaca_secret_key,
            alpaca_base_url=self.alpaca_base_url
        )
    
    @property
    def external_data(self) -> ExternalDataSettings:
        """Get external data settings"""
        return ExternalDataSettings(
            stocktwits_access_token=self.stocktwits_access_token,
            news_api_key=self.news_api_key
        )
    
    @property
    def app(self) -> AppSettings:
        """Get application settings"""
        return AppSettings(
            env=self.app_env,
            debug=self.app_debug,
            host=self.app_host,
            port=self.app_port,
            log_level=self.log_level,
            cors_origins=self.cors_origins,
            rate_limit_per_minute=self.rate_limit_per_minute,
            rate_limit_burst=self.rate_limit_burst,
            analysis_timeout_seconds=self.analysis_timeout_seconds,
            max_concurrent_analysis=self.max_concurrent_analysis,
            default_risk_profile=self.default_risk_profile,
            paper_trading_balance=self.paper_trading_balance
        )


# Global settings instance
settings = Settings()