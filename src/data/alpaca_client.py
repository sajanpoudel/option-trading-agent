"""
Alpaca Market Data Client
Real-time and historical market data integration
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import Bar, Quote, Trade
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import yfinance as yf
import requests_cache

from config.settings import settings
from config.logging import get_data_logger

logger = get_data_logger()

# Cache for yfinance requests (1 hour cache)
session = requests_cache.CachedSession('market_data_cache', expire_after=3600)


class AlpacaMarketDataClient:
    """Real-time market data client using Alpaca API with yfinance fallback"""
    
    def __init__(self):
        self.alpaca_data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key
        )
        self.alpaca_trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True  # Using paper trading
        )
        
        logger.info("Alpaca market data client initialized")
    
    async def get_current_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for symbol"""
        try:
            # Try Alpaca first for price data
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.alpaca_data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                # Get volume from yfinance since Alpaca quote doesn't have daily volume
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    volume = int(info.get('volume', 0))
                except:
                    volume = 0
                
                return {
                    'symbol': symbol,
                    'price': float(quote.ask_price + quote.bid_price) / 2,
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'volume': volume,  # Real volume from yfinance
                    'timestamp': quote.timestamp.isoformat(),
                    'source': 'alpaca_price_yfinance_volume'
                }
                
        except Exception as e:
            logger.warning(f"Alpaca quote failed for {symbol}: {e}")
        
        # Fallback to yfinance for everything
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': float(info.get('currentPrice', info.get('regularMarketPrice', 0))),
                'bid': float(info.get('bid', 0)),
                'ask': float(info.get('ask', 0)),
                'volume': int(info.get('volume', 0)),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance'
            }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return self._get_fallback_quote(symbol)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data"""
        
        try:
            # Map periods to Alpaca timeframes
            timeframe_map = {
                "1d": TimeFrame.Day,
                "1h": TimeFrame.Hour,
                "15m": TimeFrame.Minute,
                "5m": TimeFrame.Minute
            }
            
            timeframe = timeframe_map.get(interval, TimeFrame.Day)
            
            # Calculate start/end dates
            if period == "1y":
                start_date = datetime.now() - timedelta(days=365)
            elif period == "6mo":
                start_date = datetime.now() - timedelta(days=180)
            elif period == "3mo":
                start_date = datetime.now() - timedelta(days=90)
            elif period == "1mo":
                start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = datetime.now() - timedelta(days=30)
            
            # Try Alpaca first
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=datetime.now()
            )
            
            bars = self.alpaca_data_client.get_stock_bars(request)
            
            if symbol in bars.df.index.get_level_values('symbol'):
                df = bars.df[bars.df.index.get_level_values('symbol') == symbol]
                df.index = df.index.droplevel('symbol')
                return df
                
        except Exception as e:
            logger.warning(f"Alpaca historical data failed for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            logger.info(f"Retrieved {len(df)} bars for {symbol} via yfinance")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators using professional stock-indicators library"""
        
        try:
            # Get historical data
            df = await self.get_historical_data(symbol, period="1y", interval="1d")  # More data for better indicators
            
            if df.empty:
                return self._get_fallback_indicators(symbol)
            
            # Use professional indicators calculator
            from src.indicators.technical_calculator import technical_calculator
            indicators = technical_calculator.calculate_comprehensive_indicators(df, symbol)
            
            logger.info(f"Professional technical indicators calculated for {symbol}: {indicators.get('data_points', 0)} bars")
            return indicators
            
        except Exception as e:
            logger.error(f"Professional technical indicators calculation failed for {symbol}: {e}")
            # Try basic calculation as fallback
            try:
                return await self._get_basic_indicators(symbol)
            except:
                return self._get_fallback_indicators(symbol)
    
    async def _get_basic_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fallback to basic indicator calculation"""
        
        try:
            df = await self.get_historical_data(symbol, period="3mo", interval="1d")
            if df.empty:
                return self._get_fallback_indicators(symbol)
            
            # Basic calculations only
            indicators = {
                'current_price': float(df['Close'].iloc[-1]),
                'change_percent': ((float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])) / float(df['Close'].iloc[-2])) * 100 if len(df) > 1 else 0.0,
                'volume': int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 1000000,
                'ma20': float(df['Close'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df['Close'].iloc[-1]),
                'ma50': float(df['Close'].rolling(50).mean().iloc[-1]) if len(df) >= 50 else float(df['Close'].iloc[-1]),
                'resistance': float(df['High'].rolling(20).max().iloc[-1]) if len(df) >= 20 else float(df['High'].iloc[-1]),
                'support': float(df['Low'].rolling(20).min().iloc[-1]) if len(df) >= 20 else float(df['Low'].iloc[-1]),
                'source': 'basic_fallback',
                'volatility': 25.0,
                'rsi': 50.0,
                'macd': 0.0,
                'vwap': float(df['Close'].iloc[-1])
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Basic indicators calculation failed: {e}")
            return self._get_fallback_indicators(symbol)
    
    async def get_options_data(self, symbol: str) -> Dict[str, Any]:
        """Get options chain data (using yfinance)"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options expiration dates
            expiration_dates = ticker.options
            
            if not expiration_dates:
                return {'error': 'No options data available'}
            
            # Get options chain for the nearest expiration
            nearest_expiry = expiration_dates[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            # Process calls and puts
            calls_df = options_chain.calls
            puts_df = options_chain.puts
            
            # Calculate metrics
            total_call_volume = calls_df['volume'].fillna(0).sum()
            total_put_volume = puts_df['volume'].fillna(0).sum()
            put_call_ratio = total_put_volume / max(total_call_volume, 1)
            
            # Get current stock price for delta approximation
            current_price = float(ticker.info.get('currentPrice', 100))
            
            # Simple delta approximation for at-the-money options
            atm_calls = calls_df[abs(calls_df['strike'] - current_price) < 5]
            atm_puts = puts_df[abs(puts_df['strike'] - current_price) < 5]
            
            return {
                'symbol': symbol,
                'expiration': nearest_expiry,
                'current_price': current_price,
                'put_call_ratio': put_call_ratio,
                'total_call_volume': int(total_call_volume),
                'total_put_volume': int(total_put_volume),
                'call_count': len(calls_df),
                'put_count': len(puts_df),
                'atm_calls': atm_calls.to_dict('records') if not atm_calls.empty else [],
                'atm_puts': atm_puts.to_dict('records') if not atm_puts.empty else [],
                'source': 'yfinance_options'
            }
            
        except Exception as e:
            logger.error(f"Options data retrieval failed for {symbol}: {e}")
            return self._get_fallback_options_data(symbol)
    
    def _get_fallback_quote(self, symbol: str) -> Dict[str, Any]:
        """Fallback quote when all APIs fail"""
        return {
            'symbol': symbol,
            'price': 100.0,  # Default price
            'bid': 99.5,
            'ask': 100.5,
            'volume': 1000000,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
    def _get_fallback_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fallback indicators when calculation fails"""
        base_price = 100.0
        
        return {
            'current_price': base_price,
            'change_percent': 0.0,
            'ma5': base_price,
            'ma20': base_price,
            'ma50': base_price,
            'ma200': base_price,
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_upper': base_price + 5,
            'bb_middle': base_price,
            'bb_lower': base_price - 5,
            'bb_position': 0.5,
            'vwap': base_price,
            'volatility': 25.0,
            'avg_volume': 1000000,
            'current_volume': 1000000,
            'volume_ratio': 1.0,
            'resistance': base_price + 10,
            'support': base_price - 10,
            'source': 'fallback'
        }
    
    def _get_fallback_options_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback options data when APIs fail"""
        return {
            'symbol': symbol,
            'expiration': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'current_price': 100.0,
            'put_call_ratio': 1.0,
            'total_call_volume': 1000,
            'total_put_volume': 1000,
            'call_count': 20,
            'put_count': 20,
            'atm_calls': [],
            'atm_puts': [],
            'source': 'fallback'
        }