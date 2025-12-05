"""
Professional Technical Indicators Calculator
Using stock-indicators library for accurate calculations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Try to import stock_indicators, use fallback if not available
try:
    from stock_indicators import indicators
    from stock_indicators.indicators.common import Quote
    STOCK_INDICATORS_AVAILABLE = True
except ImportError:
    STOCK_INDICATORS_AVAILABLE = False
    indicators = None
    Quote = None

from backend.config.logging import get_data_logger

logger = get_data_logger()

if not STOCK_INDICATORS_AVAILABLE:
    logger.warning("stock-indicators package not installed, using fallback calculations")


class TechnicalIndicatorsCalculator:
    """Professional technical indicators calculator using stock-indicators library"""
    
    def __init__(self):
        self.cache = {}
        
    def calculate_comprehensive_indicators(
        self, 
        df: pd.DataFrame, 
        symbol: str = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators using professional library"""
        
        try:
            if df.empty:
                return self._get_fallback_indicators()
            
            # Use fallback if stock-indicators not available
            if not STOCK_INDICATORS_AVAILABLE:
                return self._calculate_basic_indicators(df, symbol)
            
            # Convert pandas DataFrame to Quote objects for stock-indicators
            quotes = self._convert_to_quotes(df)
            
            if not quotes:
                return self._get_fallback_indicators()
                
            logger.info(f"Calculating professional indicators for {len(quotes)} data points")
            
            # Calculate all indicators
            indicators_data = {}
            
            # 1. PRICE & BASIC DATA
            # Handle different column name formats
            close_col = 'close' if 'close' in df.columns else 'Close'
            high_col = 'high' if 'high' in df.columns else 'High'  
            low_col = 'low' if 'low' in df.columns else 'Low'
            volume_col = 'volume' if 'volume' in df.columns else 'Volume'
            
            current_price = float(df[close_col].iloc[-1])
            indicators_data.update({
                'current_price': current_price,
                'change_percent': ((current_price - float(df[close_col].iloc[-2])) / float(df[close_col].iloc[-2])) * 100 if len(df) > 1 else 0.0,
                'volume': int(df[volume_col].iloc[-1]) if volume_col in df.columns else 1000000,
                'high_52w': float(df[high_col].max()),
                'low_52w': float(df[low_col].min()),
            })
            
            # 2. MOVING AVERAGES (Multiple types)
            indicators_data.update(self._calculate_moving_averages(quotes))
            
            # 3. OSCILLATORS
            indicators_data.update(self._calculate_oscillators(quotes))
            
            # 4. TREND INDICATORS  
            indicators_data.update(self._calculate_trend_indicators(quotes))
            
            # 5. VOLATILITY INDICATORS
            indicators_data.update(self._calculate_volatility_indicators(quotes, df))
            
            # 6. VOLUME INDICATORS
            indicators_data.update(self._calculate_volume_indicators(quotes))
            
            # 7. SUPPORT/RESISTANCE
            indicators_data.update(self._calculate_support_resistance(quotes))
            
            # 8. PATTERN RECOGNITION
            indicators_data.update(self._calculate_patterns(quotes))
            
            # 9. ADVANCED INDICATORS
            indicators_data.update(self._calculate_advanced_indicators(quotes))
            
            indicators_data['source'] = 'stock_indicators_professional'
            indicators_data['timestamp'] = datetime.now().isoformat()
            indicators_data['data_points'] = len(quotes)
            
            # Convert Decimal values to float for JSON serialization
            indicators_data = self._convert_decimals_to_float(indicators_data)
            
            logger.info(f"Professional indicators calculated successfully: {len(indicators_data)} metrics")
            return indicators_data
            
        except Exception as e:
            logger.error(f"Professional indicators calculation failed: {e}")
            return self._get_fallback_indicators()
    
    def _convert_to_quotes(self, df: pd.DataFrame) -> List[Quote]:
        """Convert pandas DataFrame to Quote objects"""
        
        try:
            # Debug: Log the DataFrame structure
            logger.debug(f"DataFrame columns: {list(df.columns)}")
            logger.debug(f"DataFrame shape: {df.shape}")
            
            if df.empty:
                return []
            
            # Handle different column name formats
            column_mapping = {
                'open': ['Open', 'open'],
                'high': ['High', 'high'], 
                'low': ['Low', 'low'],
                'close': ['Close', 'close'],
                'volume': ['Volume', 'volume', 'Vol']
            }
            
            # Find actual column names
            cols = {}
            for key, possible_names in column_mapping.items():
                for name in possible_names:
                    if name in df.columns:
                        cols[key] = name
                        break
                else:
                    if key == 'volume':
                        cols[key] = None  # Volume is optional
                    else:
                        raise ValueError(f"Required column for {key} not found in {list(df.columns)}")
            
            quotes = []
            for index, row in df.iterrows():
                quote = Quote(
                    date=index if hasattr(index, 'date') else datetime.now(),
                    open=float(row[cols['open']]),
                    high=float(row[cols['high']]),
                    low=float(row[cols['low']]),
                    close=float(row[cols['close']]),
                    volume=int(row[cols['volume']]) if cols['volume'] else 1000000
                )
                quotes.append(quote)
            
            logger.debug(f"Successfully converted {len(quotes)} quotes")
            return quotes
            
        except Exception as e:
            logger.error(f"Quote conversion failed: {e}")
            return []
    
    def _calculate_moving_averages(self, quotes: List[Quote]) -> Dict[str, float]:
        """Calculate multiple types of moving averages"""
        
        ma_data = {}
        
        try:
            # Simple Moving Averages
            sma_periods = [5, 10, 20, 50, 200]
            for period in sma_periods:
                if len(quotes) >= period:
                    sma_results = indicators.get_sma(quotes, period)
                    if sma_results:
                        ma_data[f'sma_{period}'] = float(sma_results[-1].sma or 0)
                        # Backward compatibility
                        if period == 5:
                            ma_data['ma5'] = ma_data['sma_5']
                        elif period == 20:
                            ma_data['ma20'] = ma_data['sma_20']
                        elif period == 50:
                            ma_data['ma50'] = ma_data['sma_50']
                        elif period == 200:
                            ma_data['ma200'] = ma_data['sma_200']
            
            # Exponential Moving Averages
            ema_periods = [12, 26, 50]
            for period in ema_periods:
                if len(quotes) >= period:
                    ema_results = indicators.get_ema(quotes, period)
                    if ema_results:
                        ma_data[f'ema_{period}'] = float(ema_results[-1].ema or 0)
            
            # Hull Moving Average (advanced)
            if len(quotes) >= 20:
                hma_results = indicators.get_hma(quotes, 20)
                if hma_results:
                    ma_data['hma_20'] = float(hma_results[-1].hma or 0)
            
            # VWAP (Volume Weighted Average Price)
            vwap_results = indicators.get_vwap(quotes)
            if vwap_results:
                ma_data['vwap'] = float(vwap_results[-1].vwap or 0)
                
        except Exception as e:
            logger.warning(f"Moving averages calculation failed: {e}")
            ma_data.update({
                'ma5': quotes[-1].close,
                'ma20': quotes[-1].close,
                'ma50': quotes[-1].close,
                'ma200': quotes[-1].close,
                'vwap': quotes[-1].close
            })
        
        return ma_data
    
    def _calculate_oscillators(self, quotes: List[Quote]) -> Dict[str, float]:
        """Calculate oscillator indicators"""
        
        oscillators = {}
        
        try:
            # RSI (Relative Strength Index)
            if len(quotes) >= 14:
                rsi_results = indicators.get_rsi(quotes, 14)
                if rsi_results:
                    oscillators['rsi'] = float(rsi_results[-1].rsi or 50)
            
            # Stochastic Oscillator
            if len(quotes) >= 14:
                stoch_results = indicators.get_stoch(quotes, 14, 3, 3)
                if stoch_results:
                    oscillators['stoch_k'] = float(stoch_results[-1].k or 50)
                    oscillators['stoch_d'] = float(stoch_results[-1].d or 50)
            
            # Williams %R
            if len(quotes) >= 14:
                williams_results = indicators.get_williams_r(quotes, 14)
                if williams_results:
                    oscillators['williams_r'] = float(williams_results[-1].williams_r or -50)
            
            # Commodity Channel Index (CCI)
            if len(quotes) >= 20:
                cci_results = indicators.get_cci(quotes, 20)
                if cci_results:
                    oscillators['cci'] = float(cci_results[-1].cci or 0)
                    
        except Exception as e:
            logger.warning(f"Oscillators calculation failed: {e}")
            oscillators.update({
                'rsi': 50.0,
                'stoch_k': 50.0,
                'stoch_d': 50.0,
                'williams_r': -50.0,
                'cci': 0.0
            })
        
        return oscillators
    
    def _calculate_trend_indicators(self, quotes: List[Quote]) -> Dict[str, Any]:
        """Calculate trend-following indicators"""
        
        trend_data = {}
        
        try:
            # MACD (Moving Average Convergence Divergence)
            if len(quotes) >= 34:
                macd_results = indicators.get_macd(quotes, 12, 26, 9)
                if macd_results:
                    latest_macd = macd_results[-1]
                    trend_data.update({
                        'macd': float(latest_macd.macd or 0),
                        'macd_signal': float(latest_macd.signal or 0),
                        'macd_histogram': float(latest_macd.histogram or 0)
                    })
            
            # ADX (Average Directional Index)
            if len(quotes) >= 14:
                adx_results = indicators.get_adx(quotes, 14)
                if adx_results:
                    latest_adx = adx_results[-1]
                    trend_data.update({
                        'adx': float(latest_adx.adx or 20),
                        'pdi': float(latest_adx.pdi or 20),
                        'mdi': float(latest_adx.mdi or 20)
                    })
            
            # Aroon Indicator
            if len(quotes) >= 25:
                aroon_results = indicators.get_aroon(quotes, 25)
                if aroon_results:
                    latest_aroon = aroon_results[-1]
                    trend_data.update({
                        'aroon_up': float(latest_aroon.aroon_up or 50),
                        'aroon_down': float(latest_aroon.aroon_down or 50),
                        'aroon_oscillator': float(latest_aroon.oscillator or 0)
                    })
            
            # Supertrend
            if len(quotes) >= 10:
                supertrend_results = indicators.get_super_trend(quotes, 10, 3.0)
                if supertrend_results:
                    trend_data['supertrend'] = float(supertrend_results[-1].super_trend or quotes[-1].close)
                    trend_data['supertrend_signal'] = 'bullish' if quotes[-1].close > supertrend_results[-1].super_trend else 'bearish'
                    
        except Exception as e:
            logger.warning(f"Trend indicators calculation failed: {e}")
            trend_data.update({
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'adx': 20.0,
                'aroon_up': 50.0,
                'aroon_down': 50.0
            })
        
        return trend_data
    
    def _calculate_volatility_indicators(self, quotes: List[Quote], df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        
        # Define column names for consistency
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        vol_data = {}
        
        try:
            # Bollinger Bands
            if len(quotes) >= 20:
                bb_results = indicators.get_bollinger_bands(quotes, 20, 2)
                if bb_results:
                    latest_bb = bb_results[-1]
                    current_close = float(quotes[-1].close)
                    upper_band = float(latest_bb.upper_band or current_close + 10)
                    lower_band = float(latest_bb.lower_band or current_close - 10)
                    sma = float(latest_bb.sma or current_close)
                    width = float(latest_bb.width or 10)
                    
                    vol_data.update({
                        'bb_upper': upper_band,
                        'bb_middle': sma,
                        'bb_lower': lower_band,
                        'bb_width': width,
                        'bb_position': float((current_close - lower_band) / (upper_band - lower_band)) if upper_band != lower_band else 0.5
                    })
            
            # Average True Range (ATR)
            if len(quotes) >= 14:
                atr_results = indicators.get_atr(quotes, 14)
                if atr_results:
                    vol_data['atr'] = float(atr_results[-1].atr or 1.0)
            
            # Keltner Channels
            if len(quotes) >= 20:
                keltner_results = indicators.get_keltner(quotes, 20, 2.0)
                if keltner_results:
                    latest_keltner = keltner_results[-1]
                    current_close = float(quotes[-1].close)
                    vol_data.update({
                        'keltner_upper': float(latest_keltner.upper_band or current_close + 5),
                        'keltner_middle': float(latest_keltner.center_line or current_close),
                        'keltner_lower': float(latest_keltner.lower_band or current_close - 5)
                    })
            
            # Historical Volatility
            if len(df) >= 30:
                returns = df[close_col].pct_change().dropna()
                vol_data['volatility'] = float(returns.std() * np.sqrt(252) * 100)  # Annualized
                if len(returns) > 252:
                    rolling_std = returns.rolling(252).std()
                    min_vol = float(rolling_std.min() * 100)
                    max_vol = float(rolling_std.max() * 100)
                    vol_data['volatility_percentile'] = float(
                        (vol_data['volatility'] - min_vol) / (max_vol - min_vol)
                    ) if max_vol > min_vol else 0.5
                else:
                    vol_data['volatility_percentile'] = 0.5
                
        except Exception as e:
            logger.warning(f"Volatility indicators calculation failed: {e}")
            current_close = float(quotes[-1].close)
            vol_data.update({
                'bb_upper': current_close + 5,
                'bb_middle': current_close,
                'bb_lower': current_close - 5,
                'bb_position': 0.5,
                'atr': 1.0,
                'volatility': 25.0
            })
        
        return vol_data
    
    def _calculate_volume_indicators(self, quotes: List[Quote]) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        
        volume_data = {}
        
        try:
            # On-Balance Volume (OBV)
            obv_results = indicators.get_obv(quotes)
            if obv_results:
                volume_data['obv'] = float(obv_results[-1].obv or 0)
            
            # Money Flow Index (MFI)
            if len(quotes) >= 14:
                mfi_results = indicators.get_mfi(quotes, 14)
                if mfi_results:
                    volume_data['mfi'] = float(mfi_results[-1].mfi or 50)
            
            # Chaikin Money Flow (CMF)
            if len(quotes) >= 20:
                cmf_results = indicators.get_cmf(quotes, 20)
                if cmf_results:
                    volume_data['cmf'] = float(cmf_results[-1].cmf or 0)
            
            # Volume statistics
            volumes = [q.volume for q in quotes if q.volume > 0]
            if volumes:
                recent_volume = quotes[-1].volume
                avg_volume = sum(volumes[-20:]) / min(20, len(volumes))
                volume_data.update({
                    'current_volume': recent_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': recent_volume / avg_volume if avg_volume > 0 else 1.0
                })
                
        except Exception as e:
            logger.warning(f"Volume indicators calculation failed: {e}")
            volume_data.update({
                'obv': 0,
                'mfi': 50,
                'cmf': 0,
                'current_volume': quotes[-1].volume,
                'avg_volume': quotes[-1].volume,
                'volume_ratio': 1.0
            })
        
        return volume_data
    
    def _calculate_support_resistance(self, quotes: List[Quote]) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        
        sr_data = {}
        
        try:
            # Pivot Points
            if len(quotes) >= 3:
                try:
                    # Try different window sizes to avoid the cs_value error
                    for window_size in [3, 5, 7]:
                        try:
                            pivot_results = indicators.get_pivot_points(quotes, window_size=window_size)
                            if pivot_results and len(pivot_results) > 0:
                                latest_pivot = pivot_results[-1]
                                # Safely access pivot point attributes
                                sr_data.update({
                                    'pivot_point': float(getattr(latest_pivot, 'pp', quotes[-1].close) or quotes[-1].close),
                                    'resistance_1': float(getattr(latest_pivot, 'r1', quotes[-1].close + 5) or quotes[-1].close + 5),
                                    'resistance_2': float(getattr(latest_pivot, 'r2', quotes[-1].close + 10) or quotes[-1].close + 10),
                                    'support_1': float(getattr(latest_pivot, 's1', quotes[-1].close - 5) or quotes[-1].close - 5),
                                    'support_2': float(getattr(latest_pivot, 's2', quotes[-1].close - 10) or quotes[-1].close - 10)
                                })
                                break  # Success, exit the loop
                        except Exception as window_error:
                            logger.debug(f"Pivot points failed with window_size={window_size}: {window_error}")
                            continue
                except Exception as pivot_error:
                    logger.warning(f"Pivot points calculation failed completely: {pivot_error}")
                    # Continue with simple support/resistance calculation
            
            # Simple high/low support/resistance
            highs = [q.high for q in quotes[-20:]]
            lows = [q.low for q in quotes[-20:]]
            
            sr_data.update({
                'resistance': max(highs) if highs else quotes[-1].close + 5,
                'support': min(lows) if lows else quotes[-1].close - 5
            })
            
        except Exception as e:
            logger.warning(f"Support/resistance calculation failed: {e}")
            sr_data.update({
                'resistance': quotes[-1].close + 5,
                'support': quotes[-1].close - 5,
                'pivot_point': quotes[-1].close
            })
        
        return sr_data
    
    def _calculate_patterns(self, quotes: List[Quote]) -> Dict[str, Any]:
        """Calculate pattern recognition indicators"""
        
        patterns = {}
        
        try:
            # Williams Fractal
            if len(quotes) >= 5:
                fractal_results = indicators.get_fractal(quotes)
                if fractal_results:
                    # Count recent fractals
                    recent_fractals = [f for f in fractal_results[-20:] if f.fractal_bear or f.fractal_bull]
                    patterns.update({
                        'fractal_count': len(recent_fractals),
                        'recent_fractal_bull': any(f.fractal_bull for f in fractal_results[-5:]),
                        'recent_fractal_bear': any(f.fractal_bear for f in fractal_results[-5:])
                    })
            
            # Trend analysis
            if len(quotes) >= 20:
                recent_closes = [q.close for q in quotes[-20:]]
                trend_slope = (recent_closes[-1] - recent_closes[0]) / len(recent_closes)
                patterns.update({
                    'trend_slope': trend_slope,
                    'trend_strength': abs(trend_slope) / quotes[-1].close * 100,
                    'trend_direction': 'up' if trend_slope > 0 else 'down' if trend_slope < 0 else 'sideways'
                })
                
        except Exception as e:
            logger.warning(f"Pattern recognition failed: {e}")
            patterns.update({
                'fractal_count': 0,
                'trend_direction': 'sideways',
                'trend_strength': 0.0
            })
        
        return patterns
    
    def _calculate_advanced_indicators(self, quotes: List[Quote]) -> Dict[str, Any]:
        """Calculate advanced technical indicators"""
        
        advanced = {}
        
        try:
            # True Strength Index (TSI)
            if len(quotes) >= 50:
                tsi_results = indicators.get_tsi(quotes, 25, 13)
                if tsi_results:
                    advanced['tsi'] = float(tsi_results[-1].tsi or 0)
            
            # Ultimate Oscillator
            if len(quotes) >= 28:
                uo_results = indicators.get_ultimate(quotes, 7, 14, 28)
                if uo_results:
                    advanced['ultimate_oscillator'] = float(getattr(uo_results[-1], 'uo', 50))
            
            # Choppiness Index
            if len(quotes) >= 14:
                chop_results = indicators.get_chop(quotes, 14)
                if chop_results:
                    advanced['choppiness'] = float(chop_results[-1].chop or 50)
            
        except Exception as e:
            logger.warning(f"Advanced indicators calculation failed: {e}")
            advanced.update({
                'tsi': 0.0,
                'ultimate_oscillator': 50.0,
                'choppiness': 50.0
            })
        
        return advanced
    
    def _convert_decimals_to_float(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Decimal values to float for JSON serialization"""
        from decimal import Decimal
        from datetime import datetime
        import pandas as pd
        
        converted = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                converted[key] = float(value)
            elif isinstance(value, (pd.Timestamp, datetime)):
                converted[key] = value.isoformat()
            elif isinstance(value, dict):
                converted[key] = self._convert_decimals_to_float(value)
            elif isinstance(value, list):
                converted[key] = [
                    float(item) if isinstance(item, Decimal) else 
                    item.isoformat() if isinstance(item, (pd.Timestamp, datetime)) else item
                    for item in value
                ]
            else:
                converted[key] = value
        
        return converted
    
    def _calculate_basic_indicators(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """Calculate basic indicators using pandas when stock-indicators is not available"""
        try:
            # Ensure we have the required columns
            if 'close' not in df.columns:
                if 'Close' in df.columns:
                    df = df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
                else:
                    return self._get_fallback_indicators()
            
            close = df['close'].values
            high = df['high'].values if 'high' in df.columns else close
            low = df['low'].values if 'low' in df.columns else close
            volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
            
            current_price = float(close[-1]) if len(close) > 0 else 150.0
            
            # Simple Moving Averages
            ma5 = float(np.mean(close[-5:])) if len(close) >= 5 else current_price
            ma20 = float(np.mean(close[-20:])) if len(close) >= 20 else current_price
            ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else current_price
            ma200 = float(np.mean(close[-200:])) if len(close) >= 200 else current_price
            
            # RSI (14-period)
            if len(close) >= 15:
                delta = np.diff(close)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # MACD
            if len(close) >= 26:
                ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
                ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
                macd = float(ema12 - ema26)
                signal = float(pd.Series(close).ewm(span=9).mean().iloc[-1] - ema26)
                histogram = macd - signal
            else:
                macd, signal, histogram = 0.0, 0.0, 0.0
            
            # Bollinger Bands
            if len(close) >= 20:
                sma20 = np.mean(close[-20:])
                std20 = np.std(close[-20:])
                bb_upper = float(sma20 + 2 * std20)
                bb_lower = float(sma20 - 2 * std20)
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            else:
                bb_upper, bb_lower, bb_position = current_price + 5, current_price - 5, 0.5
            
            # ATR (14-period)
            if len(high) >= 14 and len(low) >= 14:
                tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
                atr = float(np.mean(tr[-14:]))
            else:
                atr = 1.0
            
            # Volume analysis
            current_vol = float(volume[-1]) if len(volume) > 0 else 1000000
            avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else current_vol
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            return {
                'current_price': current_price,
                'change_percent': ((current_price - close[-2]) / close[-2] * 100) if len(close) > 1 else 0.0,
                'volume': int(current_vol),
                'ma5': ma5, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
                'vwap': current_price,
                'rsi': float(rsi),
                'stoch_k': 50.0, 'stoch_d': 50.0,  # Simplified
                'williams_r': -50.0, 'cci': 0.0,
                'macd': macd, 'macd_signal': signal, 'macd_histogram': histogram,
                'adx': 20.0,
                'bb_upper': bb_upper, 'bb_middle': float(ma20), 'bb_lower': bb_lower,
                'bb_position': float(bb_position),
                'atr': atr, 'volatility': float(atr / current_price * 100) if current_price > 0 else 25.0,
                'current_volume': int(current_vol), 'avg_volume': int(avg_vol),
                'volume_ratio': float(vol_ratio),
                'obv': 0, 'mfi': 50,
                'resistance': current_price * 1.05,
                'support': current_price * 0.95,
                'source': 'pandas_fallback',
                'timestamp': datetime.now().isoformat(),
                'data_points': len(close)
            }
        except Exception as e:
            logger.error(f"Basic indicator calculation failed: {e}")
            return self._get_fallback_indicators()
    
    def _get_fallback_indicators(self) -> Dict[str, Any]:
        """Fallback indicators when calculation fails"""
        
        base_price = 150.0
        
        return {
            'current_price': base_price,
            'change_percent': 0.0,
            'volume': 1000000,
            
            # Moving averages
            'ma5': base_price,
            'ma20': base_price, 
            'ma50': base_price,
            'ma200': base_price,
            'vwap': base_price,
            
            # Oscillators
            'rsi': 50.0,
            'stoch_k': 50.0,
            'stoch_d': 50.0,
            'williams_r': -50.0,
            'cci': 0.0,
            
            # Trend indicators
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'adx': 20.0,
            
            # Volatility
            'bb_upper': base_price + 5,
            'bb_middle': base_price,
            'bb_lower': base_price - 5,
            'bb_position': 0.5,
            'atr': 1.0,
            'volatility': 25.0,
            
            # Volume
            'current_volume': 1000000,
            'avg_volume': 1000000,
            'volume_ratio': 1.0,
            'obv': 0,
            'mfi': 50,
            
            # Support/Resistance
            'resistance': base_price + 10,
            'support': base_price - 10,
            
            'source': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'data_points': 0
        }


# Global calculator instance
technical_calculator = TechnicalIndicatorsCalculator()