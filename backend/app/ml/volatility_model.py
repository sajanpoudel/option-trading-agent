"""
Prophet Volatility Forecasting Model
Advanced time series forecasting for implied volatility and market volatility prediction
"""

import asyncio
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.config.settings import settings
from backend.config.logging import get_data_logger

logger = get_data_logger()


@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    symbol: str
    forecast_horizon_days: int
    current_volatility: float
    predicted_volatility: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    confidence_intervals: Dict[str, float]
    seasonal_components: Dict[str, Any]
    market_regime: str  # 'low_vol', 'medium_vol', 'high_vol'
    forecast_accuracy: float
    key_drivers: List[str]
    timestamp: datetime


class ProphetVolatilityPredictor:
    """Advanced volatility forecasting using Facebook Prophet"""
    
    def __init__(self):
        self.models = {}  # Store models per symbol
        self.scalers = {}
        self.training_data = {}
        self.forecast_history = {}
        
        # Prophet parameters optimized for volatility data
        self.prophet_params = {
            'growth': 'linear',
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'mcmc_samples': 0,
            'uncertainty_samples': 1000
        }
        
        # Volatility thresholds for different market regimes
        self.vol_regimes = {
            'low_vol': 20.0,
            'medium_vol': 35.0,
            'high_vol': 50.0
        }
        
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, using statistical fallback")
        else:
            logger.info("Prophet Volatility Predictor initialized")
    
    async def predict_volatility(
        self, 
        symbol: str, 
        historical_data: pd.DataFrame,
        horizon_days: int = 30
    ) -> VolatilityForecast:
        """Predict volatility for given symbol and horizon"""
        
        try:
            if not PROPHET_AVAILABLE:
                return self._fallback_prediction(symbol, historical_data, horizon_days)
            
            # Prepare data for Prophet
            df = self._prepare_volatility_data(historical_data, symbol)
            
            if len(df) < 30:  # Need at least 30 days of data
                logger.warning(f"Insufficient data for {symbol}, using statistical approach")
                return self._statistical_prediction(symbol, historical_data, horizon_days)
            
            # Get or create model for this symbol
            model = self._get_or_create_model(symbol)
            
            # Train model with latest data
            model.fit(df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=horizon_days)
            
            # Add custom regressors if available
            future = self._add_custom_regressors(future, symbol)
            
            # Make forecast
            forecast = model.predict(future)
            
            # Extract current and predicted values
            current_vol = df['y'].iloc[-1] if len(df) > 0 else 20.0
            predicted_vol = forecast['yhat'].iloc[-1]
            
            # Calculate confidence intervals
            confidence_intervals = {
                'lower_80': forecast['yhat_lower'].iloc[-1],
                'upper_80': forecast['yhat_upper'].iloc[-1],
                'lower_95': forecast['yhat_lower'].iloc[-1] * 0.9,
                'upper_95': forecast['yhat_upper'].iloc[-1] * 1.1
            }
            
            # Analyze trend
            trend_data = forecast['trend'].tail(7)  # Last 7 days
            volatility_trend = self._analyze_trend(trend_data)
            
            # Extract seasonal components
            seasonal_components = self._extract_seasonal_components(forecast)
            
            # Determine market regime
            market_regime = self._classify_volatility_regime(predicted_vol)
            
            # Calculate forecast accuracy
            accuracy = self._calculate_forecast_accuracy(symbol, forecast, df)
            
            # Identify key drivers
            key_drivers = self._identify_volatility_drivers(forecast, historical_data)
            
            result = VolatilityForecast(
                symbol=symbol,
                forecast_horizon_days=horizon_days,
                current_volatility=float(current_vol),
                predicted_volatility=float(predicted_vol),
                volatility_trend=volatility_trend,
                confidence_intervals=confidence_intervals,
                seasonal_components=seasonal_components,
                market_regime=market_regime,
                forecast_accuracy=accuracy,
                key_drivers=key_drivers,
                timestamp=datetime.now()
            )
            
            # Store forecast in history
            self._store_forecast_history(symbol, result)
            
            logger.info(f"Volatility forecast for {symbol}: {predicted_vol:.1f}% ({volatility_trend})")
            return result
            
        except Exception as e:
            logger.error(f"Volatility prediction failed for {symbol}: {e}")
            return self._fallback_prediction(symbol, historical_data, horizon_days)
    
    def _prepare_volatility_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Calculate realized volatility (Parkinson estimator)
            df = df.copy()
            df['log_hl'] = np.log(df['high'] / df['low'])
            df['parkinson_vol'] = np.sqrt(252 * (df['log_hl'] ** 2) / (4 * np.log(2))) * 100
            
            # Calculate Garman-Klass volatility if we have volume
            if 'volume' in df.columns:
                df['log_co'] = np.log(df['close'] / df['close'].shift(1))
                df['gk_vol'] = np.sqrt(252 * (
                    0.5 * (df['log_hl'] ** 2) - 
                    (2 * np.log(2) - 1) * (df['log_co'] ** 2)
                )) * 100
                volatility_col = 'gk_vol'
            else:
                volatility_col = 'parkinson_vol'
            
            # Prepare Prophet dataframe
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df[volatility_col].rolling(window=5).mean()  # 5-day smoothing
            })
            
            # Remove NaN values
            prophet_df = prophet_df.dropna()
            
            # Ensure ds is datetime
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Cap extreme values
            prophet_df['y'] = np.clip(prophet_df['y'], 5, 200)  # Cap between 5% and 200%
            
            return prophet_df
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            # Fallback to simple close-based volatility
            returns = df['close'].pct_change().dropna()
            vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
            
            return pd.DataFrame({
                'ds': vol.index,
                'y': vol.values
            }).dropna()
    
    def _get_or_create_model(self, symbol: str) -> Prophet:
        """Get existing model or create new one for symbol"""
        
        if symbol not in self.models:
            model = Prophet(**self.prophet_params)
            
            # Add custom seasonalities
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5,
                mode='multiplicative'
            )
            
            # Add market hours seasonality
            model.add_seasonality(
                name='intraday',
                period=1,
                fourier_order=3,
                mode='additive'
            )
            
            self.models[symbol] = model
            logger.info(f"Created new Prophet model for {symbol}")
        
        return self.models[symbol]
    
    def _add_custom_regressors(self, future: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add custom regressors to future dataframe"""
        
        try:
            # Add market regime indicators
            future['is_weekend'] = future['ds'].dt.weekday >= 5
            future['month'] = future['ds'].dt.month
            future['quarter'] = future['ds'].dt.quarter
            
            # Add FOMC meeting indicator (simplified)
            fomc_months = [2, 5, 7, 9, 11, 12]  # Typical FOMC meeting months
            future['fomc_month'] = future['ds'].dt.month.isin(fomc_months)
            
            # Add earnings season indicator
            earnings_months = [1, 4, 7, 10]  # Quarterly earnings seasons
            future['earnings_season'] = future['ds'].dt.month.isin(earnings_months)
            
            # Add market stress indicators (would need real data in production)
            future['vix_regime'] = 0.0  # Placeholder
            future['market_trend'] = 0.0  # Placeholder
            
            return future
            
        except Exception as e:
            logger.warning(f"Failed to add custom regressors: {e}")
            return future
    
    def _analyze_trend(self, trend_data: pd.Series) -> str:
        """Analyze volatility trend from recent data"""
        
        try:
            if len(trend_data) < 2:
                return 'stable'
            
            # Calculate trend slope
            x = np.arange(len(trend_data))
            slope = np.polyfit(x, trend_data.values, 1)[0]
            
            # Classify trend
            if slope > 0.5:
                return 'increasing'
            elif slope < -0.5:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _extract_seasonal_components(self, forecast: pd.DataFrame) -> Dict[str, Any]:
        """Extract seasonal components from forecast"""
        
        try:
            components = {}
            
            # Weekly seasonality
            if 'weekly' in forecast.columns:
                weekly_peak = forecast['weekly'].tail(7).idxmax()
                weekly_effect = forecast['weekly'].tail(7).max() - forecast['weekly'].tail(7).min()
                components['weekly'] = {
                    'peak_day': weekly_peak % 7,
                    'effect_magnitude': float(weekly_effect)
                }
            
            # Monthly seasonality
            if 'monthly' in forecast.columns:
                monthly_effect = forecast['monthly'].tail(30).max() - forecast['monthly'].tail(30).min()
                components['monthly'] = {
                    'effect_magnitude': float(monthly_effect)
                }
            
            # Overall seasonality strength
            if 'weekly' in forecast.columns and 'monthly' in forecast.columns:
                total_seasonal = abs(forecast['weekly'].std()) + abs(forecast['monthly'].std())
                components['total_seasonal_strength'] = float(total_seasonal)
            
            return components
            
        except Exception as e:
            logger.warning(f"Failed to extract seasonal components: {e}")
            return {'seasonal_strength': 'unknown'}
    
    def _classify_volatility_regime(self, predicted_vol: float) -> str:
        """Classify predicted volatility into market regime"""
        
        if predicted_vol <= self.vol_regimes['low_vol']:
            return 'low_vol'
        elif predicted_vol <= self.vol_regimes['medium_vol']:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    def _calculate_forecast_accuracy(self, symbol: str, forecast: pd.DataFrame, actual: pd.DataFrame) -> float:
        """Calculate forecast accuracy using historical performance"""
        
        try:
            if symbol not in self.forecast_history:
                return 0.7  # Default accuracy
            
            # Compare recent forecasts with actual values
            history = self.forecast_history[symbol]
            if len(history) < 5:
                return 0.7
            
            # Calculate MAE for recent forecasts
            errors = []
            for h in history[-10:]:  # Last 10 forecasts
                forecast_date = datetime.fromisoformat(h['timestamp'])
                actual_vol = actual[actual.index >= forecast_date]['y'].mean()
                if not pd.isna(actual_vol):
                    error = abs(h['predicted_volatility'] - actual_vol)
                    errors.append(error)
            
            if errors:
                mae = np.mean(errors)
                # Convert MAE to accuracy score (0-1)
                accuracy = max(0.1, 1 - (mae / 50.0))  # Normalize by 50% volatility
                return min(1.0, accuracy)
            
            return 0.7
            
        except Exception:
            return 0.7
    
    def _identify_volatility_drivers(self, forecast: pd.DataFrame, historical_data: pd.DataFrame) -> List[str]:
        """Identify key drivers of volatility forecast"""
        
        drivers = []
        
        try:
            # Trend component
            trend_change = forecast['trend'].iloc[-1] - forecast['trend'].iloc[-7]
            if abs(trend_change) > 2:
                drivers.append(f"Strong trend component ({trend_change:+.1f}%)")
            
            # Seasonal effects
            if 'weekly' in forecast.columns:
                weekly_effect = forecast['weekly'].tail(7).max() - forecast['weekly'].tail(7).min()
                if weekly_effect > 3:
                    drivers.append("Significant weekly seasonality")
            
            # Market regime
            current_vol = forecast['yhat'].iloc[-1]
            if current_vol > 40:
                drivers.append("High volatility regime")
            elif current_vol < 15:
                drivers.append("Low volatility regime")
            
            # Volume impact (if available)
            if 'volume' in historical_data.columns:
                recent_volume = historical_data['volume'].tail(5).mean()
                avg_volume = historical_data['volume'].mean()
                if recent_volume > avg_volume * 1.5:
                    drivers.append("Elevated trading volume")
            
            # Price momentum
            if 'close' in historical_data.columns:
                price_change = (historical_data['close'].iloc[-1] / historical_data['close'].iloc[-20] - 1) * 100
                if abs(price_change) > 10:
                    drivers.append(f"Strong price momentum ({price_change:+.1f}%)")
            
        except Exception as e:
            logger.warning(f"Failed to identify drivers: {e}")
            drivers.append("Multiple market factors")
        
        return drivers[:5]  # Return top 5 drivers
    
    def _store_forecast_history(self, symbol: str, forecast: VolatilityForecast) -> None:
        """Store forecast in history for accuracy tracking"""
        
        try:
            if symbol not in self.forecast_history:
                self.forecast_history[symbol] = []
            
            forecast_record = {
                'timestamp': forecast.timestamp.isoformat(),
                'predicted_volatility': forecast.predicted_volatility,
                'forecast_horizon': forecast.forecast_horizon_days,
                'market_regime': forecast.market_regime,
                'trend': forecast.volatility_trend
            }
            
            self.forecast_history[symbol].append(forecast_record)
            
            # Keep only last 50 forecasts per symbol
            if len(self.forecast_history[symbol]) > 50:
                self.forecast_history[symbol] = self.forecast_history[symbol][-50:]
                
        except Exception as e:
            logger.warning(f"Failed to store forecast history: {e}")
    
    def _statistical_prediction(self, symbol: str, df: pd.DataFrame, horizon_days: int) -> VolatilityForecast:
        """Statistical fallback when Prophet is not available or data is insufficient"""
        
        try:
            # Calculate historical volatility
            returns = df['close'].pct_change().dropna()
            
            # Different window sizes for analysis
            vol_5d = returns.tail(5).std() * np.sqrt(252) * 100
            vol_20d = returns.tail(20).std() * np.sqrt(252) * 100
            vol_60d = returns.tail(60).std() * np.sqrt(252) * 100
            
            # Weighted average with more weight on recent data
            current_vol = (vol_5d * 0.5 + vol_20d * 0.3 + vol_60d * 0.2)
            
            # Simple trend analysis
            vol_series = returns.rolling(window=20).std() * np.sqrt(252) * 100
            recent_trend = vol_series.tail(5).mean() - vol_series.tail(10).head(5).mean()
            
            # Project forward with mean reversion
            predicted_vol = current_vol + (recent_trend * 0.3)  # Reduced trend impact
            
            # Mean reversion toward long-term average
            long_term_vol = vol_60d
            predicted_vol = predicted_vol * 0.7 + long_term_vol * 0.3
            
            # Determine trend
            if recent_trend > 2:
                trend = 'increasing'
            elif recent_trend < -2:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return VolatilityForecast(
                symbol=symbol,
                forecast_horizon_days=horizon_days,
                current_volatility=float(current_vol),
                predicted_volatility=float(predicted_vol),
                volatility_trend=trend,
                confidence_intervals={
                    'lower_80': float(predicted_vol * 0.8),
                    'upper_80': float(predicted_vol * 1.2),
                    'lower_95': float(predicted_vol * 0.7),
                    'upper_95': float(predicted_vol * 1.3)
                },
                seasonal_components={'method': 'statistical'},
                market_regime=self._classify_volatility_regime(predicted_vol),
                forecast_accuracy=0.6,  # Lower accuracy for statistical method
                key_drivers=[f"{horizon_days}-day statistical projection"],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Statistical prediction failed: {e}")
            return self._fallback_prediction(symbol, df, horizon_days)
    
    def _fallback_prediction(self, symbol: str, df: pd.DataFrame, horizon_days: int) -> VolatilityForecast:
        """Ultimate fallback prediction"""
        
        try:
            # Very simple volatility estimate
            if 'close' in df.columns and len(df) > 1:
                returns = df['close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 25.0
            else:
                vol = 25.0  # Default volatility
            
            return VolatilityForecast(
                symbol=symbol,
                forecast_horizon_days=horizon_days,
                current_volatility=vol,
                predicted_volatility=vol,
                volatility_trend='stable',
                confidence_intervals={
                    'lower_80': vol * 0.8,
                    'upper_80': vol * 1.2,
                    'lower_95': vol * 0.7,
                    'upper_95': vol * 1.3
                },
                seasonal_components={'method': 'fallback'},
                market_regime=self._classify_volatility_regime(vol),
                forecast_accuracy=0.5,
                key_drivers=['fallback_method'],
                timestamp=datetime.now()
            )
            
        except Exception:
            # Absolute fallback
            return VolatilityForecast(
                symbol=symbol,
                forecast_horizon_days=horizon_days,
                current_volatility=25.0,
                predicted_volatility=25.0,
                volatility_trend='stable',
                confidence_intervals={'lower_80': 20.0, 'upper_80': 30.0, 'lower_95': 17.5, 'upper_95': 32.5},
                seasonal_components={'method': 'default'},
                market_regime='medium_vol',
                forecast_accuracy=0.5,
                key_drivers=['default_values'],
                timestamp=datetime.now()
            )
    
    async def batch_predict(self, symbols: List[str], market_data: Dict[str, pd.DataFrame]) -> Dict[str, VolatilityForecast]:
        """Predict volatility for multiple symbols"""
        
        try:
            tasks = []
            for symbol in symbols:
                if symbol in market_data:
                    task = asyncio.create_task(
                        self.predict_volatility(symbol, market_data[symbol]),
                        name=f"vol_forecast_{symbol}"
                    )
                    tasks.append((symbol, task))
            
            results = {}
            for symbol, task in tasks:
                try:
                    result = await task
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Batch prediction failed for {symbol}: {e}")
                    results[symbol] = self._fallback_prediction(symbol, pd.DataFrame(), 30)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of volatility prediction models"""
        
        return {
            'prophet_available': PROPHET_AVAILABLE,
            'models_trained': len(self.models),
            'symbols_tracked': list(self.models.keys()),
            'forecast_history_count': sum(len(h) for h in self.forecast_history.values()),
            'volatility_regimes': self.vol_regimes,
            'parameters': self.prophet_params,
            'last_updated': datetime.now().isoformat()
        }


# Global instance
prophet_volatility_predictor = ProphetVolatilityPredictor()