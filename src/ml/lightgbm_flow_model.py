"""
LightGBM Options Flow Predictor
High-performance gradient boosting for options flow analysis and unusual activity detection
"""

import asyncio
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from config.settings import settings
from config.logging import get_data_logger

logger = get_data_logger()


@dataclass
class FlowPrediction:
    """Options flow prediction result"""
    flow_sentiment: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    unusual_activity_score: float
    put_call_bias: float
    volume_prediction: float
    key_indicators: List[str]
    risk_level: str
    timestamp: datetime


class LightGBMFlowPredictor:
    """Advanced options flow prediction using LightGBM"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
        
        # Model parameters optimized for options flow prediction
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,  # bullish, bearish, neutral
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Flow pattern thresholds
        self.thresholds = {
            'unusual_volume': 2.0,  # 2x normal volume
            'unusual_oi': 1.5,      # 1.5x normal OI
            'large_trade': 10000,   # Large trade threshold
            'skew_threshold': 0.3   # IV skew threshold
        }
        
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, using fallback predictor")
        else:
            logger.info("LightGBM Flow Predictor initialized")
    
    async def predict_flow(self, flow_data: Dict[str, Any]) -> FlowPrediction:
        """Predict options flow sentiment and unusual activity"""
        
        try:
            if not LIGHTGBM_AVAILABLE:
                return self._fallback_prediction(flow_data)
            
            # Extract and engineer features
            features = self._engineer_features(flow_data)
            
            if not self.is_trained:
                # Use rule-based prediction if model not trained
                return self._rule_based_prediction(features, flow_data)
            
            # Prepare features for prediction
            feature_vector = self._prepare_features(features)
            
            # Make prediction
            prediction_proba = self.model.predict(feature_vector.reshape(1, -1))
            prediction = np.argmax(prediction_proba)
            confidence = np.max(prediction_proba)
            
            # Map prediction to sentiment
            sentiment_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
            flow_sentiment = sentiment_map.get(prediction, 'neutral')
            
            # Calculate additional metrics
            unusual_score = self._calculate_unusual_activity_score(features)
            put_call_bias = features.get('put_call_ratio', 1.0)
            volume_pred = features.get('volume_ratio', 1.0)
            
            # Generate key indicators
            key_indicators = self._identify_key_indicators(features)
            
            # Determine risk level
            risk_level = self._assess_risk_level(unusual_score, confidence)
            
            result = FlowPrediction(
                flow_sentiment=flow_sentiment,
                confidence=float(confidence),
                unusual_activity_score=unusual_score,
                put_call_bias=put_call_bias,
                volume_prediction=volume_pred,
                key_indicators=key_indicators,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            logger.info(f"Flow prediction: {flow_sentiment} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Flow prediction failed: {e}")
            return self._fallback_prediction(flow_data)
    
    def _engineer_features(self, flow_data: Dict[str, Any]) -> Dict[str, float]:
        """Engineer features from raw options flow data"""
        
        features = {}
        
        try:
            # Basic flow metrics
            features['put_call_ratio'] = flow_data.get('put_call_ratio', 1.0)
            features['total_volume'] = flow_data.get('total_volume', 0)
            features['total_open_interest'] = flow_data.get('total_open_interest', 0)
            
            # Volume metrics
            call_volume = flow_data.get('call_volume', 0)
            put_volume = flow_data.get('put_volume', 0)
            total_vol = call_volume + put_volume
            
            if total_vol > 0:
                features['call_volume_pct'] = call_volume / total_vol
                features['put_volume_pct'] = put_volume / total_vol
            else:
                features['call_volume_pct'] = 0.5
                features['put_volume_pct'] = 0.5
            
            # Volume ratios compared to historical averages
            avg_volume = flow_data.get('avg_volume', total_vol)
            features['volume_ratio'] = total_vol / max(avg_volume, 1) if avg_volume > 0 else 1.0
            
            # Open Interest metrics
            call_oi = flow_data.get('call_open_interest', 0)
            put_oi = flow_data.get('put_open_interest', 0)
            total_oi = call_oi + put_oi
            
            if total_oi > 0:
                features['call_oi_pct'] = call_oi / total_oi
                features['put_oi_pct'] = put_oi / total_oi
            else:
                features['call_oi_pct'] = 0.5
                features['put_oi_pct'] = 0.5
            
            # Implied Volatility features
            iv_data = flow_data.get('implied_volatility', {})
            features['iv_rank'] = iv_data.get('rank', 50)
            features['iv_percentile'] = iv_data.get('percentile', 50)
            features['iv_skew'] = iv_data.get('skew', 0)
            
            # Large trade detection
            large_trades = flow_data.get('large_trades', [])
            features['large_trade_count'] = len(large_trades)
            features['large_trade_volume'] = sum(trade.get('volume', 0) for trade in large_trades)
            
            # Time-based features
            now = datetime.now()
            features['hour_of_day'] = now.hour
            features['day_of_week'] = now.weekday()
            features['days_to_expiry'] = self._calculate_avg_days_to_expiry(flow_data)
            
            # Options chain depth
            strikes = flow_data.get('strikes', [])
            features['strike_count'] = len(strikes)
            features['itm_otm_ratio'] = self._calculate_itm_otm_ratio(strikes, flow_data.get('current_price', 100))
            
            # Greeks-based features
            greeks = flow_data.get('greeks', {})
            features['net_delta'] = greeks.get('delta', 0)
            features['net_gamma'] = greeks.get('gamma', 0)
            features['net_theta'] = greeks.get('theta', 0)
            features['net_vega'] = greeks.get('vega', 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return self._get_default_features()
    
    def _rule_based_prediction(self, features: Dict[str, float], flow_data: Dict) -> FlowPrediction:
        """Rule-based prediction when ML model is not available"""
        
        try:
            score = 0.0
            indicators = []
            
            # Put/Call ratio analysis
            pcr = features.get('put_call_ratio', 1.0)
            if pcr < 0.7:  # Low P/C ratio = bullish
                score += 0.3
                indicators.append("Low Put/Call ratio (bullish)")
            elif pcr > 1.3:  # High P/C ratio = bearish
                score -= 0.3
                indicators.append("High Put/Call ratio (bearish)")
            
            # Volume analysis
            volume_ratio = features.get('volume_ratio', 1.0)
            if volume_ratio > self.thresholds['unusual_volume']:
                score += 0.2 if features.get('call_volume_pct', 0.5) > 0.6 else -0.2
                indicators.append("Unusual volume activity")
            
            # IV analysis
            iv_rank = features.get('iv_rank', 50)
            if iv_rank > 80:
                score -= 0.1
                indicators.append("High IV rank")
            elif iv_rank < 20:
                score += 0.1
                indicators.append("Low IV rank")
            
            # Large trades
            large_trade_count = features.get('large_trade_count', 0)
            if large_trade_count > 0:
                call_pct = features.get('call_volume_pct', 0.5)
                score += 0.2 if call_pct > 0.6 else -0.2
                indicators.append(f"Large trades detected ({large_trade_count})")
            
            # Determine sentiment
            if score > 0.2:
                sentiment = 'bullish'
                confidence = min(0.8, abs(score))
            elif score < -0.2:
                sentiment = 'bearish'
                confidence = min(0.8, abs(score))
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            # Calculate unusual activity score
            unusual_score = self._calculate_unusual_activity_score(features)
            
            # Risk level
            risk_level = 'medium'
            if unusual_score > 0.7:
                risk_level = 'high'
            elif unusual_score < 0.3:
                risk_level = 'low'
            
            return FlowPrediction(
                flow_sentiment=sentiment,
                confidence=confidence,
                unusual_activity_score=unusual_score,
                put_call_bias=pcr,
                volume_prediction=volume_ratio,
                key_indicators=indicators[:5],
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Rule-based prediction failed: {e}")
            return self._fallback_prediction(flow_data)
    
    def _calculate_unusual_activity_score(self, features: Dict[str, float]) -> float:
        """Calculate unusual activity score from features"""
        
        score = 0.0
        
        # Volume anomaly
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > self.thresholds['unusual_volume']:
            score += min(0.4, (volume_ratio - 1.0) / 5.0)
        
        # Large trades
        large_trade_count = features.get('large_trade_count', 0)
        score += min(0.3, large_trade_count / 10.0)
        
        # IV rank extremes
        iv_rank = features.get('iv_rank', 50)
        if iv_rank > 90 or iv_rank < 10:
            score += 0.2
        
        # Put/Call extremes
        pcr = features.get('put_call_ratio', 1.0)
        if pcr > 2.0 or pcr < 0.3:
            score += 0.1
        
        return min(1.0, score)
    
    def _identify_key_indicators(self, features: Dict[str, float]) -> List[str]:
        """Identify most important indicators from features"""
        
        indicators = []
        
        # Check each feature category
        if features.get('volume_ratio', 1.0) > 1.5:
            indicators.append(f"High volume ({features['volume_ratio']:.1f}x normal)")
        
        if features.get('large_trade_count', 0) > 0:
            indicators.append(f"Large trades: {int(features['large_trade_count'])}")
        
        pcr = features.get('put_call_ratio', 1.0)
        if pcr > 1.5:
            indicators.append(f"High P/C ratio ({pcr:.2f})")
        elif pcr < 0.7:
            indicators.append(f"Low P/C ratio ({pcr:.2f})")
        
        iv_rank = features.get('iv_rank', 50)
        if iv_rank > 80:
            indicators.append(f"High IV rank ({iv_rank:.0f}%)")
        elif iv_rank < 20:
            indicators.append(f"Low IV rank ({iv_rank:.0f}%)")
        
        return indicators[:5]
    
    def _assess_risk_level(self, unusual_score: float, confidence: float) -> str:
        """Assess overall risk level"""
        
        combined_score = (unusual_score + (1 - confidence)) / 2
        
        if combined_score > 0.7:
            return 'high'
        elif combined_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_avg_days_to_expiry(self, flow_data: Dict) -> float:
        """Calculate average days to expiry for the options chain"""
        
        try:
            expirations = flow_data.get('expirations', [])
            if not expirations:
                return 30.0  # Default
            
            now = datetime.now()
            days_list = []
            
            for exp in expirations:
                exp_date = exp.get('date')
                if isinstance(exp_date, str):
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    days = (exp_datetime - now).days
                    days_list.append(max(0, days))
            
            return np.mean(days_list) if days_list else 30.0
            
        except Exception:
            return 30.0
    
    def _calculate_itm_otm_ratio(self, strikes: List, current_price: float) -> float:
        """Calculate in-the-money to out-of-the-money ratio"""
        
        if not strikes:
            return 1.0
        
        itm_count = 0
        otm_count = 0
        
        for strike in strikes:
            strike_price = strike.get('strike', current_price)
            option_type = strike.get('type', 'call').lower()
            
            if option_type == 'call':
                if strike_price < current_price:
                    itm_count += 1
                else:
                    otm_count += 1
            else:  # put
                if strike_price > current_price:
                    itm_count += 1
                else:
                    otm_count += 1
        
        return itm_count / max(otm_count, 1)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when feature engineering fails"""
        
        return {
            'put_call_ratio': 1.0,
            'call_volume_pct': 0.5,
            'put_volume_pct': 0.5,
            'volume_ratio': 1.0,
            'iv_rank': 50.0,
            'large_trade_count': 0.0,
            'hour_of_day': 12.0,
            'day_of_week': 2.0,
            'days_to_expiry': 30.0
        }
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model input"""
        
        if not self.feature_names:
            self.feature_names = list(features.keys())
        
        # Ensure all expected features are present
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(features.get(name, 0.0))
        
        return np.array(feature_vector)
    
    def _fallback_prediction(self, flow_data: Dict) -> FlowPrediction:
        """Fallback prediction when all else fails"""
        
        return FlowPrediction(
            flow_sentiment='neutral',
            confidence=0.3,
            unusual_activity_score=0.0,
            put_call_bias=1.0,
            volume_prediction=1.0,
            key_indicators=['fallback_analysis'],
            risk_level='low',
            timestamp=datetime.now()
        )
    
    async def train_model(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the LightGBM model with historical options flow data"""
        
        if not LIGHTGBM_AVAILABLE:
            logger.warning("Cannot train LightGBM model - library not available")
            return False
        
        try:
            logger.info(f"Training LightGBM model with {len(training_data)} samples")
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if len(X) < 10:
                logger.warning("Insufficient training data")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            # Train model
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                valid_sets=[train_data, valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10)]
            )
            
            # Evaluate model
            y_pred = np.argmax(self.model.predict(X_test_scaled), axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'samples': len(training_data)
            })
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model"""
        
        X = []
        y = []
        
        for sample in training_data:
            features = self._engineer_features(sample['flow_data'])
            feature_vector = self._prepare_features(features)
            
            # Convert sentiment to numerical label
            sentiment = sample['label'].lower()
            if sentiment == 'bullish':
                label = 2
            elif sentiment == 'bearish':
                label = 0
            else:
                label = 1
            
            X.append(feature_vector)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance metrics"""
        
        return {
            'model_type': 'LightGBM' if LIGHTGBM_AVAILABLE else 'Rule-based',
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_history': self.training_history[-5:],  # Last 5 training sessions
            'parameters': self.lgb_params if LIGHTGBM_AVAILABLE else None,
            'thresholds': self.thresholds
        }


# Global instance
lightgbm_flow_predictor = LightGBMFlowPredictor()