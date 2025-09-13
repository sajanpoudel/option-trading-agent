"""
Ensemble Decision Model
Combines OpenAI sentiment, LightGBM flow prediction, Prophet volatility, and market data
for comprehensive trading signal generation
"""

import asyncio
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .openai_sentiment_model import openai_sentiment, OpenAISentimentAnalyzer
from .lightgbm_flow_model import lightgbm_flow_predictor, LightGBMFlowPredictor
from .prophet_volatility_model import prophet_volatility_predictor, ProphetVolatilityPredictor

from config.settings import settings
from config.logging import get_data_logger

logger = get_data_logger()


@dataclass
class EnsembleSignal:
    """Comprehensive ensemble trading signal"""
    symbol: str
    final_score: float  # -1 to 1 scale
    confidence: float   # 0 to 1 scale
    direction: str      # 'BUY', 'SELL', 'HOLD'
    strength: str       # 'strong', 'moderate', 'weak'
    component_scores: Dict[str, float]
    component_weights: Dict[str, float]
    market_regime: str
    volatility_forecast: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    key_insights: List[str]
    recommended_strategies: List[str]
    timestamp: datetime


class EnsembleDecisionModel:
    """Advanced ensemble model combining multiple ML approaches"""
    
    def __init__(self):
        # Initialize component models
        self.sentiment_analyzer = openai_sentiment
        self.flow_predictor = lightgbm_flow_predictor
        self.volatility_predictor = prophet_volatility_predictor
        
        # Base weights for different components
        self.base_weights = {
            'sentiment': 0.20,
            'flow': 0.30,
            'volatility': 0.25,
            'technical': 0.25
        }
        
        # Dynamic weight adjustments based on market conditions
        self.weight_adjustments = {
            'high_vol': {
                'sentiment': -0.05, 'flow': 0.05, 'volatility': 0.05, 'technical': -0.05
            },
            'low_vol': {
                'sentiment': 0.05, 'flow': -0.05, 'volatility': -0.05, 'technical': 0.05
            },
            'earnings_week': {
                'sentiment': 0.10, 'flow': 0.10, 'volatility': 0.05, 'technical': -0.25
            },
            'fomc_week': {
                'sentiment': 0.05, 'flow': 0.05, 'volatility': 0.15, 'technical': -0.25
            }
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.75,
            'medium': 0.50,
            'low': 0.25
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            'strong_buy': 0.65,
            'buy': 0.30,
            'sell': -0.30,
            'strong_sell': -0.65
        }
        
        logger.info("Ensemble Decision Model initialized")
    
    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None,
        options_data: Optional[Dict[str, Any]] = None
    ) -> EnsembleSignal:
        """Generate comprehensive trading signal using ensemble approach"""
        
        try:
            logger.info(f"Generating ensemble signal for {symbol}")
            start_time = datetime.now()
            
            # Detect market regime for dynamic weighting
            market_regime = await self._detect_market_regime(symbol, market_data)
            
            # Get component predictions in parallel
            component_tasks = [
                self._get_sentiment_score(sentiment_data or {}),
                self._get_flow_score(options_data or {}),
                self._get_volatility_score(symbol, market_data),
                self._get_technical_score(market_data)
            ]
            
            sentiment_score, flow_score, volatility_score, technical_score = await asyncio.gather(
                *component_tasks, return_exceptions=True
            )
            
            # Handle any exceptions in component predictions
            component_scores = {
                'sentiment': self._safe_score(sentiment_score, 0.0),
                'flow': self._safe_score(flow_score, 0.0),
                'volatility': self._safe_score(volatility_score, 0.0),
                'technical': self._safe_score(technical_score, 0.0)
            }
            
            # Adjust weights based on market regime
            adjusted_weights = self._adjust_weights_for_regime(market_regime)
            
            # Calculate weighted ensemble score
            final_score = self._calculate_ensemble_score(component_scores, adjusted_weights)
            
            # Calculate overall confidence
            confidence = self._calculate_ensemble_confidence(
                component_scores, adjusted_weights, market_regime
            )
            
            # Generate trading direction and strength
            direction, strength = self._determine_direction_and_strength(final_score, confidence)
            
            # Get volatility forecast details
            volatility_forecast = await self._get_detailed_volatility_forecast(symbol, market_data)
            
            # Assess overall risk
            risk_assessment = self._assess_ensemble_risk(
                component_scores, volatility_forecast, market_regime
            )
            
            # Generate key insights
            key_insights = self._generate_insights(
                component_scores, adjusted_weights, market_regime, volatility_forecast
            )
            
            # Recommend strategies
            recommended_strategies = self._recommend_strategies(
                direction, strength, volatility_forecast, risk_assessment
            )
            
            # Create ensemble signal
            signal = EnsembleSignal(
                symbol=symbol,
                final_score=final_score,
                confidence=confidence,
                direction=direction,
                strength=strength,
                component_scores=component_scores,
                component_weights=adjusted_weights,
                market_regime=market_regime,
                volatility_forecast=volatility_forecast,
                risk_assessment=risk_assessment,
                key_insights=key_insights,
                recommended_strategies=recommended_strategies,
                timestamp=datetime.now()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Ensemble signal generated for {symbol}: {direction} ({confidence:.2f} confidence) in {processing_time:.2f}s")
            
            return signal
            
        except Exception as e:
            logger.error(f"Ensemble signal generation failed for {symbol}: {e}")
            return self._fallback_signal(symbol)
    
    async def _detect_market_regime(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Detect current market regime for dynamic weight adjustment"""
        
        try:
            # Check VIX level if available
            vix_level = market_data.get('vix', {}).get('price', 20)
            
            # Check current volatility
            technical = market_data.get('technical', {})
            current_vol = technical.get('volatility', 25)
            
            # Check market trend
            trend_strength = technical.get('trend_strength', 0)
            
            # Determine regime
            if vix_level > 30 or current_vol > 40:
                regime = 'high_vol'
            elif vix_level < 15 or current_vol < 15:
                regime = 'low_vol'
            elif abs(trend_strength) > 0.7:
                regime = 'trending'
            else:
                regime = 'normal'
            
            # Check for special events (simplified)
            current_date = datetime.now()
            
            # FOMC week (simplified - would use real calendar)
            if current_date.month in [3, 6, 9, 12] and 15 <= current_date.day <= 21:
                regime = 'fomc_week'
            
            # Earnings season (simplified)
            if current_date.month in [1, 4, 7, 10]:
                regime = 'earnings_season'
            
            return regime
            
        except Exception as e:
            logger.warning(f"Market regime detection failed: {e}")
            return 'normal'
    
    async def _get_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Get sentiment score from OpenAI analyzer"""
        
        try:
            if not sentiment_data:
                return 0.0
            
            # Combine different sentiment sources
            text_items = []
            
            # News sentiment
            if 'news' in sentiment_data:
                for article in sentiment_data['news'][:5]:  # Top 5 articles
                    text_items.append({
                        'text': article.get('title', '') + ' ' + article.get('summary', ''),
                        'context': 'financial_news',
                        'source': 'news'
                    })
            
            # Social sentiment
            if 'social' in sentiment_data:
                for post in sentiment_data['social'][:10]:  # Top 10 posts
                    text_items.append({
                        'text': post.get('text', ''),
                        'context': 'social_media',
                        'source': 'social'
                    })
            
            if not text_items:
                return 0.0
            
            # Analyze sentiments
            sentiment_results = await self.sentiment_analyzer.analyze_multiple_texts(text_items)
            
            # Aggregate with source weights
            source_weights = {'news': 0.6, 'social': 0.4}
            aggregate_result = await self.sentiment_analyzer.aggregate_sentiments(
                sentiment_results, source_weights
            )
            
            return aggregate_result.score
            
        except Exception as e:
            logger.warning(f"Sentiment score calculation failed: {e}")
            return 0.0
    
    async def _get_flow_score(self, options_data: Dict[str, Any]) -> float:
        """Get flow score from LightGBM predictor"""
        
        try:
            if not options_data:
                return 0.0
            
            flow_prediction = await self.flow_predictor.predict_flow(options_data)
            
            # Convert categorical prediction to score
            sentiment_scores = {
                'bullish': 0.7,
                'bearish': -0.7,
                'neutral': 0.0
            }
            
            base_score = sentiment_scores.get(flow_prediction.flow_sentiment, 0.0)
            
            # Adjust by confidence
            adjusted_score = base_score * flow_prediction.confidence
            
            # Boost for unusual activity
            if flow_prediction.unusual_activity_score > 0.5:
                adjusted_score *= (1 + flow_prediction.unusual_activity_score * 0.3)
            
            return np.clip(adjusted_score, -1.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Flow score calculation failed: {e}")
            return 0.0
    
    async def _get_volatility_score(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Get volatility-based score"""
        
        try:
            # Get historical price data
            if 'price_history' not in market_data:
                return 0.0
            
            price_df = pd.DataFrame(market_data['price_history'])
            
            # Get volatility forecast
            vol_forecast = await self.volatility_predictor.predict_volatility(
                symbol, price_df, horizon_days=30
            )
            
            # Convert volatility trend to score
            trend_scores = {
                'increasing': -0.3,  # Increasing vol = bearish
                'decreasing': 0.3,   # Decreasing vol = bullish
                'stable': 0.0
            }
            
            trend_score = trend_scores.get(vol_forecast.volatility_trend, 0.0)
            
            # Adjust for regime
            if vol_forecast.market_regime == 'high_vol':
                trend_score *= 0.7  # Reduce impact in high vol
            elif vol_forecast.market_regime == 'low_vol':
                trend_score *= 1.3  # Increase impact in low vol
            
            # Adjust by forecast accuracy
            return trend_score * vol_forecast.forecast_accuracy
            
        except Exception as e:
            logger.warning(f"Volatility score calculation failed: {e}")
            return 0.0
    
    async def _get_technical_score(self, market_data: Dict[str, Any]) -> float:
        """Get technical analysis score from market data"""
        
        try:
            technical = market_data.get('technical', {})
            
            if not technical:
                return 0.0
            
            # Combine multiple technical indicators
            rsi = technical.get('rsi', 50)
            macd = technical.get('macd', 0)
            bb_position = technical.get('bollinger_position', 0.5)
            trend_strength = technical.get('trend_strength', 0)
            
            # RSI score (-1 to 1)
            rsi_score = 0.0
            if rsi > 70:
                rsi_score = -0.5  # Overbought
            elif rsi < 30:
                rsi_score = 0.5   # Oversold
            else:
                rsi_score = (50 - rsi) / 40  # Normalized around 50
            
            # MACD score
            macd_score = np.clip(macd / 5.0, -0.5, 0.5)  # Normalize MACD
            
            # Bollinger Bands score
            bb_score = (bb_position - 0.5) * 0.8  # Center around 0.5
            
            # Trend score
            trend_score = np.clip(trend_strength, -0.5, 0.5)
            
            # Weighted combination
            technical_score = (
                rsi_score * 0.30 +
                macd_score * 0.25 +
                bb_score * 0.20 +
                trend_score * 0.25
            )
            
            return np.clip(technical_score, -1.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Technical score calculation failed: {e}")
            return 0.0
    
    def _safe_score(self, score_result: Any, default: float) -> float:
        """Safely extract score from result, handling exceptions"""
        
        if isinstance(score_result, Exception):
            logger.warning(f"Component prediction failed: {score_result}")
            return default
        
        if isinstance(score_result, (int, float)):
            return float(np.clip(score_result, -1.0, 1.0))
        
        return default
    
    def _adjust_weights_for_regime(self, market_regime: str) -> Dict[str, float]:
        """Adjust component weights based on market regime"""
        
        weights = self.base_weights.copy()
        
        # Apply regime-specific adjustments
        if market_regime in self.weight_adjustments:
            adjustments = self.weight_adjustments[market_regime]
            for component, adjustment in adjustments.items():
                weights[component] += adjustment
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_ensemble_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted ensemble score"""
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            if component in weights:
                weight = weights[component]
                weighted_score += score * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_score /= total_weight
        
        return np.clip(weighted_score, -1.0, 1.0)
    
    def _calculate_ensemble_confidence(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
        market_regime: str
    ) -> float:
        """Calculate overall ensemble confidence"""
        
        # Base confidence from score magnitude
        score_magnitude = np.mean([abs(score) for score in scores.values()])
        magnitude_confidence = min(1.0, score_magnitude * 2)
        
        # Agreement between components
        score_values = list(scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)
            agreement_confidence = max(0.1, 1.0 - (score_std / 2.0))
        else:
            agreement_confidence = 0.5
        
        # Market regime adjustment
        regime_multipliers = {
            'high_vol': 0.8,    # Lower confidence in high vol
            'low_vol': 1.1,     # Higher confidence in low vol
            'trending': 1.0,
            'normal': 1.0,
            'fomc_week': 0.7,   # Lower confidence during FOMC
            'earnings_season': 0.8
        }
        
        regime_multiplier = regime_multipliers.get(market_regime, 1.0)
        
        # Combined confidence
        confidence = (magnitude_confidence * 0.4 + agreement_confidence * 0.6) * regime_multiplier
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _determine_direction_and_strength(self, score: float, confidence: float) -> Tuple[str, str]:
        """Determine trading direction and strength from score and confidence"""
        
        abs_score = abs(score)
        
        # Determine direction
        if score >= self.signal_thresholds['strong_buy']:
            direction = 'STRONG_BUY'
        elif score >= self.signal_thresholds['buy']:
            direction = 'BUY'
        elif score <= self.signal_thresholds['strong_sell']:
            direction = 'STRONG_SELL'
        elif score <= self.signal_thresholds['sell']:
            direction = 'SELL'
        else:
            direction = 'HOLD'
        
        # Determine strength based on confidence and score magnitude
        strength_score = (abs_score + confidence) / 2
        
        if strength_score >= self.confidence_thresholds['high']:
            strength = 'strong'
        elif strength_score >= self.confidence_thresholds['medium']:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        return direction, strength
    
    async def _get_detailed_volatility_forecast(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed volatility forecast information"""
        
        try:
            if 'price_history' not in market_data:
                return {'status': 'no_data'}
            
            price_df = pd.DataFrame(market_data['price_history'])
            vol_forecast = await self.volatility_predictor.predict_volatility(symbol, price_df)
            
            return {
                'current_volatility': vol_forecast.current_volatility,
                'predicted_volatility': vol_forecast.predicted_volatility,
                'trend': vol_forecast.volatility_trend,
                'regime': vol_forecast.market_regime,
                'accuracy': vol_forecast.forecast_accuracy,
                'confidence_bands': vol_forecast.confidence_intervals
            }
            
        except Exception as e:
            logger.warning(f"Detailed volatility forecast failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _assess_ensemble_risk(
        self,
        scores: Dict[str, float],
        volatility_forecast: Dict[str, Any],
        market_regime: str
    ) -> Dict[str, Any]:
        """Assess overall risk of the ensemble signal"""
        
        risk_factors = []
        risk_score = 0.0
        
        # Score disagreement risk
        score_values = list(scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)
            if score_std > 0.5:
                risk_factors.append("High disagreement between models")
                risk_score += 0.3
        
        # Volatility risk
        if volatility_forecast.get('regime') == 'high_vol':
            risk_factors.append("High volatility environment")
            risk_score += 0.2
        
        # Market regime risk
        risky_regimes = ['fomc_week', 'earnings_season', 'high_vol']
        if market_regime in risky_regimes:
            risk_factors.append(f"Risky market regime: {market_regime}")
            risk_score += 0.2
        
        # Data quality risk
        zero_scores = sum(1 for score in score_values if score == 0.0)
        if zero_scores > len(score_values) / 2:
            risk_factors.append("Low data quality or model failures")
            risk_score += 0.3
        
        # Overall risk level
        if risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors,
            'recommendation': 'Reduce position size' if risk_level == 'high' else 'Normal sizing'
        }
    
    def _generate_insights(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
        market_regime: str,
        volatility_forecast: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights from ensemble analysis"""
        
        insights = []
        
        # Dominant factor
        max_component = max(scores.items(), key=lambda x: abs(x[1]) * weights[x[0]])
        if abs(max_component[1]) > 0.3:
            direction = "bullish" if max_component[1] > 0 else "bearish"
            insights.append(f"Primary driver: {max_component[0]} analysis ({direction})")
        
        # Market regime insight
        if market_regime != 'normal':
            insights.append(f"Operating in {market_regime.replace('_', ' ')} market regime")
        
        # Volatility insight
        vol_trend = volatility_forecast.get('trend', 'stable')
        if vol_trend != 'stable':
            insights.append(f"Volatility expected to be {vol_trend}")
        
        # Score agreement
        score_values = list(scores.values())
        if len(score_values) > 1:
            agreement = 1.0 - (np.std(score_values) / 2.0)
            if agreement > 0.8:
                insights.append("High consensus across all models")
            elif agreement < 0.4:
                insights.append("Mixed signals from different analysis methods")
        
        # Zero scores warning
        zero_scores = [k for k, v in scores.items() if v == 0.0]
        if zero_scores:
            insights.append(f"Limited data for: {', '.join(zero_scores)}")
        
        return insights[:5]  # Return top 5 insights
    
    def _recommend_strategies(
        self,
        direction: str,
        strength: str,
        volatility_forecast: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Recommend options strategies based on signal and market conditions"""
        
        strategies = []
        vol_regime = volatility_forecast.get('regime', 'medium_vol')
        
        if direction in ['BUY', 'STRONG_BUY']:
            if vol_regime == 'low_vol':
                strategies.append("Long calls (low IV advantage)")
                strategies.append("Bull call spreads")
            elif vol_regime == 'high_vol':
                strategies.append("Cash-secured puts")
                strategies.append("Bull put spreads")
            else:
                strategies.append("Long calls")
                strategies.append("Call debit spreads")
                
        elif direction in ['SELL', 'STRONG_SELL']:
            if vol_regime == 'low_vol':
                strategies.append("Long puts")
                strategies.append("Bear call spreads")
            elif vol_regime == 'high_vol':
                strategies.append("Covered calls")
                strategies.append("Bear put spreads")
            else:
                strategies.append("Long puts")
                strategies.append("Put debit spreads")
                
        else:  # HOLD
            if vol_regime == 'high_vol':
                strategies.append("Iron condors")
                strategies.append("Short strangles")
            else:
                strategies.append("Iron butterflies")
                strategies.append("Calendar spreads")
        
        # Adjust for risk level
        if risk_assessment['risk_level'] == 'high':
            strategies = [s for s in strategies if 'spread' in s.lower() or 'covered' in s.lower()]
            strategies.append("Consider smaller position sizes")
        
        return strategies[:4]  # Return top 4 strategies
    
    def _fallback_signal(self, symbol: str) -> EnsembleSignal:
        """Generate fallback signal when ensemble fails"""
        
        return EnsembleSignal(
            symbol=symbol,
            final_score=0.0,
            confidence=0.3,
            direction='HOLD',
            strength='weak',
            component_scores={'sentiment': 0.0, 'flow': 0.0, 'volatility': 0.0, 'technical': 0.0},
            component_weights=self.base_weights,
            market_regime='unknown',
            volatility_forecast={'status': 'failed'},
            risk_assessment={'risk_level': 'high', 'risk_factors': ['Ensemble failure']},
            key_insights=['Ensemble model failed - using conservative approach'],
            recommended_strategies=['Hold position', 'Wait for clearer signals'],
            timestamp=datetime.now()
        )


# Global instance
ensemble_model = EnsembleDecisionModel()