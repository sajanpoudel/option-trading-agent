"""
Decision Engine Implementation
Core decision engine implementing the flowchart logic with dynamic weight assignment
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from backend.app.agents.orchestrator import OptionsOracleOrchestrator
from backend.app.ml.ensemble import EnsembleDecisionModel


@dataclass
class ScenarioWeights:
    """Dynamic weight configuration for different market scenarios"""
    technical: float
    sentiment: float
    flow: float
    history: float


@dataclass
class StrikeRecommendation:
    """Strike recommendation with risk metrics"""
    strike: float
    option_type: str  # 'call' or 'put'
    expiration: str
    delta: float
    gamma: float
    theta: float
    vega: float
    risk_score: float
    potential_return: float
    max_loss: float
    breakeven: float


class ScenarioDetector:
    """Detect market scenarios for dynamic weight adjustment"""
    
    def __init__(self):
        self.scenario_thresholds = {
            'high_volatility': {'iv_percentile': 80, 'bb_width': 0.05},
            'low_volatility': {'iv_percentile': 20, 'bb_width': 0.02},
            'earnings_approaching': {'days_to_earnings': 7},
            'strong_trend': {'adx': 25, 'rsi_extreme': 70},
            'range_bound': {'adx': 20, 'bb_squeeze': True}
        }
    
    def detect(self, technical_results: Dict) -> str:
        """Detect current market scenario"""
        try:
            scenario = technical_results.get('scenario', 'normal')
            
            # Map technical scenarios to our decision scenarios
            scenario_mapping = {
                'strong_uptrend': 'strong_trend',
                'strong_downtrend': 'strong_trend', 
                'range_bound': 'range_bound',
                'breakout': 'high_volatility',
                'potential_reversal': 'normal'
            }
            
            return scenario_mapping.get(scenario, 'normal')
            
        except Exception as e:
            logger.error(f"Scenario detection failed: {e}")
            return 'normal'


class RiskBasedStrikeSelector:
    """Select option strikes based on user risk profile"""
    
    def __init__(self):
        self.risk_profiles = {
            'conservative': {
                'delta_range': (0.15, 0.35),
                'max_loss_pct': 0.02,
                'min_theta': -0.01
            },
            'moderate': {
                'delta_range': (0.25, 0.55), 
                'max_loss_pct': 0.05,
                'min_theta': -0.03
            },
            'aggressive': {
                'delta_range': (0.45, 0.85),
                'max_loss_pct': 0.10,
                'min_theta': -0.05
            }
        }
    
    def select_strikes(self, signal: Dict, symbol: str, 
                      user_profile: Dict) -> List[StrikeRecommendation]:
        """Select optimal strikes based on risk profile and signal"""
        
        try:
            risk_level = user_profile.get('risk_level', 'moderate')
            profile_params = self.risk_profiles[risk_level]
            
            recommendations = []
            
            # Generate sample recommendations based on signal
            if signal['direction'] in ['BUY', 'STRONG_BUY']:
                recommendations.extend(self._generate_call_recommendations(
                    symbol, profile_params, signal['score']
                ))
            elif signal['direction'] in ['SELL', 'STRONG_SELL']:
                recommendations.extend(self._generate_put_recommendations(
                    symbol, profile_params, abs(signal['score'])
                ))
            else:
                # Neutral strategies for HOLD
                recommendations.extend(self._generate_neutral_recommendations(
                    symbol, profile_params
                ))
            
            # Sort by risk-adjusted return
            recommendations.sort(key=lambda x: x.potential_return / max(x.risk_score, 0.1), reverse=True)
            
            return recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Strike selection failed: {e}")
            return []
    
    def _generate_call_recommendations(self, symbol: str, profile: Dict, score: float) -> List[StrikeRecommendation]:
        """Generate call option recommendations"""
        base_price = 150.0  # Example current price
        delta_min, delta_max = profile['delta_range']
        
        recommendations = []
        
        for i, delta_target in enumerate([delta_min + 0.1, (delta_min + delta_max) / 2, delta_max - 0.1]):
            strike = base_price * (1 + (0.5 - delta_target) * 0.1)  # Approximate strike from delta
            
            rec = StrikeRecommendation(
                strike=round(strike, 2),
                option_type='call',
                expiration='2024-01-19',  # Example expiration
                delta=delta_target,
                gamma=0.01,
                theta=-0.02,
                vega=0.15,
                risk_score=0.3 + i * 0.1,
                potential_return=score * 0.15 * (1 + i * 0.1),
                max_loss=profile['max_loss_pct'],
                breakeven=strike + 5.0
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_put_recommendations(self, symbol: str, profile: Dict, score: float) -> List[StrikeRecommendation]:
        """Generate put option recommendations"""
        base_price = 150.0
        delta_min, delta_max = profile['delta_range']
        
        recommendations = []
        
        for i, delta_target in enumerate([-delta_max, -(delta_min + delta_max) / 2, -delta_min]):
            strike = base_price * (1 - (0.5 + abs(delta_target)) * 0.1)
            
            rec = StrikeRecommendation(
                strike=round(strike, 2),
                option_type='put',
                expiration='2024-01-19',
                delta=delta_target,
                gamma=0.01,
                theta=-0.02,
                vega=0.15,
                risk_score=0.3 + i * 0.1,
                potential_return=score * 0.15 * (1 + i * 0.1),
                max_loss=profile['max_loss_pct'],
                breakeven=strike - 5.0
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_neutral_recommendations(self, symbol: str, profile: Dict) -> List[StrikeRecommendation]:
        """Generate neutral strategy recommendations"""
        base_price = 150.0
        
        # Iron condor example
        rec = StrikeRecommendation(
            strike=base_price,
            option_type='iron_condor',
            expiration='2024-01-19',
            delta=0.0,
            gamma=-0.005,
            theta=0.05,  # Positive theta for neutral strategies
            vega=-0.10,
            risk_score=0.2,
            potential_return=0.08,
            max_loss=profile['max_loss_pct'],
            breakeven=base_price
        )
        
        return [rec]


class DecisionEngine:
    """Core decision engine implementing the flowchart logic"""
    
    def __init__(self):
        # Base agent weights (60% technical, 10% sentiment, 10% flow, 20% history)
        self.base_weights = ScenarioWeights(
            technical=0.60,
            sentiment=0.10,
            flow=0.10,
            history=0.20
        )
        
        self.scenario_detector = ScenarioDetector()
        self.strike_selector = RiskBasedStrikeSelector()
        self.ensemble_model = EnsembleDecisionModel()
        
        logger.info("Decision Engine initialized")
    
    async def process_stock(self, symbol: str, user_risk_profile: Dict) -> Dict:
        """Main decision processing pipeline"""
        
        logger.info(f"Processing decision for {symbol}")
        
        try:
            # Step 1: Gather agent analyses using orchestrator
            orchestrator = OptionsOracleOrchestrator()
            agent_results = await orchestrator.analyze_stock(symbol, user_risk_profile)
            
            # Step 2: Detect market scenario and adjust weights
            scenario = self.scenario_detector.detect(agent_results.get('technical', {}))
            adjusted_weights = self._adjust_weights_for_scenario(scenario)
            
            # Step 3: Calculate weighted decision
            decision_score = self._calculate_weighted_decision(agent_results, adjusted_weights)
            
            # Step 4: Generate signal using ensemble model
            market_data = {'symbol': symbol, 'price': 150.0, 'timestamp': datetime.now()}
            
            ensemble_signal = await self.ensemble_model.generate_signal(
                symbol,
                market_data,
                sentiment_data=agent_results.get('sentiment', {}),
                options_data=agent_results.get('flow', {})
            )
            
            signal = self._generate_signal(decision_score, ensemble_signal, agent_results)
            
            # Step 5: Select risk-appropriate strikes
            strike_recommendations = self.strike_selector.select_strikes(
                signal, symbol, user_risk_profile
            )
            
            # Step 6: Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                agent_results, ensemble_signal, adjusted_weights
            )
            
            result = {
                'symbol': symbol,
                'scenario': scenario,
                'agent_results': agent_results,
                'adjusted_weights': adjusted_weights.__dict__,
                'decision_score': decision_score,
                'signal': signal,
                'strike_recommendations': [rec.__dict__ for rec in strike_recommendations],
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Decision processed for {symbol}: {signal['direction']} ({confidence:.3f} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Decision processing failed for {symbol}: {e}")
            return self._fallback_decision(symbol, user_risk_profile)
    
    def _adjust_weights_for_scenario(self, scenario: str) -> ScenarioWeights:
        """Dynamically adjust weights based on market scenario"""
        
        # Start with base weights
        weights = ScenarioWeights(
            technical=self.base_weights.technical,
            sentiment=self.base_weights.sentiment,
            flow=self.base_weights.flow,
            history=self.base_weights.history
        )
        
        # Scenario-specific adjustments
        if scenario == 'high_volatility':
            # Increase technical and flow weights
            weights.technical += 0.10
            weights.flow += 0.05
            weights.sentiment -= 0.10
            weights.history -= 0.05
            
        elif scenario == 'low_volatility':
            # Increase sentiment weight in calm markets
            weights.sentiment += 0.05
            weights.history += 0.05
            weights.technical -= 0.10
            
        elif scenario == 'earnings_approaching':
            # Emphasize options flow before earnings
            weights.flow += 0.15
            weights.technical -= 0.10
            weights.history -= 0.05
            
        elif scenario == 'strong_trend':
            # Emphasize technical in trending markets
            weights.technical += 0.10
            weights.sentiment -= 0.05
            weights.flow -= 0.05
        
        # Ensure weights sum to 1.0
        total = weights.technical + weights.sentiment + weights.flow + weights.history
        if total != 1.0:
            weights.technical /= total
            weights.sentiment /= total
            weights.flow /= total
            weights.history /= total
        
        return weights
    
    def _calculate_weighted_decision(self, agent_results: Dict, weights: ScenarioWeights) -> float:
        """Calculate final weighted decision score"""
        
        try:
            # Extract scores from agent results
            technical_score = agent_results.get('technical', {}).get('weighted_score', 0.0)
            sentiment_score = agent_results.get('sentiment', {}).get('overall_sentiment', 0.0)
            flow_score = agent_results.get('flow', {}).get('flow_sentiment', 0.0)
            history_score = agent_results.get('history', {}).get('pattern_strength', 0.0)
            
            # Normalize scores to [-1, 1] range
            scores = self._normalize_scores({
                'technical': float(technical_score),
                'sentiment': self._sentiment_to_numeric(sentiment_score),
                'flow': self._sentiment_to_numeric(flow_score),
                'history': float(history_score)
            })
            
            # Calculate weighted sum
            weighted_score = (
                scores['technical'] * weights.technical +
                scores['sentiment'] * weights.sentiment +
                scores['flow'] * weights.flow +
                scores['history'] * weights.history
            )
            
            return float(weighted_score)
            
        except Exception as e:
            logger.error(f"Weighted decision calculation failed: {e}")
            return 0.0
    
    def _sentiment_to_numeric(self, sentiment: Any) -> float:
        """Convert sentiment to numeric score"""
        if isinstance(sentiment, (int, float)):
            return float(sentiment)
        
        if isinstance(sentiment, str):
            sentiment_map = {
                'very_bearish': -1.0, 'bearish': -0.5, 'neutral': 0.0,
                'bullish': 0.5, 'very_bullish': 1.0
            }
            return sentiment_map.get(sentiment.lower(), 0.0)
        
        return 0.0
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [-1, 1] range"""
        normalized = {}
        
        for key, value in scores.items():
            # Clamp to [-1, 1] range
            normalized[key] = max(-1.0, min(1.0, float(value)))
        
        return normalized
    
    def _generate_signal(self, decision_score: float, ensemble_signal, agent_results: Dict) -> Dict:
        """Generate trading signal from decision score and ensemble"""
        
        # Use ensemble signal as primary, decision score as confirmation
        final_score = (decision_score + ensemble_signal.final_score) / 2
        
        # Signal thresholds
        if final_score >= 0.6:
            direction = 'STRONG_BUY'
            strategy_type = 'aggressive_bullish'
        elif final_score >= 0.3:
            direction = 'BUY'
            strategy_type = 'moderate_bullish'
        elif final_score <= -0.6:
            direction = 'STRONG_SELL'
            strategy_type = 'aggressive_bearish'
        elif final_score <= -0.3:
            direction = 'SELL'
            strategy_type = 'moderate_bearish'
        else:
            direction = 'HOLD'
            strategy_type = 'neutral'
        
        # Determine options strategy
        options_strategy = self._select_options_strategy(
            direction, agent_results.get('technical', {}).get('scenario', 'normal')
        )
        
        return {
            'direction': direction,
            'score': final_score,
            'strategy_type': strategy_type,
            'options_strategy': options_strategy,
            'reasoning': self._generate_reasoning(agent_results, final_score)
        }
    
    def _select_options_strategy(self, direction: str, scenario: str) -> str:
        """Select optimal options strategy based on direction and scenario"""
        
        if direction in ['STRONG_BUY', 'BUY']:
            if scenario in ['high_volatility', 'breakout']:
                return 'long_call_spread'
            else:
                return 'long_call'
                
        elif direction in ['STRONG_SELL', 'SELL']:
            if scenario in ['high_volatility', 'breakout']:
                return 'long_put_spread'
            else:
                return 'long_put'
        else:
            # Neutral strategies
            if scenario == 'range_bound':
                return 'iron_condor'
            else:
                return 'straddle'
    
    def _generate_reasoning(self, agent_results: Dict, score: float) -> str:
        """Generate human-readable reasoning for the decision"""
        
        technical = agent_results.get('technical', {})
        sentiment = agent_results.get('sentiment', {})
        
        reasoning = f"Decision score: {score:.3f}. "
        reasoning += f"Technical analysis shows {technical.get('scenario', 'normal')} conditions. "
        reasoning += f"Market sentiment is {sentiment.get('overall_sentiment', 'neutral')}. "
        
        if score > 0.3:
            reasoning += "Multiple indicators suggest upward momentum."
        elif score < -0.3:
            reasoning += "Multiple indicators suggest downward pressure."
        else:
            reasoning += "Mixed signals suggest neutral positioning."
        
        return reasoning
    
    def _calculate_overall_confidence(self, agent_results: Dict, ensemble_signal, weights: ScenarioWeights) -> float:
        """Calculate overall confidence in the decision"""
        
        try:
            # Base confidence from ensemble
            base_confidence = ensemble_signal.confidence
            
            # Adjust based on agent agreement
            technical_strength = abs(agent_results.get('technical', {}).get('weighted_score', 0.0))
            sentiment_strength = 0.5  # Placeholder
            
            # Weight the confidence
            weighted_confidence = (
                base_confidence * 0.6 +  # Ensemble weight
                technical_strength * weights.technical +
                sentiment_strength * weights.sentiment
            )
            
            return max(0.0, min(1.0, weighted_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Neutral confidence
    
    def _fallback_decision(self, symbol: str, user_risk_profile: Dict) -> Dict:
        """Fallback decision when processing fails"""
        
        return {
            'symbol': symbol,
            'scenario': 'unknown',
            'agent_results': {},
            'adjusted_weights': self.base_weights.__dict__,
            'decision_score': 0.0,
            'signal': {
                'direction': 'HOLD',
                'score': 0.0,
                'strategy_type': 'neutral',
                'options_strategy': 'cash',
                'reasoning': 'Insufficient data for decision'
            },
            'strike_recommendations': [],
            'confidence': 0.1,
            'timestamp': datetime.now().isoformat()
        }