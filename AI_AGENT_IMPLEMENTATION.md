# Neural Options Oracle++ - AI Agent Implementation

## Agent System Overview

The AI Agent system is the core intelligence layer of the Neural Options Oracle++, utilizing OpenAI's Agents SDK for orchestration with specialized agents powered by GPT-4o, GPT-4o-mini, and Gemini 2.0 Flash models.

## Agent Architecture

### Master Orchestrator

```python
# agents/orchestrator/master_orchestrator.py
import asyncio
from typing import Dict, List, Any
from openai import OpenAI
import google.generativeai as genai
from redis import Redis
from datetime import datetime, timedelta

from .weight_calculator import WeightCalculator
from .scenario_detector import ScenarioDetector
from ..agents.technical_agent import TechnicalAnalysisAgent
from ..agents.sentiment_agent import SentimentAnalysisAgent
from ..agents.flow_agent import OptionsFlowAgent
from ..agents.history_agent import HistoricalPatternAgent
from ..agents.risk_agent import RiskManagementAgent
from ..agents.education_agent import EducationAgent

class MasterOrchestrator:
    """
    Master orchestrator for coordinating multiple AI agents with dynamic weight assignment
    """
    
    def __init__(self, openai_api_key: str, gemini_api_key: str, redis_url: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        genai.configure(api_key=gemini_api_key)
        self.redis_client = Redis.from_url(redis_url)
        
        # Initialize weight calculator and scenario detector
        self.weight_calculator = WeightCalculator()
        self.scenario_detector = ScenarioDetector()
        
        # Initialize agents
        self.agents = {
            'technical': TechnicalAnalysisAgent(openai_api_key),
            'sentiment': SentimentAnalysisAgent(openai_api_key),
            'flow': OptionsFlowAgent(gemini_api_key),
            'history': HistoricalPatternAgent(openai_api_key),
            'risk': RiskManagementAgent(openai_api_key),
            'education': EducationAgent(openai_api_key)
        }
        
        # Base agent weights (as per flowchart)
        self.base_weights = {
            'technical': 0.60,
            'sentiment': 0.10,
            'flow': 0.10,
            'history': 0.20
        }
        
    async def orchestrate_analysis(self, symbol: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration method that coordinates all agents
        """
        analysis_id = f"analysis_{symbol}_{int(datetime.now().timestamp())}"
        
        try:
            # Step 1: Gather market data
            market_data = await self._gather_market_data(symbol)
            
            # Step 2: Run agents in parallel
            agent_tasks = {
                'technical': self.agents['technical'].analyze(symbol, market_data),
                'sentiment': self.agents['sentiment'].analyze(symbol, market_data),
                'flow': self.agents['flow'].analyze(symbol, market_data),
                'history': self.agents['history'].analyze(symbol, market_data)
            }
            
            agent_results = {}
            for agent_name, task in agent_tasks.items():
                try:
                    agent_results[agent_name] = await task
                except Exception as e:
                    print(f"Agent {agent_name} failed: {e}")
                    agent_results[agent_name] = self._get_fallback_result(agent_name)
            
            # Step 3: Detect market scenario
            scenario = self.scenario_detector.detect_scenario(agent_results['technical'])
            
            # Step 4: Calculate dynamic weights
            adjusted_weights = self.weight_calculator.calculate_weights(
                scenario, self.base_weights, market_data
            )
            
            # Step 5: Generate decision
            decision = await self._generate_decision(
                agent_results, adjusted_weights, scenario
            )
            
            # Step 6: Get strike recommendations
            strike_recommendations = await self.agents['risk'].recommend_strikes(
                decision, symbol, user_context
            )
            
            # Step 7: Generate educational content
            educational_content = await self.agents['education'].generate_content(
                decision, agent_results, user_context
            )
            
            # Step 8: Cache results
            result = {
                'analysis_id': analysis_id,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_scenario': scenario,
                'agent_results': agent_results,
                'adjusted_weights': adjusted_weights,
                'decision': decision,
                'strike_recommendations': strike_recommendations,
                'educational_content': educational_content
            }
            
            await self._cache_analysis(analysis_id, result)
            return result
            
        except Exception as e:
            return {
                'analysis_id': analysis_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """Gather market data from multiple sources"""
        # Implementation would fetch from Alpaca, StockTwits, etc.
        pass
    
    async def _generate_decision(self, agent_results: Dict, weights: Dict, scenario: str) -> Dict:
        """Generate final trading decision using weighted agent results"""
        
        # Calculate weighted scores
        weighted_score = 0
        total_confidence = 0
        
        for agent, result in agent_results.items():
            if agent in weights:
                agent_score = result.get('score', 0)
                agent_confidence = result.get('confidence', 0)
                weight = weights[agent]
                
                weighted_score += agent_score * weight
                total_confidence += agent_confidence * weight
        
        # Determine signal direction and strength
        signal_direction = self._determine_signal_direction(weighted_score)
        signal_strength = self._determine_signal_strength(abs(weighted_score), total_confidence)
        
        return {
            'weighted_score': weighted_score,
            'confidence': total_confidence,
            'signal': {
                'direction': signal_direction,
                'strength': signal_strength,
                'reasoning': self._generate_reasoning(agent_results, weighted_score)
            },
            'scenario': scenario,
            'weights_used': weights
        }
    
    def _determine_signal_direction(self, weighted_score: float) -> str:
        """Determine trading signal direction based on weighted score"""
        if weighted_score >= 0.6:
            return 'STRONG_BUY'
        elif weighted_score >= 0.3:
            return 'BUY'
        elif weighted_score <= -0.6:
            return 'STRONG_SELL'
        elif weighted_score <= -0.3:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _determine_signal_strength(self, abs_score: float, confidence: float) -> str:
        """Determine signal strength based on score magnitude and confidence"""
        strength_score = abs_score * confidence
        
        if strength_score >= 0.7:
            return 'strong'
        elif strength_score >= 0.4:
            return 'moderate'
        else:
            return 'weak'
```

## Individual Agent Implementations

### Technical Analysis Agent

```python
# agents/agents/technical_agent.py
from openai import OpenAI
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

class TechnicalAnalysisAgent:
    """
    Technical Analysis Agent using GPT-4o for market scenario detection 
    and dynamic indicator weighting
    """
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
        # Scenario-based indicator weights
        self.scenario_weights = {
            "strong_uptrend": {
                "ma": 0.30, "rsi": 0.15, "bb": 0.10, "macd": 0.25, "vwap": 0.20
            },
            "strong_downtrend": {
                "ma": 0.30, "rsi": 0.15, "bb": 0.10, "macd": 0.25, "vwap": 0.20
            },
            "range_bound": {
                "ma": 0.15, "rsi": 0.25, "bb": 0.30, "macd": 0.15, "vwap": 0.15
            },
            "breakout": {
                "ma": 0.20, "rsi": 0.15, "bb": 0.30, "macd": 0.20, "vwap": 0.15
            },
            "potential_reversal": {
                "ma": 0.15, "rsi": 0.25, "bb": 0.20, "macd": 0.30, "vwap": 0.10
            }
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis with dynamic weighting
        """
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(market_data)
            
            # Detect market scenario using AI
            scenario = await self._detect_scenario_ai(symbol, indicators, market_data)
            
            # Get dynamic weights for the scenario
            weights = self._get_scenario_weights(scenario, market_data)
            
            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(indicators, weights)
            
            # Calculate confidence
            confidence = self._calculate_confidence(indicators, scenario)
            
            # Generate detailed analysis using AI
            analysis = await self._generate_ai_analysis(symbol, indicators, scenario, weighted_score)
            
            return {
                'agent': 'technical',
                'symbol': symbol,
                'scenario': scenario,
                'indicators': indicators,
                'weights': weights,
                'weighted_score': weighted_score,
                'confidence': confidence,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'agent': 'technical',
                'error': str(e),
                'score': 0,
                'confidence': 0
            }
    
    def _calculate_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators from market data"""
        prices = pd.DataFrame(market_data['price_data'])
        
        indicators = {}
        
        # Moving Averages
        indicators['ma'] = {
            'ma_20': prices['close'].rolling(window=20).mean().iloc[-1],
            'ma_50': prices['close'].rolling(window=50).mean().iloc[-1],
            'signal': self._calculate_ma_signal(prices)
        }
        
        # RSI
        indicators['rsi'] = {
            'value': self._calculate_rsi(prices['close']),
            'signal': self._calculate_rsi_signal(self._calculate_rsi(prices['close']))
        }
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(prices['close'])
        indicators['bb'] = {
            'upper': bb_upper,
            'lower': bb_lower,
            'middle': bb_middle,
            'width': bb_upper - bb_lower,
            'signal': self._calculate_bb_signal(prices['close'].iloc[-1], bb_upper, bb_lower)
        }
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(prices['close'])
        indicators['macd'] = {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'signal': self._calculate_macd_signal(macd_line, signal_line, histogram)
        }
        
        # VWAP
        indicators['vwap'] = {
            'value': self._calculate_vwap(prices),
            'signal': self._calculate_vwap_signal(prices)
        }
        
        # ADX for trend strength
        indicators['adx'] = {
            'value': self._calculate_adx(prices),
            'trend_strength': self._interpret_adx(self._calculate_adx(prices))
        }
        
        return indicators
    
    async def _detect_scenario_ai(self, symbol: str, indicators: Dict, market_data: Dict) -> str:
        """Use AI to detect market scenario based on technical indicators"""
        
        prompt = f"""
        As a technical analysis expert, analyze the following technical indicators for {symbol} 
        and determine the current market scenario.
        
        Technical Indicators:
        - MA20: {indicators['ma']['ma_20']:.2f}
        - MA50: {indicators['ma']['ma_50']:.2f}
        - RSI: {indicators['rsi']['value']:.2f}
        - Bollinger Band Width: {indicators['bb']['width']:.2f}
        - MACD: {indicators['macd']['macd']:.4f}
        - MACD Signal: {indicators['macd']['signal']:.4f}
        - ADX: {indicators['adx']['value']:.2f}
        - Current Price: {market_data['current_price']:.2f}
        
        Based on these indicators, classify the market scenario as one of:
        1. strong_uptrend - Clear upward momentum with strong trend
        2. strong_downtrend - Clear downward momentum with strong trend  
        3. range_bound - Sideways movement, low volatility
        4. breakout - Price breaking out of consolidation
        5. potential_reversal - Signs of trend reversal
        
        Respond with only the scenario name and a brief 2-sentence explanation.
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content
        
        # Extract scenario from AI response
        for scenario in self.scenario_weights.keys():
            if scenario in ai_response.lower():
                return scenario
        
        # Fallback to rule-based detection
        return self._detect_scenario_rules(indicators)
    
    def _detect_scenario_rules(self, indicators: Dict) -> str:
        """Fallback rule-based scenario detection"""
        current_price = indicators.get('current_price', 0)
        ma_20 = indicators['ma']['ma_20']
        ma_50 = indicators['ma']['ma_50']
        adx = indicators['adx']['value']
        bb_width = indicators['bb']['width']
        
        # Strong trend detection
        if adx > 25 and current_price > ma_20 > ma_50:
            return "strong_uptrend"
        elif adx > 25 and current_price < ma_20 < ma_50:
            return "strong_downtrend"
        # Breakout detection
        elif bb_width > np.mean([bb_width]) * 1.5:  # Simplified for example
            return "breakout"
        # Range-bound detection
        elif bb_width < np.mean([bb_width]) * 0.7:  # Simplified for example
            return "range_bound"
        else:
            return "potential_reversal"
    
    def _get_scenario_weights(self, scenario: str, market_data: Dict) -> Dict[str, float]:
        """Get dynamic weights based on scenario with volatility adjustments"""
        
        base_weights = self.scenario_weights.get(scenario, self.scenario_weights["range_bound"])
        
        # Volatility adjustment
        volatility = self._calculate_volatility(market_data)
        if volatility > 0.3:  # High volatility
            # Increase Bollinger Bands weight
            base_weights = base_weights.copy()
            base_weights['bb'] += 0.10
            
            # Proportionally reduce other weights
            reduction_per_indicator = 0.10 / (len(base_weights) - 1)
            for key in base_weights:
                if key != 'bb':
                    base_weights[key] = max(0.05, base_weights[key] - reduction_per_indicator)
        
        return base_weights
    
    def _calculate_weighted_score(self, indicators: Dict, weights: Dict) -> float:
        """Calculate final weighted technical score"""
        total_score = 0
        
        for indicator, weight in weights.items():
            if indicator in indicators:
                signal = indicators[indicator].get('signal', 0)
                total_score += signal * weight
        
        return total_score
    
    def _calculate_confidence(self, indicators: Dict, scenario: str) -> float:
        """Calculate confidence score based on indicator alignment"""
        
        # Get all indicator signals
        signals = []
        for indicator_data in indicators.values():
            if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                signals.append(indicator_data['signal'])
        
        if not signals:
            return 0.5
        
        # Calculate signal alignment (how much they agree)
        mean_signal = np.mean(signals)
        std_signal = np.std(signals)
        
        # Higher confidence when signals are aligned (low std) and strong (high abs mean)
        alignment_score = 1 - (std_signal / (abs(mean_signal) + 1))
        strength_score = min(abs(mean_signal), 1.0)
        
        base_confidence = (alignment_score + strength_score) / 2
        
        # Scenario-specific confidence adjustments
        scenario_multipliers = {
            "strong_uptrend": 1.1,
            "strong_downtrend": 1.1,
            "range_bound": 0.9,
            "breakout": 1.0,
            "potential_reversal": 0.8
        }
        
        multiplier = scenario_multipliers.get(scenario, 1.0)
        return min(base_confidence * multiplier, 1.0)
    
    async def _generate_ai_analysis(self, symbol: str, indicators: Dict, scenario: str, score: float) -> str:
        """Generate detailed AI analysis"""
        
        prompt = f"""
        As a technical analysis expert, provide a concise analysis for {symbol} based on:
        
        Market Scenario: {scenario}
        Technical Score: {score:.2f}
        Key Indicators: {self._format_indicators_for_ai(indicators)}
        
        Provide a 3-sentence analysis explaining:
        1. The current technical setup
        2. Key support/resistance levels
        3. Short-term outlook and key risks
        
        Keep it educational and suitable for options traders.
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        return response.choices[0].message.content
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.iloc[-1], lower.iloc[-1], middle.iloc[-1]
    
    # Additional technical indicator methods would go here...
```

### Sentiment Analysis Agent

```python
# agents/agents/sentiment_agent.py
from openai import OpenAI
from typing import Dict, Any
import asyncio
from datetime import datetime

class SentimentAnalysisAgent:
    """
    Sentiment Analysis Agent using GPT-4o-mini for cost-effective 
    social media and news sentiment analysis
    """
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Cost-effective model for sentiment
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment from multiple sources
        """
        try:
            # Gather sentiment data from multiple sources in parallel
            sentiment_tasks = {
                'social': self._analyze_social_sentiment(symbol),
                'news': self._analyze_news_sentiment(symbol),
                'options_positioning': self._analyze_options_sentiment(symbol, market_data)
            }
            
            sentiment_data = {}
            for source, task in sentiment_tasks.items():
                try:
                    sentiment_data[source] = await task
                except Exception as e:
                    print(f"Sentiment source {source} failed: {e}")
                    sentiment_data[source] = {'score': 0, 'confidence': 0}
            
            # Aggregate sentiment scores
            aggregate_sentiment = self._aggregate_sentiment_scores(sentiment_data)
            
            # Calculate confidence
            confidence = self._calculate_sentiment_confidence(sentiment_data)
            
            # Generate AI explanation
            explanation = await self._generate_sentiment_explanation(symbol, sentiment_data, aggregate_sentiment)
            
            return {
                'agent': 'sentiment',
                'symbol': symbol,
                'sentiment_sources': sentiment_data,
                'aggregate_sentiment': aggregate_sentiment,
                'score': aggregate_sentiment,  # For compatibility with orchestrator
                'confidence': confidence,
                'explanation': explanation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'agent': 'sentiment',
                'error': str(e),
                'score': 0,
                'confidence': 0
            }
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze social media sentiment using AI"""
        
        # In a real implementation, this would fetch data from StockTwits, Reddit, etc.
        # For now, we'll simulate the analysis
        
        mock_social_data = f"""
        Recent social media posts about ${symbol}:
        
        StockTwits:
        - "Bullish on {symbol}, great earnings beat!" (Bullish)
        - "{symbol} looking strong, breaking resistance" (Bullish)
        - "Taking profits on {symbol}, seems overextended" (Neutral/Bearish)
        
        Reddit (WSB, investing):
        - Multiple discussions about {symbol} options activity
        - Generally positive sentiment around recent developments
        - Some concerns about valuation at current levels
        """
        
        prompt = f"""
        Analyze the sentiment of these social media posts about {symbol}:
        
        {mock_social_data}
        
        Provide:
        1. Overall sentiment score (-1.0 to +1.0)
        2. Confidence level (0.0 to 1.0)
        3. Key themes (bullish/bearish factors mentioned)
        
        Format as: SCORE|CONFIDENCE|THEMES
        Example: 0.65|0.8|Earnings optimism, technical breakout, profit-taking concerns
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        
        try:
            parts = result.split('|')
            score = float(parts[0])
            confidence = float(parts[1])
            themes = parts[2] if len(parts) > 2 else ""
            
            return {
                'score': score,
                'confidence': confidence,
                'themes': themes,
                'source': 'social_media'
            }
        except:
            return {'score': 0, 'confidence': 0, 'themes': '', 'source': 'social_media'}
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze news sentiment"""
        
        # Mock news data - in real implementation, would use JigsawStack or news APIs
        mock_news_data = f"""
        Recent news about {symbol}:
        
        1. "{symbol} Reports Strong Q4 Earnings, Beats Estimates"
        2. "Analyst Upgrades {symbol} with $200 Price Target"  
        3. "Market Volatility Impacts Tech Stocks Including {symbol}"
        4. "{symbol} Announces New Product Launch for 2024"
        """
        
        prompt = f"""
        Analyze the financial news sentiment for {symbol}:
        
        {mock_news_data}
        
        Consider:
        - Earnings reports and analyst coverage
        - Product announcements and business developments  
        - Market and sector trends affecting the stock
        
        Provide sentiment score (-1.0 to +1.0), confidence (0.0 to 1.0), and key factors.
        Format as: SCORE|CONFIDENCE|FACTORS
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        
        try:
            parts = result.split('|')
            score = float(parts[0])
            confidence = float(parts[1])
            factors = parts[2] if len(parts) > 2 else ""
            
            return {
                'score': score,
                'confidence': confidence,
                'key_factors': factors,
                'source': 'news'
            }
        except:
            return {'score': 0, 'confidence': 0, 'key_factors': '', 'source': 'news'}
    
    async def _analyze_options_sentiment(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Analyze options positioning sentiment"""
        
        # Extract options data
        put_call_ratio = market_data.get('put_call_ratio', 1.0)
        unusual_volume = market_data.get('unusual_options_volume', False)
        
        # Simple sentiment based on put/call ratio
        if put_call_ratio < 0.7:
            sentiment_score = 0.5  # Bullish (more calls than puts)
        elif put_call_ratio > 1.3:
            sentiment_score = -0.5  # Bearish (more puts than calls)
        else:
            sentiment_score = 0  # Neutral
        
        # Adjust for unusual volume
        if unusual_volume:
            sentiment_score *= 1.2  # Amplify sentiment with unusual volume
        
        confidence = 0.6 if unusual_volume else 0.4
        
        return {
            'score': sentiment_score,
            'confidence': confidence,
            'put_call_ratio': put_call_ratio,
            'unusual_volume': unusual_volume,
            'source': 'options_positioning'
        }
    
    def _aggregate_sentiment_scores(self, sentiment_data: Dict) -> float:
        """Aggregate sentiment scores with weighting"""
        
        # Define source weights
        source_weights = {
            'social': 0.4,
            'news': 0.4, 
            'options_positioning': 0.2
        }
        
        total_score = 0
        total_weight = 0
        
        for source, data in sentiment_data.items():
            if source in source_weights and 'score' in data:
                weight = source_weights[source] * data.get('confidence', 0.5)
                total_score += data['score'] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_sentiment_confidence(self, sentiment_data: Dict) -> float:
        """Calculate overall sentiment confidence"""
        
        confidences = []
        for data in sentiment_data.values():
            if 'confidence' in data:
                confidences.append(data['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    async def _generate_sentiment_explanation(self, symbol: str, sentiment_data: Dict, 
                                            aggregate_score: float) -> str:
        """Generate AI explanation of sentiment analysis"""
        
        prompt = f"""
        Summarize the sentiment analysis for {symbol}:
        
        Social Sentiment: {sentiment_data.get('social', {}).get('score', 0):.2f}
        News Sentiment: {sentiment_data.get('news', {}).get('score', 0):.2f}
        Options Sentiment: {sentiment_data.get('options_positioning', {}).get('score', 0):.2f}
        
        Overall Sentiment: {aggregate_score:.2f}
        
        Provide a 2-sentence summary explaining the key sentiment drivers and their implications for options trading.
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        return response.choices[0].message.content
```

### Options Flow Agent (Gemini)

```python
# agents/agents/flow_agent.py
import google.generativeai as genai
from typing import Dict, Any
from datetime import datetime
import asyncio

class OptionsFlowAgent:
    """
    Options Flow Agent using Gemini 2.0 Flash for fast options analysis
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze options flow and unusual activity
        """
        try:
            # Calculate flow metrics
            flow_metrics = self._calculate_flow_metrics(market_data)
            
            # Detect unusual activity
            unusual_activity = self._detect_unusual_activity(market_data)
            
            # Generate flow prediction using Gemini
            flow_prediction = await self._generate_flow_prediction(symbol, flow_metrics, unusual_activity)
            
            # Calculate overall flow score
            flow_score = self._calculate_flow_score(flow_metrics, unusual_activity)
            
            # Calculate confidence
            confidence = self._calculate_flow_confidence(flow_metrics, unusual_activity)
            
            return {
                'agent': 'flow',
                'symbol': symbol,
                'flow_metrics': flow_metrics,
                'unusual_activity': unusual_activity,
                'flow_prediction': flow_prediction,
                'score': flow_score,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'agent': 'flow',
                'error': str(e),
                'score': 0,
                'confidence': 0
            }
    
    def _calculate_flow_metrics(self, market_data: Dict) -> Dict[str, Any]:
        """Calculate options flow metrics"""
        
        options_data = market_data.get('options_data', {})
        
        # Put/Call ratio
        put_volume = options_data.get('put_volume', 0)
        call_volume = options_data.get('call_volume', 1)  # Avoid division by zero
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        # Volume ratio (current vs average)
        current_volume = options_data.get('total_volume', 0)
        avg_volume = options_data.get('avg_volume_20d', 1)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Gamma exposure (simplified calculation)
        total_gamma = options_data.get('total_gamma', 0)
        
        # Implied volatility metrics
        iv_rank = options_data.get('iv_rank', 50)  # Percentile rank
        iv_skew = options_data.get('iv_skew', 0)   # Put-call IV difference
        
        return {
            'put_call_ratio': put_call_ratio,
            'volume_ratio': volume_ratio,
            'gamma_exposure': total_gamma,
            'iv_rank': iv_rank,
            'iv_skew': iv_skew,
            'total_volume': current_volume
        }
    
    def _detect_unusual_activity(self, market_data: Dict) -> List[Dict]:
        """Detect unusual options activity"""
        
        options_chain = market_data.get('options_chain', [])
        unusual_activity = []
        
        for option in options_chain:
            volume = option.get('volume', 0)
            open_interest = option.get('open_interest', 1)
            avg_volume = option.get('avg_volume', 0)
            
            # Unusual volume criteria
            volume_threshold = max(avg_volume * 3, 1000)  # 3x average or 1000 minimum
            oi_ratio = volume / open_interest if open_interest > 0 else 0
            
            if volume > volume_threshold or oi_ratio > 0.5:
                unusual_score = min((volume / volume_threshold) * (1 + oi_ratio), 2.0)
                
                unusual_activity.append({
                    'strike': option.get('strike'),
                    'expiration': option.get('expiration'),
                    'option_type': option.get('type'),
                    'volume': volume,
                    'open_interest': open_interest,
                    'unusual_score': unusual_score,
                    'reason': 'high_volume' if volume > volume_threshold else 'high_oi_ratio'
                })
        
        # Sort by unusual score descending
        return sorted(unusual_activity, key=lambda x: x['unusual_score'], reverse=True)[:10]
    
    async def _generate_flow_prediction(self, symbol: str, flow_metrics: Dict, 
                                      unusual_activity: List[Dict]) -> Dict:
        """Generate flow prediction using Gemini"""
        
        unusual_summary = ""
        if unusual_activity:
            unusual_summary = f"""
            Unusual Activity Detected:
            {chr(10).join([f"- {act['strike']}{act['option_type'].upper()} {act['expiration']}: Volume {act['volume']}, Score {act['unusual_score']:.2f}" 
                          for act in unusual_activity[:5]])}
            """
        
        prompt = f"""
        Analyze options flow for {symbol}:
        
        Flow Metrics:
        - Put/Call Ratio: {flow_metrics['put_call_ratio']:.2f}
        - Volume Ratio (vs 20d avg): {flow_metrics['volume_ratio']:.2f}
        - IV Rank: {flow_metrics['iv_rank']:.1f}%
        - IV Skew: {flow_metrics['iv_skew']:.2f}
        - Total Volume: {flow_metrics['total_volume']:,}
        
        {unusual_summary}
        
        Provide:
        1. Flow interpretation (bullish/bearish/neutral)
        2. Key insights about unusual activity
        3. Implications for stock movement
        
        Keep response concise and actionable for options traders.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            return {
                'prediction': response.text,
                'flow_direction': self._extract_flow_direction(response.text),
                'key_insights': self._extract_insights(response.text)
            }
        except Exception as e:
            return {
                'prediction': f"Flow analysis error: {str(e)}",
                'flow_direction': 'neutral',
                'key_insights': []
            }
    
    def _calculate_flow_score(self, flow_metrics: Dict, unusual_activity: List) -> float:
        """Calculate overall flow score"""
        
        score = 0
        
        # Put/call ratio contribution
        pcr = flow_metrics['put_call_ratio']
        if pcr < 0.7:
            score += 0.3  # Bullish (more calls)
        elif pcr > 1.3:
            score -= 0.3  # Bearish (more puts)
        
        # Volume ratio contribution
        vol_ratio = flow_metrics['volume_ratio']
        if vol_ratio > 2.0:
            score += 0.2 * min(vol_ratio / 2.0, 2.0)  # High volume is bullish, cap at 2x
        
        # IV rank contribution (high IV can be bearish)
        iv_rank = flow_metrics['iv_rank']
        if iv_rank > 80:
            score -= 0.1  # Very high IV suggests fear
        elif iv_rank < 20:
            score += 0.1  # Low IV suggests complacency
        
        # Unusual activity contribution
        if unusual_activity:
            # Check if more calls or puts in unusual activity
            call_activity = sum(1 for act in unusual_activity if act['option_type'] == 'call')
            put_activity = len(unusual_activity) - call_activity
            
            if call_activity > put_activity:
                score += 0.2
            elif put_activity > call_activity:
                score -= 0.2
        
        return max(-1.0, min(1.0, score))  # Clamp between -1 and 1
    
    def _calculate_flow_confidence(self, flow_metrics: Dict, unusual_activity: List) -> float:
        """Calculate confidence in flow analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more volume
        vol_ratio = flow_metrics['volume_ratio']
        if vol_ratio > 1.5:
            confidence += 0.2
        
        # Higher confidence with unusual activity
        if unusual_activity:
            avg_unusual_score = sum(act['unusual_score'] for act in unusual_activity) / len(unusual_activity)
            confidence += min(avg_unusual_score / 5.0, 0.3)
        
        # Adjust for IV rank (extreme values more reliable)
        iv_rank = flow_metrics['iv_rank']
        if iv_rank > 80 or iv_rank < 20:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_flow_direction(self, prediction_text: str) -> str:
        """Extract flow direction from AI response"""
        text_lower = prediction_text.lower()
        
        if 'bullish' in text_lower or 'positive' in text_lower:
            return 'bullish'
        elif 'bearish' in text_lower or 'negative' in text_lower:
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_insights(self, prediction_text: str) -> List[str]:
        """Extract key insights from AI response"""
        # Simple extraction - in practice, could use more sophisticated NLP
        sentences = prediction_text.split('.')
        insights = []
        
        for sentence in sentences[:3]:  # Take first 3 sentences as insights
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short sentences
                insights.append(sentence)
        
        return insights
```

This AI agent implementation provides a robust foundation for the Neural Options Oracle++ system with proper error handling, dynamic weighting, and educational explanations while using the most appropriate AI models for each task.