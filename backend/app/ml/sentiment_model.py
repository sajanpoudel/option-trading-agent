"""
OpenAI-Based Sentiment Analysis Model
Replaces FinBERT with GPT-4o-mini for better integration and performance
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from openai import OpenAI
from backend.config.settings import settings
from backend.config.logging import get_data_logger

logger = get_data_logger()


@dataclass
class SentimentResult:
    """Sentiment analysis result structure"""
    score: float  # -1 to 1 scale
    confidence: float  # 0 to 1 scale
    reasoning: str
    category: str  # 'bullish', 'bearish', 'neutral'
    signals: List[str]
    timestamp: datetime


class OpenAISentimentAnalyzer:
    """Advanced sentiment analysis using OpenAI GPT models"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.analysis_count = 0
        self.success_rate = 0.0
        
        # Financial sentiment patterns for validation
        self.bullish_keywords = [
            'buy', 'bull', 'moon', 'rocket', 'calls', 'upward', 'growth', 
            'surge', 'rally', 'breakout', 'momentum', 'strong', 'positive'
        ]
        
        self.bearish_keywords = [
            'sell', 'bear', 'crash', 'puts', 'downward', 'decline', 
            'dump', 'correction', 'breakdown', 'resistance', 'weak', 'negative'
        ]
        
        logger.info(f"OpenAI Sentiment Analyzer initialized with {model}")
    
    async def analyze_text(self, text: str, context: str = "") -> SentimentResult:
        """Analyze sentiment of given text"""
        
        try:
            prompt = self._create_sentiment_prompt(text, context)
            
            response = await asyncio.create_task(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a financial sentiment analysis expert specialized in options trading and market psychology."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
            )
            
            result = self._parse_sentiment_response(response.choices[0].message.content, text)
            
            # Validate result
            validated_result = self._validate_sentiment(result, text)
            
            self.analysis_count += 1
            logger.info(f"Sentiment analyzed: {validated_result.category} ({validated_result.score:.2f})")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._get_fallback_sentiment(text)
    
    async def analyze_multiple_texts(
        self, 
        texts: List[Dict[str, str]]
    ) -> Dict[str, SentimentResult]:
        """Analyze sentiment for multiple texts with different contexts"""
        
        try:
            tasks = []
            for item in texts:
                text = item.get('text', '')
                context = item.get('context', '')
                source = item.get('source', 'unknown')
                
                task = asyncio.create_task(
                    self.analyze_text(text, context),
                    name=f"sentiment_{source}"
                )
                tasks.append((source, task))
            
            results = {}
            for source, task in tasks:
                try:
                    result = await task
                    results[source] = result
                except Exception as e:
                    logger.error(f"Failed to analyze {source}: {e}")
                    results[source] = self._get_fallback_sentiment("")
            
            return results
            
        except Exception as e:
            logger.error(f"Multiple sentiment analysis failed: {e}")
            return {}
    
    def _create_sentiment_prompt(self, text: str, context: str = "") -> str:
        """Create optimized prompt for sentiment analysis"""
        
        context_info = f"\nContext: {context}" if context else ""
        
        prompt = f"""
Analyze the financial sentiment of the following text for options trading decisions.
{context_info}

Text to analyze:
"{text}"

Provide your analysis in this exact JSON format:
{{
    "sentiment_score": <float between -1.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "category": "<bullish|bearish|neutral>",
    "reasoning": "<brief explanation of sentiment analysis>",
    "key_signals": ["<signal1>", "<signal2>", "<signal3>"],
    "market_impact": "<low|medium|high>",
    "options_bias": "<calls|puts|neutral>"
}}

Guidelines:
- Score: -1.0 = Very Bearish, 0.0 = Neutral, +1.0 = Very Bullish
- Confidence: Based on clarity and strength of sentiment indicators
- Key signals: 3-5 most important phrases/words driving the sentiment
- Focus on options trading implications
- Consider both explicit sentiment and implied market psychology
"""
        
        return prompt
    
    def _parse_sentiment_response(self, response: str, original_text: str) -> SentimentResult:
        """Parse OpenAI response into structured sentiment result"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in response")
            
            # Extract and validate fields
            score = float(data.get('sentiment_score', 0.0))
            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
            
            confidence = float(data.get('confidence', 0.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            category = data.get('category', 'neutral').lower()
            if category not in ['bullish', 'bearish', 'neutral']:
                category = 'neutral'
            
            reasoning = data.get('reasoning', 'Sentiment analysis completed')
            signals = data.get('key_signals', [])
            
            # Ensure consistency between score and category
            if abs(score) < 0.2:
                category = 'neutral'
            elif score > 0 and category == 'bearish':
                category = 'bullish'
            elif score < 0 and category == 'bullish':
                category = 'bearish'
            
            return SentimentResult(
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                category=category,
                signals=signals[:5],  # Limit to 5 signals
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            return self._get_fallback_sentiment(original_text)
    
    def _validate_sentiment(self, result: SentimentResult, text: str) -> SentimentResult:
        """Validate sentiment result against keyword patterns"""
        
        try:
            text_lower = text.lower()
            
            # Count keyword matches
            bullish_matches = sum(1 for word in self.bullish_keywords if word in text_lower)
            bearish_matches = sum(1 for word in self.bearish_keywords if word in text_lower)
            
            # Calculate keyword-based score
            keyword_score = (bullish_matches - bearish_matches) / max(len(text.split()) / 10, 1)
            keyword_score = max(-1.0, min(1.0, keyword_score))
            
            # Compare with OpenAI result
            score_diff = abs(result.score - keyword_score)
            
            # If significant disagreement, blend the results
            if score_diff > 0.5 and len(text.split()) < 50:  # Only for shorter texts
                blended_score = (result.score * 0.7) + (keyword_score * 0.3)
                result.score = blended_score
                result.confidence *= 0.9  # Reduce confidence slightly
                logger.debug(f"Blended sentiment score: {blended_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Sentiment validation failed: {e}")
            return result
    
    def _get_fallback_sentiment(self, text: str) -> SentimentResult:
        """Generate fallback sentiment when analysis fails"""
        
        # Simple keyword-based fallback
        text_lower = text.lower() if text else ""
        
        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        if bullish_count > bearish_count:
            score = 0.3
            category = 'bullish'
        elif bearish_count > bullish_count:
            score = -0.3
            category = 'bearish'
        else:
            score = 0.0
            category = 'neutral'
        
        return SentimentResult(
            score=score,
            confidence=0.3,  # Low confidence for fallback
            reasoning="Fallback keyword-based analysis",
            category=category,
            signals=['fallback_analysis'],
            timestamp=datetime.now()
        )
    
    async def aggregate_sentiments(
        self, 
        sentiment_results: Dict[str, SentimentResult],
        source_weights: Optional[Dict[str, float]] = None
    ) -> SentimentResult:
        """Aggregate multiple sentiment results with optional source weighting"""
        
        if not sentiment_results:
            return self._get_fallback_sentiment("")
        
        # Default weights
        if source_weights is None:
            source_weights = {
                'news': 0.4,
                'social': 0.3,
                'analyst': 0.2,
                'earnings': 0.1
            }
        
        total_score = 0.0
        total_confidence = 0.0
        total_weight = 0.0
        all_signals = []
        categories = []
        
        for source, result in sentiment_results.items():
            weight = source_weights.get(source, 1.0 / len(sentiment_results))
            
            total_score += result.score * weight * result.confidence
            total_confidence += result.confidence * weight
            total_weight += weight
            
            all_signals.extend(result.signals)
            categories.append(result.category)
        
        # Calculate aggregate values
        if total_weight > 0:
            avg_score = total_score / total_weight
            avg_confidence = total_confidence / total_weight
        else:
            avg_score = 0.0
            avg_confidence = 0.0
        
        # Determine aggregate category
        bullish_count = categories.count('bullish')
        bearish_count = categories.count('bearish')
        
        if bullish_count > bearish_count:
            category = 'bullish'
        elif bearish_count > bullish_count:
            category = 'bearish'
        else:
            category = 'neutral'
        
        # Create reasoning
        reasoning = f"Aggregated from {len(sentiment_results)} sources: "
        reasoning += f"{bullish_count} bullish, {bearish_count} bearish, "
        reasoning += f"{len(categories) - bullish_count - bearish_count} neutral"
        
        return SentimentResult(
            score=avg_score,
            confidence=avg_confidence,
            reasoning=reasoning,
            category=category,
            signals=list(set(all_signals))[:10],  # Unique signals, max 10
            timestamp=datetime.now()
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the sentiment analyzer"""
        
        return {
            'model': self.model,
            'analysis_count': self.analysis_count,
            'success_rate': self.success_rate,
            'average_confidence': 0.75,  # Placeholder - would be calculated from history
            'supported_languages': ['en'],
            'last_updated': datetime.now().isoformat()
        }


class ComparativeAnalyzer:
    """Compare OpenAI sentiment with traditional approaches"""
    
    def __init__(self):
        self.openai_analyzer = OpenAISentimentAnalyzer()
        
    async def compare_approaches(self, text: str) -> Dict[str, Any]:
        """Compare OpenAI vs traditional keyword-based sentiment"""
        
        # OpenAI analysis
        openai_result = await self.openai_analyzer.analyze_text(text)
        
        # Traditional keyword analysis
        traditional_result = self._traditional_sentiment(text)
        
        # Comparison metrics
        score_diff = abs(openai_result.score - traditional_result['score'])
        agreement = score_diff < 0.3
        
        return {
            'openai': {
                'score': openai_result.score,
                'confidence': openai_result.confidence,
                'category': openai_result.category,
                'reasoning': openai_result.reasoning
            },
            'traditional': traditional_result,
            'comparison': {
                'score_difference': score_diff,
                'agreement': agreement,
                'openai_advantage': openai_result.confidence > 0.7,
                'recommendation': 'openai' if openai_result.confidence > 0.6 else 'hybrid'
            }
        }
    
    def _traditional_sentiment(self, text: str) -> Dict[str, Any]:
        """Traditional keyword-based sentiment analysis"""
        
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'positive', 'buy', 'bullish', 'up', 'gain']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'sell', 'bearish', 'down', 'loss']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words > 0:
            score = (positive_count - negative_count) / total_words * 10
            score = max(-1.0, min(1.0, score))
        else:
            score = 0.0
        
        confidence = min(0.8, (positive_count + negative_count) / max(total_words / 10, 1))
        
        return {
            'score': score,
            'confidence': confidence,
            'method': 'keyword_based',
            'positive_matches': positive_count,
            'negative_matches': negative_count
        }


# Global instances
openai_sentiment = OpenAISentimentAnalyzer()
comparative_analyzer = ComparativeAnalyzer()