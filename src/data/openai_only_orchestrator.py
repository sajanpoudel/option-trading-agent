"""
OpenAI-Only Data Orchestrator
Streamlined architecture using only OpenAI web search + OptionsProfitCalculator
Eliminates JigsawStack and complex scraping dependencies
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from openai import AsyncOpenAI
import re

from config.settings import settings
from config.logging import get_data_logger
from .alpaca_client import AlpacaMarketDataClient

logger = get_data_logger()

@dataclass
class OptionsAnalysis:
    """Complete options analysis result"""
    symbol: str
    total_call_volume: int
    total_put_volume: int
    put_call_ratio: float
    unusual_activity: bool
    key_strikes: List[Dict[str, Any]]
    expiration_analysis: Dict[str, Any]
    flow_sentiment: str
    confidence: float

@dataclass 
class MarketIntelligence:
    """Comprehensive market intelligence"""
    symbol: str
    timestamp: datetime
    options_analysis: OptionsAnalysis
    news_sentiment: Dict[str, Any]
    social_sentiment: Dict[str, Any]
    technical_signals: Dict[str, Any]
    market_outlook: Dict[str, Any]
    confidence_score: float

class OptionsProfitCalculatorAPI:
    """Enhanced OptionsProfitCalculator integration"""
    
    def __init__(self):
        self.base_url = "https://www.optionsprofitcalculator.com/ajax/getOptions"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_comprehensive_options_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive options data with detailed analysis"""
        try:
            # Get raw options data
            raw_data = await self._fetch_options_data(symbol)
            
            if not raw_data or 'options' not in raw_data:
                logger.warning(f"No options data received for {symbol}")
                return {}
            
            # Parse and analyze options data
            analysis = self._analyze_options_chain(raw_data, symbol)
            
            logger.info(f"Retrieved comprehensive options data for {symbol}: {analysis['total_contracts']} contracts across {len(analysis['expirations'])} expirations")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting comprehensive options data for {symbol}: {e}")
            return {}
    
    async def _fetch_options_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw options data from API"""
        try:
            params = {
                'stock': symbol.upper(),
                'reqId': 3
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Options API error: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return {}
    
    def _analyze_options_chain(self, raw_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Comprehensive analysis of options chain data"""
        
        options_data = raw_data.get('options', {})
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'expirations': [],
            'total_contracts': 0,
            'total_call_volume': 0,
            'total_put_volume': 0,
            'total_call_oi': 0,
            'total_put_oi': 0,
            'unusual_activity': [],
            'key_strikes': [],
            'volume_by_expiration': {},
            'oi_by_expiration': {}
        }
        
        for expiration_date, exp_data in options_data.items():
            exp_analysis = {
                'expiration': expiration_date,
                'calls': [],
                'puts': [],
                'total_call_vol': 0,
                'total_put_vol': 0,
                'total_call_oi': 0,
                'total_put_oi': 0
            }
            
            # Analyze calls
            if 'c' in exp_data:
                for strike_str, option_data in exp_data['c'].items():
                    strike = float(strike_str)
                    volume = int(option_data.get('v', 0))
                    oi = int(option_data.get('oi', 0))
                    last = float(option_data.get('l', 0))
                    bid = float(option_data.get('b', 0))
                    ask = float(option_data.get('a', 0))
                    
                    call_info = {
                        'strike': strike,
                        'volume': volume,
                        'open_interest': oi,
                        'last': last,
                        'bid': bid,
                        'ask': ask,
                        'spread': ask - bid if ask > bid else 0,
                        'type': 'call'
                    }
                    
                    exp_analysis['calls'].append(call_info)
                    exp_analysis['total_call_vol'] += volume
                    exp_analysis['total_call_oi'] += oi
                    
                    # Detect unusual activity
                    if volume > 1000 or (volume > 0 and volume > oi * 2):
                        analysis['unusual_activity'].append({
                            'strike': strike,
                            'expiration': expiration_date,
                            'type': 'call',
                            'volume': volume,
                            'open_interest': oi,
                            'reason': 'high_volume' if volume > 1000 else 'volume_vs_oi'
                        })
            
            # Analyze puts
            if 'p' in exp_data:
                for strike_str, option_data in exp_data['p'].items():
                    strike = float(strike_str)
                    volume = int(option_data.get('v', 0))
                    oi = int(option_data.get('oi', 0))
                    last = float(option_data.get('l', 0))
                    bid = float(option_data.get('b', 0))
                    ask = float(option_data.get('a', 0))
                    
                    put_info = {
                        'strike': strike,
                        'volume': volume,
                        'open_interest': oi,
                        'last': last,
                        'bid': bid,
                        'ask': ask,
                        'spread': ask - bid if ask > bid else 0,
                        'type': 'put'
                    }
                    
                    exp_analysis['puts'].append(put_info)
                    exp_analysis['total_put_vol'] += volume
                    exp_analysis['total_put_oi'] += oi
                    
                    # Detect unusual activity
                    if volume > 1000 or (volume > 0 and volume > oi * 2):
                        analysis['unusual_activity'].append({
                            'strike': strike,
                            'expiration': expiration_date,
                            'type': 'put',
                            'volume': volume,
                            'open_interest': oi,
                            'reason': 'high_volume' if volume > 1000 else 'volume_vs_oi'
                        })
            
            # Add to main analysis
            analysis['expirations'].append(exp_analysis)
            analysis['total_call_volume'] += exp_analysis['total_call_vol']
            analysis['total_put_volume'] += exp_analysis['total_put_vol']
            analysis['total_call_oi'] += exp_analysis['total_call_oi']
            analysis['total_put_oi'] += exp_analysis['total_put_oi']
            analysis['volume_by_expiration'][expiration_date] = {
                'calls': exp_analysis['total_call_vol'],
                'puts': exp_analysis['total_put_vol']
            }
            analysis['oi_by_expiration'][expiration_date] = {
                'calls': exp_analysis['total_call_oi'],
                'puts': exp_analysis['total_put_oi']
            }
        
        # Calculate key metrics
        total_volume = analysis['total_call_volume'] + analysis['total_put_volume']
        analysis['total_contracts'] = total_volume
        analysis['put_call_ratio'] = (analysis['total_put_volume'] / max(analysis['total_call_volume'], 1))
        
        # Identify key strikes (highest volume/OI)
        all_options = []
        for exp in analysis['expirations']:
            all_options.extend(exp['calls'])
            all_options.extend(exp['puts'])
        
        # Sort by volume and take top strikes
        sorted_by_volume = sorted(all_options, key=lambda x: x['volume'], reverse=True)
        analysis['key_strikes'] = sorted_by_volume[:10]  # Top 10 by volume
        
        return analysis

class OpenAIMarketIntelligence:
    """OpenAI-powered market intelligence using web search"""
    
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.options_api = OptionsProfitCalculatorAPI()
        self.alpaca_client = AlpacaMarketDataClient()
    
    async def get_comprehensive_intelligence(self, symbol: str) -> MarketIntelligence:
        """Get comprehensive market intelligence for a symbol"""
        
        try:
            logger.info(f"Starting comprehensive intelligence gathering for {symbol}")
            
            # Run all analyses in parallel
            tasks = [
                self._get_options_intelligence(symbol),
                self._get_news_intelligence(symbol),
                self._get_social_intelligence(symbol),
                self._get_technical_intelligence(symbol),
                self._get_market_outlook(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            options_analysis = results[0] if not isinstance(results[0], Exception) else {}
            news_sentiment = results[1] if not isinstance(results[1], Exception) else {}
            social_sentiment = results[2] if not isinstance(results[2], Exception) else {}
            technical_signals = results[3] if not isinstance(results[3], Exception) else {}
            market_outlook = results[4] if not isinstance(results[4], Exception) else {}
            
            # Create comprehensive analysis
            overall_confidence = self._calculate_confidence_score(results)
            
            return MarketIntelligence(
                symbol=symbol,
                timestamp=datetime.now(),
                options_analysis=options_analysis,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                technical_signals=technical_signals,
                market_outlook=market_outlook,
                confidence_score=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive intelligence for {symbol}: {e}")
            return self._get_fallback_intelligence(symbol)
    
    async def _get_options_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive options intelligence"""
        
        try:
            # Get real options data
            async with self.options_api as api:
                options_data = await api.get_comprehensive_options_data(symbol)
            
            if not options_data:
                return {}
            
            # Analyze with OpenAI
            prompt = f"""
            Analyze this comprehensive options data for {symbol}:
            
            REAL OPTIONS DATA:
            - Total Call Volume: {options_data.get('total_call_volume', 0):,}
            - Total Put Volume: {options_data.get('total_put_volume', 0):,}
            - Put/Call Ratio: {options_data.get('put_call_ratio', 0):.2f}
            - Total Call OI: {options_data.get('total_call_oi', 0):,}
            - Total Put OI: {options_data.get('total_put_oi', 0):,}
            - Unusual Activity Events: {len(options_data.get('unusual_activity', []))}
            
            TOP VOLUME STRIKES:
            {self._format_key_strikes(options_data.get('key_strikes', [])[:5])}
            
            EXPIRATION ANALYSIS:
            {self._format_expiration_data(options_data.get('volume_by_expiration', {}))}
            
            Provide professional options flow analysis in JSON format:
            {{
                "flow_sentiment": "bullish|bearish|neutral",
                "unusual_activity_assessment": "high|medium|low|none", 
                "key_insights": [list of key insights],
                "gamma_exposure": "high|medium|low",
                "volatility_expectation": "high|medium|low",
                "institutional_positioning": "bullish|bearish|neutral",
                "retail_vs_institutional": "retail_heavy|institutional_heavy|balanced",
                "expiration_focus": "string describing key expiration dates",
                "strike_concentration": "string describing strike concentration",
                "flow_quality_score": float_0_to_1
            }}
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional options flow analyst with expertise in detecting institutional positioning and unusual activity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            analysis = self._extract_json_response(response.choices[0].message.content)
            analysis['raw_options_data'] = options_data  # Include raw data
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in options intelligence for {symbol}: {e}")
            return {}
    
    async def _get_news_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get news intelligence using OpenAI web search"""
        
        prompt = f"""
        Search for the latest financial news about {symbol} from the past 24 hours.
        
        Focus on:
        - Earnings reports and guidance
        - Analyst upgrades/downgrades  
        - SEC filings and regulatory news
        - Company announcements
        - Market-moving events
        - Options-related news
        
        Provide analysis in JSON format:
        {{
            "sentiment_score": float_-1_to_1,
            "key_headlines": [list of important headlines with sentiment],
            "market_impact": "bullish|bearish|neutral",
            "catalyst_events": [list of upcoming catalysts],
            "analyst_activity": "upgrades|downgrades|neutral",
            "earnings_proximity": "pre_earnings|post_earnings|far_from_earnings",
            "news_quality": "high|medium|low",
            "confidence": float_0_to_1
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial news analyst specializing in market-moving events and their impact on options trading."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            return self._extract_json_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in news intelligence for {symbol}: {e}")
            return {}
    
    async def _get_social_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get social sentiment intelligence using OpenAI web search"""
        
        prompt = f"""
        Search social media and forums for sentiment about {symbol}:
        
        Sources to analyze:
        - Reddit (r/wallstreetbets, r/investing, r/stocks, r/options)
        - StockTwits discussions and sentiment indicators
        - Twitter/X financial discussions
        - Discord trading communities
        - Options-focused forums
        
        Provide analysis in JSON format:
        {{
            "overall_sentiment": float_-1_to_1,
            "bullish_percentage": float,
            "bearish_percentage": float,
            "sentiment_intensity": "high|medium|low",
            "trending_topics": [list of trending discussion topics],
            "retail_positioning": "bullish|bearish|neutral",
            "options_sentiment": "calls_popular|puts_popular|neutral",
            "social_momentum": "increasing|decreasing|stable",
            "confidence": float_0_to_1
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a social media sentiment analyst specializing in retail investor sentiment and options trading discussions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return self._extract_json_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in social intelligence for {symbol}: {e}")
            return {}
    
    async def _get_technical_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis intelligence"""
        
        try:
            # Get real technical data
            technical_data = await self.alpaca_client.get_technical_indicators(symbol)
            
            prompt = f"""
            Analyze the technical indicators for {symbol}:
            
            REAL TECHNICAL DATA:
            - Current Price: ${technical_data.get('current_price', 0):.2f}
            - RSI: {technical_data.get('rsi', 50):.2f}
            - MACD: {technical_data.get('macd', 0):.4f}
            - MACD Signal: {technical_data.get('macd_signal', 0):.4f}
            - BB Upper: ${technical_data.get('bb_upper', 0):.2f}
            - BB Lower: ${technical_data.get('bb_lower', 0):.2f}
            - Volume: {technical_data.get('volume', 0):,}
            - Volatility: {technical_data.get('volatility', 0):.2f}%
            - Support: ${technical_data.get('support', 0):.2f}
            - Resistance: ${technical_data.get('resistance', 0):.2f}
            
            Provide technical analysis in JSON format:
            {{
                "trend_direction": "bullish|bearish|neutral",
                "trend_strength": "strong|moderate|weak",
                "momentum": "positive|negative|neutral",
                "volatility_regime": "high|medium|low",
                "support_resistance": {{
                    "key_support": float,
                    "key_resistance": float,
                    "breakout_potential": "high|medium|low"
                }},
                "technical_signals": [list of key technical signals],
                "options_implications": "good_for_calls|good_for_puts|neutral",
                "confidence": float_0_to_1
            }}
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a technical analyst specializing in options trading and technical indicators."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            analysis = self._extract_json_response(response.choices[0].message.content)
            analysis['raw_technical_data'] = technical_data  # Include raw data
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in technical intelligence for {symbol}: {e}")
            return {}
    
    async def _get_market_outlook(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market outlook using web search"""
        
        prompt = f"""
        Provide comprehensive market outlook for {symbol}:
        
        Search for and analyze:
        - Institutional analyst ratings and price targets
        - Sector performance and comparison
        - Macro economic factors affecting the stock
        - Upcoming events (earnings, ex-dividend, etc.)
        - Options market maker positioning
        - Volatility forecasts
        
        Provide outlook in JSON format:
        {{
            "price_target_consensus": float,
            "analyst_rating": "strong_buy|buy|hold|sell|strong_sell",
            "sector_outlook": "outperform|underperform|inline",
            "volatility_forecast": "expanding|contracting|stable",
            "upcoming_catalysts": [list of upcoming events],
            "risk_factors": [list of key risks],
            "opportunities": [list of key opportunities],
            "time_horizon_outlook": {{
                "short_term": "bullish|bearish|neutral",
                "medium_term": "bullish|bearish|neutral",
                "long_term": "bullish|bearish|neutral"
            }},
            "confidence": float_0_to_1
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a comprehensive market strategist with expertise in equity analysis and options market dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            return self._extract_json_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in market outlook for {symbol}: {e}")
            return {}
    
    def _extract_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from OpenAI response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from OpenAI response")
            return {}
    
    def _format_key_strikes(self, strikes: List[Dict]) -> str:
        """Format key strikes for prompt"""
        if not strikes:
            return "No significant volume strikes"
        
        formatted = []
        for strike in strikes:
            formatted.append(f"${strike['strike']:.0f} {strike['type'].upper()}: Vol={strike['volume']:,}, OI={strike['open_interest']:,}")
        
        return "\n".join(formatted)
    
    def _format_expiration_data(self, exp_data: Dict) -> str:
        """Format expiration data for prompt"""
        if not exp_data:
            return "No expiration data"
        
        formatted = []
        for exp_date, volumes in exp_data.items():
            call_vol = volumes.get('calls', 0)
            put_vol = volumes.get('puts', 0)
            formatted.append(f"{exp_date}: Calls={call_vol:,}, Puts={put_vol:,}")
        
        return "\n".join(formatted)
    
    def _calculate_confidence_score(self, results: List[Any]) -> float:
        """Calculate overall confidence score"""
        successful = sum(1 for r in results if not isinstance(r, Exception) and r)
        total = len(results)
        return successful / total if total > 0 else 0.0
    
    def _get_fallback_intelligence(self, symbol: str) -> MarketIntelligence:
        """Fallback intelligence when analysis fails"""
        return MarketIntelligence(
            symbol=symbol,
            timestamp=datetime.now(),
            options_analysis={},
            news_sentiment={},
            social_sentiment={},
            technical_signals={},
            market_outlook={},
            confidence_score=0.0
        )

# Test function
async def test_openai_orchestrator():
    """Test OpenAI-only orchestrator with real options data"""
    print("🧠 Testing OpenAI-Only Market Intelligence System")
    
    intelligence = OpenAIMarketIntelligence(settings.openai_api_key)
    
    symbol = "AAPL"
    print(f"📊 Getting comprehensive intelligence for {symbol}...")
    
    try:
        result = await intelligence.get_comprehensive_intelligence(symbol)
        
        print(f"✅ Analysis completed with {result.confidence_score:.1%} confidence")
        print(f"📈 Options Analysis: {len(result.options_analysis)} fields")
        print(f"📰 News Sentiment: {len(result.news_sentiment)} fields")
        print(f"💭 Social Sentiment: {len(result.social_sentiment)} fields")
        print(f"📊 Technical Signals: {len(result.technical_signals)} fields")
        print(f"🎯 Market Outlook: {len(result.market_outlook)} fields")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_openai_orchestrator())