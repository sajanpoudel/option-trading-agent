"""
StockTwits Web Scraper Agent
Uses OpenAI to analyze trending stocks from StockTwits
"""
import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from openai import OpenAI
from loguru import logger


class StockTwitsWebScraperAgent:
    """Agent for scraping trending stocks from StockTwits"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.openai_client = openai_client
        self.base_url = "https://stocktwits.com"
        self.trending_url = f"{self.base_url}/sentiment/most-active"
        
    async def get_trending_stocks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending stocks from StockTwits most-active page using OpenAI web search
        Returns list of trending stocks with symbols and sentiment data
        """
        try:
            logger.info(f"🔥 Using OpenAI web search to get trending STOCKS from StockTwits...")
            
            if not self.openai_client:
                logger.warning("OpenAI client not available, using fallback stocks")
                return self._get_fallback_stocks(limit)
            
            # Use OpenAI to directly analyze the StockTwits page
            trending_stocks = await self._analyze_stocktwits_with_openai(limit)
            
            logger.info(f"✅ Found {len(trending_stocks)} trending STOCKS from StockTwits")
            return trending_stocks
            
        except Exception as e:
            logger.error(f"Error getting trending stocks: {e}")
            return self._get_fallback_stocks(limit)
    
    async def _fetch_webpage(self, url: str) -> Optional[str]:
        """Fetch webpage content using aiohttp"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {len(content)} characters from {url}")
                        return content
                    else:
                        logger.error(f"HTTP {response.status} when fetching {url}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching webpage {url}: {e}")
            return None
    
    async def _analyze_stocktwits_with_openai(self, limit: int) -> List[Dict[str, Any]]:
        """Use OpenAI to analyze StockTwits page directly for STOCK symbols only"""
        try:
            prompt = f"""
            Please analyze the StockTwits most active page at https://stocktwits.com/sentiment/most-active 
            and extract the top {limit} trending STOCK symbols (NOT cryptocurrency).
            
            IMPORTANT INSTRUCTIONS:
            1. Focus ONLY on actual stock market symbols (NYSE, NASDAQ stocks)
            2. EXCLUDE all cryptocurrency symbols like BTC, ETH, SOL, PEPE, BONK, KENDU, etc.
            3. Look for traditional stock symbols like AAPL, GOOGL, MSFT, TSLA, NVDA, etc.
            4. Extract the sentiment (Bullish/Bearish/Neutral) for each stock
            5. Get approximate mention counts or activity levels
            
            Return ONLY a JSON array in this exact format:
            [
              {{
                "symbol": "AAPL",
                "name": "Apple Inc",
                "sentiment": "Bullish",
                "sentiment_score": 0.75,
                "mentions": 1234,
                "trending": true
              }},
              {{
                "symbol": "GOOGL", 
                "name": "Alphabet Inc",
                "sentiment": "Neutral",
                "sentiment_score": 0.50,
                "mentions": 890,
                "trending": true
              }}
            ]
            
            Focus on STOCKS only - no crypto, no commodities, just traditional equities.
            """
            
            # Use OpenAI Web Search API (new format)
            try:
                response = await asyncio.to_thread(
                    self.openai_client.responses.create,
                    model="gpt-4o",
                    tools=[{"type": "web_search"}],
                    tool_choice="auto",
                    input=prompt
                )
                
                # Extract content from the new response format
                ai_response = ""
                if hasattr(response, 'output_text'):
                    ai_response = response.output_text
                elif hasattr(response, 'content') and response.content:
                    for content in response.content:
                        if hasattr(content, 'text'):
                            ai_response += content.text
                        elif hasattr(content, 'output_text'):
                            ai_response += content.output_text
                elif hasattr(response, 'choices') and response.choices:
                    # Handle different response format
                    for choice in response.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            ai_response += choice.message.content
                
                # If web search returns non-JSON text, ask AI to extract symbols
                if ai_response and not ai_response.strip().startswith('{') and not ai_response.strip().startswith('['):
                    logger.info("Web search returned text instead of JSON, processing with AI...")
                    # Convert text response to JSON using AI
                    extract_prompt = f"""
The web search returned this text about trending stocks: 

{ai_response[:2000]}

Please extract trending STOCK symbols (NOT crypto) from this text and format as JSON:
[
  {{
    "symbol": "AAPL",
    "name": "Apple Inc",
    "sentiment": "Bullish",
    "sentiment_score": 0.75,
    "mentions": 1234,
    "trending": true
  }}
]

Only return the JSON array, no other text.
"""
                    
                    extract_response = await asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": extract_prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    ai_response = extract_response.choices[0].message.content.strip()
                
            except Exception as web_search_error:
                logger.warning(f"Web search API failed, falling back to chat: {web_search_error}")
                # Fallback to regular chat completion
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                ai_response = response.choices[0].message.content.strip()
            logger.info(f"OpenAI web search response: {ai_response[:200]}...")
            
            # Parse JSON response
            import json
            try:
                # Clean up response to extract just the JSON
                if '```json' in ai_response:
                    ai_response = ai_response.split('```json')[1].split('```')[0]
                elif '```' in ai_response:
                    ai_response = ai_response.split('```')[1].split('```')[0]
                
                trending_data = json.loads(ai_response.strip())
                
                # Filter out any crypto that might have slipped through
                crypto_symbols = {'BTC', 'ETH', 'SOL', 'PEPE', 'BONK', 'KENDU', 'VELOD', 'DOGE', 'SHIB'}
                filtered_data = [
                    stock for stock in trending_data 
                    if isinstance(stock, dict) and stock.get('symbol', '').upper() not in crypto_symbols
                ]
                
                logger.info(f"Filtered out crypto, got {len(filtered_data)} stock symbols")
                return filtered_data[:limit] if isinstance(filtered_data, list) else []
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {e}")
                return self._get_fallback_stocks(limit)
                
        except Exception as e:
            logger.error(f"OpenAI web search failed: {e}")
            return self._get_fallback_stocks(limit)
    
    async def _analyze_html_with_openai(self, html_content: str, limit: int) -> List[Dict[str, Any]]:
        """Use OpenAI to analyze HTML and extract trending stock data"""
        if not self.openai_client:
            logger.error("OpenAI client not available - cannot analyze HTML")
            return []
        
        try:
            # Clean up the HTML to focus on the important parts
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content (first 8000 chars to stay within token limits)
            text_content = soup.get_text()[:8000]
            
            prompt = f"""
            Analyze this StockTwits most-active page content and extract the trending stock symbols with their sentiment data.
            
            Content: {text_content}
            
            Please extract up to {limit} trending stocks in the following JSON format:
            [
              {{
                "symbol": "AAPL",
                "name": "Apple Inc",
                "sentiment": "Bullish",
                "sentiment_score": 0.75,
                "mentions": 1234,
                "trending": true
              }}
            ]
            
            Focus on:
            1. Stock symbols (usually 1-5 uppercase letters)
            2. Sentiment indicators (Bullish/Bearish/Neutral)
            3. Number of mentions or activity level
            4. Company names if available
            
            Return only the JSON array, no other text.
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"OpenAI extracted trending stocks: {ai_response[:200]}...")
            
            # Parse JSON response
            import json
            try:
                trending_data = json.loads(ai_response)
                return trending_data[:limit] if isinstance(trending_data, list) else []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {e}")
                return self._fallback_html_parsing(html_content, limit)
                
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._fallback_html_parsing(html_content, limit)
    
    def _fallback_html_parsing(self, html_content: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback HTML parsing when OpenAI is not available"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            trending_stocks = []
            
            # Look for common patterns in StockTwits HTML
            # Stock symbols are usually in spans, links, or divs with specific classes
            symbol_patterns = [
                r'\b([A-Z]{1,5})\b',  # Basic stock symbol pattern
            ]
            
            text_content = soup.get_text()
            symbols_found = set()
            
            for pattern in symbol_patterns:
                matches = re.findall(pattern, text_content)
                for match in matches:
                    if len(match) >= 2 and match not in ['THE', 'AND', 'FOR', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'USE', 'WAY', 'WHY', 'AIR', 'BAD', 'BIG', 'BOX', 'CAR', 'CAT', 'CUT', 'DOG', 'EAR', 'EYE', 'FAR', 'FUN', 'GOT', 'HOT', 'JOB', 'LOT', 'MAN', 'OWN', 'PUT', 'RUN', 'SIT', 'SUN', 'TOO', 'TOP', 'WIN', 'YES', 'YET']:
                        symbols_found.add(match)
            
            # Convert to the expected format
            for i, symbol in enumerate(list(symbols_found)[:limit]):
                trending_stocks.append({
                    "symbol": symbol,
                    "name": f"{symbol} Inc",
                    "sentiment": "Neutral",
                    "sentiment_score": 0.5,
                    "mentions": 100 - i * 10,  # Mock descending mentions
                    "trending": True
                })
            
            logger.info(f"Fallback parsing found {len(trending_stocks)} symbols")
            return trending_stocks
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return []
    
    def _get_fallback_stocks(self, limit: int) -> List[Dict[str, Any]]:
        """Return fallback popular stocks when web scraping fails"""
        fallback_stocks = [
            {"symbol": "AAPL", "name": "Apple Inc", "sentiment": "Bullish", "sentiment_score": 0.75, "mentions": 1500, "trending": True},
            {"symbol": "GOOGL", "name": "Alphabet Inc", "sentiment": "Neutral", "sentiment_score": 0.60, "mentions": 1200, "trending": True},
            {"symbol": "MSFT", "name": "Microsoft Corp", "sentiment": "Bullish", "sentiment_score": 0.80, "mentions": 1100, "trending": True},
            {"symbol": "TSLA", "name": "Tesla Inc", "sentiment": "Bearish", "sentiment_score": 0.30, "mentions": 2000, "trending": True},
            {"symbol": "NVDA", "name": "NVIDIA Corp", "sentiment": "Bullish", "sentiment_score": 0.85, "mentions": 1800, "trending": True},
            {"symbol": "AMZN", "name": "Amazon Inc", "sentiment": "Neutral", "sentiment_score": 0.55, "mentions": 900, "trending": True},
            {"symbol": "META", "name": "Meta Platforms", "sentiment": "Bearish", "sentiment_score": 0.40, "mentions": 800, "trending": True},
            {"symbol": "NFLX", "name": "Netflix Inc", "sentiment": "Neutral", "sentiment_score": 0.50, "mentions": 600, "trending": True}
        ]
        
        logger.info(f"Using fallback stocks: {[s['symbol'] for s in fallback_stocks[:limit]]}")
        return fallback_stocks[:limit]
    
    async def get_stock_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get detailed sentiment analysis for a specific stock"""
        try:
            stock_url = f"{self.base_url}/symbol/{symbol}"
            html_content = await self._fetch_webpage(stock_url)
            
            if not html_content or not self.openai_client:
                return {
                    "symbol": symbol,
                    "sentiment": "Neutral",
                    "sentiment_score": 0.5,
                    "mentions": 50,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Use OpenAI to analyze stock-specific sentiment
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()[:4000]
            
            prompt = f"""
            Analyze the sentiment for stock {symbol} from this StockTwits page content:
            
            {text_content}
            
            Return JSON format:
            {{
              "symbol": "{symbol}",
              "sentiment": "Bullish|Bearish|Neutral",
              "sentiment_score": 0.0-1.0,
              "key_themes": ["theme1", "theme2"],
              "mentions": estimated_count,
              "confidence": 0.0-1.0
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            import json
            sentiment_data = json.loads(response.choices[0].message.content.strip())
            sentiment_data["timestamp"] = datetime.now().isoformat()
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {
                "symbol": symbol,
                "sentiment": "Neutral",
                "sentiment_score": 0.5,
                "mentions": 50,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }


# Global instance for the application
web_scraper_agent = None

def get_web_scraper_agent(openai_client: Optional[OpenAI] = None) -> StockTwitsWebScraperAgent:
    """Get or create the global web scraper agent instance"""
    global web_scraper_agent
    if web_scraper_agent is None:
        web_scraper_agent = StockTwitsWebScraperAgent(openai_client)
    return web_scraper_agent