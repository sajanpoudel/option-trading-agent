# Neural Options Oracle++ - Testing Strategy

## Testing Overview

The Neural Options Oracle++ system requires comprehensive testing across multiple layers:
- AI Agent Testing (mocked LLM responses)
- API Integration Testing
- Real-time Data Pipeline Testing
- Frontend Component Testing
- End-to-End Workflow Testing

## Testing Architecture

### 1. Unit Testing

#### AI Agent Testing
```python
# tests/unit/test_agents.py
import pytest
from unittest.mock import AsyncMock, patch
from agents.technical_agent import TechnicalAnalysisAgent
from agents.sentiment_agent import SentimentAnalysisAgent

class TestTechnicalAnalysisAgent:
    
    @pytest.fixture
    def mock_openai_response(self):
        return {
            "choices": [{
                "message": {
                    "content": """
                    {
                        "scenario": "strong_uptrend",
                        "indicators": {
                            "ma": {"signal": 0.8, "value": 155.2},
                            "rsi": {"signal": 0.6, "value": 68.5},
                            "bb": {"signal": 0.4, "value": "upper"},
                            "macd": {"signal": 0.9, "value": 2.3},
                            "vwap": {"signal": 0.7, "value": 154.8}
                        },
                        "weights": {
                            "ma": 0.30, "rsi": 0.15, "bb": 0.10, 
                            "macd": 0.25, "vwap": 0.20
                        },
                        "weighted_score": 0.72,
                        "confidence": 0.85
                    }
                    """
                }
            }]
        }
    
    @patch('openai.ChatCompletion.acreate')
    async def test_analyze_technical_indicators(self, mock_openai, mock_openai_response):
        """Test technical analysis with mocked OpenAI response"""
        
        # Setup mock
        mock_openai.return_value = mock_openai_response
        
        # Create agent
        agent = TechnicalAnalysisAgent()
        
        # Test analysis
        result = await agent.analyze_technical_indicators(
            symbol="AAPL", 
            timeframe="1D"
        )
        
        # Assertions
        assert result["scenario"] == "strong_uptrend"
        assert result["weighted_score"] == 0.72
        assert result["confidence"] == 0.85
        assert result["weights"]["ma"] == 0.30
        
        # Verify OpenAI was called with correct parameters
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args
        assert "AAPL" in call_args[1]["messages"][-1]["content"]

class TestSentimentAnalysisAgent:
    
    @pytest.fixture
    def mock_jigsawstack_response(self):
        return {
            "sentiment": "positive",
            "confidence": 0.78,
            "score": 0.65,
            "keywords": ["bullish", "earnings beat", "strong guidance"]
        }
    
    @patch('jigsawstack.sentiment.analyze')
    async def test_analyze_social_sentiment(self, mock_jigsawstack, mock_jigsawstack_response):
        """Test sentiment analysis with mocked JigsawStack"""
        
        # Setup mock
        mock_jigsawstack.return_value = mock_jigsawstack_response
        
        # Create agent
        agent = SentimentAnalysisAgent()
        
        # Test analysis
        result = await agent.analyze_social_sentiment("AAPL")
        
        # Assertions
        assert result["aggregate_sentiment"] > 0.5
        assert "bullish" in result["keywords"]
        assert result["confidence"] == 0.78
        
        # Verify JigsawStack was called
        mock_jigsawstack.assert_called()
```

#### Decision Engine Testing
```python
# tests/unit/test_decision_engine.py
import pytest
from src.core.decision_engine import DecisionEngine
from unittest.mock import AsyncMock

class TestDecisionEngine:
    
    @pytest.fixture
    def mock_agent_results(self):
        return {
            "technical": {
                "scenario": "strong_uptrend",
                "weighted_score": 0.72,
                "confidence": 0.85
            },
            "sentiment": {
                "aggregate_sentiment": 0.65,
                "confidence": 0.78
            },
            "flow": {
                "ml_prediction": 0.58,
                "unusual_activity": True
            },
            "history": {
                "pattern_score": 0.45,
                "similar_patterns": 15
            }
        }
    
    async def test_calculate_weighted_decision(self, mock_agent_results):
        """Test weighted decision calculation"""
        
        engine = DecisionEngine()
        
        # Test with default weights
        weights = {
            "technical": 0.60,
            "sentiment": 0.10,
            "flow": 0.10,
            "history": 0.20
        }
        
        decision_score = engine.calculate_weighted_decision(
            mock_agent_results, weights
        )
        
        # Expected: 0.72*0.6 + 0.65*0.1 + 0.58*0.1 + 0.45*0.2 = 0.645
        expected_score = 0.645
        assert abs(decision_score - expected_score) < 0.01
    
    async def test_generate_signal(self, mock_agent_results):
        """Test signal generation from decision score"""
        
        engine = DecisionEngine()
        
        # Test strong buy signal
        signal = engine.generate_signal(0.7, mock_agent_results)
        assert signal["direction"] == "STRONG_BUY"
        assert signal["strategy_type"] == "aggressive_bullish"
        
        # Test sell signal
        signal = engine.generate_signal(-0.4, mock_agent_results)
        assert signal["direction"] == "SELL"
        assert signal["strategy_type"] == "moderate_bearish"
        
        # Test hold signal
        signal = engine.generate_signal(0.1, mock_agent_results)
        assert signal["direction"] == "HOLD"
        assert signal["strategy_type"] == "neutral"
```

### 2. Integration Testing

#### Database Integration Testing
```python
# tests/integration/test_database.py
import pytest
import asyncio
from config.database import SupabaseManager
from datetime import datetime, timedelta

class TestSupabaseIntegration:
    
    @pytest.fixture
    async def db_manager(self):
        """Create test database manager"""
        manager = SupabaseManager()
        # Use test database or ensure cleanup
        yield manager
        # Cleanup test data
        await self.cleanup_test_data(manager)
    
    async def test_browser_session_lifecycle(self, db_manager):
        """Test browser session creation and retrieval"""
        
        # Create session
        session_token = await db_manager.create_browser_session(
            ip_address="127.0.0.1",
            user_agent="test-browser",
            device_info={"platform": "test"}
        )
        
        assert session_token is not None
        
        # Retrieve session
        session = await db_manager.get_session(session_token)
        
        assert session is not None
        assert session["ip_address"] == "127.0.0.1"
        assert session["is_active"] is True
    
    async def test_trading_signal_storage(self, db_manager):
        """Test trading signal storage and retrieval"""
        
        signal_data = {
            "symbol": "AAPL",
            "direction": "BUY",
            "strength": "strong",
            "confidence_score": 0.85,
            "market_scenario": "strong_uptrend",
            "agent_weights": {"technical": 0.6, "sentiment": 0.1},
            "technical_analysis": {"rsi": 68.5, "ma_signal": 0.8},
            "strike_recommendations": [
                {"strike": 155, "expiration": "2024-02-16", "type": "call"}
            ]
        }
        
        # Save signal
        signal_id = await db_manager.save_trading_signal(signal_data)
        assert signal_id is not None
        
        # Verify signal was saved
        result = db_manager.client.table("trading_signals")\
            .select("*")\
            .eq("id", signal_id)\
            .single()\
            .execute()
            
        assert result.data["symbol"] == "AAPL"
        assert result.data["confidence_score"] == 0.85
    
    async def test_position_pnl_updates(self, db_manager):
        """Test real-time position P&L updates"""
        
        # Create test position
        position_data = {
            "symbol": "AAPL",
            "position_type": "option",
            "option_type": "call",
            "strike_price": 155.0,
            "quantity": 1,
            "entry_price": 2.50
        }
        
        position_id = await db_manager.create_position(position_data)
        assert position_id is not None
        
        # Update P&L
        updated_position = await db_manager.update_position_pnl(
            position_id, 3.25
        )
        
        # Verify P&L calculation (3.25 - 2.50) * 1 * 100 = $75
        assert updated_position["unrealized_pnl"] == 75.0
        assert updated_position["current_price"] == 3.25
```

#### API Integration Testing
```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, AsyncMock

class TestAPIIntegration:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('agents.orchestrator.OptionsOracleOrchestrator.analyze_stock')
    def test_analyze_stock_endpoint(self, mock_analyze, client):
        """Test stock analysis API endpoint"""
        
        # Mock agent orchestrator response
        mock_analyze.return_value = {
            "symbol": "AAPL",
            "signal": {
                "direction": "BUY",
                "score": 0.72,
                "confidence": 0.85
            },
            "agent_results": {
                "technical": {"weighted_score": 0.72},
                "sentiment": {"aggregate_sentiment": 0.65}
            }
        }
        
        # Make API request
        response = client.post("/api/v1/analysis/analyze/AAPL")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["signal"]["direction"] == "BUY"
        assert data["signal"]["score"] == 0.72
        
        # Verify mock was called
        mock_analyze.assert_called_once_with("AAPL")
    
    def test_portfolio_summary_endpoint(self, client):
        """Test portfolio summary endpoint"""
        
        response = client.get("/api/v1/portfolio/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_positions" in data
        assert "total_value" in data
        assert "portfolio_greeks" in data
```

### 3. AI Model Testing

#### Mock LLM Responses for Consistent Testing
```python
# tests/mocks/llm_responses.py
class MockLLMResponses:
    
    TECHNICAL_ANALYSIS_RESPONSES = {
        "AAPL_bullish": {
            "scenario": "strong_uptrend",
            "indicators": {
                "ma": {"signal": 0.8, "value": 155.2},
                "rsi": {"signal": 0.6, "value": 68.5},
                "bb": {"signal": 0.4, "value": "upper"},
                "macd": {"signal": 0.9, "value": 2.3},
                "vwap": {"signal": 0.7, "value": 154.8}
            },
            "weighted_score": 0.72,
            "confidence": 0.85
        },
        "TSLA_bearish": {
            "scenario": "strong_downtrend",
            "indicators": {
                "ma": {"signal": -0.7, "value": 245.8},
                "rsi": {"signal": -0.5, "value": 35.2},
                "bb": {"signal": -0.3, "value": "lower"},
                "macd": {"signal": -0.8, "value": -1.8},
                "vwap": {"signal": -0.6, "value": 248.5}
            },
            "weighted_score": -0.64,
            "confidence": 0.78
        }
    }
    
    SENTIMENT_RESPONSES = {
        "positive": {
            "sentiment": "positive",
            "confidence": 0.78,
            "score": 0.65,
            "keywords": ["bullish", "earnings beat", "strong guidance"]
        },
        "negative": {
            "sentiment": "negative", 
            "confidence": 0.82,
            "score": -0.58,
            "keywords": ["bearish", "concerns", "downgrade"]
        }
    }

# Usage in tests
@pytest.fixture
def mock_technical_response():
    return MockLLMResponses.TECHNICAL_ANALYSIS_RESPONSES["AAPL_bullish"]
```

### 4. Performance Testing

#### Load Testing
```python
# tests/performance/test_load.py
import asyncio
import pytest
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

class TestSystemPerformance:
    
    async def test_concurrent_analysis_requests(self):
        """Test system performance under concurrent analysis requests"""
        
        async def make_analysis_request(session, symbol):
            start_time = time.time()
            async with session.post(f"/api/v1/analysis/analyze/{symbol}") as response:
                data = await response.json()
                end_time = time.time()
                return {
                    "symbol": symbol,
                    "response_time": end_time - start_time,
                    "status": response.status,
                    "success": response.status == 200
                }
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"] * 10  # 50 requests
        
        async with aiohttp.ClientSession("http://localhost:8080") as session:
            tasks = [make_analysis_request(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
        
        # Performance assertions
        successful_requests = [r for r in results if r["success"]]
        assert len(successful_requests) >= 45  # 90% success rate
        
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        assert avg_response_time < 5.0  # Average response time under 5 seconds
        
        max_response_time = max(r["response_time"] for r in successful_requests)
        assert max_response_time < 15.0  # No request over 15 seconds
    
    async def test_websocket_performance(self):
        """Test WebSocket real-time updates performance"""
        
        import websockets
        
        messages_received = 0
        start_time = time.time()
        
        async def websocket_client():
            nonlocal messages_received
            uri = "ws://localhost:8080/ws/positions"
            
            async with websockets.connect(uri) as websocket:
                # Listen for 30 seconds
                timeout = 30
                end_time = start_time + timeout
                
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        messages_received += 1
                    except asyncio.TimeoutError:
                        continue
        
        # Run WebSocket client
        await websocket_client()
        
        # Performance assertions
        elapsed_time = time.time() - start_time
        messages_per_second = messages_received / elapsed_time
        
        assert messages_per_second >= 1.0  # At least 1 message per second
```

### 5. End-to-End Testing

#### Complete User Journey Testing
```python
# tests/e2e/test_user_journey.py
import pytest
from playwright.async_api import async_playwright
import asyncio

class TestCompleteUserJourney:
    
    @pytest.fixture
    async def browser_page(self):
        """Setup browser for E2E testing"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            yield page
            await browser.close()
    
    async def test_complete_analysis_flow(self, browser_page):
        """Test complete user flow from stock selection to position tracking"""
        
        page = browser_page
        
        # Navigate to application
        await page.goto("http://localhost:3000")
        
        # Wait for page to load
        await page.wait_for_selector('[data-testid="stock-selector"]')
        
        # Enter stock symbol
        await page.fill('[data-testid="stock-input"]', 'AAPL')
        await page.click('[data-testid="analyze-button"]')
        
        # Wait for analysis to complete
        await page.wait_for_selector('[data-testid="analysis-results"]', timeout=30000)
        
        # Verify analysis results are displayed
        technical_score = await page.text_content('[data-testid="technical-score"]')
        assert technical_score is not None
        
        # Check for signal generation
        signal_direction = await page.text_content('[data-testid="signal-direction"]')
        assert signal_direction in ["BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"]
        
        # Verify strike recommendations are shown
        strikes_visible = await page.is_visible('[data-testid="strike-recommendations"]')
        assert strikes_visible
        
        # Test educational content generation
        education_visible = await page.is_visible('[data-testid="educational-content"]')
        assert education_visible
        
        # If signal is bullish, test paper trade execution
        if signal_direction in ["BUY", "STRONG_BUY"]:
            await page.click('[data-testid="execute-paper-trade"]')
            
            # Wait for position creation
            await page.wait_for_selector('[data-testid="position-created"]', timeout=10000)
            
            # Verify position appears in portfolio
            position_visible = await page.is_visible('[data-testid="portfolio-position"]')
            assert position_visible
    
    async def test_3d_visualization_interaction(self, browser_page):
        """Test 3D payoff visualization interaction"""
        
        page = browser_page
        
        # Navigate and analyze stock
        await page.goto("http://localhost:3000")
        await page.fill('[data-testid="stock-input"]', 'AAPL')
        await page.click('[data-testid="analyze-button"]')
        
        # Wait for 3D visualization to load
        await page.wait_for_selector('[data-testid="payoff-3d"]', timeout=15000)
        
        # Test 3D interaction (mouse events)
        canvas = await page.query_selector('[data-testid="payoff-3d"] canvas')
        assert canvas is not None
        
        # Simulate mouse interaction with 3D scene
        await canvas.hover()
        await page.mouse.down()
        await page.mouse.move(100, 100)
        await page.mouse.up()
        
        # Verify visualization updated
        # (This would require specific 3D scene testing)
```

### 6. Testing Configuration

#### pytest Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --asyncio-mode=auto
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    api: API tests
    db: Database tests
```

#### Test Dependencies
```txt
# tests/requirements.txt
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1
pytest-playwright==0.4.0
aiohttp==3.8.5
websockets==11.0.3
factory-boy==3.3.0
faker==19.3.0
```

### 7. Continuous Integration

#### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: neural_oracle_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/requirements.txt
    
    - name: Set up environment
      run: |
        cp .env.test .env
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -m unit --cov=src
    
    - name: Run integration tests  
      run: |
        pytest tests/integration/ -m integration
      env:
        SUPABASE_URL: ${{ secrets.SUPABASE_TEST_URL }}
        SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_TEST_KEY }}
    
    - name: Run API tests
      run: |
        pytest tests/integration/test_api.py -m api
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 8. Test Data Management

#### Test Fixtures and Factories
```python
# tests/factories.py
import factory
from datetime import datetime, timedelta
import uuid

class TradingSignalFactory(factory.Factory):
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    symbol = "AAPL"
    signal_type = "hybrid"
    direction = "BUY"
    strength = "strong"
    confidence_score = 0.85
    market_scenario = "strong_uptrend"
    agent_weights = factory.Dict({
        "technical": 0.60,
        "sentiment": 0.10,
        "flow": 0.10,
        "history": 0.20
    })
    technical_analysis = factory.Dict({
        "rsi": 68.5,
        "ma_signal": 0.8,
        "scenario": "strong_uptrend"
    })
    created_at = factory.LazyFunction(datetime.now)
    expires_at = factory.LazyFunction(lambda: datetime.now() + timedelta(hours=24))

class PositionFactory(factory.Factory):
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    symbol = "AAPL"
    position_type = "option"
    option_type = "call"
    strike_price = 155.0
    expiration_date = factory.LazyFunction(lambda: datetime.now().date() + timedelta(days=30))
    quantity = 1
    entry_price = 2.50
    current_price = 2.50
    status = "open"
```

This comprehensive testing strategy ensures the Neural Options Oracle++ system is thoroughly validated across all components while maintaining fast development cycles through mocked LLM responses and comprehensive test coverage.