# Neural Options Oracle++ - Architecture Details

## Service-Level Architecture

### 1. API Gateway Service (`api-gateway/`)

**Purpose**: Central entry point for all client requests with routing, authentication, and WebSocket management.

**Technology Stack**:
- FastAPI 0.104+
- Python 3.11+
- Pydantic for request/response models
- JWT authentication
- WebSocket support

**Key Components**:
```python
# File Structure
api-gateway/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── jwt_handler.py      # JWT token management
│   │   └── middleware.py       # Auth middleware
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analysis.py         # Stock analysis endpoints
│   │   ├── trading.py          # Trading endpoints
│   │   ├── education.py        # Educational endpoints
│   │   └── websocket.py        # WebSocket handlers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py         # Pydantic request models
│   │   └── responses.py        # Pydantic response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── orchestrator_client.py  # Agent orchestrator client
│   │   ├── decision_client.py       # Decision engine client
│   │   └── trading_client.py        # Trading service client
│   └── utils/
│       ├── __init__.py
│       ├── rate_limiter.py     # Rate limiting implementation
│       └── websocket_manager.py # WebSocket connection management
├── requirements.txt
└── Dockerfile
```

**API Endpoints**:
```python
# Core endpoints
POST /api/v1/auth/login
GET  /api/v1/auth/profile
POST /api/v1/analysis/stock/{symbol}
GET  /api/v1/trading/positions
POST /api/v1/trading/execute
GET  /api/v1/education/lessons
WS   /ws  # WebSocket endpoint for real-time updates
```

**Docker Configuration**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

### 2. Agent Orchestrator Service (`agent-orchestrator/`)

**Purpose**: Manages AI agents using OpenAI Agents SDK with dynamic weight assignment and agent coordination.

**Technology Stack**:
- OpenAI Agents SDK
- OpenAI Python client
- Google AI Python SDK (Gemini)
- Redis for inter-agent communication

**Key Components**:
```python
# File Structure
agent-orchestrator/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Service entry point
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── master_orchestrator.py  # Main orchestration logic
│   │   ├── weight_calculator.py    # Dynamic weight assignment
│   │   └── scenario_detector.py    # Market scenario detection
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Base agent class
│   │   ├── technical_agent.py      # Technical analysis (GPT-4o)
│   │   ├── sentiment_agent.py      # Sentiment analysis (GPT-4o-mini)
│   │   ├── flow_agent.py           # Options flow (Gemini 2.0)
│   │   ├── history_agent.py        # Historical patterns (GPT-4o)
│   │   ├── risk_agent.py           # Risk management (GPT-4o)
│   │   └── education_agent.py      # Educational content (GPT-4o-mini)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agent_responses.py      # Agent response models
│   │   └── market_data.py          # Market data models
│   └── utils/
│       ├── __init__.py
│       ├── redis_client.py         # Redis communication
│       └── ai_client.py            # AI model clients
├── requirements.txt
└── Dockerfile
```

**Agent Implementation Example**:
```python
# agents/technical_agent.py
from openai import OpenAI
from typing import Dict, Any

class TechnicalAnalysisAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
    async def analyze(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """
        Analyze technical indicators with dynamic scenario weighting
        """
        # Calculate technical indicators
        indicators = self._calculate_indicators(market_data)
        
        # Detect market scenario
        scenario = self._detect_scenario(indicators)
        
        # Get scenario-specific weights
        weights = self._get_scenario_weights(scenario)
        
        # Generate analysis using OpenAI
        prompt = self._build_analysis_prompt(symbol, indicators, scenario, weights)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return {
            "scenario": scenario,
            "indicators": indicators,
            "weights": weights,
            "analysis": response.choices[0].message.content,
            "confidence": self._calculate_confidence(indicators)
        }
    
    def _get_scenario_weights(self, scenario: str) -> Dict[str, float]:
        """Dynamic weight assignment based on market scenario"""
        weights = {
            "strong_uptrend": {
                "ma": 0.30, "rsi": 0.15, "bb": 0.10, 
                "macd": 0.25, "vwap": 0.20
            },
            "strong_downtrend": {
                "ma": 0.30, "rsi": 0.15, "bb": 0.10, 
                "macd": 0.25, "vwap": 0.20
            },
            "range_bound": {
                "ma": 0.15, "rsi": 0.25, "bb": 0.30, 
                "macd": 0.15, "vwap": 0.15
            },
            "breakout": {
                "ma": 0.20, "rsi": 0.15, "bb": 0.30, 
                "macd": 0.20, "vwap": 0.15
            },
            "potential_reversal": {
                "ma": 0.15, "rsi": 0.25, "bb": 0.20, 
                "macd": 0.30, "vwap": 0.10
            }
        }
        return weights.get(scenario, weights["range_bound"])
```

---

### 3. Market Data Service (`market-data/`)

**Purpose**: Real-time market data collection, processing, and distribution.

**Technology Stack**:
- Alpaca Trade API for market data
- StockTwits API for social data
- JigsawStack for data enhancement
- WebSocket clients for real-time streaming
- InfluxDB for time series storage

**Key Components**:
```python
# File Structure
market-data/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Service entry point
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── alpaca_collector.py     # Alpaca market data
│   │   ├── stocktwits_collector.py # StockTwits social data
│   │   └── jigsawstack_collector.py # JigsawStack enhancement
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── price_processor.py      # Price data processing
│   │   ├── options_processor.py    # Options chain processing
│   │   └── sentiment_processor.py  # Social sentiment processing
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── redis_storage.py        # Real-time cache
│   │   └── influx_storage.py       # Time series storage
│   └── streams/
│       ├── __init__.py
│       ├── market_stream.py        # Market data streaming
│       └── websocket_publisher.py  # WebSocket broadcasting
├── requirements.txt
└── Dockerfile
```

**Real-time Data Pipeline**:
```python
# streams/market_stream.py
import asyncio
from alpaca.data.live import StockDataStream
from typing import Dict, Callable

class MarketDataStream:
    def __init__(self, api_key: str, secret_key: str):
        self.stream = StockDataStream(api_key, secret_key)
        self.subscribers = {}
        
    async def start_streaming(self, symbols: list):
        """Start real-time market data streaming"""
        await self.stream.subscribe_quotes(self._handle_quote, *symbols)
        await self.stream.subscribe_trades(self._handle_trade, *symbols)
        await self.stream.run()
        
    async def _handle_quote(self, quote):
        """Process incoming quote data"""
        processed_data = {
            "symbol": quote.symbol,
            "bid": quote.bid,
            "ask": quote.ask,
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "timestamp": quote.timestamp
        }
        
        # Broadcast to subscribers
        await self._broadcast_data("quote", processed_data)
        
    async def _broadcast_data(self, data_type: str, data: Dict):
        """Broadcast data to all subscribers"""
        for subscriber in self.subscribers.get(data_type, []):
            await subscriber(data)
```

---

### 4. Decision Engine Service (`decision-engine/`)

**Purpose**: Core decision-making logic with dynamic weight assignment and signal generation.

**Technology Stack**:
- Python 3.11+
- NumPy for numerical computations
- Pandas for data manipulation
- PostgreSQL for signal storage

**Key Components**:
```python
# File Structure
decision-engine/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Service entry point
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── decision_engine.py  # Main decision logic
│   │   ├── weight_adjuster.py  # Dynamic weight adjustment
│   │   ├── signal_generator.py # Trading signal generation
│   │   └── scenario_classifier.py # Market scenario classification
│   ├── models/
│   │   ├── __init__.py
│   │   ├── decision_models.py  # Decision data models
│   │   └── signal_models.py    # Signal data models
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── risk_calculator.py  # Risk metrics calculation
│   │   ├── greeks_calculator.py # Options Greeks calculation
│   │   └── probability_calculator.py # Probability of profit
│   └── database/
│       ├── __init__.py
│       └── postgres_client.py  # PostgreSQL operations
├── requirements.txt
└── Dockerfile
```

**Decision Engine Logic**:
```python
# engine/decision_engine.py
import numpy as np
from typing import Dict, List

class DecisionEngine:
    def __init__(self):
        self.base_weights = {
            'technical': 0.60,
            'sentiment': 0.10,
            'flow': 0.10,
            'history': 0.20
        }
        
    async def process_decision(self, symbol: str, agent_results: Dict) -> Dict:
        """Main decision processing pipeline"""
        
        # Detect market scenario
        scenario = self._detect_scenario(agent_results)
        
        # Adjust weights based on scenario
        adjusted_weights = self._adjust_weights(scenario)
        
        # Calculate weighted decision score
        decision_score = self._calculate_weighted_score(agent_results, adjusted_weights)
        
        # Generate trading signal
        signal = self._generate_signal(decision_score, agent_results)
        
        return {
            'symbol': symbol,
            'scenario': scenario,
            'weights': adjusted_weights,
            'decision_score': decision_score,
            'signal': signal,
            'confidence': self._calculate_confidence(agent_results)
        }
        
    def _adjust_weights(self, scenario: str) -> Dict[str, float]:
        """Dynamically adjust agent weights based on market scenario"""
        weights = self.base_weights.copy()
        
        if scenario == 'high_volatility':
            # Increase technical analysis weight in volatile markets
            weights['technical'] += 0.10
            weights['sentiment'] -= 0.05
            weights['flow'] -= 0.05
            
        elif scenario == 'low_volatility':
            # Increase sentiment and flow weights in calm markets
            weights['sentiment'] += 0.05
            weights['flow'] += 0.05
            weights['technical'] -= 0.10
            
        elif scenario == 'earnings_approaching':
            # Emphasize options flow before earnings
            weights['flow'] += 0.15
            weights['technical'] -= 0.10
            weights['history'] -= 0.05
            
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
```

---

### 5. Trading Execution Service (`trading-execution/`)

**Purpose**: Paper trading execution, position management, and real-time P&L tracking.

**Technology Stack**:
- Alpaca Trade API (Paper Trading)
- PostgreSQL for position storage
- Redis for real-time data
- WebSocket for live updates

**Key Components**:
```python
# File Structure
trading-execution/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Service entry point
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── alpaca_executor.py  # Alpaca paper trading
│   │   ├── position_manager.py # Position management
│   │   └── risk_monitor.py     # Real-time risk monitoring
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── pnl_calculator.py   # P&L calculation
│   │   ├── greeks_calculator.py # Options Greeks
│   │   └── risk_calculator.py  # Risk metrics
│   ├── models/
│   │   ├── __init__.py
│   │   ├── position_models.py  # Position data models
│   │   └── order_models.py     # Order data models
│   └── database/
│       ├── __init__.py
│       └── postgres_client.py  # Database operations
├── requirements.txt
└── Dockerfile
```

**Trading Execution Logic**:
```python
# execution/alpaca_executor.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class AlpacaExecutor:
    def __init__(self, api_key: str, secret_key: str):
        self.client = TradingClient(api_key, secret_key, paper=True)
        
    async def execute_options_trade(self, symbol: str, recommendation: Dict) -> Dict:
        """Execute options trade based on recommendation"""
        
        # Build option symbol (e.g., AAPL240315C00150000)
        option_symbol = self._build_option_symbol(
            symbol,
            recommendation['strike'],
            recommendation['expiration'],
            recommendation['option_type']
        )
        
        # Create order request
        order_request = MarketOrderRequest(
            symbol=option_symbol,
            qty=recommendation['quantity'],
            side=OrderSide.BUY if recommendation['side'] == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order
        order = self.client.submit_order(order_request)
        
        # Calculate initial Greeks
        greeks = self._calculate_greeks(recommendation)
        
        return {
            'order_id': order.id,
            'symbol': symbol,
            'option_symbol': option_symbol,
            'status': order.status,
            'filled_qty': order.filled_qty,
            'filled_avg_price': order.filled_avg_price,
            'greeks': greeks,
            'timestamp': order.created_at
        }
```

---

### 6. Educational Service (`educational/`)

**Purpose**: Adaptive learning content generation and trade explanation system.

**Technology Stack**:
- OpenAI GPT-4o-mini for content generation
- PostgreSQL for progress tracking
- Interactive quiz system

**Key Components**:
```python
# File Structure
educational/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Service entry point
│   ├── content/
│   │   ├── __init__.py
│   │   ├── content_generator.py    # AI content generation
│   │   ├── explanation_engine.py   # Trade explanations
│   │   └── quiz_generator.py       # Interactive quizzes
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── adaptive_engine.py      # Adaptive learning logic
│   │   ├── progress_tracker.py     # Learning progress
│   │   └── curriculum_manager.py   # Curriculum management
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── strategy_simulator.py   # Options strategy simulation
│   │   └── monte_carlo.py          # Monte Carlo simulations
│   └── models/
│       ├── __init__.py
│       ├── learning_models.py      # Learning data models
│       └── content_models.py       # Content data models
├── requirements.txt
└── Dockerfile
```

---

### 7. Frontend Service (`frontend/`)

**Purpose**: Interactive React dashboard with 3D visualizations and real-time updates.

**Technology Stack**:
- Next.js 14 with TypeScript
- Three.js for 3D visualizations
- Socket.IO for WebSocket connections
- Tailwind CSS for styling
- Recharts for financial charts

**Key Components**:
```typescript
// File Structure
frontend/
├── components/
│   ├── dashboard/
│   │   ├── TradingDashboard.tsx        # Main dashboard
│   │   ├── AgentAnalysisPanel.tsx      # Agent results display
│   │   ├── DecisionEngineViz.tsx       # Decision visualization
│   │   └── RealTimeMonitoring.tsx      # Live position monitoring
│   ├── visualizations/
│   │   ├── PayoffSurface3D.tsx         # 3D payoff surface
│   │   ├── GreeksVisualization.tsx     # Greeks charts
│   │   └── TechnicalCharts.tsx         # Technical analysis charts
│   ├── education/
│   │   ├── LearningModule.tsx          # Educational content
│   │   ├── InteractiveQuiz.tsx         # Quiz component
│   │   └── StrategySimulator.tsx       # Strategy simulation
│   └── ui/
│       ├── StockSelector.tsx           # Stock selection
│       ├── RiskProfileSelector.tsx     # Risk profile settings
│       └── TradingControls.tsx         # Trading execution controls
├── hooks/
│   ├── useWebSocket.ts                 # WebSocket connection
│   ├── useAgentData.ts                 # Agent data management
│   ├── usePositions.ts                 # Position tracking
│   └── useEducation.ts                 # Educational content
├── utils/
│   ├── calculations.ts                 # Financial calculations
│   ├── chartHelpers.ts                 # Chart utilities
│   └── formatters.ts                   # Data formatting
├── pages/
│   ├── index.tsx                       # Main dashboard
│   ├── education.tsx                   # Education portal
│   ├── portfolio.tsx                   # Portfolio management
│   └── api/                            # API routes
└── styles/
    ├── globals.css                     # Global styles
    └── components.css                  # Component styles
```

---

## Data Flow Architecture

### Request Flow
1. **Client Request** → API Gateway (Port 8080)
2. **Authentication** → JWT validation
3. **Route to Service** → Based on endpoint
4. **Service Processing** → Business logic execution
5. **Data Storage** → PostgreSQL/Redis/InfluxDB
6. **Response** → JSON response to client

### Real-time Data Flow
1. **Market Data** → Market Data Service (Port 8082)
2. **Processing** → Data normalization and enrichment
3. **Caching** → Redis for fast access
4. **Broadcasting** → WebSocket to all connected clients
5. **Storage** → InfluxDB for historical data

### AI Agent Flow
1. **Market Analysis Request** → Agent Orchestrator (Port 8081)
2. **Agent Coordination** → Multi-agent processing
3. **External AI APIs** → OpenAI/Gemini API calls
4. **Weight Adjustment** → Dynamic scenario-based weighting
5. **Decision Engine** → Signal generation
6. **Trading Execution** → Paper trade execution

## Security Architecture

### Authentication & Authorization
```python
# JWT-based authentication
{
    "sub": "user_id",
    "exp": timestamp,
    "iat": timestamp,
    "scope": ["read", "write", "trade"],
    "risk_level": "moderate"
}
```

### API Security
- Rate limiting: 100 requests/minute per user
- Input validation: Pydantic models
- SQL injection prevention: Parameterized queries
- CORS configuration: Specific origin allowlist

### Data Security
- Database encryption at rest
- TLS 1.3 for data in transit
- API key rotation: 90-day cycle
- Audit logging: All trade actions logged

This architecture provides a scalable, maintainable, and secure foundation for the Neural Options Oracle++ system with clear separation of concerns and modern development practices.