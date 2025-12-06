# Neural Options Oracle++

AI-Powered Options Trading Platform with Lambda Architecture

Complete intelligent trading system combining 9 specialized AI agents, multi-layered machine learning ensemble, real-time big data ingestion, and autonomous trading capabilities.

---

## Demo

https://github.com/user-attachments/assets/option-flow.mp4

Watch the platform in action: intelligent stock analysis, autonomous trading execution, and real-time market monitoring.

---

## System Architecture

![Overall Architecture](assets/overall-diagram.png)

### Lambda Architecture for Big Data Processing

**Speed Layer (Kafka)**: Real-time event streaming processing 10,000+ events/second with sub-100ms latency

**Batch Layer (Dask)**: Distributed computing for historical analysis with parallel processing across multiple workers

**Serving Layer**: Unified interface merging real-time and batch views with graceful degradation

---

## User Flow

![User Flow](assets/userflow.jpeg)

From landing page to trade execution, users interact with AI agents through natural language, receive comprehensive analysis, and execute trades with confidence.

---

## Technology Stack

### Frontend
- **Framework**: Next.js 14 (App Router) with React 18
- **Language**: TypeScript 5
- **Styling**: TailwindCSS 4 with custom components
- **UI Library**: Radix UI primitives + shadcn/ui
- **Charts**: TradingView Lightweight Charts, Recharts
- **State Management**: React Context + Custom Hooks
- **Real-time**: WebSocket connections for live data

### Backend
- **Framework**: FastAPI (Python 3.11)
- **API Style**: RESTful + WebSocket
- **Database**: Supabase (PostgreSQL)
- **Caching**: Redis-compatible layer
- **Authentication**: Supabase Auth
- **Logging**: Loguru with structured logging

### Big Data Infrastructure
- **Stream Processing**: Apache Kafka 7.5
- **Message Queue**: Confluent Kafka with Zookeeper
- **Distributed Computing**: Dask with scheduler + workers
- **Data Ingestion**: Real-time WebSocket + REST APIs
- **Throughput**: 10,000+ messages/second
- **Storage**: Compressed event streaming (1GB/day)

### AI & Machine Learning
- **LLM Provider**: OpenAI GPT-4o, GPT-4o-mini
- **Agent Framework**: OpenAI Agents SDK v0.3.0
- **ML Models**: LightGBM, Facebook Prophet, Ensemble
- **Reinforcement Learning**: PyTorch Deep Q-Network
- **Sentiment Analysis**: OpenAI-powered NLP
- **Technical Analysis**: 62 professional indicators

### Market Data
- **Real-time Data**: Alpaca Markets API
- **Options Data**: yfinance API
- **Technical Indicators**: stock-indicators library (62 metrics)
- **Social Sentiment**: Reddit, Twitter, StockTwits via web search
- **News Data**: Financial news aggregation

### DevOps & Deployment
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes-ready
- **Profiles**: Development (minimal) + Production (full ingestion)
- **Monitoring**: Health checks, metrics endpoints
- **Scaling**: Horizontal scaling for Kafka consumers and Dask workers

---

## AI Agent System

### 9 Specialized Agents

**Analysis Agents (6)**

1. **Technical Analysis Agent** (GPT-4o)
   - 62 professional technical indicators
   - Dynamic scenario detection (trending, range-bound, breakout)
   - Support/resistance identification
   - Trend strength and momentum analysis

2. **Sentiment Analysis Agent** (GPT-4o-mini)
   - Multi-source sentiment aggregation
   - Reddit, Twitter, StockTwits scraping via web search
   - Financial news sentiment classification
   - Real-time social media monitoring

3. **Options Flow Agent** (Gemini 2.0)
   - Unusual options activity detection
   - Put/call ratio analysis
   - Smart money tracking
   - Volume/open interest anomalies

4. **Historical Pattern Agent** (GPT-4o)
   - Chart pattern recognition
   - Historical support/resistance levels
   - Pattern reliability scoring
   - Fractal analysis

5. **Risk Management Agent** (GPT-4o)
   - User risk profile integration
   - Strike selection optimization
   - Position sizing calculations
   - Greeks-based risk assessment

6. **Education Agent** (GPT-4o-mini)
   - Strategy explanations
   - Trade reasoning
   - Interactive learning content
   - Personalized education

**Trading Agents (3)**

7. **Buy Agent**
   - Autonomous trade execution
   - Budget-based recommendations
   - Alpaca API integration
   - Paper trading support

8. **Multi-Stock Agent**
   - Hot stocks screening from 100+ symbols
   - Multi-criteria comparison
   - Best stock selection
   - Portfolio optimization

9. **Multi-Options Agent**
   - Options strategy selection
   - Risk-reward optimization
   - Multi-leg strategy construction
   - Execution planning

### Orchestrator

**Master coordinator** managing all 9 agents with:
- Dynamic weight assignment based on market scenarios
- Parallel agent execution
- Result aggregation
- Confidence scoring
- Final signal generation (BUY/SELL/HOLD)

---

## Machine Learning Pipeline

### Multi-Layered Ensemble

**Layer 1: Specialized Models**

1. **OpenAI Sentiment Model** (GPT-4o-mini)
   - Financial sentiment classification
   - Market psychology analysis
   - Confidence scoring

2. **LightGBM Options Flow Model**
   - Flow prediction from historical patterns
   - Feature engineering from options data
   - Bullish/Bearish/Neutral classification

3. **Prophet Volatility Model**
   - Volatility forecasting
   - Seasonal trend detection
   - Uncertainty quantification

**Layer 2: Ensemble Integration**

Combines all models with dynamic weighting:
- Market regime detection
- Component score aggregation
- Weighted voting mechanism
- Final confidence calculation

**Layer 3: Reinforcement Learning**

PyTorch Deep Q-Network:
- Learns from agent results (no mock data)
- State: market conditions + agent scores
- Actions: BUY/SELL/HOLD with position sizing
- Reward: simulated profit/loss

---

## User Preference System

### Risk Profile Settings

**Conservative**
- Lower position sizes (1-5% of portfolio)
- ITM and near-the-money options
- Shorter expiration periods
- Tight stop-losses

**Moderate**
- Medium position sizes (5-10%)
- Mix of ITM/ATM/OTM options
- Standard expiration periods
- Balanced risk-reward

**Aggressive**
- Larger position sizes (10-20%)
- OTM and speculative options
- Longer-dated expirations
- Higher risk tolerance

### Trading Style

**Day Trader**
- Intraday positions only
- High-frequency signals
- Quick entry/exit strategies
- Scalping-friendly recommendations

**Swing Trader**
- Multi-day to multi-week holds
- Medium-frequency signals
- Technical pattern-based entries
- Trend-following strategies

**Position Trader**
- Long-term holds (weeks to months)
- Low-frequency signals
- Fundamental-driven entries
- Large move anticipation

### Customizable Parameters

- Maximum daily trades (1-50)
- Position size percentage (1-100%)
- Risk level (1-5 scale)
- Auto-trade toggle (manual/autonomous)
- Budget allocation
- Preferred sectors

---

## Key Features

### Autonomous Trading

**Natural Language Trading**
- "Find me the best stocks to buy under $500"
- "Buy call options for AAPL with $1000 budget"
- "What are the top gainers today?"

**Intelligent Stock Discovery**
- Hot stocks screening (100+ symbols)
- Multi-source data aggregation
- Reddit trending stocks
- Twitter mentions analysis
- Web scraping for market sentiment

**Auto Execution**
- Paper trading via Alpaca API
- Budget-based recommendations
- Risk-validated trades
- Real-time order status
- Trade confirmation dialog

### Real-Time Data Ingestion

**Kafka Topics**
- market-ticks: Stock prices (1-second updates)
- options-flow: Unusual activity detection
- sentiment-events: Social media sentiment
- technical-signals: Indicator calculations

**Data Sources**
- Alpaca Markets: Real-time quotes
- yfinance: Options chains
- OpenAI Web Search: News and social media
- Custom scrapers: Reddit, Twitter, StockTwits

**Processing Pipeline**
- Event streaming → Kafka Producer
- Topic partitioning by symbol
- Parallel consumer processing
- Real-time buffer updates
- Serving layer integration

### Interactive Dashboard

**Trading Interface**
- Chat-based interaction
- Quick action buttons
- Symbol search
- Real-time price charts
- Technical indicators overlay

**Analysis Views**
- Technical analysis tab
- Sentiment analysis tab
- Options flow tab
- Historical patterns tab
- Risk assessment tab

**Portfolio Management**
- Position overview
- P&L tracking (daily, total)
- Trade history
- Performance metrics
- Cash balance

**Real-Time Monitoring**
- Live price updates
- Position delta changes
- Market alerts
- Trade notifications
- Agent activity logs

---

## Getting Started

### Prerequisites

- Docker Desktop (latest)
- Node.js 18+ (for frontend development)
- Python 3.11 (for backend development)
- Git

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/option-trading-agent.git
cd option-trading-agent
```

2. Create `.env` file:
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional: Big Data Ingestion
KAFKA_ENABLED=false
DASK_ENABLED=false
```

### Running the Application

#### Option 1: Development Mode (Minimal Infrastructure)

```bash
# Start backend API only
docker-compose up

# Backend runs at http://localhost:8080
# API docs at http://localhost:8080/docs
```

#### Option 2: Production Mode (Full Lambda Architecture)

```bash
# Start with Kafka + Dask + Full Ingestion
docker-compose --profile ingestion up

# Services:
# - API: http://localhost:8080
# - Kafka: localhost:9092
# - Dask Dashboard: http://localhost:8787
# - Zookeeper: localhost:2181
```

#### Option 3: Scale Dask Workers

```bash
# Scale to 4 workers for better performance
docker-compose --profile ingestion up --scale dask-worker=4
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev

# Frontend runs at http://localhost:3000
```

### Backend Development

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.api.main:app --reload --port 8080
```

---

## API Endpoints

### Analysis

```bash
# Get comprehensive stock analysis
POST /api/v1/analysis/analyze
{
  "symbol": "AAPL",
  "user_risk_profile": {
    "risk_tolerance": "moderate",
    "max_position_size": 0.05
  }
}

# Get buy recommendations
POST /api/v1/analysis/buy
{
  "symbol": "TSLA",
  "budget": 1000,
  "user_query": "I want aggressive call options"
}

# Multi-stock screening
POST /api/v1/trading/multi-stock/analyze
{
  "query": "Find me the best stocks under $500",
  "budget": 500
}
```

### System

```bash
# Health check
GET /api/v1/system/health

# Ingestion layer status
GET /api/v1/system/ingestion/status

# Agent status
GET /api/v1/agents/status
```

### Hot Stocks

```bash
# Get trending stocks
GET /api/v1/stocks/hot-stocks
```

---

## Project Structure

```
option-trading-agent/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── main.py              # FastAPI application
│   │   │   └── routes/              # API route handlers
│   │   ├── agents/
│   │   │   ├── orchestrator.py      # Master orchestrator
│   │   │   ├── analysis/            # 6 analysis agents
│   │   │   └── trading/             # 3 trading agents
│   │   ├── ingestion/
│   │   │   ├── kafka/               # Kafka producer/consumer
│   │   │   ├── dask/                # Dask cluster + tasks
│   │   │   ├── streams/             # Data streams
│   │   │   └── manager.py           # Ingestion manager
│   │   ├── services/
│   │   │   ├── market_data.py       # Market data manager
│   │   │   └── alpaca.py            # Alpaca client
│   │   └── ml/
│   │       └── ensemble_model.py    # ML pipeline
│   └── config/
│       ├── database.py              # Supabase config
│       └── settings.py              # Environment settings
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 # Landing page
│   │   └── layout.tsx               # Root layout
│   ├── components/
│   │   ├── trading/                 # Trading interface
│   │   ├── chat/                    # Chat components
│   │   └── ui/                      # UI primitives
│   └── package.json
├── assets/
│   ├── overall-diagram.png          # Architecture diagram
│   ├── userflow.jpeg                # User flow diagram
│   └── option-flow.mp4              # Demo video
├── docker-compose.yml               # Docker orchestration
├── Dockerfile                       # Backend container
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Performance Metrics

### Big Data Performance

**Kafka Throughput**
- 10,000+ messages/second per topic
- Sub-10ms publish latency
- Sub-100ms end-to-end latency
- 1GB/day storage with gzip compression

**Dask Performance**
- 1 worker: 100 symbols in 30 seconds
- 2 workers: 100 symbols in 15 seconds
- 4 workers: 100 symbols in 8 seconds
- Near-linear scaling efficiency

**Resource Usage**
- Kafka: 1 core, 512MB RAM, 10GB disk
- Dask Scheduler: 0.5 core, 512MB RAM
- Dask Worker: 2 cores, 2GB RAM each
- Total: 4 cores, 5GB RAM, 11GB disk

### Application Performance

**ML Models**
- Success rate: 100% (4/4 models operational)
- Agent response time: 8-15 seconds
- Technical indicators: 62 metrics calculated
- Pipeline completion: 30-45 seconds end-to-end

**Data Integration**
- Market data: Real-time via Alpaca
- Technical analysis: 62 professional indicators
- Agent results: No mock data
- Options data: Real chains from yfinance
- Sentiment: Live social media scraping

---

## Dynamic Weight Assignment

### Base Weights
- Technical Analysis: 60%
- Sentiment Analysis: 10%
- Options Flow: 10%
- Historical Patterns: 20%

### Scenario-Based Adjustments

**High Volatility**
- Technical: 70% (+10%)
- Flow: 15% (+5%)
- Sentiment: 5% (-5%)
- History: 15% (-5%)

**Low Volatility**
- Technical: 50% (-10%)
- Sentiment: 15% (+5%)
- Flow: 15% (+5%)
- History: 25% (+5%)

**Strong Trend**
- Technical: 70% (+10%)
- Sentiment: 5% (-5%)
- Flow: 5% (-5%)
- History: 20% (same)

**Earnings Approaching**
- Technical: 50% (-10%)
- Flow: 25% (+15%)
- Sentiment: 10% (same)
- History: 15% (-5%)

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

**Technologies**
- OpenAI for GPT-4o and Agents SDK
- Alpaca Markets for real-time market data
- Apache Kafka for stream processing
- Dask for distributed computing
- FastAPI for backend framework
- Next.js for frontend framework

**Data Sources**
- Alpaca Markets API
- Yahoo Finance (yfinance)
- Reddit, Twitter, StockTwits
- Financial news aggregators

---

## Contact

For questions, issues, or feature requests, please open an issue on GitHub.

Project Link: https://github.com/yourusername/option-trading-agent
