# Backend

FastAPI backend with AI agent orchestration system.

## 📁 Structure

```
backend/
├── app/
│   ├── agents/
│   │   ├── analysis/      # 6 analysis agents
│   │   └── trading/       # 3 trading agents
│   ├── api/               # FastAPI routes
│   ├── core/              # Decision engine + intent router
│   ├── services/          # Market data + Alpaca
│   ├── ml/                # ML models (ensemble, sentiment, flow, volatility)
│   └── indicators/        # Technical indicators
├── config/                # Settings, database, logging
├── scripts/               # Utility scripts
└── main.py               # Entry point
```

## 🚀 Run

```bash
cd backend
python3 main.py
```

Server runs on `http://localhost:8080`

## 📡 API Endpoints

- `/api/analysis` - Market analysis
- `/api/trading` - Trading operations
- `/api/portfolio` - Portfolio management
- `/api/education` - Educational content
- `/api/system` - Health checks
- `/chat` - AI chat interface

## 🤖 Agent System

**Analysis Agents** (`app/agents/analysis/`):
- `technical.py` - Technical indicators
- `sentiment.py` - Sentiment analysis
- `flow.py` - Options flow
- `historical.py` - Pattern recognition
- `education.py` - Content generation
- `risk.py` - Risk assessment

**Trading Agents** (`app/agents/trading/`):
- `buy.py` - Buy signal execution
- `multi_stock.py` - Multi-stock analysis
- `multi_options.py` - Options strategies

**Orchestrator** (`app/agents/orchestrator.py`):
- Coordinates all agents
- Dynamic weight assignment
- Scenario detection

## 🔧 Configuration

Environment variables in `/.env`:
```
OPENAI_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
SUPABASE_URL=
SUPABASE_KEY=
```

## 📦 Imports

```python
# Import agents
from backend.app.agents import TechnicalAnalysisAgent, BuyAgent
from backend.app.agents.orchestrator import OptionsOracleOrchestrator

# Import services
from backend.app.services.market_data import MarketDataManager

# Import core
from backend.app.core.decision_engine import DecisionEngine
```

## 🧪 Testing

```bash
pytest tests/
```
