# Neural Options Oracle++

AI-driven options trading platform with multi-agent analysis system.

##  Quick Start

### Backend
```bash
cd backend
python3 main.py
# Server: http://localhost:8080
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# App: http://localhost:3000
```

## 📁 Structure

```
option-trading-agent/
├── backend/           # Python backend (FastAPI)
│   ├── app/
│   │   ├── agents/   # 9 AI agents (analysis + trading)
│   │   ├── api/      # REST endpoints
│   │   ├── core/     # Decision engine
│   │   ├── services/ # Market data, Alpaca
│   │   ├── ml/       # ML models
│   │   └── indicators/
│   ├── config/       # Settings
│   └── main.py       # Entry point
├── frontend/         # Next.js frontend
└── logs/            # Application logs
```

## 🤖 AI Agents

**Analysis Agents:**
- Technical Analysis (RSI, MACD, Bollinger Bands)
- Sentiment Analysis (News, social media)
- Options Flow (Unusual activity)
- Historical Patterns
- Education (Learning content)
- Risk Management

**Trading Agents:**
- Buy Agent (Signal execution)
- Multi-Stock Analysis
- Multi-Options Strategy

## 🔧 Configuration

Create `.env` file:
```
OPENAI_API_KEY=your_key
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

## 📊 Features

- Real-time market data (Alpaca API)
- Multi-agent AI analysis
- Paper trading execution
- Interactive 3D visualizations
- Educational content generation
- Risk-based recommendations

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- OpenAI GPT-4
- Pandas, NumPy
- Supabase

**Frontend:**
- Next.js 14
- TypeScript
- Three.js
- Tailwind CSS

---

See `backend/README.md` and `frontend/README.md` for detailed setup.
