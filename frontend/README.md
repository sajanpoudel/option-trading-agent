# Frontend

Next.js 14 frontend with interactive trading dashboard.

## 🚀 Run

```bash
cd frontend
npm install
npm run dev
```

App runs on `http://localhost:3000`

## 📁 Structure

```
frontend/
├── src/
│   ├── app/           # Next.js 14 app router
│   ├── components/    # React components
│   ├── lib/           # Utilities
│   └── styles/        # CSS/Tailwind
├── public/            # Static assets
└── package.json
```

## ✨ Features

- **Real-time Dashboard** - Live market data and agent analysis
- **AI Chat Interface** - Natural language trading queries
- **3D Visualizations** - Interactive payoff diagrams (Three.js)
- **Agent Analytics** - Multi-agent decision breakdown
- **Portfolio Tracking** - Real-time P&L monitoring
- **Educational Content** - Interactive learning modules

## 🎨 Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Three.js** - 3D visualizations
- **Framer Motion** - Animations
- **Recharts** - Data visualization

## 🔌 API Integration

Frontend connects to backend at `http://localhost:8080`

```typescript
// Example API call
const response = await fetch('http://localhost:8080/api/analysis', {
  method: 'POST',
  body: JSON.stringify({ symbol: 'AAPL' })
});
```

## 📱 Key Components

- `Dashboard` - Main trading interface
- `AgentAnalysis` - AI agent results display
- `ChatInterface` - Natural language queries
- `PortfolioView` - Position tracking
- `OptionsChain` - Options data table
- `PayoffDiagram` - 3D strategy visualization

## 🛠️ Development

```bash
# Development
npm run dev

# Build
npm run build

# Production
npm start
```
