# Neural Options Oracle++ - Frontend Architecture

## Frontend Technology Stack

### Core Technologies
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type safety and enhanced developer experience
- **Tailwind CSS**: Utility-first CSS framework
- **Three.js**: 3D visualizations for payoff surfaces
- **Socket.IO Client**: Real-time WebSocket communication
- **Recharts**: Financial charting library
- **Framer Motion**: Animation library
- **React Hook Form**: Form management
- **Zustand**: Lightweight state management

### Supporting Libraries
- **@react-three/fiber**: React renderer for Three.js
- **@react-three/drei**: Three.js helpers
- **lucide-react**: Icon library
- **date-fns**: Date manipulation
- **numeral**: Number formatting
- **react-hot-toast**: Notification system

## Application Architecture

### File Structure
```
frontend/
├── app/                           # Next.js App Router
│   ├── (auth)/                   # Auth route group
│   │   ├── login/
│   │   └── register/
│   ├── dashboard/                # Main dashboard
│   ├── education/                # Educational portal
│   ├── portfolio/                # Portfolio management
│   ├── api/                      # API routes
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/                    # React components
│   ├── ui/                       # Base UI components
│   ├── dashboard/                # Dashboard-specific components
│   ├── education/                # Educational components
│   ├── trading/                  # Trading-related components
│   ├── visualizations/           # 3D visualizations
│   └── layout/                   # Layout components
├── hooks/                        # Custom React hooks
├── lib/                          # Utilities and configurations
├── stores/                       # Zustand stores
├── types/                        # TypeScript type definitions
├── utils/                        # Utility functions
└── public/                       # Static assets
```

## Component Architecture

### UI Component System

#### Base Components (`components/ui/`)
```typescript
// components/ui/button.tsx
import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes, forwardRef } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    return (
      <button
        className={cn(
          "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
          {
            "bg-primary text-primary-foreground hover:bg-primary/90": variant === "default",
            "bg-destructive text-destructive-foreground hover:bg-destructive/90": variant === "destructive",
            "border border-input bg-background hover:bg-accent hover:text-accent-foreground": variant === "outline",
          },
          {
            "h-10 px-4 py-2": size === "default",
            "h-9 rounded-md px-3": size === "sm",
            "h-11 rounded-md px-8": size === "lg",
            "h-10 w-10": size === "icon",
          },
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);

export { Button };
```

#### Card Component
```typescript
// components/ui/card.tsx
import { cn } from "@/lib/utils";
import { HTMLAttributes, forwardRef } from "react";

const Card = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-lg border bg-card text-card-foreground shadow-sm",
        className
      )}
      {...props}
    />
  )
);

const CardHeader = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("flex flex-col space-y-1.5 p-6", className)}
      {...props}
    />
  )
);

const CardTitle = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn(
        "text-2xl font-semibold leading-none tracking-tight",
        className
      )}
      {...props}
    />
  )
);

export { Card, CardHeader, CardTitle };
```

### Dashboard Components (`components/dashboard/`)

#### Main Trading Dashboard
```typescript
// components/dashboard/TradingDashboard.tsx
'use client';

import { useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAgentData } from '@/hooks/useAgentData';
import { usePositions } from '@/hooks/usePositions';
import { AgentAnalysisPanel } from './AgentAnalysisPanel';
import { DecisionEngineViz } from './DecisionEngineViz';
import { RealTimeMonitoring } from './RealTimeMonitoring';
import { StockSelector } from './StockSelector';
import { StrikeRecommendations } from './StrikeRecommendations';
import { PayoffSurface3D } from '@/components/visualizations/PayoffSurface3D';

export function TradingDashboard() {
  const { socket, isConnected } = useWebSocket();
  const { agentResults, requestAnalysis } = useAgentData();
  const { positions, portfolioSummary } = usePositions();

  useEffect(() => {
    if (socket) {
      socket.emit('subscribe', {
        channels: ['market_data', 'positions', 'analysis'],
        symbols: ['AAPL', 'MSFT', 'TSLA']
      });
    }
  }, [socket]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-blue-900 p-4">
      {/* Header */}
      <header className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">
          🧠 Neural Options Oracle++
        </h1>
        <p className="text-slate-300">
          AI-Powered Options Trading Intelligence
        </p>
        <div className="mt-2 flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-slate-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Stock Selection */}
      <div className="grid grid-cols-12 gap-4 mb-6">
        <div className="col-span-8">
          <StockSelector onStockSelect={requestAnalysis} />
        </div>
        <div className="col-span-4">
          <RiskProfileSelector />
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-12 gap-4">
        {/* Agent Analysis */}
        <div className="col-span-8 space-y-4">
          <AgentAnalysisPanel results={agentResults} />
          <DecisionEngineViz 
            signal={agentResults?.signal}
            weights={agentResults?.adjustedWeights}
          />
        </div>

        {/* 3D Payoff Visualization */}
        <div className="col-span-4 bg-slate-800 rounded-lg p-4 h-96">
          <h3 className="text-xl font-bold text-white mb-4">
            Strategy Payoff 3D
          </h3>
          <PayoffSurface3D positions={positions} />
        </div>

        {/* Real-time Monitoring */}
        <div className="col-span-6 bg-slate-800 rounded-lg p-4">
          <RealTimeMonitoring 
            positions={positions}
            portfolioSummary={portfolioSummary}
          />
        </div>

        {/* Educational Module */}
        <div className="col-span-6 bg-slate-800 rounded-lg p-4">
          <EducationalModule 
            signal={agentResults?.signal}
            positions={positions}
          />
        </div>

        {/* Strike Recommendations */}
        <div className="col-span-12 bg-slate-800 rounded-lg p-4">
          <StrikeRecommendations 
            recommendations={agentResults?.strikeRecommendations}
          />
        </div>
      </div>
    </div>
  );
}
```

#### Agent Analysis Panel
```typescript
// components/dashboard/AgentAnalysisPanel.tsx
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { AgentResults } from '@/types/trading';
import { TrendingUp, TrendingDown, Activity, Brain } from 'lucide-react';

interface AgentAnalysisPanelProps {
  results: AgentResults | null;
}

export function AgentAnalysisPanel({ results }: AgentAnalysisPanelProps) {
  if (!results) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Agent Analysis</CardTitle>
          <p className="text-slate-500">Select a stock to begin analysis</p>
        </CardHeader>
      </Card>
    );
  }

  const agents = [
    {
      name: 'Technical',
      weight: 60,
      data: results.technical,
      icon: TrendingUp,
      color: 'blue'
    },
    {
      name: 'Sentiment',
      weight: 10,
      data: results.sentiment,
      icon: Brain,
      color: 'purple'
    },
    {
      name: 'Flow',
      weight: 10,
      data: results.flow,
      icon: Activity,
      color: 'orange'
    },
    {
      name: 'History',
      weight: 20,
      data: results.history,
      icon: TrendingDown,
      color: 'green'
    }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-6 h-6" />
          AI Agent Analysis
        </CardTitle>
      </CardHeader>
      
      <div className="grid grid-cols-4 gap-4 p-6">
        {agents.map((agent) => (
          <div key={agent.name} className="bg-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <agent.icon className={`w-5 h-5 text-${agent.color}-400`} />
              <h3 className="font-semibold text-white">
                {agent.name} ({agent.weight}%)
              </h3>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-300">Score:</span>
                <span className={`font-bold ${
                  agent.data?.score > 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {agent.data?.score?.toFixed(2) || 'N/A'}
                </span>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-slate-300">Confidence:</span>
                <span className="text-white">
                  {((agent.data?.confidence || 0) * 100).toFixed(0)}%
                </span>
              </div>
              
              <Progress 
                value={(agent.data?.confidence || 0) * 100}
                className="h-2"
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
```

### Visualization Components (`components/visualizations/`)

#### 3D Payoff Surface
```typescript
// components/visualizations/PayoffSurface3D.tsx
'use client';

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { useRef, useMemo } from 'react';
import { Mesh, PlaneGeometry, MeshStandardMaterial } from 'three';
import { Position } from '@/types/trading';

interface PayoffSurface3DProps {
  positions: Position[];
}

function PayoffMesh({ positions }: { positions: Position[] }) {
  const meshRef = useRef<Mesh>(null);
  
  const geometry = useMemo(() => {
    const geo = new PlaneGeometry(10, 10, 50, 50);
    const vertices = geo.attributes.position.array as Float32Array;
    
    // Calculate payoff for each vertex
    for (let i = 0; i < vertices.length; i += 3) {
      const stockPrice = (vertices[i] + 5) * 20; // Map to stock price range
      const payoff = calculatePortfolioPayoff(positions, stockPrice);
      vertices[i + 2] = payoff / 1000; // Scale for visualization
    }
    
    geo.attributes.position.needsUpdate = true;
    geo.computeVertexNormals();
    
    return geo;
  }, [positions]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.1) * 0.1;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        color="#ff69b4"
        wireframe
        transparent
        opacity={0.8}
      />
    </mesh>
  );
}

function calculatePortfolioPayoff(positions: Position[], stockPrice: number): number {
  return positions.reduce((totalPayoff, position) => {
    if (position.optionType === 'call') {
      const intrinsicValue = Math.max(0, stockPrice - position.strikePrice);
      return totalPayoff + (intrinsicValue - position.entryPrice) * position.quantity;
    } else if (position.optionType === 'put') {
      const intrinsicValue = Math.max(0, position.strikePrice - stockPrice);
      return totalPayoff + (intrinsicValue - position.entryPrice) * position.quantity;
    }
    return totalPayoff;
  }, 0);
}

export function PayoffSurface3D({ positions }: PayoffSurface3DProps) {
  if (!positions || positions.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400">
        No positions to visualize
      </div>
    );
  }

  return (
    <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <PayoffMesh positions={positions} />
      <Grid 
        args={[20, 20]} 
        position={[0, -2, 0]} 
        cellSize={1} 
        cellThickness={0.5} 
        cellColor="#334155"
        sectionSize={5} 
        sectionThickness={1} 
        sectionColor="#475569"
      />
      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
    </Canvas>
  );
}
```

### Real-time Monitoring Component

```typescript
// components/dashboard/RealTimeMonitoring.tsx
import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useWebSocket } from '@/hooks/useWebSocket';
import { Position, PortfolioSummary } from '@/types/trading';
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react';

interface RealTimeMonitoringProps {
  positions: Position[];
  portfolioSummary: PortfolioSummary;
}

export function RealTimeMonitoring({ positions, portfolioSummary }: RealTimeMonitoringProps) {
  const { socket } = useWebSocket();
  const [realtimePnL, setRealtimePnL] = useState(portfolioSummary?.totalPnL || 0);

  useEffect(() => {
    if (socket) {
      socket.on('position_update', (data) => {
        setRealtimePnL(prev => prev + data.pnl_change);
      });

      return () => {
        socket.off('position_update');
      };
    }
  }, [socket]);

  const pnlPercentage = portfolioSummary?.totalValue 
    ? (realtimePnL / portfolioSummary.totalValue) * 100 
    : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-6 h-6" />
          Real-time Monitoring
        </CardTitle>
      </CardHeader>

      <div className="p-6 space-y-4">
        {/* Portfolio Summary */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="w-5 h-5 text-green-400" />
              <h3 className="font-semibold text-white">Total Value</h3>
            </div>
            <p className="text-2xl font-bold text-white">
              ${portfolioSummary?.totalValue?.toLocaleString() || '0'}
            </p>
          </div>

          <div className="bg-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              {pnlPercentage >= 0 ? (
                <TrendingUp className="w-5 h-5 text-green-400" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-400" />
              )}
              <h3 className="font-semibold text-white">P&L</h3>
            </div>
            <p className={`text-2xl font-bold ${
              realtimePnL >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              ${realtimePnL.toLocaleString()} ({pnlPercentage.toFixed(2)}%)
            </p>
          </div>
        </div>

        {/* Greeks Summary */}
        <div className="bg-slate-700 rounded-lg p-4">
          <h3 className="font-semibold text-white mb-3">Portfolio Greeks</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-slate-300">Delta:</span>
              <span className="float-right text-white">
                {portfolioSummary?.portfolioGreeks?.delta?.toFixed(2) || '0.00'}
              </span>
            </div>
            <div>
              <span className="text-slate-300">Gamma:</span>
              <span className="float-right text-white">
                {portfolioSummary?.portfolioGreeks?.gamma?.toFixed(2) || '0.00'}
              </span>
            </div>
            <div>
              <span className="text-slate-300">Theta:</span>
              <span className="float-right text-white">
                {portfolioSummary?.portfolioGreeks?.theta?.toFixed(2) || '0.00'}
              </span>
            </div>
            <div>
              <span className="text-slate-300">Vega:</span>
              <span className="float-right text-white">
                {portfolioSummary?.portfolioGreeks?.vega?.toFixed(2) || '0.00'}
              </span>
            </div>
          </div>
        </div>

        {/* Active Positions */}
        <div className="bg-slate-700 rounded-lg p-4">
          <h3 className="font-semibold text-white mb-3">Active Positions</h3>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {positions.map((position) => (
              <div key={position.id} className="flex justify-between items-center py-2 border-b border-slate-600 last:border-b-0">
                <div>
                  <span className="text-white font-medium">{position.symbol}</span>
                  <span className="text-slate-300 ml-2">
                    {position.strikePrice}C {position.expirationDate}
                  </span>
                </div>
                <div className="text-right">
                  <div className={`font-semibold ${
                    position.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    ${position.unrealizedPnL.toFixed(2)}
                  </div>
                  <div className="text-sm text-slate-400">
                    {position.unrealizedPnLPercent.toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}
```

## State Management

### Zustand Stores (`stores/`)

#### Trading Store
```typescript
// stores/tradingStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { Position, TradingSignal, PortfolioSummary } from '@/types/trading';

interface TradingState {
  positions: Position[];
  signals: TradingSignal[];
  portfolioSummary: PortfolioSummary | null;
  selectedSymbol: string | null;
  
  // Actions
  setPositions: (positions: Position[]) => void;
  updatePosition: (positionId: string, updates: Partial<Position>) => void;
  addSignal: (signal: TradingSignal) => void;
  setPortfolioSummary: (summary: PortfolioSummary) => void;
  setSelectedSymbol: (symbol: string | null) => void;
}

export const useTradingStore = create<TradingState>()(
  devtools(
    (set, get) => ({
      positions: [],
      signals: [],
      portfolioSummary: null,
      selectedSymbol: null,

      setPositions: (positions) => set({ positions }),
      
      updatePosition: (positionId, updates) => set((state) => ({
        positions: state.positions.map((position) =>
          position.id === positionId ? { ...position, ...updates } : position
        )
      })),
      
      addSignal: (signal) => set((state) => ({
        signals: [signal, ...state.signals.slice(0, 99)] // Keep last 100 signals
      })),
      
      setPortfolioSummary: (summary) => set({ portfolioSummary: summary }),
      
      setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol })
    }),
    { name: 'trading-store' }
  )
);
```

## Custom Hooks

### WebSocket Hook
```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuthStore } from '@/stores/authStore';
import { useTradingStore } from '@/stores/tradingStore';

export function useWebSocket() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const { token } = useAuthStore();
  const { updatePosition, addSignal } = useTradingStore();

  useEffect(() => {
    if (!token) return;

    const newSocket = io(process.env.NEXT_PUBLIC_WS_URL!, {
      auth: { token },
      transports: ['websocket']
    });

    newSocket.on('connect', () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    });

    newSocket.on('position_update', (data) => {
      updatePosition(data.position_id, {
        currentPrice: data.current_price,
        unrealizedPnL: data.unrealized_pnl,
        unrealizedPnLPercent: data.unrealized_pnl_percent,
        greeks: data.greeks
      });
    });

    newSocket.on('signal_generated', (data) => {
      addSignal(data);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [token]);

  return { socket, isConnected };
}
```

### Agent Data Hook
```typescript
// hooks/useAgentData.ts
import { useState, useCallback } from 'react';
import { AgentResults } from '@/types/trading';
import { apiClient } from '@/lib/apiClient';

export function useAgentData() {
  const [agentResults, setAgentResults] = useState<AgentResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestAnalysis = useCallback(async (symbol: string, options?: {
    timeframe?: string;
    analysisDepth?: 'quick' | 'standard' | 'full';
    includeEducation?: boolean;
  }) => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiClient.post(`/api/v1/analysis/stock/${symbol}`, {
        timeframe: options?.timeframe || '1D',
        analysis_depth: options?.analysisDepth || 'standard',
        include_education: options?.includeEducation ?? true
      });

      setAgentResults(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    agentResults,
    loading,
    error,
    requestAnalysis
  };
}
```

## Type Definitions

### Trading Types
```typescript
// types/trading.ts
export interface Position {
  id: string;
  symbol: string;
  optionSymbol?: string;
  positionType: 'stock' | 'option';
  optionType?: 'call' | 'put';
  strikePrice: number;
  expirationDate: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  greeks?: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
  status: 'open' | 'closed' | 'expired';
  entryDate: string;
}

export interface TradingSignal {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL' | 'HOLD' | 'STRONG_BUY' | 'STRONG_SELL';
  strength: 'weak' | 'moderate' | 'strong';
  confidence: number;
  timestamp: string;
  reasoning: string;
}

export interface AgentResults {
  technical: {
    scenario: string;
    indicators: Record<string, any>;
    weights: Record<string, number>;
    score: number;
    confidence: number;
  };
  sentiment: {
    score: number;
    confidence: number;
    sources: Record<string, any>;
  };
  flow: {
    score: number;
    confidence: number;
    metrics: Record<string, any>;
  };
  history: {
    score: number;
    confidence: number;
    pattern: string;
  };
  signal: TradingSignal;
  strikeRecommendations: StrikeRecommendation[];
  adjustedWeights: Record<string, number>;
}

export interface StrikeRecommendation {
  rank: number;
  strike: number;
  expiration: string;
  optionType: 'call' | 'put';
  delta: number;
  probabilityOfProfit: number;
  maxProfit: number;
  maxLoss: number;
  riskRewardRatio: number;
  cost: number;
  reasoning: string;
}

export interface PortfolioSummary {
  totalPositions: number;
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  portfolioGreeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
}
```

This comprehensive frontend architecture provides a modern, type-safe, and scalable foundation for the Neural Options Oracle++ user interface with real-time capabilities, 3D visualizations, and educational components.