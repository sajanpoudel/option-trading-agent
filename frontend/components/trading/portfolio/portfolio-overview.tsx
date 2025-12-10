"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { XAxis, YAxis, ResponsiveContainer, Area, AreaChart } from "recharts"
import { TrendingUp, TrendingDown, Activity, Eye, EyeOff } from "lucide-react"

interface Position {
  symbol: string
  quantity: number
  avgCost: number
  currentPrice: number
  marketValue: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  dayChange: number
  dayChangePercent: number
}

interface PortfolioData {
  totalValue: number
  totalPnL: number
  totalPnLPercent: number
  dayChange: number
  dayChangePercent: number
  buyingPower: number
  positions: Position[]
}

const mockPortfolioHistory = [
  { time: "9:30", value: 10000, pnl: 0 },
  { time: "10:00", value: 10150, pnl: 150 },
  { time: "10:30", value: 10080, pnl: 80 },
  { time: "11:00", value: 10220, pnl: 220 },
  { time: "11:30", value: 10180, pnl: 180 },
  { time: "12:00", value: 10350, pnl: 350 },
  { time: "12:30", value: 10290, pnl: 290 },
  { time: "1:00", value: 10420, pnl: 420 },
  { time: "1:30", value: 10380, pnl: 380 },
  { time: "2:00", value: 10510, pnl: 510 },
  { time: "2:30", value: 10470, pnl: 470 },
  { time: "3:00", value: 10580, pnl: 580 },
  { time: "3:30", value: 10520, pnl: 520 },
  { time: "4:00", value: 10650, pnl: 650 },
]

export function PortfolioOverview() {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null)
  const [isBalanceVisible, setIsBalanceVisible] = useState(true)
  const [selectedTimeframe, setSelectedTimeframe] = useState("1D")

  useEffect(() => {
    const mockData: PortfolioData = {
      totalValue: 10650.0,
      totalPnL: 650.0,
      totalPnLPercent: 6.5,
      dayChange: 420.5,
      dayChangePercent: 4.1,
      buyingPower: 2500.0,
      positions: [
        {
          symbol: "AAPL",
          quantity: 10,
          avgCost: 150.0,
          currentPrice: 165.5,
          marketValue: 1655.0,
          unrealizedPnL: 155.0,
          unrealizedPnLPercent: 10.33,
          dayChange: 25.5,
          dayChangePercent: 1.57,
        },
        {
          symbol: "TSLA",
          quantity: 5,
          avgCost: 220.0,
          currentPrice: 245.8,
          marketValue: 1229.0,
          unrealizedPnL: 129.0,
          unrealizedPnLPercent: 11.73,
          dayChange: 45.8,
          dayChangePercent: 3.87,
        },
        {
          symbol: "NVDA",
          quantity: 8,
          avgCost: 180.0,
          currentPrice: 195.25,
          marketValue: 1562.0,
          unrealizedPnL: 122.0,
          unrealizedPnLPercent: 8.47,
          dayChange: 38.25,
          dayChangePercent: 2.44,
        },
      ],
    }
    setPortfolioData(mockData)
  }, [])

  if (!portfolioData) return null

  const formatCurrency = (amount: number) => {
    if (!isBalanceVisible) return "••••••"
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(amount)
  }

  const formatPercent = (percent: number) => {
    if (!isBalanceVisible) return "••••"
    return `${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%`
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Header */}
      <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 border-green-200 dark:border-green-800">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Portfolio</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => setIsBalanceVisible(!isBalanceVisible)}>
              {isBalanceVisible ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Total Value */}
          <div>
            <div className="text-3xl font-bold text-green-600 dark:text-green-400">
              {formatCurrency(portfolioData.totalValue)}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div
                className={`flex items-center gap-1 ${
                  portfolioData.dayChange >= 0 ? "text-green-600" : "text-red-600"
                }`}
              >
                {portfolioData.dayChange >= 0 ? (
                  <TrendingUp className="h-4 w-4" />
                ) : (
                  <TrendingDown className="h-4 w-4" />
                )}
                <span className="font-medium">{formatCurrency(Math.abs(portfolioData.dayChange))}</span>
                <span className="text-sm">({formatPercent(portfolioData.dayChangePercent)})</span>
              </div>
              <span className="text-sm text-muted-foreground">Today</span>
            </div>
          </div>

          {/* Timeframe Selector */}
          <div className="flex gap-2 pt-2">
            {["1D", "1W", "1M", "3M", "1Y", "ALL"].map((timeframe) => (
              <Button
                key={timeframe}
                variant={selectedTimeframe === timeframe ? "default" : "ghost"}
                size="sm"
                onClick={() => setSelectedTimeframe(timeframe)}
                className="px-3 py-1 h-8"
              >
                {timeframe}
              </Button>
            ))}
          </div>

          {/* Portfolio Chart */}
          <div className="h-48 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockPortfolioHistory}>
                <defs>
                  <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="time" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#6b7280', fontSize: 12 }}
                  dy={10}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#6b7280', fontSize: 12 }}
                  dx={-10}
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#10b981" 
                  strokeWidth={2} 
                  fill="url(#portfolioGradient)"
                  animationDuration={500}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4 pt-2">
            <div className="text-center">
              <div className="text-sm text-muted-foreground">Total P&L</div>
              <div className={`font-semibold ${portfolioData.totalPnL >= 0 ? "text-green-600" : "text-red-600"}`}>
                {formatCurrency(portfolioData.totalPnL)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-muted-foreground">Buying Power</div>
              <div className="font-semibold">{formatCurrency(portfolioData.buyingPower)}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-muted-foreground">Positions</div>
              <div className="font-semibold">{portfolioData.positions.length}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Positions List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Positions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {portfolioData.positions.map((position) => (
              <div
                key={position.symbol}
                className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div>
                    <div className="font-semibold">{position.symbol}</div>
                    <div className="text-sm text-muted-foreground">{position.quantity} shares</div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="font-semibold">{formatCurrency(position.marketValue)}</div>
                  <div
                    className={`text-sm flex items-center gap-1 justify-end ${
                      position.unrealizedPnL >= 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {position.unrealizedPnL >= 0 ? (
                      <TrendingUp className="h-3 w-3" />
                    ) : (
                      <TrendingDown className="h-3 w-3" />
                    )}
                    {formatCurrency(Math.abs(position.unrealizedPnL))}
                    <span className="text-xs">({formatPercent(position.unrealizedPnLPercent)})</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
