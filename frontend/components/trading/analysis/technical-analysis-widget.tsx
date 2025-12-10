"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import { TrendingUp, TrendingDown, Activity, Target, Zap, Shield, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import { type StockData } from "@/lib/api-service"

interface TechnicalAnalysisWidgetProps {
  symbol: string
  stockData: StockData
  className?: string
}

export function TechnicalAnalysisWidget({ symbol, stockData, className }: TechnicalAnalysisWidgetProps) {
  if (!stockData) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">No data available for {symbol}</p>
        </CardContent>
      </Card>
    )
  }

  const { technicals, chartData } = stockData
  const { rsi, macd, movingAverage20, movingAverage50, support, resistance, volumeAvg, bollingerUpper, bollingerLower } = technicals

  // Generate RSI data for chart
  const rsiData = chartData.map((item, index) => ({
    time: item.time,
    rsi: Math.max(0, Math.min(100, 50 + (Math.random() - 0.5) * 20 + (index - 50) * 0.2)),
    price: item.price
  }))

  // Generate MACD data
  const macdData = chartData.map((item, index) => ({
    time: item.time,
    macd: (Math.random() - 0.5) * 2,
    signal: (Math.random() - 0.5) * 1.5,
    histogram: (Math.random() - 0.5) * 1
  }))

  // Generate volume data
  const volumeData = chartData.map(item => ({
    time: item.time,
    volume: item.volume,
    avgVolume: volumeAvg
  }))

  const formatTime = (timeString: string) => {
    const date = new Date(timeString)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getRSIStatus = (rsi: number) => {
    if (rsi > 70) return { status: 'Overbought', color: 'text-red-500', bg: 'bg-red-500/10' }
    if (rsi < 30) return { status: 'Oversold', color: 'text-green-500', bg: 'bg-green-500/10' }
    return { status: 'Neutral', color: 'text-yellow-500', bg: 'bg-yellow-500/10' }
  }

  const getMACDStatus = (macd: string) => {
    switch (macd) {
      case 'bullish': return { status: 'Bullish', color: 'text-green-500', bg: 'bg-green-500/10' }
      case 'bearish': return { status: 'Bearish', color: 'text-red-500', bg: 'bg-red-500/10' }
      default: return { status: 'Neutral', color: 'text-yellow-500', bg: 'bg-yellow-500/10' }
    }
  }

  const rsiStatus = getRSIStatus(rsi)
  const macdStatus = getMACDStatus(macd)

  return (
    <div className={cn("space-y-4", className)}>
      {/* Key Indicators Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Technical Indicators
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* RSI */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">RSI (14)</span>
                <Badge className={cn("text-xs", rsiStatus.bg, rsiStatus.color)}>
                  {rsiStatus.status}
                </Badge>
              </div>
              <div className="text-2xl font-bold">{rsi.toFixed(1)}</div>
              <Progress value={rsi} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0</span>
                <span>50</span>
                <span>100</span>
              </div>
            </div>

            {/* MACD */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">MACD</span>
                <Badge className={cn("text-xs", macdStatus.bg, macdStatus.color)}>
                  {macdStatus.status}
                </Badge>
              </div>
              <div className="text-2xl font-bold capitalize">{macd}</div>
              <div className="text-sm text-muted-foreground">
                Momentum indicator
              </div>
            </div>

            {/* Moving Averages */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">MA Trend</span>
                <Badge variant="outline" className="text-xs">
                  {movingAverage20 > movingAverage50 ? 'Bullish' : 'Bearish'}
                </Badge>
              </div>
              <div className="space-y-1">
                <div className="text-sm">
                  MA20: <span className="font-semibold">${movingAverage20.toFixed(2)}</span>
                </div>
                <div className="text-sm">
                  MA50: <span className="font-semibold">${movingAverage50.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Charts */}
      <Tabs defaultValue="rsi" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="rsi">RSI</TabsTrigger>
          <TabsTrigger value="macd">MACD</TabsTrigger>
          <TabsTrigger value="volume">Volume</TabsTrigger>
        </TabsList>

        <TabsContent value="rsi" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Relative Strength Index (RSI)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rsiData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis 
                      dataKey="time" 
                      tickFormatter={formatTime}
                      className="text-xs"
                    />
                    <YAxis domain={[0, 100]} className="text-xs" />
                    <Tooltip
                      labelFormatter={(value) => `Time: ${formatTime(value as string)}`}
                      formatter={(value: number) => [`${value.toFixed(1)}`, 'RSI']}
                    />
                    <Line
                      type="monotone"
                      dataKey="rsi"
                      stroke="#8884d8"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="70"
                      stroke="#ef4444"
                      strokeDasharray="5 5"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="30"
                      stroke="#22c55e"
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="macd" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">MACD (Moving Average Convergence Divergence)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={macdData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis 
                      dataKey="time" 
                      tickFormatter={formatTime}
                      className="text-xs"
                    />
                    <YAxis className="text-xs" />
                    <Tooltip
                      labelFormatter={(value) => `Time: ${formatTime(value as string)}`}
                      formatter={(value: number, name: string) => [
                        value.toFixed(3), 
                        name === 'macd' ? 'MACD' : name === 'signal' ? 'Signal' : 'Histogram'
                      ]}
                    />
                    <Line
                      type="monotone"
                      dataKey="macd"
                      stroke="#8884d8"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="signal"
                      stroke="#ff7300"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Bar dataKey="histogram" fill="#8884d8" opacity={0.6} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="volume" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Volume Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={volumeData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis 
                      dataKey="time" 
                      tickFormatter={formatTime}
                      className="text-xs"
                    />
                    <YAxis className="text-xs" />
                    <Tooltip
                      labelFormatter={(value) => `Time: ${formatTime(value as string)}`}
                      formatter={(value: number, name: string) => [
                        `${(value / 1000000).toFixed(1)}M`, 
                        name === 'volume' ? 'Volume' : 'Avg Volume'
                      ]}
                    />
                    <Bar dataKey="volume" fill="#8884d8" opacity={0.7} />
                    <Line
                      type="monotone"
                      dataKey="avgVolume"
                      stroke="#ff7300"
                      strokeWidth={2}
                      dot={false}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Support & Resistance Levels */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Key Levels
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <div className="text-sm font-medium text-muted-foreground mb-2">Support Levels</div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 rounded bg-red-500/10">
                    <span className="text-sm">Primary Support</span>
                    <span className="font-semibold text-red-500">${support.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 rounded bg-orange-500/10">
                    <span className="text-sm">Bollinger Lower</span>
                    <span className="font-semibold text-orange-500">${bollingerLower.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="text-sm font-medium text-muted-foreground mb-2">Resistance Levels</div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 rounded bg-green-500/10">
                    <span className="text-sm">Primary Resistance</span>
                    <span className="font-semibold text-green-500">${resistance.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 rounded bg-blue-500/10">
                    <span className="text-sm">Bollinger Upper</span>
                    <span className="font-semibold text-blue-500">${bollingerUpper.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Assessment */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Risk Assessment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-500 mb-1">Medium</div>
              <div className="text-sm text-muted-foreground">Risk Level</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500 mb-1">Good</div>
              <div className="text-sm text-muted-foreground">Liquidity</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500 mb-1">Stable</div>
              <div className="text-sm text-muted-foreground">Volatility</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
