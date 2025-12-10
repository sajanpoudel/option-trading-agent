"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts"
import { Activity, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react"

interface Position {
  id: string
  symbol: string
  optionType: "Call" | "Put"
  strike: number
  expiry: string
  quantity: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercent: number
  delta: number
  gamma: number
  theta: number
  vega: number
  riskLevel: "Low" | "Medium" | "High"
}

interface PnLDataPoint {
  time: string
  pnl: number
  timestamp: number
}

export function RealTimeMonitoring({ selectedStock }: { selectedStock: string }) {
  const [positions, setPositions] = useState<Position[]>([])
  const [pnlHistory, setPnlHistory] = useState<PnLDataPoint[]>([])
  const [totalPnL, setTotalPnL] = useState(0)
  const [portfolioGreeks, setPortfolioGreeks] = useState({
    delta: 0,
    gamma: 0,
    theta: 0,
    vega: 0,
  })

  // Generate mock positions
  const generateMockPositions = (): Position[] => [
    {
      id: "1",
      symbol: selectedStock,
      optionType: "Call",
      strike: 180,
      expiry: "2024-01-19",
      quantity: 5,
      entryPrice: 3.2,
      currentPrice: 3.8,
      pnl: (3.8 - 3.2) * 5 * 100,
      pnlPercent: ((3.8 - 3.2) / 3.2) * 100,
      delta: 0.65,
      gamma: 0.08,
      theta: -0.12,
      vega: 0.25,
      riskLevel: "Medium",
    },
    {
      id: "2",
      symbol: selectedStock,
      optionType: "Put",
      strike: 170,
      expiry: "2024-01-26",
      quantity: -3,
      entryPrice: 2.1,
      currentPrice: 1.8,
      pnl: (2.1 - 1.8) * 3 * 100,
      pnlPercent: ((2.1 - 1.8) / 2.1) * 100,
      delta: -0.35,
      gamma: 0.06,
      theta: -0.08,
      vega: 0.18,
      riskLevel: "Low",
    },
    {
      id: "3",
      symbol: selectedStock,
      optionType: "Call",
      strike: 190,
      expiry: "2024-01-12",
      quantity: -2,
      entryPrice: 1.8,
      currentPrice: 2.3,
      pnl: (1.8 - 2.3) * 2 * 100,
      pnlPercent: ((1.8 - 2.3) / 1.8) * 100,
      delta: -0.45,
      gamma: -0.05,
      theta: 0.09,
      vega: -0.15,
      riskLevel: "High",
    },
  ]

  // Simulate real-time updates
  useEffect(() => {
    const mockPositions = generateMockPositions()
    setPositions(mockPositions)

    const interval = setInterval(() => {
      setPositions((prevPositions) =>
        prevPositions.map((position) => {
          const priceChange = (Math.random() - 0.5) * 0.2
          const newCurrentPrice = Math.max(0.1, position.currentPrice + priceChange)
          const newPnL = (newCurrentPrice - position.entryPrice) * position.quantity * 100
          const newPnLPercent = ((newCurrentPrice - position.entryPrice) / position.entryPrice) * 100

          return {
            ...position,
            currentPrice: newCurrentPrice,
            pnl: newPnL,
            pnlPercent: newPnLPercent,
            delta: position.delta + (Math.random() - 0.5) * 0.02,
            gamma: position.gamma + (Math.random() - 0.5) * 0.005,
            theta: position.theta + (Math.random() - 0.5) * 0.01,
            vega: position.vega + (Math.random() - 0.5) * 0.01,
          }
        }),
      )
    }, 2000)

    return () => clearInterval(interval)
  }, [selectedStock])

  // Update P&L history and portfolio Greeks
  useEffect(() => {
    const newTotalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0)
    setTotalPnL(newTotalPnL)

    // Add to P&L history
    const now = new Date()
    const timeString = now.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })

    setPnlHistory((prev) => {
      const newHistory = [
        ...prev,
        {
          time: timeString,
          pnl: newTotalPnL,
          timestamp: now.getTime(),
        },
      ].slice(-20) // Keep last 20 data points
      return newHistory
    })

    // Calculate portfolio Greeks
    const newPortfolioGreeks = positions.reduce(
      (acc, pos) => ({
        delta: acc.delta + pos.delta * pos.quantity,
        gamma: acc.gamma + pos.gamma * pos.quantity,
        theta: acc.theta + pos.theta * pos.quantity,
        vega: acc.vega + pos.vega * pos.quantity,
      }),
      { delta: 0, gamma: 0, theta: 0, vega: 0 },
    )
    setPortfolioGreeks(newPortfolioGreeks)
  }, [positions])

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? "text-green-400" : "text-red-400"
  }

  const getRiskBadgeVariant = (risk: string) => {
    switch (risk) {
      case "High":
        return "destructive"
      case "Medium":
        return "secondary"
      default:
        return "outline"
    }
  }

  const getGreekGaugeColor = (value: number, type: string) => {
    if (type === "theta") return value < -0.5 ? "bg-red-500" : value < -0.2 ? "bg-yellow-500" : "bg-green-500"
    if (type === "delta")
      return Math.abs(value) > 100 ? "bg-red-500" : Math.abs(value) > 50 ? "bg-yellow-500" : "bg-green-500"
    return "bg-blue-500"
  }

  const getGreekGaugeValue = (value: number, type: string) => {
    if (type === "delta") return Math.min(Math.abs(value), 200) / 2
    if (type === "theta") return Math.min(Math.abs(value), 2) * 50
    return Math.min(Math.abs(value), 1) * 100
  }

  return (
    <Card className="col-span-12 md:col-span-6 lg:col-span-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Real-Time Monitoring
          <Badge variant={totalPnL >= 0 ? "default" : "destructive"} className="ml-auto">
            {totalPnL >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}$
            {totalPnL.toFixed(0)}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Active Positions Table */}
        <div className="space-y-2">
          <h4 className="font-medium text-sm">Active Positions</h4>
          <div className="border rounded-lg max-h-48 overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-16">Type</TableHead>
                  <TableHead className="w-16">Strike</TableHead>
                  <TableHead className="w-20">P&L</TableHead>
                  <TableHead className="w-16">Delta</TableHead>
                  <TableHead className="w-16">Risk</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.id} className="text-xs">
                    <TableCell>
                      <Badge variant={position.optionType === "Call" ? "default" : "secondary"} className="text-xs">
                        {position.optionType}
                      </Badge>
                    </TableCell>
                    <TableCell>${position.strike}</TableCell>
                    <TableCell>
                      <div className="flex flex-col">
                        <span className={getPnLColor(position.pnl)}>${position.pnl.toFixed(0)}</span>
                        <span className={`text-xs ${getPnLColor(position.pnl)}`}>
                          {position.pnlPercent > 0 ? "+" : ""}
                          {position.pnlPercent.toFixed(1)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className={position.delta > 0 ? "text-green-400" : "text-red-400"}>
                        {position.delta.toFixed(2)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getRiskBadgeVariant(position.riskLevel)} className="text-xs">
                        {position.riskLevel}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>

        {/* Real-time P&L Chart */}
        <div className="space-y-2">
          <h4 className="font-medium text-sm">P&L Trend</h4>
          <div className="h-32 border rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pnlHistory}>
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload[0]) {
                      const value = payload[0].value as number
                      return (
                        <div className="bg-card border rounded-lg p-2 shadow-lg">
                          <p className="text-sm font-medium">Time: {label}</p>
                          <p className={`text-sm ${getPnLColor(value)}`}>P&L: ${value.toFixed(2)}</p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="pnl"
                  stroke={totalPnL >= 0 ? "#4CAF50" : "#F44336"}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Portfolio Greeks Gauges */}
        <div className="space-y-2">
          <h4 className="font-medium text-sm">Portfolio Greeks</h4>
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(portfolioGreeks).map(([greek, value]) => (
              <div key={greek} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="capitalize">{greek}</span>
                  <span className={value > 0 ? "text-green-400" : "text-red-400"}>{value.toFixed(2)}</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${getGreekGaugeColor(value, greek)}`}
                    style={{ width: `${getGreekGaugeValue(value, greek)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Alerts */}
        <div className="border-t pt-3">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="h-4 w-4 text-yellow-400" />
            <h4 className="font-medium text-sm">Risk Alerts</h4>
          </div>
          <div className="space-y-1 text-xs">
            {Math.abs(portfolioGreeks.delta) > 100 && (
              <div className="flex items-center gap-2 text-red-400">
                <div className="w-1 h-1 rounded-full bg-red-400" />
                High Delta Exposure: {portfolioGreeks.delta.toFixed(1)}
              </div>
            )}
            {portfolioGreeks.theta < -1 && (
              <div className="flex items-center gap-2 text-yellow-400">
                <div className="w-1 h-1 rounded-full bg-yellow-400" />
                High Time Decay: {portfolioGreeks.theta.toFixed(2)}
              </div>
            )}
            {positions.filter((p) => p.riskLevel === "High").length > 0 && (
              <div className="flex items-center gap-2 text-orange-400">
                <div className="w-1 h-1 rounded-full bg-orange-400" />
                {positions.filter((p) => p.riskLevel === "High").length} High Risk Position(s)
              </div>
            )}
            {Math.abs(portfolioGreeks.delta) <= 50 &&
              portfolioGreeks.theta > -0.5 &&
              positions.filter((p) => p.riskLevel === "High").length === 0 && (
                <div className="flex items-center gap-2 text-green-400">
                  <div className="w-1 h-1 rounded-full bg-green-400" />
                  Portfolio within risk parameters
                </div>
              )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
