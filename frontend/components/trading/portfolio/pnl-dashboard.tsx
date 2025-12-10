"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts"
import { TrendingUp, TrendingDown, DollarSign, Target, BarChart3, Eye, EyeOff } from "lucide-react"
import { cn } from "@/lib/utils"
import { useState, useEffect } from "react"
import { apiService } from "@/lib/api-service"

interface PnLDashboardProps {
  className?: string
}

export function PnLDashboard({ className }: PnLDashboardProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [portfolio, setPortfolio] = useState({
    totalValue: 25000,
    todayPnL: 0,
    totalPnL: 0,
    cashBalance: 5000,
    positions: []
  })
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    const loadPortfolio = async () => {
      try {
        const portfolioData = await apiService.getPortfolio()
        setPortfolio(portfolioData)
      } catch (error) {
        console.error('Failed to load portfolio:', error)
        // Keep fallback data
      } finally {
        setLoading(false)
      }
    }
    
    loadPortfolio()
  }, [])
  
  // Generate P&L data over time
  const generatePnLData = () => {
    const data = []
    const startValue = 25000
    const currentValue = portfolio.totalValue
    const totalPnL = portfolio.totalPnL
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      
      // Simulate realistic P&L progression
      const baseValue = startValue + (totalPnL * (30 - i) / 30)
      const volatility = (Math.random() - 0.5) * 200
      const value = Math.max(baseValue + volatility, startValue * 0.8)
      
      data.push({
        date: date.toISOString().split('T')[0],
        value: Math.round(value * 100) / 100,
        pnl: Math.round((value - startValue) * 100) / 100,
        pnlPercent: Math.round(((value - startValue) / startValue) * 10000) / 100
      })
    }
    
    return data
  }

  const pnlData = generatePnLData()
  const totalValue = portfolio.totalValue
  const todayPnL = portfolio.todayPnL
  const totalPnL = portfolio.totalPnL
  const pnlPercent = (totalPnL / (totalValue - totalPnL)) * 100
  const isPositive = totalPnL >= 0

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Main P&L Card - Robinhood Style */}
      <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white text-lg">Portfolio</CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
              className="text-slate-300 hover:text-white"
            >
              {showDetails ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Total Value */}
          <div className="text-center">
            <div className="text-3xl font-bold text-white">
              {formatCurrency(totalValue)}
            </div>
            <div className={cn(
              "flex items-center justify-center gap-1 text-sm font-medium",
              isPositive ? "text-green-400" : "text-red-400"
            )}>
              {isPositive ? (
                <TrendingUp className="h-4 w-4" />
              ) : (
                <TrendingDown className="h-4 w-4" />
              )}
              <span>
                {formatCurrency(totalPnL)} ({formatPercent(pnlPercent)})
              </span>
            </div>
          </div>

          {/* P&L Chart */}
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={pnlData}>
                <defs>
                  <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={isPositive ? "#22c55e" : "#ef4444"}
                  strokeWidth={2}
                  fill="url(#pnlGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Today's P&L */}
          <div className="flex justify-between items-center pt-2 border-t border-slate-700">
            <div className="text-slate-300 text-sm">Today's P&L</div>
            <div className={cn(
              "font-semibold",
              todayPnL >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {formatCurrency(todayPnL)}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed View */}
      {showDetails && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Cash Balance */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <DollarSign className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Cash Balance</span>
              </div>
              <div className="text-2xl font-bold">
                {formatCurrency(portfolio.cashBalance)}
              </div>
            </CardContent>
          </Card>

          {/* Active Positions */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Active Positions</span>
              </div>
              <div className="text-2xl font-bold">
                {portfolio.positions.length}
              </div>
              <div className="text-xs text-muted-foreground">
                {portfolio.positions.map(p => p.symbol).join(', ')}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Position Details */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {portfolio.positions.map((position, index) => {
              const isPositionPositive = position.pnl >= 0
              return (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center gap-3">
                    <div>
                      <div className="font-semibold">{position.symbol}</div>
                      <div className="text-sm text-muted-foreground">
                        {position.quantity} shares @ {formatCurrency(position.averageCost)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">
                      {formatCurrency(position.totalValue)}
                    </div>
                    <div className={cn(
                      "text-sm flex items-center gap-1",
                      isPositionPositive ? "text-green-500" : "text-red-500"
                    )}>
                      {isPositionPositive ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      <span>
                        {formatCurrency(position.pnl)} ({formatPercent(position.pnlPercent)})
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}