"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ScatterChart, Scatter, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip } from "recharts"
import { TrendingUp, TrendingDown, Zap, Eye, Play, AlertCircle } from "lucide-react"

interface TradingSignal {
  id: string
  symbol: string
  direction: "BUY" | "SELL"
  confidence: number
  strike: number
  expiration: string
  entryPrice: number
  exitRules: string
  timestamp: Date
  positionSize: number
  reasoning: string
  riskReward: string
}

export function TradingSignals({ selectedStock }: { selectedStock: string }) {
  const [signals, setSignals] = useState<TradingSignal[]>([])
  const [selectedSignal, setSelectedSignal] = useState<TradingSignal | null>(null)
  const [newSignalAnimation, setNewSignalAnimation] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // Fetch real trading signals from backend
  const fetchTradingSignals = async () => {
    try {
      setLoading(true)
      const response = await fetch(`http://localhost:8080/api/v1/signals/${selectedStock}`)
      if (response.ok) {
        const data = await response.json()
        const fetchedSignals = data.signals || []
        
        // Transform backend signals to frontend format
        const transformedSignals: TradingSignal[] = fetchedSignals.map((signal: any) => ({
          id: signal.id,
          symbol: signal.symbol,
          direction: signal.direction,
          confidence: signal.confidence,
          strike: signal.strike,
          expiration: signal.expiration,
          entryPrice: signal.entryPrice,
          exitRules: signal.exitRules,
          timestamp: new Date(signal.timestamp),
          positionSize: signal.positionSize,
          reasoning: signal.reasoning,
          riskReward: signal.riskReward
        }))
        
        setSignals(transformedSignals)
      } else {
        console.error('Failed to fetch trading signals:', response.status)
        setSignals([])
      }
    } catch (error) {
      console.error('Error fetching trading signals:', error)
      setSignals([])
    } finally {
      setLoading(false)
    }
  }

  // Fallback mock signals only for demonstration
  const generateMockSignals = (): TradingSignal[] => [
    {
      id: "1",
      symbol: selectedStock,
      direction: "BUY",
      confidence: 85,
      strike: 185,
      expiration: "2024-01-19",
      entryPrice: 3.2,
      exitRules: "Take profit at 50% or theta decay",
      timestamp: new Date(),
      positionSize: 5,
      reasoning:
        "Strong technical breakout with unusual call flow and positive sentiment convergence. RSI showing bullish divergence.",
      riskReward: "1:2.5",
    },
    {
      id: "2",
      symbol: selectedStock,
      direction: "BUY",
      confidence: 72,
      strike: 190,
      expiration: "2024-01-26",
      entryPrice: 2.85,
      exitRules: "Stop loss at 30% or support break",
      timestamp: new Date(Date.now() - 300000),
      positionSize: 3,
      reasoning:
        "Earnings momentum play with historical seasonal strength. Options flow showing smart money positioning.",
      riskReward: "1:3.0",
    },
    {
      id: "3",
      symbol: selectedStock,
      direction: "SELL",
      confidence: 68,
      strike: 175,
      expiration: "2024-01-12",
      entryPrice: 1.95,
      exitRules: "Cover at 25% profit or time decay",
      timestamp: new Date(Date.now() - 600000),
      positionSize: 2,
      reasoning: "Overbought conditions with negative divergence. High IV crush opportunity post-earnings.",
      riskReward: "1:1.8",
    },
  ]

  useEffect(() => {
    if (selectedStock) {
      fetchTradingSignals()
      
      // Refresh signals every 30 seconds
      const interval = setInterval(fetchTradingSignals, 30000)
      return () => clearInterval(interval)
    }
  }, [selectedStock])

  const scatterData = signals.map((signal) => ({
    confidence: signal.confidence,
    strength: signal.direction === "BUY" ? signal.confidence : -signal.confidence,
    size: signal.positionSize * 10,
    color: signal.direction === "BUY" ? "#4CAF50" : "#F44336",
    signal,
  }))

  const getDirectionIcon = (direction: "BUY" | "SELL") => {
    return direction === "BUY" ? (
      <TrendingUp className="h-4 w-4 text-green-400" />
    ) : (
      <TrendingDown className="h-4 w-4 text-red-400" />
    )
  }

  const getDirectionBadge = (direction: "BUY" | "SELL") => {
    return (
      <Badge variant={direction === "BUY" ? "default" : "destructive"} className="gap-1">
        {getDirectionIcon(direction)}
        {direction}
      </Badge>
    )
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-green-400"
    if (confidence >= 60) return "text-yellow-400"
    return "text-red-400"
  }

  return (
    <Card>
    
  
    </Card>
  )
}
