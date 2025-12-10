"use client"

import { useState, useEffect, useCallback } from "react"
import { getComprehensiveAnalysis, transformAnalysisResponseToStockData } from "@/lib/api-service"

interface RealTimeData {
  [symbol: string]: {
    price: number
    change: number
    changePercent: number
    volume: number
    lastUpdate: Date
  }
}

export function useRealTimeData() {
  const [data, setData] = useState<RealTimeData>({})
  const [isConnected, setIsConnected] = useState(false)

  const updateStockData = useCallback(async (symbol: string) => {
    try {
      const analysisResponse = await getComprehensiveAnalysis(symbol)
      const stockData = await transformAnalysisResponseToStockData(analysisResponse)
      if (!stockData) return

    // Use real data with slight randomization for real-time effect
    const volatility = 0.01 // 1% max change per update
    const randomChange = (Math.random() - 0.5) * volatility
    const newPrice = stockData.stock.price * (1 + randomChange)
    const change = newPrice - stockData.stock.price
    const changePercent = (change / stockData.stock.price) * 100

    setData(prev => ({
      ...prev,
      [symbol]: {
        price: Math.round(newPrice * 100) / 100,
        change: Math.round(change * 100) / 100,
        changePercent: Math.round(changePercent * 100) / 100,
        volume: stockData.stock.volume + Math.floor(Math.random() * 10000), // Slight volume variation
        lastUpdate: new Date()
      }
    }))
    } catch (error) {
      console.error(`Failed to update stock data for ${symbol}:`, error)
    }
  }, [])

  const startRealTimeUpdates = useCallback(() => {
    setIsConnected(true)
    
    // Update all stocks every 2-5 seconds
    const updateInterval = setInterval(() => {
      const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'SPY']
      symbols.forEach(symbol => {
        updateStockData(symbol)
      })
    }, Math.random() * 3000 + 2000) // Random interval between 2-5 seconds

    return () => {
      clearInterval(updateInterval)
      setIsConnected(false)
    }
  }, [updateStockData])

  const stopRealTimeUpdates = useCallback(() => {
    setIsConnected(false)
  }, [])

  const getCurrentPrice = useCallback((symbol: string) => {
    return data[symbol]?.price || 0
  }, [data])

  const getCurrentChange = useCallback((symbol: string) => {
    return data[symbol]?.change || 0
  }, [data])

  const getCurrentChangePercent = useCallback((symbol: string) => {
    return data[symbol]?.changePercent || 0
  }, [data])

  useEffect(() => {
    const cleanup = startRealTimeUpdates()
    return cleanup
  }, [startRealTimeUpdates])

  return {
    data,
    isConnected,
    startRealTimeUpdates,
    stopRealTimeUpdates,
    getCurrentPrice,
    getCurrentChange,
    getCurrentChangePercent,
    updateStockData
  }
}
