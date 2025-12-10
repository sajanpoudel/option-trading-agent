"use client"

import { createContext, useContext, useState, useEffect, type ReactNode } from "react"
import { apiService } from "@/lib/api-service"

export type TradingMode = "manual" | "autonomous"

interface Position {
  symbol: string
  quantity: number
  averageCost: number
  currentPrice: number
  totalValue: number
  pnl: number
  pnlPercent: number
}

interface Trade {
  id: string
  symbol: string
  type: 'buy' | 'sell'
  quantity: number
  price: number
  timestamp: Date
  status: 'pending' | 'executed' | 'cancelled'
  pnl?: number
}

interface TradingState {
  mode: TradingMode
  isActive: boolean
  riskLevel: number
  positionSize: number
  maxDailyTrades: string
  currentTrades: number
  totalPnL: number
  todayPnL: number
  positions: Position[]
  trades: Trade[]
  totalValue: number
  cashBalance: number
}

interface TradingContextType {
  state: TradingState
  setMode: (mode: TradingMode) => void
  setActive: (active: boolean) => void
  updateRiskLevel: (level: number) => void
  updatePositionSize: (size: number) => void
  updateMaxDailyTrades: (max: string) => void
  executeTrade: (type: "buy" | "sell", symbol: string, quantity: number, price: number) => void
  addToPnL: (amount: number) => void
  updatePositions: (positions: Position[]) => void
  addTrade: (trade: Trade) => void
}

const TradingContext = createContext<TradingContextType | undefined>(undefined)

export function TradingProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<TradingState>({
    mode: "manual",
    isActive: true, // Enable trading by default
    riskLevel: 3,
    positionSize: 25,
    maxDailyTrades: "5",
    currentTrades: 0,
    totalPnL: 0,
    todayPnL: 0,
    positions: [],
    trades: [],
    totalValue: 0,
    cashBalance: 0,
  })

  // Load portfolio data on mount
  useEffect(() => {
    const loadPortfolio = async () => {
      try {
        const portfolio = await apiService.getPortfolio()
        setState(prev => ({
          ...prev,
          totalPnL: portfolio.totalPnL,
          todayPnL: portfolio.todayPnL,
          positions: portfolio.positions,
          totalValue: portfolio.totalValue,
          cashBalance: portfolio.cashBalance,
        }))
      } catch (error) {
        console.error('Failed to load portfolio:', error)
      }
    }
    
    loadPortfolio()
  }, [])

  const setMode = (mode: TradingMode) => {
    setState((prev) => ({ ...prev, mode }))
  }

  const setActive = (active: boolean) => {
    setState((prev) => ({ ...prev, isActive: active }))
  }

  const updateRiskLevel = (level: number) => {
    setState((prev) => ({ ...prev, riskLevel: level }))
  }

  const updatePositionSize = (size: number) => {
    setState((prev) => ({ ...prev, positionSize: size }))
  }

  const updateMaxDailyTrades = (max: string) => {
    setState((prev) => ({ ...prev, maxDailyTrades: max }))
  }

  const executeTrade = (type: "buy" | "sell", symbol: string, quantity: number, price: number) => {
    console.log(`[v0] Executing ${type} trade: ${quantity} shares of ${symbol} at $${price}`)
    
    const trade: Trade = {
      id: Date.now().toString(),
      symbol,
      type,
      quantity,
      price,
      timestamp: new Date(),
      status: 'executed'
    }

    setState((prev) => {
      const newTrades = [...prev.trades, trade]
      const tradeValue = quantity * price
      
      let newPositions = [...prev.positions]
      let newCashBalance = prev.cashBalance
      let newTotalValue = prev.totalValue
      let newTotalPnL = prev.totalPnL
      let newTodayPnL = prev.todayPnL

      if (type === 'buy') {
        // Check if we have enough cash
        if (newCashBalance < tradeValue) {
          console.warn('Insufficient cash for trade')
          return prev
        }

        // Deduct cash for purchase
        newCashBalance -= tradeValue
        
        // Update or create position
        const existingPositionIndex = newPositions.findIndex(p => p.symbol === symbol)
        if (existingPositionIndex >= 0) {
          const existing = newPositions[existingPositionIndex]
          const newQuantity = existing.quantity + quantity
          const newAverageCost = ((existing.averageCost * existing.quantity) + (price * quantity)) / newQuantity
          const newTotalValue = newQuantity * price
          const newPnL = newTotalValue - (newQuantity * newAverageCost)
          
          newPositions[existingPositionIndex] = {
            symbol,
            quantity: newQuantity,
            averageCost: newAverageCost,
            currentPrice: price,
            totalValue: newTotalValue,
            pnl: newPnL,
            pnlPercent: (newPnL / (newQuantity * newAverageCost)) * 100
          }
        } else {
          newPositions.push({
            symbol,
            quantity,
            averageCost: price,
            currentPrice: price,
            totalValue: tradeValue,
            pnl: 0,
            pnlPercent: 0
          })
        }
      } else {
        // Check if we have enough shares to sell
        const existingPositionIndex = newPositions.findIndex(p => p.symbol === symbol)
        if (existingPositionIndex >= 0) {
          const existing = newPositions[existingPositionIndex]
          if (existing.quantity < quantity) {
            console.warn('Insufficient shares to sell')
            return prev
          }
        } else {
          console.warn('No position to sell')
          return prev
        }

        // Add cash from sale
        newCashBalance += tradeValue
        
        // Update or remove position
        if (existingPositionIndex >= 0) {
          const existing = newPositions[existingPositionIndex]
          const newQuantity = existing.quantity - quantity
          
          if (newQuantity <= 0) {
            // Remove position entirely
            newPositions.splice(existingPositionIndex, 1)
          } else {
            // Update position
            const newTotalValue = newQuantity * price
            const newPnL = newTotalValue - (newQuantity * existing.averageCost)
            
            newPositions[existingPositionIndex] = {
              ...existing,
              quantity: newQuantity,
              currentPrice: price,
              totalValue: newTotalValue,
              pnl: newPnL,
              pnlPercent: (newPnL / (newQuantity * existing.averageCost)) * 100
            }
          }
        }
      }

      // Recalculate total value and P&L
      const positionsValue = newPositions.reduce((sum, pos) => sum + pos.totalValue, 0)
      newTotalValue = newCashBalance + positionsValue
      
      // Calculate total P&L
      const totalCost = newPositions.reduce((sum, pos) => sum + (pos.averageCost * pos.quantity), 0)
      newTotalPnL = positionsValue - totalCost
      
      // Simulate today's P&L (in real app, this would be calculated differently)
      newTodayPnL = newTotalPnL * 0.1 // 10% of total P&L as today's gain

      return {
        ...prev,
        currentTrades: prev.currentTrades + 1,
        trades: newTrades,
        positions: newPositions,
        cashBalance: newCashBalance,
        totalValue: newTotalValue,
        totalPnL: newTotalPnL,
        todayPnL: newTodayPnL
      }
    })
  }

  const addToPnL = (amount: number) => {
    setState((prev) => ({
      ...prev,
      totalPnL: prev.totalPnL + amount,
      todayPnL: prev.todayPnL + amount,
    }))
  }

  const updatePositions = (positions: Position[]) => {
    setState((prev) => ({
      ...prev,
      positions,
      totalValue: prev.cashBalance + positions.reduce((sum, pos) => sum + pos.totalValue, 0)
    }))
  }

  const addTrade = (trade: Trade) => {
    setState((prev) => ({
      ...prev,
      trades: [...prev.trades, trade]
    }))
  }

  return (
    <TradingContext.Provider
      value={{
        state,
        setMode,
        setActive,
        updateRiskLevel,
        updatePositionSize,
        updateMaxDailyTrades,
        executeTrade,
        addToPnL,
        updatePositions,
        addTrade,
      }}
    >
      {children}
    </TradingContext.Provider>
  )
}

export function useTrading() {
  const context = useContext(TradingContext)
  if (context === undefined) {
    throw new Error("useTrading must be used within a TradingProvider")
  }
  return context
}
