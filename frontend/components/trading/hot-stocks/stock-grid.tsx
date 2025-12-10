"use client"

import { StockCard } from "./stock-card"

interface HotStock {
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  volume: string
  sparklineData: { value: number }[]
  aiScore: number
  signals: string[]
  trending: boolean
}

interface StockGridProps {
  stocks: HotStock[]
  onStockSelect: (stock: string) => void
}

export function StockGrid({ stocks, onStockSelect }: StockGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-6">
      {stocks.map((stock) => (
        <StockCard key={stock.symbol} stock={stock} onSelect={() => onStockSelect(stock.symbol)} />
      ))}
    </div>
  )
}
