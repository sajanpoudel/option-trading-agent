"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, TrendingUp, TrendingDown } from "lucide-react"

interface StockData {
  price: number
  change: number
  changePercent: number
  aiScore: number
  sentiment: string
  riskLevel: string
  signals: string[]
}

interface AnalysisHeaderProps {
  selectedStock: string
  stockData: StockData
  onBack: () => void
}

export function AnalysisHeader({ selectedStock, stockData, onBack }: AnalysisHeaderProps) {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30">
      <div className="container mx-auto px-4 sm:px-6 py-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={onBack} className="shrink-0">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div className="min-w-0">
              <h1 className="text-xl sm:text-2xl font-bold text-foreground">{selectedStock} Analysis</h1>
              <p className="text-sm text-muted-foreground truncate">Real-time AI-powered analysis</p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 w-full sm:w-auto">
            <div className="text-right">
              <div className="text-xl sm:text-2xl font-bold">${stockData.price.toFixed(2)}</div>
              <div
                className={`flex items-center gap-1 text-sm ${
                  stockData.change >= 0 ? "text-green-400" : "text-red-400"
                }`}
              >
                {stockData.change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                {stockData.change > 0 ? "+" : ""}
                {stockData.change.toFixed(2)} ({stockData.changePercent > 0 ? "+" : ""}
                {stockData.changePercent.toFixed(1)}%)
              </div>
            </div>

            <div className="flex gap-2">
              <Badge variant="default" className="text-green-400">
                AI Score: {stockData.aiScore}
              </Badge>
              <Badge variant="outline">{stockData.sentiment}</Badge>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
