"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, ResponsiveContainer } from "recharts"
import { TrendingUp, TrendingDown, Flame } from "lucide-react"

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

interface StockCardProps {
  stock: HotStock
  onSelect: () => void
}

export function StockCard({ stock, onSelect }: StockCardProps) {
  const getPriceChangeColor = (change: number) => {
    return change >= 0 ? "text-green-400" : "text-red-400"
  }

  const getPriceChangeIcon = (change: number) => {
    return change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />
  }

  const getAIScoreColor = (score: number) => {
    if (score >= 85) return "text-green-400"
    if (score >= 70) return "text-yellow-400"
    return "text-red-400"
  }

  return (
    <Card className="hover:shadow-lg transition-all duration-200 cursor-pointer group hover:border-primary/50">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <CardTitle className="text-lg font-bold">{stock.symbol}</CardTitle>
              {stock.trending && (
                <Badge variant="outline" className="text-xs px-2 py-0 border-orange-500/50 text-orange-400">
                  <Flame className="h-3 w-3 mr-1" />
                  Hot
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground truncate">{stock.name}</p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Price Info */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xl sm:text-2xl font-bold">${stock.price.toFixed(2)}</div>
            <div className={`flex items-center gap-1 text-sm ${getPriceChangeColor(stock.change)}`}>
              {getPriceChangeIcon(stock.change)}
              {stock.change > 0 ? "+" : ""}
              {stock.change.toFixed(2)} ({stock.changePercent > 0 ? "+" : ""}
              {stock.changePercent.toFixed(1)}%)
            </div>
          </div>
          <div className="w-20 sm:w-24 h-10 sm:h-12">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stock.sparklineData}>
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={stock.change >= 0 ? "#4CAF50" : "#F44336"}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Market Data */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-muted-foreground text-xs">Volume</div>
            <div className="font-medium">{stock.volume}</div>
          </div>
        </div>

      </CardContent>
    </Card>
  )
}
