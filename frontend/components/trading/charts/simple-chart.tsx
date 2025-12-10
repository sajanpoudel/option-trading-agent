"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, Volume2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { type StockData } from "@/lib/api-service"
import { TradingViewChart } from "./tradingview-chart"

interface SimpleChartProps {
  symbol: string
  stockData: StockData
  className?: string
}

export function SimpleChart({ symbol, stockData, className }: SimpleChartProps) {
  if (!stockData) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">
            No data available for {symbol}
          </p>
        </CardContent>
      </Card>
    )
  }

  const { stock, technicals } = stockData
  const isPositive = stock.change >= 0

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="text-lg">
              {stock.symbol} - {stock.name}
            </CardTitle>
            <div className="flex items-center gap-3">
              <span className="text-2xl font-bold">
                ${stock.price.toFixed(2)}
              </span>
              <div className={cn(
                "flex items-center gap-1 text-sm",
                isPositive ? "text-green-500" : "text-red-500"
              )}>
                {isPositive ? (
                  <TrendingUp className="h-4 w-4" />
                ) : (
                  <TrendingDown className="h-4 w-4" />
                )}
                <span>
                  {isPositive ? '+' : ''}{stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
          <div className="text-right space-y-1">
            <div className="flex items-center gap-2">
              <Volume2 className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">
                {(stock.volume / 1000000).toFixed(1)}M
              </span>
            </div>
            <Badge variant={isPositive ? "default" : "destructive"} className="text-xs">
              {stock.sector}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[600px] w-full">
          <TradingViewChart
            symbol={symbol}
            height={600}
            theme="dark"
            style="1"
            interval="D"
            allow_symbol_change={false}
            hide_side_toolbar={false}
            hide_top_toolbar={false}
            studies={[]}
          />
        </div>
        
        {/* Key Levels */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t">
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Support</div>
            <div className="text-sm font-bold text-red-500">
              ${technicals.support.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Resistance</div>
            <div className="text-sm font-bold text-green-500">
              ${technicals.resistance.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">MA(20)</div>
            <div className="text-sm font-bold text-blue-500">
              ${technicals.movingAverage20.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">MA(50)</div>
            <div className="text-sm font-bold text-yellow-500">
              ${technicals.movingAverage50.toFixed(2)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
