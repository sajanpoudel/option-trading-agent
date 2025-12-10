"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Activity, 
  BarChart3, 
  Volume2,
  Target,
  AlertTriangle
} from "lucide-react"
import { type StockData } from "@/lib/api-service"
import { cn } from "@/lib/utils"

interface TechnicalIndicatorsProps {
  selectedStock: string
  stockData: StockData | null
  isLoading?: boolean
  className?: string
}

export function TechnicalIndicators({ selectedStock, stockData, isLoading = false, className }: TechnicalIndicatorsProps) {
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-32">
          <p className="text-muted-foreground">Loading technical data for {selectedStock}...</p>
        </CardContent>
      </Card>
    )
  }

  if (!stockData) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-32">
          <p className="text-muted-foreground">
            No technical data available for {selectedStock}
          </p>
        </CardContent>
      </Card>
    )
  }

  const { stock, technicals, aiAnalysis } = stockData

  const getRSIStatus = (rsi: number) => {
    if (rsi >= 70) return { status: 'Overbought', color: 'text-red-500', icon: AlertTriangle }
    if (rsi <= 30) return { status: 'Oversold', color: 'text-green-500', icon: AlertTriangle }
    return { status: 'Neutral', color: 'text-yellow-500', icon: Minus }
  }

  const getMACDIcon = (macd: string) => {
    switch (macd) {
      case 'bullish': return TrendingUp
      case 'bearish': return TrendingDown
      default: return Minus
    }
  }

  const getMACDColor = (macd: string) => {
    switch (macd) {
      case 'bullish': return 'text-green-500'
      case 'bearish': return 'text-red-500'
      default: return 'text-yellow-500'
    }
  }

  const rsiInfo = getRSIStatus(technicals.rsi)
  const MACDIcon = getMACDIcon(technicals.macd)
  const macdColor = getMACDColor(technicals.macd)

  const volumeRatio = stock.volume / technicals.volumeAvg
  const isHighVolume = volumeRatio > 1.2

  return (
    <div className={cn("space-y-4", className)}>
      {/* Main Technical Indicators */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Technical Indicators
            <Badge variant="outline" className="ml-auto">
              {selectedStock}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* RSI */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">RSI (14)</span>
                <rsiInfo.icon className={cn("h-4 w-4", rsiInfo.color)} />
              </div>
              <div className="space-y-1">
                <Progress value={technicals.rsi} className="h-2" />
                <div className="flex justify-between items-center">
                  <span className="text-lg font-bold">{technicals.rsi.toFixed(1)}</span>
                  <Badge variant="outline" className={cn("text-xs", rsiInfo.color)}>
                    {rsiInfo.status}
                  </Badge>
                </div>
              </div>
            </div>

            {/* MACD */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">MACD</span>
                <MACDIcon className={cn("h-4 w-4", macdColor)} />
              </div>
              <div className="space-y-1">
                <div className="text-lg font-bold capitalize">{technicals.macd}</div>
                <Badge variant="outline" className={cn("text-xs", macdColor)}>
                  {technicals.macd === 'bullish' ? 'Buy Signal' : 
                   technicals.macd === 'bearish' ? 'Sell Signal' : 'No Signal'}
                </Badge>
              </div>
            </div>

            {/* Volume */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Volume</span>
                <Volume2 className={cn("h-4 w-4", isHighVolume ? "text-green-500" : "text-gray-500")} />
              </div>
              <div className="space-y-1">
                <div className="text-lg font-bold">
                  {(stock.volume / 1000000).toFixed(1)}M
                </div>
                <Badge variant="outline" className={cn(
                  "text-xs",
                  isHighVolume ? "text-green-500" : "text-gray-500"
                )}>
                  {isHighVolume ? 'High Volume' : 'Normal'}
                </Badge>
              </div>
            </div>

            {/* Moving Averages */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">MA Trend</span>
                <Activity className={cn("h-4 w-4", 
                  stock.price > technicals.movingAverage20 ? "text-green-500" : "text-red-500"
                )} />
              </div>
              <div className="space-y-1">
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span>MA(20):</span>
                    <span>${technicals.movingAverage20.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>MA(50):</span>
                    <span>${technicals.movingAverage50.toFixed(2)}</span>
                  </div>
                </div>
                <Badge variant="outline" className={cn(
                  "text-xs",
                  stock.price > technicals.movingAverage20 && technicals.movingAverage20 > technicals.movingAverage50
                    ? "text-green-500" : "text-red-500"
                )}>
                  {stock.price > technicals.movingAverage20 && technicals.movingAverage20 > technicals.movingAverage50
                    ? 'Bullish' : 'Bearish'}
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Support & Resistance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Key Levels
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center space-y-1">
              <div className="text-xs text-muted-foreground">Support</div>
              <div className="text-lg font-bold text-red-500">
                ${technicals.support.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">
                -{(((stock.price - technicals.support) / stock.price) * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="text-center space-y-1">
              <div className="text-xs text-muted-foreground">Resistance</div>
              <div className="text-lg font-bold text-green-500">
                ${technicals.resistance.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">
                +{(((technicals.resistance - stock.price) / stock.price) * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="text-center space-y-1">
              <div className="text-xs text-muted-foreground">Bollinger Upper</div>
              <div className="text-lg font-bold">
                ${technicals.bollingerUpper.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">
                +{(((technicals.bollingerUpper - stock.price) / stock.price) * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="text-center space-y-1">
              <div className="text-xs text-muted-foreground">Bollinger Lower</div>
              <div className="text-lg font-bold">
                ${technicals.bollingerLower.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">
                -{(((stock.price - technicals.bollingerLower) / stock.price) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Analysis Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            AI Technical Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Overall Signal</span>
              <Badge 
                variant={aiAnalysis.recommendation.includes('buy') ? 'default' : 
                        aiAnalysis.recommendation.includes('sell') ? 'destructive' : 'outline'}
                className="font-semibold"
              >
                {aiAnalysis.recommendation.toUpperCase().replace('_', ' ')}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Confidence Level</span>
              <div className="flex items-center gap-2">
                <Progress value={aiAnalysis.confidence} className="w-16 h-2" />
                <span className="text-sm font-bold">{aiAnalysis.confidence}%</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Risk Level</span>
              <Badge 
                variant="outline" 
                className={cn(
                  aiAnalysis.riskLevel === 'high' ? 'text-red-500' :
                  aiAnalysis.riskLevel === 'medium' ? 'text-yellow-500' : 'text-green-500'
                )}
              >
                {aiAnalysis.riskLevel.toUpperCase()}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
