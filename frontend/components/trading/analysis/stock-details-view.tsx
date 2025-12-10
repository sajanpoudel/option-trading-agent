"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, TrendingUp, TrendingDown, Volume2, DollarSign, Target, Activity } from "lucide-react"
import { cn } from "@/lib/utils"
import { getComprehensiveAnalysis, transformAnalysisResponseToStockData, type StockData } from "@/lib/api-service"
import { IntradayChart } from "../charts/intraday-chart"
import { TechnicalAnalysisWidget } from "./technical-analysis-widget"
import { useTrading } from "../controls/trading-context"

interface StockDetailsViewProps {
  symbol: string
  onBack: () => void
  className?: string
}

export function StockDetailsView({ symbol, onBack, className }: StockDetailsViewProps) {
  const { state, executeTrade } = useTrading()
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        setIsLoading(true)
        const analysisResponse = await getComprehensiveAnalysis(symbol)
        const transformedData = await transformAnalysisResponseToStockData(analysisResponse)
        setStockData(transformedData)
      } catch (error) {
        console.error('Error fetching stock data:', error)
      } finally {
        setIsLoading(false)
      }
    }
    fetchStockData()
  }, [symbol])
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">Loading data for {symbol}...</p>
        </CardContent>
      </Card>
    )
  }

  if (!stockData) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">No data available for {symbol}</p>
        </CardContent>
      </Card>
    )
  }

  const { stock, technicals, aiAnalysis } = stockData
  const isPositive = stock.change >= 0

  const handleQuickTrade = (type: 'buy' | 'sell') => {
    const quantity = Math.floor((25420.50 * (state.positionSize / 100)) / stock.price)
    executeTrade(type, symbol, quantity, stock.price)
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <div>
            <h1 className="text-2xl font-bold">{stock.symbol}</h1>
            <p className="text-muted-foreground">{stock.name}</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-3xl font-bold">
            ${stock.price.toFixed(2)}
          </div>
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

      {/* Quick Actions */}
      <div className="flex gap-2">
        <Button
          onClick={() => handleQuickTrade('buy')}
          className="flex items-center gap-2"
          disabled={!state.isActive}
        >
          <TrendingUp className="h-4 w-4" />
          Quick Buy
        </Button>
        <Button
          variant="destructive"
          onClick={() => handleQuickTrade('sell')}
          className="flex items-center gap-2"
          disabled={!state.isActive}
        >
          <TrendingDown className="h-4 w-4" />
          Quick Sell
        </Button>
        <Badge variant="outline" className="ml-auto">
          {state.mode} mode
        </Badge>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technical">Technical</TabsTrigger>
          <TabsTrigger value="analysis">AI Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Intraday Chart */}
          <IntradayChart symbol={symbol} stockData={stockData} />

          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Volume2 className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Volume</span>
                </div>
                <div className="text-2xl font-bold">
                  {(stock.volume / 1000000).toFixed(1)}M
                </div>
                <div className="text-xs text-muted-foreground">
                  {stock.volume > technicals.volumeAvg ? 'Above avg' : 'Below avg'}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Sector</span>
                </div>
                <div className="text-2xl font-bold">{stock.sector}</div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Support</span>
                </div>
                <div className="text-2xl font-bold text-red-500">
                  ${technicals.support.toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">
                  {((stock.price - technicals.support) / stock.price * 100).toFixed(1)}% away
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Resistance</span>
                </div>
                <div className="text-2xl font-bold text-green-500">
                  ${technicals.resistance.toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">
                  {((technicals.resistance - stock.price) / stock.price * 100).toFixed(1)}% away
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Company Description */}
          <Card>
            <CardHeader>
              <CardTitle>About {stock.symbol}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                {stock.description}
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="technical" className="space-y-6">
          <TechnicalAnalysisWidget symbol={symbol} stockData={stockData} />
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          {/* AI Analysis Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                AI Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-500 mb-1">
                    {aiAnalysis.confidence}%
                  </div>
                  <div className="text-sm text-muted-foreground">Confidence</div>
                </div>
                <div className="text-center">
                  <div className={cn(
                    "text-3xl font-bold mb-1",
                    aiAnalysis.recommendation.includes('buy') ? "text-green-500" : 
                    aiAnalysis.recommendation.includes('sell') ? "text-red-500" : "text-yellow-500"
                  )}>
                    {aiAnalysis.recommendation.toUpperCase().replace('_', ' ')}
                  </div>
                  <div className="text-sm text-muted-foreground">Recommendation</div>
                </div>
                <div className="text-center">
                  <div className={cn(
                    "text-3xl font-bold mb-1 capitalize",
                    aiAnalysis.sentiment === 'bullish' ? "text-green-500" : 
                    aiAnalysis.sentiment === 'bearish' ? "text-red-500" : "text-yellow-500"
                  )}>
                    {aiAnalysis.sentiment}
                  </div>
                  <div className="text-sm text-muted-foreground">Sentiment</div>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="text-sm font-medium mb-2">Analysis Reasoning:</div>
                  <ul className="space-y-1">
                    {aiAnalysis.reasoning.map((reason, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                        <span className="text-blue-500 mt-1">•</span>
                        {reason}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <span className="text-sm font-medium">Risk Level</span>
                  <Badge 
                    variant={aiAnalysis.riskLevel === 'low' ? 'default' : 
                            aiAnalysis.riskLevel === 'medium' ? 'secondary' : 'destructive'}
                  >
                    {aiAnalysis.riskLevel.toUpperCase()}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Trading Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle>Trading Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {state.mode === 'autonomous' ? (
                  <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="h-4 w-4 text-blue-500" />
                      <span className="font-medium text-blue-500">Autonomous Mode Active</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      AI will automatically execute trades based on analysis when conditions are optimal.
                    </p>
                  </div>
                ) : (
                  <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="h-4 w-4 text-yellow-500" />
                      <span className="font-medium text-yellow-500">Manual Mode Active</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Review the analysis above and use the Quick Buy/Sell buttons to execute trades manually.
                    </p>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3 rounded border">
                    <div className="text-sm font-medium mb-1">Entry Strategy</div>
                    <div className="text-xs text-muted-foreground">
                      {aiAnalysis.recommendation.includes('buy') 
                        ? 'Consider entering on pullbacks to support levels'
                        : aiAnalysis.recommendation.includes('sell')
                        ? 'Consider exiting on rallies to resistance levels'
                        : 'Wait for clearer directional signals'
                      }
                    </div>
                  </div>
                  <div className="p-3 rounded border">
                    <div className="text-sm font-medium mb-1">Position Size</div>
                    <div className="text-xs text-muted-foreground">
                      Recommended: {state.positionSize}% of portfolio
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
