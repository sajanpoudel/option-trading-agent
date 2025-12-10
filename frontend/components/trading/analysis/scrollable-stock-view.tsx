"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ArrowLeft, TrendingUp, TrendingDown, Volume2, DollarSign, Target, Activity, Bot, User, BarChart3, PieChart, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import { getComprehensiveAnalysis, transformAnalysisResponseToStockData, StockData } from "@/lib/api-service"
import { SimpleChart } from "../charts/simple-chart"
import { TechnicalAnalysisWidget } from "./technical-analysis-widget"
import { useTrading } from "../controls/trading-context"
import { TradingSettings } from "../controls/trading-settings"
import { AIAgentAnalysis } from "@/components/ai-agent-analysis"
import { TradingSignals } from "@/components/trading-signals"
import { PayoffSurface3D } from "@/components/payoff-surface-3d"
import { RealTimeMonitoring } from "@/components/real-time-monitoring"
import { EducationalModule } from "@/components/educational-module"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useState, useEffect } from "react"

interface ScrollableStockViewProps {
  symbol: string
  onBack: () => void
  className?: string
}

export function ScrollableStockView({ symbol, onBack, className }: ScrollableStockViewProps) {
  const { state, executeTrade } = useTrading()
  const [showTradeSuccess, setShowTradeSuccess] = useState(false)
  const [tradeMessage, setTradeMessage] = useState("")
  const [quantity, setQuantity] = useState(1)
  const [sellQuantity, setSellQuantity] = useState(1)
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("technical")
  
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        setIsLoading(true)
        // Use the direct analysis API for better data structure
        const analysisResponse = await getComprehensiveAnalysis(symbol)
        const transformedData = await transformAnalysisResponseToStockData(analysisResponse)
        setStockData(transformedData)
      } catch (error) {
        console.error('Error fetching stock data:', error)
        // Set fallback data
        setStockData({
          stock: {
            symbol,
            name: `${symbol} Inc`,
            price: 100,
            change: 0,
            changePercent: 0,
            volume: 0,
            sector: 'Unknown',
            description: 'Data unavailable'
          },
          technicals: {
            rsi: 50,
            macd: 'neutral',
            movingAverage20: 100,
            movingAverage50: 100,
            support: 95,
            resistance: 105,
            volumeAvg: 0,
            bollingerUpper: 105,
            bollingerLower: 95
          },
          chartData: [],
          aiAnalysis: {
            confidence: 0,
            recommendation: 'hold',
            reasoning: ['Data unavailable'],
            sentiment: 'neutral',
            riskLevel: 'high'
          }
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchStockData()
  }, [symbol])

  // Listen for chat analysis data to override the default API call
  useEffect(() => {
    const handleChatAnalysisData = (event: any) => {
      const { symbol: chatSymbol, analysisData, response } = event.detail
      
      if (chatSymbol === symbol && analysisData) {
        console.log(`🔄 Using chat analysis data for ${symbol}`, analysisData)
        
        // Transform chat analysis data to stock data format
        try {
          const transformedData = transformChatAnalysisToStockData(analysisData, response, symbol)
          setStockData(transformedData)
          setIsLoading(false)
        } catch (error) {
          console.error('Error transforming chat analysis data:', error)
          // Fall back to the API call if transformation fails
        }
      }
    }

    window.addEventListener('setChatAnalysisData', handleChatAnalysisData)
    
    return () => {
      window.removeEventListener('setChatAnalysisData', handleChatAnalysisData)
    }
  }, [symbol])

  // Helper function to transform chat analysis data
  const transformChatAnalysisToStockData = (analysisData: any, _response: string, symbol: string) => {
    // Extract data from the analysis response
    const toolResults = analysisData?.tool_results || []
    const stockAnalysis = toolResults.find((result: any) => result.tool === "analyze_stock")
    
    if (stockAnalysis && stockAnalysis.analysis_result) {
      const analysis = stockAnalysis.analysis_result
      const signal = analysis.signal || {}
      
      return {
        stock: {
          symbol: symbol,
          name: `${symbol} Corporation`,
          price: analysis.current_price || 177.82,
          change: signal.price_change || 0.65,
          changePercent: signal.price_change_percent || 0.37,
          volume: analysis.volume || 121500000,
          sector: 'Technology',
          description: analysis.description || `${symbol} stock analysis`
        },
        technicals: {
          rsi: analysis.agent_results?.technical?.metrics?.RSI || 55.5,
          macd: analysis.agent_results?.technical?.metrics?.MACD_Signal || 'neutral',
          movingAverage20: analysis.agent_results?.technical?.metrics?.MA_20 || 155.20,
          movingAverage50: analysis.agent_results?.technical?.metrics?.MA_50 || 155.20,
          support: analysis.support_levels?.[0] || 164.07,
          resistance: analysis.resistance_levels?.[0] || 184.47,
          volumeAvg: analysis.volume_avg || 0,
          bollingerUpper: analysis.agent_results?.technical?.metrics?.BB_Upper || 181.38,
          bollingerLower: analysis.agent_results?.technical?.metrics?.BB_Lower || 174.26
        },
        chartData: [],
        aiAnalysis: {
          confidence: analysis.confidence || 0.65,
          recommendation: signal.direction?.toLowerCase() || 'hold',
          reasoning: [analysis.agent_results?.technical?.explanation || 'Analysis based on market conditions'],
          sentiment: analysis.agent_results?.sentiment?.sentiment || 'neutral',
          riskLevel: analysis.risk_level || 'medium'
        }
      }
    }
    
    throw new Error('Invalid analysis data structure')
  }
  
  if (isLoading || !stockData) {
    return (
      <Card className={className}>
        <CardContent className="p-0">
          {/* Loading Skeleton */}
          <div className="relative overflow-hidden">
            {/* Skeleton Content */}
            <div className="p-6 space-y-6">
              {/* Header Skeleton with Loading Indicator */}
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary border-t-transparent"></div>
                  <div className="h-8 w-32 bg-muted rounded-lg animate-pulse"></div>
                  <div className="h-6 w-20 bg-green-200 rounded animate-pulse"></div>
                </div>
                <div className="flex gap-4">
                  <div className="h-6 w-24 bg-muted rounded animate-pulse"></div>
                  <div className="h-6 w-20 bg-muted rounded animate-pulse"></div>
                  <div className="h-6 w-16 bg-muted rounded animate-pulse"></div>
                </div>
              </div>
              
              {/* Tabs Skeleton */}
              <div className="flex gap-2 border-b">
                <div className="h-8 w-20 bg-muted rounded-t animate-pulse"></div>
                <div className="h-8 w-24 bg-muted rounded-t animate-pulse"></div>
                <div className="h-8 w-20 bg-muted rounded-t animate-pulse"></div>
              </div>
              
              {/* Chart Skeleton */}
              <div className="h-64 bg-muted rounded-lg animate-pulse relative">
                <div className="absolute inset-4 border-2 border-dashed border-muted-foreground/20 rounded"></div>
              </div>
              
              {/* Analysis Cards Skeleton */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="p-4 border rounded-lg space-y-3">
                    <div className="h-5 w-32 bg-muted rounded animate-pulse"></div>
                    <div className="h-8 w-20 bg-green-200 rounded animate-pulse"></div>
                    <div className="space-y-2">
                      <div className="h-3 w-full bg-muted rounded animate-pulse"></div>
                      <div className="h-3 w-3/4 bg-muted rounded animate-pulse"></div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* AI Agent Analysis Skeleton */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-5 w-5 bg-blue-200 rounded animate-pulse"></div>
                  <div className="h-6 w-40 bg-muted rounded animate-pulse"></div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="p-3 border rounded-lg space-y-2">
                      <div className="h-8 w-8 bg-muted rounded-full animate-pulse mx-auto"></div>
                      <div className="h-3 w-16 bg-muted rounded animate-pulse mx-auto"></div>
                      <div className="h-2 w-12 bg-muted rounded animate-pulse mx-auto"></div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Technical Indicators Skeleton */}
              <div className="space-y-3">
                <div className="h-6 w-48 bg-muted rounded animate-pulse"></div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                    <div key={i} className="space-y-1">
                      <div className="h-4 w-12 bg-muted rounded animate-pulse"></div>
                      <div className="h-6 w-16 bg-muted rounded animate-pulse"></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const { stock, technicals, aiAnalysis } = stockData
  
  // Add defensive programming for incomplete stock data
  if (!stock) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <p className="text-muted-foreground">Stock data not available for {symbol}</p>
        </CardContent>
      </Card>
    )
  }
  
  const isPositive = (stock.change || 0) >= 0

  // Get current position for this stock
  const currentPosition = state.positions.find(p => p.symbol === symbol)
  const hasPosition = currentPosition && currentPosition.quantity > 0

  const handleQuickTrade = (type: 'buy' | 'sell') => {
    console.log(`Attempting ${type} trade for ${symbol}`)
    console.log('Current state:', { 
      cashBalance: state.cashBalance, 
      positionSize: state.positionSize, 
      stockPrice: stock.price,
      isActive: state.isActive,
      quantity: type === 'buy' ? quantity : sellQuantity
    })
    
    if (!state.isActive) {
      setTradeMessage(`❌ Trading is disabled. Please enable trading in the mode toggle.`)
      setShowTradeSuccess(true)
      setTimeout(() => setShowTradeSuccess(false), 3000)
      return
    }
    
    const tradeQuantity = type === 'buy' ? quantity : sellQuantity
    
    if (type === 'buy') {
      if (tradeQuantity <= 0) {
        setTradeMessage(`❌ Please enter a valid quantity to buy.`)
        setShowTradeSuccess(true)
        setTimeout(() => setShowTradeSuccess(false), 3000)
        return
      }
      
      if (state.cashBalance < (tradeQuantity * stock.price)) {
        setTradeMessage(`❌ Insufficient cash. Required: $${(tradeQuantity * stock.price).toFixed(2)}, Available: $${state.cashBalance.toFixed(2)}`)
        setShowTradeSuccess(true)
        setTimeout(() => setShowTradeSuccess(false), 3000)
        return
      }
      
      executeTrade(type, symbol, tradeQuantity, stock.price)
      setTradeMessage(`✅ Bought ${tradeQuantity} shares of ${symbol} at $${stock.price.toFixed(2)}`)
      setShowTradeSuccess(true)
      setTimeout(() => setShowTradeSuccess(false), 3000)
      setQuantity(1) // Reset quantity after successful trade
    } else if (type === 'sell') {
      if (!hasPosition) {
        setTradeMessage(`❌ No position in ${symbol} to sell.`)
        setShowTradeSuccess(true)
        setTimeout(() => setShowTradeSuccess(false), 3000)
        return
      }
      
      if (tradeQuantity <= 0) {
        setTradeMessage(`❌ Please enter a valid quantity to sell.`)
        setShowTradeSuccess(true)
        setTimeout(() => setShowTradeSuccess(false), 3000)
        return
      }
      
      if (tradeQuantity > currentPosition.quantity) {
        setTradeMessage(`❌ Cannot sell ${tradeQuantity} shares. You only have ${currentPosition.quantity} shares.`)
        setShowTradeSuccess(true)
        setTimeout(() => setShowTradeSuccess(false), 3000)
        return
      }
      
      executeTrade(type, symbol, tradeQuantity, stock.price)
      setTradeMessage(`✅ Sold ${tradeQuantity} shares of ${symbol} at $${stock.price.toFixed(2)}`)
      setShowTradeSuccess(true)
      setTimeout(() => setShowTradeSuccess(false), 3000)
      setSellQuantity(1) // Reset quantity after successful trade
    }
  }


  return (
    <div className={cn("min-h-screen bg-background", className)}>
      {/* Fixed Header */}
      <div className="sticky top-0 z-50 bg-background/95 backdrop-blur border-b">
        <div className="flex items-center justify-between p-4">
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
              <h1 className="text-xl font-bold">{stock.symbol}</h1>
              <p className="text-sm text-muted-foreground">{stock.name}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-2xl font-bold">
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
            <TradingSettings />
          </div>
        </div>
      </div>

        {/* Trade Success Message */}
        {showTradeSuccess && (
          <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50">
            <div className={cn(
              "px-4 py-2 rounded-lg shadow-lg text-white",
              tradeMessage.includes("❌") ? "bg-red-500" : "bg-green-500"
            )}>
              {tradeMessage}
            </div>
          </div>
        )}

        {/* Trading Disabled Notice */}
        {!state.isActive && (
          <div className="fixed top-20 right-4 z-50">
            <div className="bg-yellow-500 text-white px-4 py-2 rounded-lg shadow-lg">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                <span>Trading disabled. Click the settings button to enable.</span>
              </div>
            </div>
          </div>
        )}

      {/* Main Content - Scrollable */}
      <div className="space-y-6 p-4">
        {/* Quick Actions */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Mode</div>
                  <Badge variant={state.mode === "autonomous" ? "default" : "secondary"}>
                    {state.mode === "autonomous" ? (
                      <>
                        <Bot className="h-3 w-3 mr-1" />
                        Autonomous
                      </>
                    ) : (
                      <>
                        <User className="h-3 w-3 mr-1" />
                        Manual
                      </>
                    )}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Trading</div>
                  <Badge variant={state.isActive ? "default" : "destructive"}>
                    {state.isActive ? "Active" : "Disabled"}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Cash</div>
                  <div className="font-semibold">${state.cashBalance.toFixed(2)}</div>
                </div>
                {hasPosition && (
                  <div className="text-center">
                    <div className="text-sm text-muted-foreground">Position</div>
                    <div className="font-semibold">{currentPosition.quantity} shares</div>
                  </div>
                )}
              </div>
              <div className="flex gap-4">
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      placeholder="Quantity"
                      value={quantity}
                      onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
                      min="1"
                      className="w-24 h-10"
                    />
                    <Button
                      onClick={() => handleQuickTrade('buy')}
                      className={cn(
                        "flex items-center gap-2 font-semibold",
                        !state.isActive ? "opacity-50 cursor-not-allowed" : "hover:bg-green-600"
                      )}
                      disabled={!state.isActive || state.cashBalance < (stock.price * quantity) || quantity <= 0}
                      size="lg"
                    >
                      <TrendingUp className="h-4 w-4" />
                      Buy
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Total: ${(quantity * stock.price).toFixed(2)}
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      placeholder="Quantity"
                      value={sellQuantity}
                      onChange={(e) => setSellQuantity(parseInt(e.target.value) || 0)}
                      min="1"
                      max={hasPosition ? currentPosition.quantity : 0}
                      className="w-24 h-10"
                    />
                    <Button
                      variant="destructive"
                      onClick={() => handleQuickTrade('sell')}
                      className={cn(
                        "flex items-center gap-2 font-semibold",
                        !state.isActive ? "opacity-50 cursor-not-allowed" : "hover:bg-red-600"
                      )}
                      disabled={!state.isActive || !hasPosition || sellQuantity <= 0 || sellQuantity > (currentPosition?.quantity || 0)}
                      size="lg"
                    >
                      <TrendingDown className="h-4 w-4" />
                      Sell
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {hasPosition ? `Max: ${currentPosition.quantity} shares` : 'No position'}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Professional Trading Chart */}
        <div className="relative">
          <SimpleChart symbol={symbol} stockData={stockData} />
          <div className="absolute top-2 right-2">
            <Badge variant="outline" className="text-xs">
              Professional Chart
            </Badge>
          </div>
        </div>

        {/* P&L and Position Details */}
        {hasPosition && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="h-5 w-5" />
                Position Details & P&L
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Position Stats */}
                <div className="space-y-4">
                  <h4 className="font-semibold text-sm text-muted-foreground">Position Information</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs text-muted-foreground">Shares Owned</div>
                      <div className="text-lg font-bold">{currentPosition.quantity}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Average Cost</div>
                      <div className="text-lg font-bold">${currentPosition.averageCost.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Total Invested</div>
                      <div className="text-lg font-bold">${(currentPosition.averageCost * currentPosition.quantity).toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Current Value</div>
                      <div className="text-lg font-bold">${currentPosition.totalValue.toFixed(2)}</div>
                    </div>
                  </div>
                </div>

                {/* P&L Stats */}
                <div className="space-y-4">
                  <h4 className="font-semibold text-sm text-muted-foreground">Profit & Loss</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Unrealized P&L</span>
                      <div className={cn(
                        "text-lg font-bold",
                        currentPosition.pnl >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {currentPosition.pnl >= 0 ? '+' : ''}${currentPosition.pnl.toFixed(2)}
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">P&L Percentage</span>
                      <div className={cn(
                        "text-lg font-bold",
                        currentPosition.pnlPercent >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {currentPosition.pnlPercent >= 0 ? '+' : ''}{currentPosition.pnlPercent.toFixed(2)}%
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Current Price</span>
                      <div className="text-lg font-bold">${stock.price.toFixed(2)}</div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Price Change</span>
                      <div className={cn(
                        "text-sm font-medium",
                        stock.change >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recent Trades */}
              <div className="mt-6 pt-6 border-t">
                <h4 className="font-semibold text-sm text-muted-foreground mb-4">Recent Trades</h4>
                <div className="space-y-2">
                  {state.trades
                    .filter(trade => trade.symbol === symbol)
                    .slice(-3)
                    .reverse()
                    .map((trade) => (
                      <div key={trade.id} className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className={cn(
                            "w-2 h-2 rounded-full",
                            trade.type === 'buy' ? "bg-green-500" : "bg-red-500"
                          )} />
                          <div>
                            <div className="font-medium text-sm">
                              {trade.type.toUpperCase()} {trade.quantity} shares
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {new Date(trade.timestamp).toLocaleDateString()} at {new Date(trade.timestamp).toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-bold">${trade.price.toFixed(2)}</div>
                          <div className="text-xs text-muted-foreground">
                            Total: ${(trade.price * trade.quantity).toFixed(2)}
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </CardContent>
          </Card>
        )}


        {/* Key Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>Key Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Volume2 className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Volume</span>
                </div>
                <div className="text-2xl font-bold">
                  {(stock.volume / 1000000).toFixed(1)}M
                </div>
                <div className="text-xs text-muted-foreground">
                  {stock.volume > technicals.volumeAvg ? 'Above avg' : 'Below avg'}
                </div>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Sector</span>
                </div>
                <div className="text-2xl font-bold">{stock.sector}</div>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Support</span>
                </div>
                <div className="text-2xl font-bold text-red-500">
                  ${technicals.support.toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">
                  {((stock.price - technicals.support) / stock.price * 100).toFixed(1)}% away
                </div>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Resistance</span>
                </div>
                <div className="text-2xl font-bold text-green-500">
                  ${technicals.resistance.toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">
                  {((technicals.resistance - stock.price) / stock.price * 100).toFixed(1)}% away
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* AI Analysis */}
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
                  {Array.isArray(aiAnalysis.reasoning) ? aiAnalysis.reasoning.map((reason, index) => (
                    <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      {reason}
                    </li>
                  )) : (
                    <li className="text-sm text-muted-foreground flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      {aiAnalysis.reasoning || 'Analysis completed'}
                    </li>
                  )}
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

        {/* Comprehensive Analysis Tabs */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Comprehensive Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="technical" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Technical
                </TabsTrigger>
                <TabsTrigger value="monitoring" className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Monitoring
                </TabsTrigger>
                <TabsTrigger value="education" className="flex items-center gap-2">
                  Education
                </TabsTrigger>
              </TabsList>

              <TabsContent value="technical" className="space-y-6">
                <div className="grid grid-cols-1 gap-6">
                  <TradingSignals selectedStock={symbol} />
                  <TechnicalAnalysisWidget symbol={symbol} stockData={stockData} />
                </div>
              </TabsContent>

              <TabsContent value="monitoring" className="space-y-6">
                <RealTimeMonitoring selectedStock={symbol} />
              </TabsContent>

              <TabsContent value="education" className="space-y-6">
                <EducationalModule selectedStock={symbol} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

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
      </div>
    </div>
  )
}
