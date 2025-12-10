"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { LineChart, Line, ResponsiveContainer } from "recharts"
import { TrendingUp, TrendingDown, Search, Flame, Eye, ArrowRight } from "lucide-react"

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

interface HotStocksViewProps {
  onStockSelect: (stock: string) => void
}

export function HotStocksView({ onStockSelect }: HotStocksViewProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [hotStocks, setHotStocks] = useState<HotStock[]>([])
  const [loading, setLoading] = useState(true)

  // Fetch real hot stocks data from backend
  useEffect(() => {
    const fetchHotStocks = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8080/api/v1/stocks/hot-stocks')
        if (response.ok) {
          const data = await response.json()
          setHotStocks(data.stocks || [])
        } else {
          console.error('Failed to fetch hot stocks:', response.status)
          // Fallback to empty array - no mock data
          setHotStocks([])
        }
      } catch (error) {
        console.error('Error fetching hot stocks:', error)
        setHotStocks([])
      } finally {
        setLoading(false)
      }
    }

    fetchHotStocks()
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchHotStocks, 30000)
    return () => clearInterval(interval)
  }, [])

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

  const filteredStocks = hotStocks.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.name.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Neural Options Oracle++</h1>
              <p className="text-muted-foreground mt-1">AI-Powered Trading Intelligence</p>
            </div>
            <Badge variant="default" className="gap-2">
              <Flame className="h-4 w-4" />
              Live Market Data
            </Badge>
          </div>

          {/* Search Bar */}
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Hot Stocks Grid */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-6">
            <Flame className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Hot Stocks</h2>
            <Badge variant="secondary" className="text-xs">
              {filteredStocks.filter((s) => s.trending).length} Trending
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredStocks.map((stock) => (
              <Card key={stock.symbol} className="hover:shadow-lg transition-all duration-200 cursor-pointer group">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <CardTitle className="text-lg">{stock.symbol}</CardTitle>
                        {stock.trending && (
                          <Badge variant="outline" className="text-xs px-2 py-0">
                            <Flame className="h-3 w-3 mr-1" />
                            Hot
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">{stock.name}</p>
                    </div>
                    <div className={`text-right ${getAIScoreColor(stock.aiScore)}`}>
                      <div className="text-sm font-medium">AI Score</div>
                      <div className="text-lg font-bold">{stock.aiScore}</div>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  {/* Price Info */}
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-2xl font-bold">${stock.price.toFixed(2)}</div>
                      <div className={`flex items-center gap-1 text-sm ${getPriceChangeColor(stock.change)}`}>
                        {getPriceChangeIcon(stock.change)}
                        {stock.change > 0 ? "+" : ""}
                        {stock.change.toFixed(2)} ({stock.changePercent > 0 ? "+" : ""}
                        {stock.changePercent.toFixed(1)}%)
                      </div>
                    </div>
                    <div className="w-24 h-12">
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
                      <div className="text-muted-foreground">Volume</div>
                      <div className="font-medium">{stock.volume}</div>
                    </div>
                  </div>

                  {/* AI Signals */}
                  <div>
                    <div className="text-sm text-muted-foreground mb-2">AI Signals</div>
                    <div className="flex flex-wrap gap-1">
                      {stock.signals.map((signal, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {signal}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Action Button */}
                  <Button
                    onClick={() => onStockSelect(stock.symbol)}
                    className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                    variant="outline"
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    Analyze {stock.symbol}
                    <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-sm text-muted-foreground">Market Sentiment</div>
              <div className="text-2xl font-bold text-green-400">Bullish</div>
              <div className="text-xs text-muted-foreground">73% positive signals</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-sm text-muted-foreground">Active Signals</div>
              <div className="text-2xl font-bold">24</div>
              <div className="text-xs text-muted-foreground">Across all stocks</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-sm text-muted-foreground">Win Rate</div>
              <div className="text-2xl font-bold text-green-400">78.5%</div>
              <div className="text-xs text-muted-foreground">Last 30 days</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-sm text-muted-foreground">Volatility Index</div>
              <div className="text-2xl font-bold text-yellow-400">Medium</div>
              <div className="text-xs text-muted-foreground">VIX: 18.2</div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
