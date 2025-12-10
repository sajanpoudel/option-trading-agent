"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Activity,
  BarChart3,
  PieChart,
  AlertTriangle,
  BookOpen,
  TrendingUp,
  Brain,
  Target,
  ChevronRight,
  Zap,
  LineChart,
} from "lucide-react"
import { AIAgentAnalysis } from "@/components/ai-agent-analysis"
import { TradingSignals } from "@/components/trading-signals"
import { PayoffSurface3D } from "@/components/payoff-surface-3d"
import { RealTimeMonitoring } from "@/components/real-time-monitoring"
import { EducationalModule } from "@/components/educational-module"
import { TechnicalIndicators } from "./technical-indicators"
import { TradingViewChart } from "../charts/tradingview-chart"
import { MiniChart } from "../charts/mini-chart"
import { TechnicalAnalysisWidget } from "../charts/technical-analysis-widget"
import { useState, useEffect } from "react"
import { getComprehensiveAnalysis, transformAnalysisResponseToStockData, getHotStocks, type StockData } from "@/lib/api-service"

interface AnalysisCardsProps {
  selectedStock: string
}

export function AnalysisCards({ selectedStock }: AnalysisCardsProps) {
  const [expandedCard, setExpandedCard] = useState<string | null>(null)
  const [expandAll, setExpandAll] = useState(false)
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const toggleCard = (cardId: string) => {
    setExpandedCard(expandedCard === cardId ? null : cardId)
  }

  // Load stock data from API
  useEffect(() => {
    const loadStockData = async () => {
      try {
        setLoading(true)
        setError(null)
        const analysisResponse = await getComprehensiveAnalysis(selectedStock)
        const transformedData = await transformAnalysisResponseToStockData(analysisResponse)
        setStockData(transformedData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load stock data')
        console.error('Error loading stock data:', err)
      } finally {
        setLoading(false)
      }
    }

    loadStockData()
  }, [selectedStock])

  useEffect(() => {
    const handleExpandAll = (event: CustomEvent) => {
      if (event.detail.stock === selectedStock) {
        setExpandAll(true)
        setTimeout(() => setExpandedCard("charts"), 100)
        setTimeout(() => setExpandedCard("ai-agents"), 200)
        setTimeout(() => setExpandedCard("signals"), 300)
        setTimeout(() => setExpandedCard("technical"), 400)
        setTimeout(() => setExpandedCard("options"), 500)
      }
    }

    window.addEventListener("expandAllAnalysis", handleExpandAll as EventListener)
    return () => window.removeEventListener("expandAllAnalysis", handleExpandAll as EventListener)
  }, [selectedStock])

  const isCardExpanded = (cardId: string) => {
    return expandAll || expandedCard === cardId
  }

  // Show loading state
  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="h-5 w-5 bg-gray-300 rounded"></div>
                  <div className="h-6 w-16 bg-gray-300 rounded"></div>
                </div>
                <div className="h-6 w-24 bg-gray-300 rounded"></div>
              </CardHeader>
              <CardContent>
                <div className="h-8 w-20 bg-gray-300 rounded mb-2"></div>
                <div className="h-4 w-32 bg-gray-300 rounded"></div>
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="text-center py-8">
          <p className="text-muted-foreground">Loading analysis for {selectedStock}...</p>
        </div>
      </div>
    )
  }

  // Show error state
  if (error || !stockData) {
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <div className="text-red-500 mb-4">
            <AlertTriangle className="h-12 w-12 mx-auto" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Failed to load data</h3>
          <p className="text-muted-foreground mb-4">
            {error || `No data available for ${selectedStock}`}
          </p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="group hover:shadow-lg transition-all duration-300">
        <CardHeader className="cursor-pointer" onClick={() => toggleCard("charts")}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600">
                <LineChart className="h-5 w-5 text-white" />
              </div>
              <div>
                <CardTitle className="text-xl">Live Charts & Analysis</CardTitle>
                <CardDescription>Real-time TradingView charts and technical analysis</CardDescription>
              </div>
            </div>
            <ChevronRight className={`h-5 w-5 transition-transform ${isCardExpanded("charts") ? "rotate-90" : ""}`} />
          </div>
        </CardHeader>
        {isCardExpanded("charts") && (
          <CardContent className="pt-0">
            <Tabs defaultValue="main-chart" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="main-chart">Main Chart</TabsTrigger>
                <TabsTrigger value="technical">Technical Analysis</TabsTrigger>
                <TabsTrigger value="mini-charts">Overview</TabsTrigger>
              </TabsList>

              <TabsContent value="main-chart" className="mt-4">
                <div className="rounded-lg overflow-hidden">
                  <TradingViewChart
                    symbol={selectedStock}
                    height={700}
                    theme="dark"
                    style="1"
                    interval="D"
                    allow_symbol_change={false}
                    studies={[]}
                  />
                </div>
              </TabsContent>

              <TabsContent value="technical" className="mt-4">
                <div className="rounded-lg overflow-hidden">
                  <TechnicalAnalysisWidget symbol={selectedStock} height={400} theme="light" />
                </div>
              </TabsContent>

              <TabsContent value="mini-charts" className="mt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium mb-2">12-Month Overview</h4>
                    <MiniChart symbol={selectedStock} height={200} theme="light" />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-2">Intraday Performance</h4>
                    <MiniChart symbol={selectedStock} height={200} theme="light" />
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        )}
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* AI Score Card */}
        <Card className="border-l-4 border-l-green-500 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <TrendingUp className="h-5 w-5 text-green-600" />
              <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                {loading ? "Loading..." : stockData?.aiAnalysis.recommendation.replace('_', ' ') || "Hold"}
              </Badge>
            </div>
            <CardTitle className="text-lg">AI Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {loading ? "..." : `${stockData?.aiAnalysis.confidence || 0}/100`}
            </div>
            <p className="text-sm text-muted-foreground">
              {loading ? "Loading sentiment..." : `${stockData?.aiAnalysis.sentiment || "neutral"} sentiment detected`}
            </p>
          </CardContent>
        </Card>

        {/* Price Action Card */}
        <Card className="border-l-4 border-l-blue-500 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <Activity className="h-5 w-5 text-blue-600" />
              <Badge variant="outline">Live</Badge>
            </div>
            <CardTitle className="text-lg">Price Action</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? "..." : `$${stockData?.stock.price.toFixed(2) || "0.00"}`}
            </div>
            <p className={`text-sm ${stockData?.stock.change >= 0 ? "text-green-600" : "text-red-600"}`}>
              {loading ? "Loading..." : `${stockData?.stock.change >= 0 ? "+" : ""}$${stockData?.stock.change.toFixed(2) || "0.00"} (${stockData?.stock.changePercent >= 0 ? "+" : ""}${stockData?.stock.changePercent.toFixed(2) || "0.00"}%)`}
            </p>
          </CardContent>
        </Card>

        {/* Risk Level Card */}
        <Card className="border-l-4 border-l-purple-500 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <Target className="h-5 w-5 text-purple-600" />
              <Badge variant="secondary">
                {loading ? "Loading..." : stockData?.aiAnalysis.riskLevel || "Medium"}
              </Badge>
            </div>
            <CardTitle className="text-lg">Risk Level</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">
              {loading ? "..." : `${(100 - (stockData?.aiAnalysis.confidence || 0)).toFixed(1)}/100`}
            </div>
            <p className="text-sm text-muted-foreground">
              {loading ? "Loading..." : `${stockData?.aiAnalysis.riskLevel || "Medium"} volatility`}
            </p>
          </CardContent>
        </Card>

        {/* Signals Card */}
        <Card className="border-l-4 border-l-orange-500 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-950 dark:to-orange-900">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <Zap className="h-5 w-5 text-orange-600" />
              <Badge variant="secondary">
                {loading ? "Loading..." : `${stockData?.aiAnalysis.reasoning.length || 0} Active`}
              </Badge>
            </div>
            <CardTitle className="text-lg">Signals</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">
              {loading ? "..." : stockData?.aiAnalysis.confidence >= 80 ? "Strong" : stockData?.aiAnalysis.confidence >= 60 ? "Moderate" : "Weak"}
            </div>
            <p className="text-sm text-muted-foreground">
              {loading ? "Loading..." : `${stockData?.aiAnalysis.confidence >= 70 ? "High" : "Medium"} confidence`}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="group hover:shadow-lg transition-all duration-300">
          <CardHeader className="cursor-pointer" onClick={() => toggleCard("ai-agents")}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
                  <Brain className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">AI Agent Analysis</CardTitle>
                  <CardDescription>Multi-agent sentiment & technical analysis</CardDescription>
                </div>
              </div>
              <ChevronRight
                className={`h-5 w-5 transition-transform ${isCardExpanded("ai-agents") ? "rotate-90" : ""}`}
              />
            </div>
          </CardHeader>
          {isCardExpanded("ai-agents") && (
            <CardContent className="pt-0">
              <AIAgentAnalysis selectedStock={selectedStock} />
            </CardContent>
          )}
        </Card>

        <Card className="group hover:shadow-lg transition-all duration-300">
          <CardHeader className="cursor-pointer" onClick={() => toggleCard("signals")}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600">
                  <Activity className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">Trading Signals</CardTitle>
                  <CardDescription>Real-time buy/sell recommendations</CardDescription>
                </div>
              </div>
              <ChevronRight
                className={`h-5 w-5 transition-transform ${isCardExpanded("signals") ? "rotate-90" : ""}`}
              />
            </div>
          </CardHeader>
          {isCardExpanded("signals") && (
            <CardContent className="pt-0">
              <TradingSignals selectedStock={selectedStock} />
            </CardContent>
          )}
        </Card>

        <Card className="group hover:shadow-lg transition-all duration-300">
          <CardHeader className="cursor-pointer" onClick={() => toggleCard("technical")}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-orange-500 to-red-600">
                  <BarChart3 className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">Technical Indicators</CardTitle>
                  <CardDescription>RSI, MACD, Bollinger Bands & more</CardDescription>
                </div>
              </div>
              <ChevronRight
                className={`h-5 w-5 transition-transform ${isCardExpanded("technical") ? "rotate-90" : ""}`}
              />
            </div>
          </CardHeader>
          {isCardExpanded("technical") && (
            <CardContent className="pt-0">
              <TechnicalIndicators selectedStock={selectedStock} stockData={stockData} isLoading={loading} />
            </CardContent>
          )}
        </Card>

        <Card className="group hover:shadow-lg transition-all duration-300">
          <CardHeader className="cursor-pointer" onClick={() => toggleCard("options")}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600">
                  <PieChart className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">Options Analysis</CardTitle>
                  <CardDescription>3D payoff surface & Greeks</CardDescription>
                </div>
              </div>
              <ChevronRight
                className={`h-5 w-5 transition-transform ${isCardExpanded("options") ? "rotate-90" : ""}`}
              />
            </div>
          </CardHeader>
          {isCardExpanded("options") && (
            <CardContent className="pt-0">
              <PayoffSurface3D selectedStock={selectedStock} />
            </CardContent>
          )}
        </Card>
      </div>

      <div className="space-y-6">
        <Card className="group hover:shadow-lg transition-all duration-300">
          <CardHeader className="cursor-pointer" onClick={() => toggleCard("monitoring")}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-red-500 to-orange-600">
                  <AlertTriangle className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">Real-Time Monitoring</CardTitle>
                  <CardDescription>Live portfolio tracking & risk alerts</CardDescription>
                </div>
              </div>
              <ChevronRight
                className={`h-5 w-5 transition-transform ${isCardExpanded("monitoring") ? "rotate-90" : ""}`}
              />
            </div>
          </CardHeader>
          {isCardExpanded("monitoring") && (
            <CardContent className="pt-0">
              <RealTimeMonitoring selectedStock={selectedStock} />
            </CardContent>
          )}
        </Card>

        <Card className="group hover:shadow-lg transition-all duration-300">
          <CardHeader className="cursor-pointer" onClick={() => toggleCard("education")}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-indigo-500 to-blue-600">
                  <BookOpen className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">Learning Center</CardTitle>
                  <CardDescription>Educational content & trading insights</CardDescription>
                </div>
              </div>
              <ChevronRight
                className={`h-5 w-5 transition-transform ${isCardExpanded("education") ? "rotate-90" : ""}`}
              />
            </div>
          </CardHeader>
          {isCardExpanded("education") && (
            <CardContent className="pt-0">
              <EducationalModule selectedStock={selectedStock} />
            </CardContent>
          )}
        </Card>
      </div>
    </div>
  )
}
