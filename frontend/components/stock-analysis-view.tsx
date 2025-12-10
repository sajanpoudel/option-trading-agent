"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, TrendingUp, TrendingDown, Activity, BarChart3, PieChart, AlertTriangle } from "lucide-react"
import { AIAgentAnalysis } from "@/components/ai-agent-analysis"
import { TradingSignals } from "@/components/trading-signals"
import { PayoffSurface3D } from "@/components/payoff-surface-3d"
import { RealTimeMonitoring } from "@/components/real-time-monitoring"
import { EducationalModule } from "@/components/educational-module"

interface StockAnalysisViewProps {
  analysisData: {
    stock: string
    query: string
    timestamp: string
  }
  onBack: () => void
}

export function StockAnalysisView({ analysisData, onBack }: StockAnalysisViewProps) {
  const [activeTab, setActiveTab] = useState("overview")

  const mockAnalysis = {
    price: 456.78,
    change: 12.45,
    changePercent: 2.8,
    aiScore: 92,
    sentiment: "Bullish",
    riskLevel: "Medium",
    signals: ["Strong Buy", "Bullish Momentum", "High Volume"],
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="icon" onClick={onBack}>
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <div>
                <h1 className="text-xl font-semibold text-foreground">{analysisData.stock} Analysis</h1>
                <p className="text-sm text-muted-foreground">"{analysisData.query}"</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-2xl font-bold">${mockAnalysis.price.toFixed(2)}</div>
                <div
                  className={`flex items-center gap-1 text-sm ${mockAnalysis.change >= 0 ? "text-green-400" : "text-red-400"}`}
                >
                  {mockAnalysis.change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  +{mockAnalysis.change.toFixed(2)} ({mockAnalysis.changePercent.toFixed(1)}%)
                </div>
              </div>
              <Badge variant="default" className="text-green-400">
                AI Score: {mockAnalysis.aiScore}
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Analysis Content */}
      <main className="container mx-auto px-6 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="technical" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Technical
            </TabsTrigger>
            <TabsTrigger value="options" className="flex items-center gap-2">
              <PieChart className="h-4 w-4" />
              Options
            </TabsTrigger>
            <TabsTrigger value="monitoring" className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Monitoring
            </TabsTrigger>
            <TabsTrigger value="education" className="flex items-center gap-2">
              Education
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AIAgentAnalysis selectedStock={analysisData.stock} />
              <TradingSignals selectedStock={analysisData.stock} />
            </div>
          </TabsContent>

          <TabsContent value="technical" className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <AIAgentAnalysis selectedStock={analysisData.stock} />
              <Card>
                <CardHeader>
                  <CardTitle>Technical Indicators</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">RSI</div>
                      <div className="text-lg font-bold text-yellow-400">68.5</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">MACD</div>
                      <div className="text-lg font-bold text-green-400">Bullish</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">Volume</div>
                      <div className="text-lg font-bold">45.2M</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">Support</div>
                      <div className="text-lg font-bold">$445</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="options" className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <PayoffSurface3D selectedStock={analysisData.stock} />
              <TradingSignals selectedStock={analysisData.stock} />
            </div>
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-6">
            <RealTimeMonitoring selectedStock={analysisData.stock} />
          </TabsContent>

          <TabsContent value="education" className="space-y-6">
            <EducationalModule selectedStock={analysisData.stock} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
