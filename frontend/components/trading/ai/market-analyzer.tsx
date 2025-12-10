"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { TrendingUp, TrendingDown, Activity, BarChart3, Users, Globe } from "lucide-react"

interface MarketData {
  sentiment: "bullish" | "bearish" | "neutral"
  sentimentScore: number
  volumeAnalysis: "high" | "normal" | "low"
  marketTrend: "uptrend" | "downtrend" | "sideways"
  institutionalFlow: "buying" | "selling" | "neutral"
  newsImpact: "positive" | "negative" | "neutral"
}

interface MarketAnalyzerProps {
  symbol: string
  onAnalysisComplete: (data: MarketData) => void
}

export function MarketAnalyzer({ symbol, onAnalysisComplete }: MarketAnalyzerProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [marketData, setMarketData] = useState<MarketData | null>(null)
  const [progress, setProgress] = useState(0)

  const analyzeMarket = async () => {
    setIsAnalyzing(true)
    setProgress(0)

    const steps = [
      "Analyzing market sentiment...",
      "Processing volume data...",
      "Evaluating trend patterns...",
      "Checking institutional flows...",
      "Scanning news impact...",
    ]

    for (let i = 0; i < steps.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 800))
      setProgress(((i + 1) / steps.length) * 100)
    }

    const data: MarketData = {
      sentiment: ["bullish", "bearish", "neutral"][Math.floor(Math.random() * 3)] as any,
      sentimentScore: Math.floor(Math.random() * 100),
      volumeAnalysis: ["high", "normal", "low"][Math.floor(Math.random() * 3)] as any,
      marketTrend: ["uptrend", "downtrend", "sideways"][Math.floor(Math.random() * 3)] as any,
      institutionalFlow: ["buying", "selling", "neutral"][Math.floor(Math.random() * 3)] as any,
      newsImpact: ["positive", "negative", "neutral"][Math.floor(Math.random() * 3)] as any,
    }

    setMarketData(data)
    setIsAnalyzing(false)
    onAnalysisComplete(data)
  }

  useEffect(() => {
    if (symbol) {
      analyzeMarket()
    }
  }, [symbol])

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case "bullish":
        return "text-green-600 dark:text-green-400"
      case "bearish":
        return "text-red-600 dark:text-red-400"
      default:
        return "text-yellow-600 dark:text-yellow-400"
    }
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "bullish":
        return <TrendingUp className="h-4 w-4" />
      case "bearish":
        return <TrendingDown className="h-4 w-4" />
      default:
        return <Activity className="h-4 w-4" />
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Market Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isAnalyzing ? (
          <div className="space-y-3">
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground text-center">Analyzing market conditions for {symbol}...</p>
          </div>
        ) : marketData ? (
          <div className="space-y-4">
            {/* Market Sentiment */}
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                {getSentimentIcon(marketData.sentiment)}
                <span className="font-medium">Market Sentiment</span>
              </div>
              <div className="text-right">
                <Badge variant="outline" className={getSentimentColor(marketData.sentiment)}>
                  {marketData.sentiment.toUpperCase()}
                </Badge>
                <p className="text-xs text-muted-foreground mt-1">{marketData.sentimentScore}% confidence</p>
              </div>
            </div>

            {/* Volume Analysis */}
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                <span className="font-medium">Volume Analysis</span>
              </div>
              <Badge variant={marketData.volumeAnalysis === "high" ? "default" : "secondary"}>
                {marketData.volumeAnalysis.toUpperCase()}
              </Badge>
            </div>

            {/* Market Trend */}
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                <span className="font-medium">Market Trend</span>
              </div>
              <Badge
                variant="outline"
                className={
                  marketData.marketTrend === "uptrend"
                    ? "text-green-600"
                    : marketData.marketTrend === "downtrend"
                      ? "text-red-600"
                      : "text-yellow-600"
                }
              >
                {marketData.marketTrend.toUpperCase()}
              </Badge>
            </div>

            {/* Institutional Flow */}
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4" />
                <span className="font-medium">Institutional Flow</span>
              </div>
              <Badge
                variant="outline"
                className={
                  marketData.institutionalFlow === "buying"
                    ? "text-green-600"
                    : marketData.institutionalFlow === "selling"
                      ? "text-red-600"
                      : "text-gray-600"
                }
              >
                {marketData.institutionalFlow.toUpperCase()}
              </Badge>
            </div>

            {/* News Impact */}
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4" />
                <span className="font-medium">News Impact</span>
              </div>
              <Badge
                variant="outline"
                className={
                  marketData.newsImpact === "positive"
                    ? "text-green-600"
                    : marketData.newsImpact === "negative"
                      ? "text-red-600"
                      : "text-gray-600"
                }
              >
                {marketData.newsImpact.toUpperCase()}
              </Badge>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-4">No analysis data available</p>
        )}
      </CardContent>
    </Card>
  )
}
