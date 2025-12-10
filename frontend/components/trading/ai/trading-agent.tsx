"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { useTrading } from "@/components/trading/controls/trading-context"
import { Brain, TrendingUp, TrendingDown, AlertCircle, CheckCircle, Clock, Zap } from "lucide-react"

interface TradingSignal {
  action: "buy" | "sell" | "hold"
  confidence: number
  reasoning: string
  targetPrice: number
  stopLoss: number
  timeframe: string
}

interface AnalysisStep {
  id: string
  name: string
  status: "pending" | "analyzing" | "complete"
  result?: string
}

interface TradingAgentProps {
  symbol: string
  currentPrice: number
  onTradeRecommendation: (signal: TradingSignal) => void
}

export function TradingAgent({ symbol, currentPrice, onTradeRecommendation }: TradingAgentProps) {
  const { state } = useTrading()
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [currentSignal, setCurrentSignal] = useState<TradingSignal | null>(null)
  const [analysisSteps, setAnalysisSteps] = useState<AnalysisStep[]>([
    { id: "technical", name: "Technical Analysis", status: "pending" },
    { id: "sentiment", name: "Market Sentiment", status: "pending" },
    { id: "volume", name: "Volume Analysis", status: "pending" },
    { id: "risk", name: "Risk Assessment", status: "pending" },
    { id: "decision", name: "Trading Decision", status: "pending" },
  ])

  const startAnalysis = async () => {
    setIsAnalyzing(true)
    setAnalysisProgress(0)
    setCurrentSignal(null)

    for (let i = 0; i < analysisSteps.length; i++) {
      setAnalysisSteps((prev) =>
        prev.map((step, index) =>
          index === i ? { ...step, status: "analyzing" } : index < i ? { ...step, status: "complete" } : step,
        ),
      )

      // Simulate analysis time
      await new Promise((resolve) => setTimeout(resolve, 1500))

      setAnalysisProgress(((i + 1) / analysisSteps.length) * 100)
    }

    const signal = generateTradingSignal()
    setCurrentSignal(signal)
    setIsAnalyzing(false)

    setAnalysisSteps((prev) => prev.map((step) => ({ ...step, status: "complete" })))

    onTradeRecommendation(signal)
  }

  const generateTradingSignal = (): TradingSignal => {
    const actions: ("buy" | "sell" | "hold")[] = ["buy", "sell", "hold"]
    const randomAction = actions[Math.floor(Math.random() * actions.length)]
    const confidence = Math.floor(Math.random() * 40) + 60 // 60-100%

    const priceVariation = currentPrice * 0.05 // 5% variation
    const targetPrice =
      randomAction === "buy"
        ? currentPrice + priceVariation
        : randomAction === "sell"
          ? currentPrice - priceVariation
          : currentPrice

    const stopLoss =
      randomAction === "buy" ? currentPrice * 0.95 : randomAction === "sell" ? currentPrice * 1.05 : currentPrice

    return {
      action: randomAction,
      confidence,
      reasoning: getReasoningForAction(randomAction, confidence),
      targetPrice: Number(targetPrice.toFixed(2)),
      stopLoss: Number(stopLoss.toFixed(2)),
      timeframe: "1-3 days",
    }
  }

  const getReasoningForAction = (action: string, confidence: number): string => {
    const reasons = {
      buy: [
        "Strong upward momentum detected with increasing volume",
        "Technical indicators showing oversold conditions with reversal signals",
        "Positive market sentiment and institutional buying pressure",
      ],
      sell: [
        "Overbought conditions with bearish divergence patterns",
        "Resistance level reached with declining volume",
        "Market sentiment turning negative with profit-taking signals",
      ],
      hold: [
        "Mixed signals in technical analysis requiring more data",
        "Sideways consolidation pattern with unclear direction",
        "Waiting for clearer market direction and volume confirmation",
      ],
    }

    const actionReasons = reasons[action as keyof typeof reasons]
    return actionReasons[Math.floor(Math.random() * actionReasons.length)]
  }

  useEffect(() => {
    if (state.mode === "autonomous" && state.isActive && currentSignal && currentSignal.action !== "hold") {
      const timer = setTimeout(() => {
        console.log(`[v0] Auto-executing ${currentSignal.action} signal for ${symbol}`)
        // Auto-execute trade logic would go here
      }, 3000)

      return () => clearTimeout(timer)
    }
  }, [currentSignal, state.mode, state.isActive, symbol])

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          AI Trading Agent
          <Badge variant={state.isActive ? "default" : "secondary"}>
            {state.mode === "autonomous" ? "Autonomous" : "Manual"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Analysis Controls */}
        <div className="flex items-center gap-3">
          <Button onClick={startAnalysis} disabled={isAnalyzing} className="flex items-center gap-2">
            {isAnalyzing ? (
              <>
                <Clock className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Zap className="h-4 w-4" />
                Analyze {symbol}
              </>
            )}
          </Button>

          {isAnalyzing && (
            <div className="flex-1">
              <Progress value={analysisProgress} className="h-2" />
            </div>
          )}
        </div>

        {/* Analysis Steps */}
        {isAnalyzing && (
          <div className="space-y-2">
            {analysisSteps.map((step) => (
              <div key={step.id} className="flex items-center gap-3 p-2 rounded-lg bg-muted/50">
                {step.status === "complete" && <CheckCircle className="h-4 w-4 text-green-500" />}
                {step.status === "analyzing" && <Clock className="h-4 w-4 text-blue-500 animate-spin" />}
                {step.status === "pending" && (
                  <div className="h-4 w-4 rounded-full border-2 border-muted-foreground/30" />
                )}
                <span className={`text-sm ${step.status === "complete" ? "text-green-600 dark:text-green-400" : ""}`}>
                  {step.name}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Trading Signal */}
        {currentSignal && (
          <div className="space-y-4 p-4 border rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2">
                {currentSignal.action === "buy" && <TrendingUp className="h-4 w-4 text-green-500" />}
                {currentSignal.action === "sell" && <TrendingDown className="h-4 w-4 text-red-500" />}
                {currentSignal.action === "hold" && <AlertCircle className="h-4 w-4 text-yellow-500" />}
                {currentSignal.action.toUpperCase()} Signal
              </h3>
              <Badge variant={currentSignal.confidence > 80 ? "default" : "secondary"}>
                {currentSignal.confidence}% Confidence
              </Badge>
            </div>

            <p className="text-sm text-muted-foreground">{currentSignal.reasoning}</p>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Target Price:</span>
                <p className="font-medium">${currentSignal.targetPrice}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Stop Loss:</span>
                <p className="font-medium">${currentSignal.stopLoss}</p>
              </div>
            </div>

            <div className="text-xs text-muted-foreground">Timeframe: {currentSignal.timeframe}</div>

            {state.mode === "autonomous" && state.isActive && currentSignal.action !== "hold" && (
              <div className="flex items-center gap-2 p-2 bg-blue-100 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                <Clock className="h-4 w-4 text-blue-600 animate-pulse" />
                <span className="text-sm text-blue-800 dark:text-blue-200">Auto-executing in 3 seconds...</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
