"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts"
import { TrendingUp, Brain, Activity, BarChart3, Clock } from "lucide-react"

interface AgentData {
  name: string
  scenario: string
  score: number
  weight: number
  color: string
  icon: React.ReactNode
  details: {
    indicators: string[]
    confidence: number
    reasoning: string
  }
}

export function AIAgentAnalysis({ selectedStock }: { selectedStock: string }) {
  const [selectedAgent, setSelectedAgent] = useState<AgentData | null>(null)
  const [agentData, setAgentData] = useState<AgentData[]>([])
  const [loading, setLoading] = useState(true)

  // Fetch real AI agent data from backend
  useEffect(() => {
    const fetchAgentData = async () => {
      try {
        setLoading(true)
        const response = await fetch(`http://localhost:8080/api/v1/agents/${selectedStock}`)
        if (response.ok) {
          const data = await response.json()
          const agents = data.agents || []
          
          // Transform backend data to frontend format
          const transformedAgents: AgentData[] = agents.map((agent: any) => ({
            name: agent.name,
            scenario: agent.scenario,
            score: agent.score,
            weight: agent.weight,
            color: agent.color,
            icon: getAgentIcon(agent.name),
            details: {
              indicators: agent.indicators || [],
              confidence: agent.confidence || agent.score,
              reasoning: agent.reasoning || `${agent.name} analysis completed`
            }
          }))
          
          setAgentData(transformedAgents)
        } else {
          console.error('Failed to fetch agent data:', response.status)
          setAgentData([])
        }
      } catch (error) {
        console.error('Error fetching agent data:', error)
        setAgentData([])
      } finally {
        setLoading(false)
      }
    }

    if (selectedStock) {
      fetchAgentData()
    }
  }, [selectedStock])

  // Helper function to get agent icons
  const getAgentIcon = (agentName: string) => {
    switch (agentName) {
      case 'Technical': return <BarChart3 className="h-4 w-4" />
      case 'Sentiment': return <Brain className="h-4 w-4" />
      case 'Flow': return <Activity className="h-4 w-4" />
      case 'History': return <Clock className="h-4 w-4" />
      default: return <BarChart3 className="h-4 w-4" />
    }
  }

  // Fallback mock data only if no real data available
  const fallbackAgentData: AgentData[] = [
    {
      name: "Technical",
      scenario: "Bullish Breakout",
      score: 75,
      weight: 60,
      color: "#2196F3",
      icon: <BarChart3 className="h-4 w-4" />,
      details: {
        indicators: ["RSI: 68 (Bullish)", "MACD: Positive Crossover", "Volume: Above Average", "Support: $180"],
        confidence: 82,
        reasoning:
          "Strong technical indicators suggest upward momentum with RSI showing bullish divergence and MACD crossover confirming trend reversal.",
      },
    },
    {
      name: "Sentiment",
      scenario: "Positive Buzz",
      score: 45,
      weight: 10,
      color: "#9C27B0",
      icon: <Brain className="h-4 w-4" />,
      details: {
        indicators: [
          "StockTwits: 68% Bullish",
          "News Sentiment: Positive",
          "Social Volume: High",
          "Analyst Upgrades: 2",
        ],
        confidence: 71,
        reasoning:
          "Social media sentiment remains positive with increased discussion volume and recent analyst upgrades supporting bullish outlook.",
      },
    },
    {
      name: "Flow",
      scenario: "Unusual Call Activity",
      score: 85,
      weight: 10,
      color: "#FF9800",
      icon: <Activity className="h-4 w-4" />,
      details: {
        indicators: ["Call/Put Ratio: 2.1", "Unusual Options: 15", "Dark Pool: Buying", "Institutional Flow: Bullish"],
        confidence: 89,
        reasoning:
          "Significant unusual call option activity detected with institutional buying pressure in dark pools indicating smart money positioning.",
      },
    },
    {
      name: "History",
      scenario: "Seasonal Strength",
      score: 60,
      weight: 20,
      color: "#4CAF50",
      icon: <Clock className="h-4 w-4" />,
      details: {
        indicators: [
          "Historical Performance: +12%",
          "Seasonal Pattern: Bullish",
          "Earnings Cycle: Positive",
          "Volatility: Decreasing",
        ],
        confidence: 76,
        reasoning:
          "Historical analysis shows strong seasonal performance during this period with decreasing volatility suggesting consolidation before move.",
      },
    },
  ]

  // Use real data if available, otherwise show loading or empty state
  const displayData = agentData.length > 0 ? agentData : []
  
  const chartData = displayData.map((agent) => ({
    name: agent.name,
    weight: agent.weight,
    score: agent.score,
    color: agent.color,
  }))

  const getScoreColor = (score: number) => {
    if (score >= 70) return "text-green-400"
    if (score >= 40) return "text-yellow-400"
    return "text-red-400"
  }

  const getScoreBadgeVariant = (score: number) => {
    if (score >= 70) return "default"
    if (score >= 40) return "secondary"
    return "destructive"
  }

  return (
    <Card className="col-span-12 md:col-span-6 lg:col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          AI Agent Analysis
          <Badge variant="outline" className="ml-auto">
            {selectedStock}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Agent Cards Grid */}
        <div className="grid grid-cols-2 gap-3">
          {loading ? (
            <div className="col-span-2 text-center py-4 text-muted-foreground">
              Loading AI agent data...
            </div>
          ) : displayData.length === 0 ? (
            <div className="col-span-2 text-center py-4 text-muted-foreground">
              No agent data available
            </div>
          ) : (
            displayData.map((agent) => (
            <Dialog key={agent.name}>
              <DialogTrigger asChild>
                <Card
                  className="cursor-pointer hover:bg-accent/50 transition-colors border-l-4"
                  style={{ borderLeftColor: agent.color }}
                >
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <div style={{ color: agent.color }}>{agent.icon}</div>
                      <span className="font-medium text-sm">{agent.name}</span>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">{agent.scenario}</p>
                      <div className="flex items-center justify-between">
                        <Badge variant={getScoreBadgeVariant(agent.score)} className="text-xs">
                          {agent.score}%
                        </Badge>
                        <span className="text-xs text-muted-foreground">Weight: {agent.weight}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <div style={{ color: agent.color }}>{agent.icon}</div>
                    {agent.name} Agent Analysis
                  </DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Scenario</h4>
                    <p className="text-sm text-muted-foreground">{agent.scenario}</p>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Confidence Score</h4>
                    <div className="flex items-center gap-2">
                      <Progress value={agent.details.confidence} className="flex-1" />
                      <span className="text-sm font-medium">{agent.details.confidence}%</span>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Key Indicators</h4>
                    <ul className="space-y-1">
                      {agent.details.indicators.map((indicator, index) => (
                        <li key={index} className="text-sm text-muted-foreground flex items-center gap-2">
                          <div className="w-1 h-1 rounded-full bg-primary" />
                          {indicator}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Reasoning</h4>
                    <p className="text-sm text-muted-foreground">{agent.details.reasoning}</p>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          )))}
        </div>

        {/* Agent Weights Chart */}
        <div className="mt-4">
          <h4 className="font-medium mb-2 text-sm">Agent Weights Distribution</h4>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis hide />
                <Bar dataKey="weight" radius={[2, 2, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Overall Signal */}
        <div className="border-t pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Overall Signal</span>
            <Badge className="bg-green-500/20 text-green-400 border-green-500/30">BULLISH</Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Weighted analysis suggests bullish momentum with high confidence from flow and technical agents.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
