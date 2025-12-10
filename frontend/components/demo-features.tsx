"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  TrendingUp, 
  Bot, 
  BarChart3, 
  MessageSquare, 
  Target, 
  DollarSign,
  Activity,
  Shield,
  Zap
} from "lucide-react"

export function DemoFeatures() {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null)

  const features = [
    {
      id: "ai-trading",
      title: "AI Trading Assistant",
      description: "Intelligent trading recommendations powered by advanced AI algorithms",
      icon: Bot,
      color: "text-blue-500",
      bgColor: "bg-blue-50 dark:bg-blue-950",
      features: [
        "Real-time market analysis",
        "Automated trade suggestions",
        "Risk assessment and management",
        "Portfolio optimization"
      ]
    },
    {
      id: "professional-charts",
      title: "Professional Charts",
      description: "TradingView-style charts with advanced technical analysis tools",
      icon: BarChart3,
      color: "text-green-500",
      bgColor: "bg-green-50 dark:bg-green-950",
      features: [
        "Interactive price charts",
        "Technical indicators (RSI, MACD, Moving Averages)",
        "Support and resistance levels",
        "Real-time data updates"
      ]
    },
    {
      id: "chat-interface",
      title: "Interactive Chat",
      description: "Natural language interface for trading and market analysis",
      icon: MessageSquare,
      color: "text-purple-500",
      bgColor: "bg-purple-50 dark:bg-purple-950",
      features: [
        "Ask questions about stocks",
        "Get instant market insights",
        "Execute trades through chat",
        "Portfolio analysis and advice"
      ]
    },
    {
      id: "risk-management",
      title: "Risk Management",
      description: "Advanced risk controls and position sizing tools",
      icon: Shield,
      color: "text-orange-500",
      bgColor: "bg-orange-50 dark:bg-orange-950",
      features: [
        "Position size controls",
        "Risk level settings",
        "Stop-loss management",
        "Portfolio diversification"
      ]
    },
    {
      id: "real-time-monitoring",
      title: "Real-time Monitoring",
      description: "Live market data and portfolio tracking",
      icon: Activity,
      color: "text-red-500",
      bgColor: "bg-red-50 dark:bg-red-950",
      features: [
        "Live price updates",
        "P&L tracking",
        "Trade execution monitoring",
        "Market alerts and notifications"
      ]
    },
    {
      id: "portfolio-management",
      title: "Portfolio Management",
      description: "Comprehensive portfolio tracking and analysis",
      icon: DollarSign,
      color: "text-emerald-500",
      bgColor: "bg-emerald-50 dark:bg-emerald-950",
      features: [
        "Portfolio overview dashboard",
        "Performance analytics",
        "Trade history tracking",
        "P&L visualization"
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            Neural Options Oracle++
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
            The future of AI-powered trading is here. Experience professional-grade tools 
            with intelligent automation and real-time market insights.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="text-lg px-8 py-6">
              <Zap className="mr-2 h-5 w-5" />
              Start Trading Now
            </Button>
            <Button variant="outline" size="lg" className="text-lg px-8 py-6">
              <Target className="mr-2 h-5 w-5" />
              View Demo
            </Button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {features.map((feature) => {
            const Icon = feature.icon
            return (
              <Card 
                key={feature.id}
                className={`cursor-pointer transition-all duration-300 hover:shadow-lg ${
                  selectedFeature === feature.id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => setSelectedFeature(
                  selectedFeature === feature.id ? null : feature.id
                )}
              >
                <CardHeader className="pb-3">
                  <div className={`w-12 h-12 rounded-lg ${feature.bgColor} flex items-center justify-center mb-4`}>
                    <Icon className={`h-6 w-6 ${feature.color}`} />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    {feature.description}
                  </p>
                </CardHeader>
                {selectedFeature === feature.id && (
                  <CardContent className="pt-0">
                    <ul className="space-y-2">
                      {feature.features.map((item, index) => (
                        <li key={index} className="flex items-center text-sm">
                          <div className="w-1.5 h-1.5 rounded-full bg-primary mr-3" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                )}
              </Card>
            )
          })}
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-16">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary mb-2">99.9%</div>
            <div className="text-sm text-muted-foreground">Uptime</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary mb-2">50ms</div>
            <div className="text-sm text-muted-foreground">Latency</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary mb-2">24/7</div>
            <div className="text-sm text-muted-foreground">Monitoring</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary mb-2">AI</div>
            <div className="text-sm text-muted-foreground">Powered</div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center">
          <Card className="max-w-2xl mx-auto">
            <CardContent className="p-8">
              <h2 className="text-2xl font-bold mb-4">Ready to Get Started?</h2>
              <p className="text-muted-foreground mb-6">
                Join thousands of traders who trust Neural Options Oracle++ for their trading needs.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button size="lg" className="text-lg px-8">
                  <TrendingUp className="mr-2 h-5 w-5" />
                  Start Trading
                </Button>
                <Button variant="outline" size="lg" className="text-lg px-8">
                  Learn More
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}