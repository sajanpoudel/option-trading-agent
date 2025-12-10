"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { DemoFeatures } from "@/components/demo-features"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { ArrowRight, Bot, User } from "lucide-react"

export default function TradingDashboard() {
  const [showDemo, setShowDemo] = useState(false)
  const [startTrading, setStartTrading] = useState(false)

  if (startTrading) {
    return <MainLayout />
  }

  if (showDemo) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-4xl font-bold">OptionsOracle</h1>
            <div className="flex gap-2">
              <Button onClick={() => setShowDemo(false)} variant="outline">
                <ArrowRight className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
              <Button onClick={() => setStartTrading(true)}>
                <User className="h-4 w-4 mr-2" />
                Start Trading
              </Button>
            </div>
          </div>
          <DemoFeatures />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Landing Page */}
      <div className="flex flex-col items-center justify-center min-h-screen space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
            OptionsOracle
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl">
            AI-Powered Trading Platform with Interactive Chat Interface
          </p>
          <p className="text-muted-foreground max-w-3xl">
            Experience the future of trading with autonomous AI agents, real-time analysis, 
            and seamless mode switching between manual and automated trading.
          </p>
        </div>

        <div className="flex gap-4">
          <Button size="lg" onClick={() => setShowDemo(true)} variant="outline">
            <Bot className="h-5 w-5 mr-2" />
            View Features
          </Button>
          <Button size="lg" onClick={() => setStartTrading(true)}>
            <User className="h-5 w-5 mr-2" />
            Start Trading
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12 max-w-4xl">
          <div className="text-center space-y-2">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
              <Bot className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-semibold">Autonomous Trading</h3>
            <p className="text-sm text-muted-foreground">
              AI handles analysis and execution automatically
            </p>
          </div>
          <div className="text-center space-y-2">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
              <User className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-semibold">Manual Control</h3>
            <p className="text-sm text-muted-foreground">
              Full control with AI-powered recommendations
            </p>
          </div>
          <div className="text-center space-y-2">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
              <ArrowRight className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-semibold">Seamless Switching</h3>
            <p className="text-sm text-muted-foreground">
              Switch modes instantly based on your needs
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
