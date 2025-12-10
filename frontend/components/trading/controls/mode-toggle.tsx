"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Bot, User, Play, Pause, Settings } from "lucide-react"
import { cn } from "@/lib/utils"
import { useTrading } from "./trading-context"

interface ModeToggleProps {
  className?: string
}

export function ModeToggle({ className }: ModeToggleProps) {
  const { state, setMode, setActive } = useTrading()

  const handleModeChange = (mode: "manual" | "autonomous") => {
    setMode(mode)
  }

  const handleActiveToggle = (active: boolean) => {
    setActive(active)
  }

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Trading Mode
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Mode Selection */}
        <div className="space-y-3">
          <div className="text-sm font-medium">Trading Mode</div>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant={state.mode === "manual" ? "default" : "outline"}
              onClick={() => handleModeChange("manual")}
              className="flex items-center gap-2 h-auto p-4"
            >
              <User className="h-4 w-4" />
              <div className="text-left">
                <div className="font-medium">Manual</div>
                <div className="text-xs opacity-70">You control trades</div>
              </div>
            </Button>
            <Button
              variant={state.mode === "autonomous" ? "default" : "outline"}
              onClick={() => handleModeChange("autonomous")}
              className="flex items-center gap-2 h-auto p-4"
            >
              <Bot className="h-4 w-4" />
              <div className="text-left">
                <div className="font-medium">Autonomous</div>
                <div className="text-xs opacity-70">AI controls trades</div>
              </div>
            </Button>
          </div>
        </div>

        {/* Active Toggle */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
          <div className="space-y-1">
            <div className="text-sm font-medium">Trading Active</div>
            <div className="text-xs text-muted-foreground">
              {state.isActive ? "Trading is enabled" : "Trading is disabled"}
            </div>
          </div>
          <Switch
            checked={state.isActive}
            onCheckedChange={handleActiveToggle}
          />
        </div>

        {/* Current Status */}
        <div className="space-y-2">
          <div className="text-sm font-medium">Current Status</div>
          <div className="flex items-center gap-2">
            <Badge 
              variant={state.mode === "autonomous" ? "default" : "secondary"}
              className="flex items-center gap-1"
            >
              {state.mode === "autonomous" ? (
                <Bot className="h-3 w-3" />
              ) : (
                <User className="h-3 w-3" />
              )}
              {state.mode === "autonomous" ? "Autonomous" : "Manual"}
            </Badge>
            <Badge 
              variant={state.isActive ? "default" : "destructive"}
              className="flex items-center gap-1"
            >
              {state.isActive ? (
                <Play className="h-3 w-3" />
              ) : (
                <Pause className="h-3 w-3" />
              )}
              {state.isActive ? "Active" : "Inactive"}
            </Badge>
          </div>
        </div>

        {/* Mode Descriptions */}
        <div className="space-y-3 pt-2 border-t">
          {state.mode === "manual" ? (
            <div className="space-y-2">
              <div className="text-sm font-medium text-blue-500">Manual Mode</div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>• You have full control over all trading decisions</p>
                <p>• AI provides analysis and recommendations</p>
                <p>• You manually execute buy/sell orders</p>
                <p>• Perfect for learning and cautious trading</p>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-sm font-medium text-green-500">Autonomous Mode</div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>• AI analyzes markets and executes trades automatically</p>
                <p>• Trades based on technical analysis and risk parameters</p>
                <p>• Monitors positions and adjusts as needed</p>
                <p>• Ideal for hands-off trading strategies</p>
              </div>
            </div>
          )}
        </div>

        {/* Risk Parameters Display */}
        <div className="space-y-2 pt-2 border-t">
          <div className="text-sm font-medium">Current Settings</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Risk Level:</span>
              <span className="font-medium">{state.riskLevel}/10</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Position Size:</span>
              <span className="font-medium">{state.positionSize}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Max Daily Trades:</span>
              <span className="font-medium">{state.maxDailyTrades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Current Trades:</span>
              <span className="font-medium">{state.currentTrades}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
