"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Settings, DollarSign, Target, Shield, Activity } from "lucide-react"
import { useTrading } from "./trading-context"

export function TradingSettings() {
  const { state, setActive, setMode, updateRiskLevel, updatePositionSize, updateMaxDailyTrades } = useTrading()
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="p-2"
      >
        <Settings className="h-4 w-4" />
      </Button>

      {isOpen && (
        <Card className="absolute top-12 right-0 z-50 w-80 shadow-lg">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Trading Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Trading Toggle */}
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label className="text-sm font-medium">Trading</Label>
                <div className="flex items-center gap-2">
                  <Activity className="h-3 w-3" />
                  <span className="text-xs text-muted-foreground">
                    {state.isActive ? "Enabled" : "Disabled"}
                  </span>
                </div>
              </div>
              <Switch
                checked={state.isActive}
                onCheckedChange={setActive}
              />
            </div>

            {/* Trading Mode */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Trading Mode</Label>
              <div className="flex gap-2">
                <Button
                  variant={state.mode === "manual" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMode("manual")}
                  className="flex-1"
                >
                  Manual
                </Button>
                <Button
                  variant={state.mode === "autonomous" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMode("autonomous")}
                  className="flex-1"
                >
                  Autonomous
                </Button>
              </div>
            </div>

            {/* Risk Level */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Risk Level</Label>
                <Badge variant="outline">{state.riskLevel}/5</Badge>
              </div>
              <Slider
                value={[state.riskLevel]}
                onValueChange={([value]) => updateRiskLevel(value)}
                max={5}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Conservative</span>
                <span>Aggressive</span>
              </div>
            </div>

            {/* Position Size */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Position Size</Label>
                <Badge variant="outline">{state.positionSize}%</Badge>
              </div>
              <Slider
                value={[state.positionSize]}
                onValueChange={([value]) => updatePositionSize(value)}
                max={100}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>1%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Max Daily Trades */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Max Daily Trades</Label>
              <Input
                type="number"
                value={state.maxDailyTrades}
                onChange={(e) => updateMaxDailyTrades(e.target.value)}
                min="1"
                max="50"
                className="h-8"
              />
            </div>

            {/* Current Status */}
            <div className="pt-2 border-t space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Cash Balance</span>
                <span className="font-medium">${state.cashBalance.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Total P&L</span>
                <span className={`font-medium ${state.totalPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {state.totalPnL >= 0 ? '+' : ''}${state.totalPnL.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Today's P&L</span>
                <span className={`font-medium ${state.todayPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {state.todayPnL >= 0 ? '+' : ''}${state.todayPnL.toFixed(2)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
