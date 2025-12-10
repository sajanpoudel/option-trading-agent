"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlertTriangle, Bot, User, Settings, DollarSign, Shield } from "lucide-react"

export type TradingMode = "manual" | "autonomous"

interface TradingModeProps {
  mode: TradingMode
  onModeChange: (mode: TradingMode) => void
  isActive: boolean
  onToggleActive: (active: boolean) => void
}

export function TradingModeSelector({ mode, onModeChange, isActive, onToggleActive }: TradingModeProps) {
  const [riskLevel, setRiskLevel] = useState([3])
  const [positionSize, setPositionSize] = useState([25])
  const [maxDailyTrades, setMaxDailyTrades] = useState("5")

  return (
    <Card className="w-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Trading Controls
          </CardTitle>
          <Badge variant={isActive ? "default" : "secondary"} className="flex items-center gap-1">
            {isActive ? (
              <>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Active
              </>
            ) : (
              "Inactive"
            )}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Trading Mode Toggle */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h3 className="font-medium">Trading Mode</h3>
              {mode === "autonomous" && (
                <Badge variant="outline" className="text-xs">
                  <Bot className="h-3 w-3 mr-1" />
                  AI Controlled
                </Badge>
              )}
            </div>
            <Switch
              checked={mode === "autonomous"}
              onCheckedChange={(checked) => onModeChange(checked ? "autonomous" : "manual")}
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Button
              variant={mode === "manual" ? "default" : "outline"}
              onClick={() => onModeChange("manual")}
              className="flex items-center gap-2"
            >
              <User className="h-4 w-4" />
              Manual
            </Button>
            <Button
              variant={mode === "autonomous" ? "default" : "outline"}
              onClick={() => onModeChange("autonomous")}
              className="flex items-center gap-2"
            >
              <Bot className="h-4 w-4" />
              Autonomous
            </Button>
          </div>
        </div>

        {/* Risk Management */}
        <div className="space-y-4">
          <h3 className="font-medium flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Risk Management
          </h3>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm text-muted-foreground">Risk Level</label>
                <span className="text-sm font-medium">{riskLevel[0]}/10</span>
              </div>
              <Slider value={riskLevel} onValueChange={setRiskLevel} max={10} min={1} step={1} className="w-full" />
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm text-muted-foreground">Position Size</label>
                <span className="text-sm font-medium">{positionSize[0]}%</span>
              </div>
              <Slider
                value={positionSize}
                onValueChange={setPositionSize}
                max={100}
                min={1}
                step={1}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">Max Daily Trades</label>
              <Select value={maxDailyTrades} onValueChange={setMaxDailyTrades}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 Trade</SelectItem>
                  <SelectItem value="3">3 Trades</SelectItem>
                  <SelectItem value="5">5 Trades</SelectItem>
                  <SelectItem value="10">10 Trades</SelectItem>
                  <SelectItem value="unlimited">Unlimited</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* Trading Status */}
        <div className="space-y-3">
          <h3 className="font-medium flex items-center gap-2">
            <DollarSign className="h-4 w-4" />
            Trading Status
          </h3>

          <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
            <span className="text-sm">Enable Trading</span>
            <Switch checked={isActive} onCheckedChange={onToggleActive} />
          </div>

          {mode === "autonomous" && (
            <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 rounded-lg">
              <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5" />
              <div className="text-sm text-amber-800 dark:text-amber-200">
                <p className="font-medium">Autonomous Mode Active</p>
                <p className="text-xs mt-1">AI will execute trades automatically based on analysis</p>
              </div>
            </div>
          )}
        </div>

        {/* Emergency Stop */}
        <Button variant="destructive" className="w-full" onClick={() => onToggleActive(false)} disabled={!isActive}>
          Emergency Stop
        </Button>
      </CardContent>
    </Card>
  )
}
