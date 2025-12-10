"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useTrading } from "@/components/trading/controls/trading-context"
import { ShoppingCart, TrendingDown, Clock, AlertTriangle } from "lucide-react"

interface TradeExecutionProps {
  symbol: string
  currentPrice: number
  recommendation?: {
    action: "buy" | "sell" | "hold"
    confidence: number
    targetPrice: number
    stopLoss: number
  }
  onExecuteTrade: (trade: TradeOrder) => void
}

interface TradeOrder {
  symbol: string
  action: "buy" | "sell"
  orderType: "market" | "limit" | "stop"
  quantity: number
  price?: number
  stopPrice?: number
}

export function TradeExecutionPanel({ symbol, currentPrice, recommendation, onExecuteTrade }: TradeExecutionProps) {
  const { state } = useTrading()
  const [orderType, setOrderType] = useState<"market" | "limit" | "stop">("market")
  const [quantity, setQuantity] = useState<string>("10")
  const [limitPrice, setLimitPrice] = useState<string>(currentPrice.toString())
  const [stopPrice, setStopPrice] = useState<string>("")
  const [isExecuting, setIsExecuting] = useState(false)

  const calculateOrderValue = () => {
    const qty = Number.parseInt(quantity) || 0
    const price = orderType === "market" ? currentPrice : Number.parseFloat(limitPrice) || currentPrice
    return qty * price
  }

  const handleExecuteTrade = async (action: "buy" | "sell") => {
    if (!quantity || Number.parseInt(quantity) <= 0) return

    setIsExecuting(true)

    const trade: TradeOrder = {
      symbol,
      action,
      orderType,
      quantity: Number.parseInt(quantity),
      price: orderType === "limit" ? Number.parseFloat(limitPrice) : undefined,
      stopPrice: orderType === "stop" ? Number.parseFloat(stopPrice) : undefined,
    }

    await new Promise((resolve) => setTimeout(resolve, 1500))

    onExecuteTrade(trade)
    setIsExecuting(false)
  }

  const isValidOrder = () => {
    const qty = Number.parseInt(quantity)
    if (!qty || qty <= 0) return false

    if (orderType === "limit" && (!limitPrice || Number.parseFloat(limitPrice) <= 0)) return false
    if (orderType === "stop" && (!stopPrice || Number.parseFloat(stopPrice) <= 0)) return false

    return true
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShoppingCart className="h-5 w-5" />
          Trade Execution
          <Badge variant={state.isActive ? "default" : "secondary"}>{state.isActive ? "Active" : "Inactive"}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* AI Recommendation Banner */}
        {recommendation && recommendation.action !== "hold" && (
          <div
            className={`p-3 rounded-lg border ${
              recommendation.action === "buy"
                ? "bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-800"
                : "bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              {recommendation.action === "buy" ? (
                <ShoppingCart className="h-4 w-4 text-green-600" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-600" />
              )}
              <span className="font-medium text-sm">AI Recommends: {recommendation.action.toUpperCase()}</span>
              <Badge variant="outline" className="text-xs">
                {recommendation.confidence}% confidence
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              Target: ${recommendation.targetPrice} | Stop Loss: ${recommendation.stopLoss}
            </p>
          </div>
        )}

        {/* Order Configuration */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="quantity">Quantity</Label>
              <Input
                id="quantity"
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                placeholder="Enter shares"
                min="1"
              />
            </div>
            <div>
              <Label htmlFor="orderType">Order Type</Label>
              <Select value={orderType} onValueChange={(value: any) => setOrderType(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="market">Market Order</SelectItem>
                  <SelectItem value="limit">Limit Order</SelectItem>
                  <SelectItem value="stop">Stop Order</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {orderType === "limit" && (
            <div>
              <Label htmlFor="limitPrice">Limit Price</Label>
              <Input
                id="limitPrice"
                type="number"
                value={limitPrice}
                onChange={(e) => setLimitPrice(e.target.value)}
                placeholder="Enter limit price"
                step="0.01"
              />
            </div>
          )}

          {orderType === "stop" && (
            <div>
              <Label htmlFor="stopPrice">Stop Price</Label>
              <Input
                id="stopPrice"
                type="number"
                value={stopPrice}
                onChange={(e) => setStopPrice(e.target.value)}
                placeholder="Enter stop price"
                step="0.01"
              />
            </div>
          )}
        </div>

        {/* Order Summary */}
        <div className="p-3 bg-muted/50 rounded-lg space-y-2">
          <div className="flex justify-between text-sm">
            <span>Current Price:</span>
            <span className="font-medium">${currentPrice.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Estimated Value:</span>
            <span className="font-medium">${calculateOrderValue().toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Order Type:</span>
            <span className="font-medium capitalize">{orderType}</span>
          </div>
        </div>

        {/* Trading Buttons */}
        <div className="grid grid-cols-2 gap-3">
          <Button
            onClick={() => handleExecuteTrade("buy")}
            disabled={!isValidOrder() || !state.isActive || isExecuting}
            className="bg-green-600 hover:bg-green-700 text-white"
          >
            {isExecuting ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <ShoppingCart className="h-4 w-4 mr-2" />
                BUY {symbol}
              </>
            )}
          </Button>

          <Button
            onClick={() => handleExecuteTrade("sell")}
            disabled={!isValidOrder() || !state.isActive || isExecuting}
            variant="destructive"
          >
            {isExecuting ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <TrendingDown className="h-4 w-4 mr-2" />
                SELL {symbol}
              </>
            )}
          </Button>
        </div>

        {/* Warnings */}
        {!state.isActive && (
          <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5" />
            <div className="text-sm text-amber-800 dark:text-amber-200">
              <p className="font-medium">Trading Disabled</p>
              <p className="text-xs mt-1">Enable trading in the controls panel to execute orders</p>
            </div>
          </div>
        )}

        {state.mode === "autonomous" && (
          <div className="text-xs text-muted-foreground text-center p-2 bg-blue-50 dark:bg-blue-950/20 rounded">
            Autonomous mode active - AI will execute recommended trades automatically
          </div>
        )}
      </CardContent>
    </Card>
  )
}
