"use client"

import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ShoppingCart, TrendingDown, AlertTriangle, CheckCircle } from "lucide-react"

interface TradeOrder {
  symbol: string
  action: "buy" | "sell"
  orderType: "market" | "limit" | "stop"
  quantity: number
  price?: number
  stopPrice?: number
}

interface TradeConfirmationProps {
  isOpen: boolean
  onClose: () => void
  trade: TradeOrder | null
  currentPrice: number
  onConfirm: () => void
}

export function TradeConfirmationDialog({ isOpen, onClose, trade, currentPrice, onConfirm }: TradeConfirmationProps) {
  const [isExecuting, setIsExecuting] = useState(false)
  const [isExecuted, setIsExecuted] = useState(false)

  if (!trade) return null

  const handleConfirm = async () => {
    setIsExecuting(true)

    await new Promise((resolve) => setTimeout(resolve, 2000))

    setIsExecuting(false)
    setIsExecuted(true)
    onConfirm()

    setTimeout(() => {
      setIsExecuted(false)
      onClose()
    }, 2000)
  }

  const getExecutionPrice = () => {
    if (trade.orderType === "market") return currentPrice
    if (trade.orderType === "limit") return trade.price || currentPrice
    return trade.stopPrice || currentPrice
  }

  const getTotalValue = () => {
    return trade.quantity * getExecutionPrice()
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {trade.action === "buy" ? (
              <ShoppingCart className="h-5 w-5 text-green-600" />
            ) : (
              <TrendingDown className="h-5 w-5 text-red-600" />
            )}
            Confirm {trade.action.toUpperCase()} Order
          </DialogTitle>
          <DialogDescription>Please review your order details before confirming</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Order Details */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Symbol:</span>
              <Badge variant="outline" className="font-mono">
                {trade.symbol}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Action:</span>
              <Badge variant={trade.action === "buy" ? "default" : "destructive"}>{trade.action.toUpperCase()}</Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Order Type:</span>
              <span className="font-medium capitalize">{trade.orderType}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Quantity:</span>
              <span className="font-medium">{trade.quantity} shares</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                {trade.orderType === "market" ? "Current Price:" : "Execution Price:"}
              </span>
              <span className="font-medium">${getExecutionPrice().toFixed(2)}</span>
            </div>
          </div>

          <Separator />

          {/* Total Value */}
          <div className="flex justify-between items-center text-lg font-semibold">
            <span>Total Value:</span>
            <span className={trade.action === "buy" ? "text-green-600" : "text-red-600"}>
              ${getTotalValue().toFixed(2)}
            </span>
          </div>

          {/* Warnings */}
          {trade.orderType === "market" && (
            <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 rounded-lg">
              <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5" />
              <div className="text-sm text-amber-800 dark:text-amber-200">
                <p className="font-medium">Market Order</p>
                <p className="text-xs mt-1">Price may vary from current quote due to market conditions</p>
              </div>
            </div>
          )}

          {/* Success State */}
          {isExecuted && (
            <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800 rounded-lg">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <div className="text-sm text-green-800 dark:text-green-200">
                <p className="font-medium">Order Executed Successfully!</p>
                <p className="text-xs mt-1">
                  Your {trade.action} order for {trade.symbol} has been processed
                </p>
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={onClose} disabled={isExecuting}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={isExecuting || isExecuted}
            className={trade.action === "buy" ? "bg-green-600 hover:bg-green-700" : ""}
            variant={trade.action === "sell" ? "destructive" : "default"}
          >
            {isExecuting ? "Executing..." : isExecuted ? "Executed" : `Confirm ${trade.action.toUpperCase()}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
