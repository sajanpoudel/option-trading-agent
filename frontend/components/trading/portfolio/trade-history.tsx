"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { History, TrendingUp, TrendingDown, Clock, CheckCircle } from "lucide-react"

interface Trade {
  id: string
  symbol: string
  action: "buy" | "sell"
  quantity: number
  price: number
  total: number
  timestamp: Date
  status: "executed" | "pending" | "cancelled"
  pnl?: number
}

const mockTrades: Trade[] = [
  {
    id: "1",
    symbol: "AAPL",
    action: "buy",
    quantity: 10,
    price: 150.0,
    total: 1500.0,
    timestamp: new Date("2024-01-15T10:30:00"),
    status: "executed",
    pnl: 155.0,
  },
  {
    id: "2",
    symbol: "TSLA",
    action: "buy",
    quantity: 5,
    price: 220.0,
    total: 1100.0,
    timestamp: new Date("2024-01-15T11:45:00"),
    status: "executed",
    pnl: 129.0,
  },
  {
    id: "3",
    symbol: "NVDA",
    action: "sell",
    quantity: 3,
    price: 185.5,
    total: 556.5,
    timestamp: new Date("2024-01-15T14:20:00"),
    status: "executed",
    pnl: -25.5,
  },
  {
    id: "4",
    symbol: "MSFT",
    action: "buy",
    quantity: 8,
    price: 380.0,
    total: 3040.0,
    timestamp: new Date("2024-01-15T15:10:00"),
    status: "pending",
  },
]

export function TradeHistory() {
  const [selectedTab, setSelectedTab] = useState("all")

  const filteredTrades = mockTrades.filter((trade) => {
    if (selectedTab === "all") return true
    if (selectedTab === "executed") return trade.status === "executed"
    if (selectedTab === "pending") return trade.status === "pending"
    return true
  })

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(amount)
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "executed":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "pending":
        return <Clock className="h-4 w-4 text-yellow-500" />
      default:
        return null
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "executed":
        return "text-green-600 dark:text-green-400"
      case "pending":
        return "text-yellow-600 dark:text-yellow-400"
      case "cancelled":
        return "text-red-600 dark:text-red-400"
      default:
        return "text-muted-foreground"
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Trade History
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="all">All Trades</TabsTrigger>
            <TabsTrigger value="executed">Executed</TabsTrigger>
            <TabsTrigger value="pending">Pending</TabsTrigger>
          </TabsList>

          <TabsContent value={selectedTab} className="mt-4">
            <div className="space-y-3">
              {filteredTrades.map((trade) => (
                <div
                  key={trade.id}
                  className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(trade.status)}
                      <Badge variant={trade.action === "buy" ? "default" : "destructive"} className="text-xs">
                        {trade.action.toUpperCase()}
                      </Badge>
                    </div>

                    <div>
                      <div className="font-semibold">{trade.symbol}</div>
                      <div className="text-sm text-muted-foreground">
                        {trade.quantity} shares @ {formatCurrency(trade.price)}
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-semibold">{formatCurrency(trade.total)}</div>
                    <div className="flex items-center gap-2 text-sm">
                      <span className={getStatusColor(trade.status)}>{trade.status}</span>
                      <span className="text-muted-foreground">{formatTime(trade.timestamp)}</span>
                    </div>
                    {trade.pnl !== undefined && (
                      <div
                        className={`text-sm flex items-center gap-1 justify-end mt-1 ${
                          trade.pnl >= 0 ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {trade.pnl >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                        {formatCurrency(Math.abs(trade.pnl))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {filteredTrades.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">No trades found for the selected filter</div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
