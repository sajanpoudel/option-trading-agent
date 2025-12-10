"use client"

import { Card, CardContent } from "@/components/ui/card"

export function MarketStats() {
  const stats = [
    {
      label: "S&P 500",
      value: "+0.8%",
      subtext: "4,783.35",
      color: "text-green-400",
    },
    {
      label: "NASDAQ",
      value: "+1.2%",
      subtext: "15,055.65",
      color: "text-green-400",
    },
    {
      label: "VIX",
      value: "18.2",
      subtext: "Volatility Index",
      color: "text-yellow-400",
    },
    {
      label: "10Y Treasury",
      value: "4.25%",
      subtext: "-0.05 Today",
      color: "text-foreground",
    },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <Card key={index}>
          <CardContent className="p-4">
            <div className="text-sm text-muted-foreground">{stat.label}</div>
            <div className={`text-xl sm:text-2xl font-bold ${stat.color}`}>{stat.value}</div>
            <div className="text-xs text-muted-foreground">{stat.subtext}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
