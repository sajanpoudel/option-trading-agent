"use client"

import { Badge } from "@/components/ui/badge"
import { Flame } from "lucide-react"

export function HotStocksHeader() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30">
      <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground">Options Oracle</h1>
            <p className="text-muted-foreground mt-1 text-sm sm:text-base">Real-Time Market Analysis</p>
          </div>
          <Badge variant="default" className="gap-2 self-start sm:self-auto">
            <Flame className="h-4 w-4" />
            Live Market Data
          </Badge>
        </div>
      </div>
    </header>
  )
}
