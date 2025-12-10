"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ChatHeaderProps {
  selectedStock: string
  onToggleCollapse: () => void
  isCollapsed: boolean
  className?: string
}

export function ChatHeader({ selectedStock, onToggleCollapse, isCollapsed, className }: ChatHeaderProps) {
  return (
    <div className={cn("flex items-center justify-between p-4 border-b border-border bg-card/50", className)}>
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        <div>
          <h3 className="font-semibold text-sm">AI Trading Assistant</h3>
          {selectedStock && <p className="text-xs text-muted-foreground">Analyzing {selectedStock}</p>}
        </div>
      </div>

      <Button variant="ghost" size="sm" onClick={onToggleCollapse} className="h-8 w-8 p-0 lg:hidden">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </Button>
    </div>
  )
}
