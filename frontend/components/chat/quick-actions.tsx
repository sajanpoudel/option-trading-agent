"use client"

import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { TrendingUp, Activity, BarChart3, Zap, ChevronDown } from "lucide-react"
import { useState } from "react"

interface QuickActionsProps {
  onAction: (action: string) => void
  className?: string
}

const quickActions = [
  {
    label: "Hot Stocks",
    action: "show hot stocks",
    icon: TrendingUp,
    description: "Trending opportunities",
  },
  {
    label: "Analyze AAPL",
    action: "analyze AAPL",
    icon: Activity,
    description: "Full technical analysis",
  },
  {
    label: "Analyze TSLA",
    action: "analyze TSLA",
    icon: BarChart3,
    description: "Complete breakdown",
  },
  {
    label: "Analyze NVDA",
    action: "analyze NVDA",
    icon: Zap,
    description: "AI & charts",
  },
]

export function QuickActions({ onAction, className }: QuickActionsProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className={cn("w-full", className)}
    >
      <CollapsibleTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "w-full flex items-center justify-between p-2 text-xs",
            "text-muted-foreground hover:text-primary-foreground",
            "hover:bg-primary hover:text-primary-foreground",
            "transition-all duration-200"
          )}
        >
          <span className="font-medium">Quick Actions</span>
          <ChevronDown 
            className={cn(
              "h-4 w-4 transition-transform duration-200",
              isOpen && "transform rotate-180"
            )} 
          />
        </Button>
      </CollapsibleTrigger>

      <CollapsibleContent className="pt-2">
        <div className="grid grid-cols-2 gap-2">
          {quickActions.map((action) => {
            const IconComponent = action.icon
            return (
              <Button
                key={action.action}
                variant="outline"
                size="sm"
                onClick={() => onAction(action.action)}
                className={cn(
                  "text-xs h-auto p-2 justify-start flex-col items-start gap-1",
                  "bg-background hover:bg-primary",
                  "text-foreground hover:text-primary-foreground",
                  "border-border hover:border-primary",
                  "group transition-all duration-200"
                )}
              >
                <div className="flex items-center gap-1.5 w-full">
                  <IconComponent className="h-3 w-3" />
                  <span className="font-medium">{action.label}</span>
                </div>
                <span className="text-[10px] text-muted-foreground group-hover:text-primary-foreground/90">
                  {action.description}
                </span>
              </Button>
            )
          })}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}