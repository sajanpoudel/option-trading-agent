"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Activity, BarChart3, PieChart, AlertTriangle, BookOpen } from "lucide-react"
import { AIAgentAnalysis } from "@/components/ai-agent-analysis"
import { TradingSignals } from "@/components/trading-signals"
import { PayoffSurface3D } from "@/components/payoff-surface-3d"
import { RealTimeMonitoring } from "@/components/real-time-monitoring"
import { EducationalModule } from "@/components/educational-module"
import { TechnicalIndicators } from "./technical-indicators"

interface AnalysisTabsProps {
  selectedStock: string
  activeTab: string
  onTabChange: (tab: string) => void
}

export function AnalysisTabs({ selectedStock, activeTab, onTabChange }: AnalysisTabsProps) {
  return (
    <Tabs value={activeTab} onValueChange={onTabChange} className="space-y-6">
      <TabsList className="grid w-full grid-cols-2 sm:grid-cols-5 h-auto">
        <TabsTrigger value="overview" className="flex items-center gap-2 text-xs sm:text-sm">
          <Activity className="h-4 w-4" />
          <span className="hidden sm:inline">Overview</span>
        </TabsTrigger>
        <TabsTrigger value="technical" className="flex items-center gap-2 text-xs sm:text-sm">
          <BarChart3 className="h-4 w-4" />
          <span className="hidden sm:inline">Technical</span>
        </TabsTrigger>
        <TabsTrigger value="options" className="flex items-center gap-2 text-xs sm:text-sm">
          <PieChart className="h-4 w-4" />
          <span className="hidden sm:inline">Options</span>
        </TabsTrigger>
        <TabsTrigger value="monitoring" className="flex items-center gap-2 text-xs sm:text-sm">
          <AlertTriangle className="h-4 w-4" />
          <span className="hidden sm:inline">Monitor</span>
        </TabsTrigger>
        <TabsTrigger value="education" className="flex items-center gap-2 text-xs sm:text-sm">
          <BookOpen className="h-4 w-4" />
          <span className="hidden sm:inline">Learn</span>
        </TabsTrigger>
      </TabsList>

      <TabsContent value="overview" className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AIAgentAnalysis selectedStock={selectedStock} />
          <TradingSignals selectedStock={selectedStock} />
        </div>
      </TabsContent>

      <TabsContent value="technical" className="space-y-6">
        <div className="grid grid-cols-1 gap-6">
          <AIAgentAnalysis selectedStock={selectedStock} />
          <TechnicalIndicators selectedStock={selectedStock} />
        </div>
      </TabsContent>

      <TabsContent value="options" className="space-y-6">
        <div className="grid grid-cols-1 gap-6">
          <PayoffSurface3D selectedStock={selectedStock} />
          <TradingSignals selectedStock={selectedStock} />
        </div>
      </TabsContent>

      <TabsContent value="monitoring" className="space-y-6">
        <RealTimeMonitoring selectedStock={selectedStock} />
      </TabsContent>

      <TabsContent value="education" className="space-y-6">
        <EducationalModule selectedStock={selectedStock} />
      </TabsContent>
    </Tabs>
  )
}
