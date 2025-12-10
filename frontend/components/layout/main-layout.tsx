"use client"

import { useState, useEffect } from "react"
import { ChatContainer } from "@/components/chat/chat-container"
import { HotStocksContainer } from "@/components/trading/hot-stocks/hot-stocks-container"
import { ScrollableStockView } from "@/components/trading/analysis/scrollable-stock-view"
import { PnLDashboard } from "@/components/trading/portfolio/pnl-dashboard"
import { PortfolioOverview } from "@/components/trading/portfolio/portfolio-overview"
import { TradingProvider } from "@/components/trading/controls/trading-context"
import { cn } from "@/lib/utils"

export type ViewType = "hot-stocks" | "portfolio" | "stock-details"

interface MainLayoutProps {
  className?: string
}

export function MainLayout({ className }: MainLayoutProps) {
  const [currentView, setCurrentView] = useState<ViewType>("hot-stocks")
  const [selectedStock, setSelectedStock] = useState<string>("")
  const [isChatCollapsed, setIsChatCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024)
      if (window.innerWidth < 1024) {
        setIsChatCollapsed(true)
      }
    }

    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  const handleStockSelect = (stock: string) => {
    setSelectedStock(stock)
    setCurrentView("stock-details")
  }

  const handleViewChange = (view: ViewType) => {
    setCurrentView(view)
  }

  const toggleChat = () => {
    setIsChatCollapsed(!isChatCollapsed)
  }

  const handleBackToHotStocks = () => {
    setCurrentView("hot-stocks")
    setSelectedStock("")
  }

  return (
    <TradingProvider>
      <div className={cn("flex h-screen bg-background overflow-hidden", className)}>
        {/* Main Content Area */}
        <div
          className={cn(
            "flex-1 transition-all duration-300 ease-in-out overflow-hidden",
            !isChatCollapsed && !isMobile ? "mr-[480px]" : "mr-0",
          )}
        >
          <div className="h-full overflow-auto touch-scroll custom-scrollbar">
            {currentView === "hot-stocks" && (
              <div className="space-y-8">
                <div className="container mx-auto px-4 sm:px-6 py-6">
                  <PortfolioOverview />
                </div>
                <HotStocksContainer onStockSelect={handleStockSelect} />
              </div>
            )}
            {currentView === "stock-details" && selectedStock && (
              <ScrollableStockView
                symbol={selectedStock}
                onBack={handleBackToHotStocks}
              />
            )}
            {currentView === "portfolio" && (
              <div className="p-6">
                <PnLDashboard />
              </div>
            )}
          </div>
        </div>

        {/* Chat Sidebar */}
        <div
          className={cn(
            "fixed right-0 top-0 h-full transition-all duration-300 ease-in-out z-50 sidebar-mobile flex",
            isMobile ? "w-full" : "w-[480px]",
            isChatCollapsed ? "translate-x-[calc(100%-24px)]" : "translate-x-0",
          )}
        >
          {/* Collapse Toggle Button */}
          <button
            onClick={toggleChat}
            className={cn(
              "absolute left-0 top-1/2 -translate-x-full -translate-y-1/2",
              "h-24 w-6 bg-card border-l border-y border-border rounded-l-lg",
              "flex items-center justify-center",
              "hover:bg-muted/50 transition-colors",
              "group"
            )}
          >
            <div className={cn(
              "text-xs text-muted-foreground font-medium rotate-180 whitespace-nowrap",
              "transition-opacity duration-200",
              !isChatCollapsed && "opacity-0 group-hover:opacity-100",
            )} style={{ writingMode: "vertical-rl" }}>
              AI Assistant
            </div>
            <div className={cn(
              "absolute inset-0 flex items-center justify-center",
              "text-muted-foreground transition-opacity duration-200",
              isChatCollapsed ? "opacity-100" : "opacity-0"
            )}>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </div>
          </button>

          {/* Main Chat Container */}
          <div className={cn(
            "flex-1 bg-card border-l border-border",
            "h-full flex flex-col",
            isMobile && "chat-mobile"
          )}>
            <ChatContainer
              selectedStock={selectedStock}
              onStockSelect={handleStockSelect}
              onViewChange={handleViewChange}
              onToggleCollapse={toggleChat}
              isCollapsed={isChatCollapsed}
            />
          </div>
        </div>

        {/* Mobile Chat Toggle Button */}
        <button
          onClick={toggleChat}
          className={cn(
            "fixed bottom-6 right-6 z-50",
            "w-14 h-14 bg-primary text-primary-foreground rounded-full",
            "flex items-center justify-center shadow-lg",
            "hover:bg-primary/90 transition-all duration-200",
            "focus-ring tap-highlight-none",
            "animate-pulse-glow",
            isMobile ? "block" : "lg:hidden",
          )}
          aria-label={isChatCollapsed ? "Open chat" : "Close chat"}
        >
          {isChatCollapsed ? (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.955 8.955 0 01-2.697-.413l-3.178 1.059a1 1 0 01-1.271-1.271l1.059-3.178A8.955 8.955 0 013 12a8 8 0 018-8 8 8 0 018 8z"
              />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
        </button>

        {/* Mobile Overlay */}
        {isMobile && !isChatCollapsed && (
          <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={toggleChat} />
        )}
      </div>
    </TradingProvider>
  )
}