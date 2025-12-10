"use client"

import { ScrollArea } from "@/components/ui/scroll-area"
import { forwardRef, useImperativeHandle, useRef, useEffect, useState } from "react"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { TrendingUp, TrendingDown, ArrowRight, Play, Pause, Bot, User, Loader2 } from "lucide-react"
import { getChatAnalysis, transformChatResponseToStockData, type StockData } from "@/lib/api-service"

// StockButton component for async stock data loading
function StockButton({ symbol, onStockSelect }: { symbol: string; onStockSelect: (symbol: string) => void }) {
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadStockData = async () => {
      try {
        const chatResponse = await getChatAnalysis(`analyze ${symbol}`)
        const transformedData = transformChatResponseToStockData(chatResponse)
        setStockData(transformedData)
      } catch (error) {
        console.error(`Failed to load stock data for ${symbol}:`, error)
      } finally {
        setLoading(false)
      }
    }
    loadStockData()
  }, [symbol])

  if (loading) {
    return (
      <Button variant="outline" size="sm" disabled>
        {symbol}...
      </Button>
    )
  }

  if (!stockData) {
    return (
      <Button variant="outline" size="sm" disabled>
        {symbol} (N/A)
      </Button>
    )
  }

  const isPositive = stockData.stock.change >= 0

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => onStockSelect(symbol)}
      className={cn(
        "group relative overflow-hidden transition-all duration-300",
        "bg-card/40 hover:bg-card/60 backdrop-blur-sm",
        "border border-border/50 hover:border-border",
        "shadow-sm hover:shadow-md",
        "hover:scale-105 active:scale-95"
      )}
    >
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1">
          <span className="font-semibold text-sm">{symbol}</span>
          <span className="text-xs text-muted-foreground">
            ${stockData.stock.price.toFixed(2)}
          </span>
        </div>
        <div className={cn(
          "flex items-center gap-1 text-xs",
          isPositive ? "text-green-500" : "text-red-500"
        )}>
          {isPositive ? (
            <TrendingUp className="h-3 w-3" />
          ) : (
            <TrendingDown className="h-3 w-3" />
          )}
          <span>
            {isPositive ? '+' : ''}{stockData.stock.changePercent.toFixed(1)}%
          </span>
        </div>
        <ArrowRight className="h-3 w-3 text-muted-foreground group-hover:text-foreground transition-colors" />
      </div>
    </Button>
  )
}

interface Message {
  id: string
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  stockMention?: string
  actionType?: "analysis" | "trade" | "portfolio" | "general" | "STOCK_ANALYSIS" | "technical_analysis" | "TECHNICAL_ANALYSIS" | "stock_analysis" | "PORTFOLIO_MANAGEMENT"
  interactiveElements?: {
    stockButtons?: string[]
    tradeActions?: { 
      symbol: string; 
      type: 'buy' | 'sell';
      recommendation?: any;
      requiresConfirmation?: boolean;
      analysis?: string;
      budget?: number;
    }[]
    confirmationRequired?: boolean
  }
}

interface MessageListProps {
  messages: Message[]
  isTyping: boolean
  className?: string
  onStockSelect?: (stock: string) => void
  onTradeAction?: (symbol: string, action: 'buy' | 'sell') => void
  onConfirmTrade?: (confirmed: boolean) => void
}

export interface MessageListRef {
  scrollToBottom: () => void
}

export const MessageList = forwardRef<MessageListRef, MessageListProps>(({ 
  messages, 
  isTyping, 
  className, 
  onStockSelect, 
  onTradeAction, 
  onConfirmTrade 
}, ref) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useImperativeHandle(ref, () => ({
    scrollToBottom: () => {
      // Try multiple scroll methods
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({
          behavior: "smooth",
          block: "end",
        })
      }
      
      // Also try scrolling the ScrollArea directly
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight
        }
      }
    }
  }))

  // Auto-scroll when messages change
  useEffect(() => {
    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({
          behavior: "smooth",
          block: "end",
        })
      }
      
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight
        }
      }
    }

    // Scroll immediately and with delays
    scrollToBottom()
    setTimeout(scrollToBottom, 100)
    setTimeout(scrollToBottom, 300)
  }, [messages, isTyping])
  const renderInteractiveElements = (message: Message) => {
    if (!message.interactiveElements) return null

    return (
      <div className="mt-4 space-y-3">
        {message.interactiveElements.stockButtons && (
          <div className="flex flex-wrap gap-2">
            {message.interactiveElements.stockButtons.map((symbol) => (
              <StockButton 
                key={symbol} 
                symbol={symbol} 
                onStockSelect={onStockSelect || (() => {})} 
              />
            ))}
          </div>
        )}

        {/* Removed separate Execute button - using only Confirm/Cancel buttons */}

        {message.interactiveElements.confirmationRequired && (
          <div className="flex gap-2 mt-4">
            <Button
              size="sm"
              variant="default"
              onClick={() => {
                console.log("🔘 Confirm Trade button clicked!")
                onConfirmTrade?.(true)
              }}
              className={cn(
                "flex items-center gap-2 bg-green-500 hover:bg-green-600",
                "transition-all duration-300 hover:shadow-lg hover:translate-y-[-1px]",
                "shadow-sm shadow-green-500/20"
              )}
              disabled={isTyping}
            >
              {isTyping ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-3 w-3 animate-spin" /> Executing...
                </span>
              ) : (
                <>
                  <span className="text-lg leading-none">✓</span>
                  Confirm Trade
                </>
              )}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onConfirmTrade?.(false)}
              className={cn(
                "flex items-center gap-2",
                "transition-all duration-300 hover:shadow-md hover:translate-y-[-1px]",
                "border-red-500/30 hover:border-red-500/50 hover:bg-red-500/5"
              )}
              disabled={isTyping}
            >
              <span className="text-lg leading-none text-red-500">✗</span>
              Cancel
            </Button>
          </div>
        )}
      </div>
    )
  }

  return (
    <ScrollArea ref={scrollAreaRef} className={cn("flex-1 h-full overflow-y-auto", className)} type="always">
      <div className="flex flex-col min-h-full">
        {messages.map((message, index) => (
          <div 
            key={message.id} 
            className={cn(
              "group relative animate-in fade-in-0 slide-in-from-bottom-2 duration-300",
              message.sender === "user" ? "bg-muted/50" : "bg-transparent",
              "border-b border-border/5"
            )}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="w-full max-w-[880px] mx-auto px-6 py-6">
              <div className={cn(
                "text-sm",
                message.sender === "user" 
                  ? "text-primary-foreground" 
                  : "text-foreground"
              )}>

                <MarkdownRenderer 
                  content={message.content.split('\n').filter(line => line.trim()).join('\n')}
                  className={cn(
                    message.sender === "user" ? "prose-invert" : "",
                    "min-w-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
                    "[&>p:empty]:hidden [&>p]:my-1"
                  )}
                />
                
                {message.stockMention && (
                  <div className="mt-4 flex items-center gap-2">
                    <div className="px-2 py-1 bg-primary/10 rounded text-xs font-mono text-primary">
                      📊 {message.stockMention}
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 px-2 text-xs hover:bg-primary/10"
                      onClick={() => onStockSelect?.(message.stockMention!)}
                    >
                      View Details <ArrowRight className="h-3 w-3 ml-1" />
                    </Button>
                  </div>
                )}

                {renderInteractiveElements(message)}

                <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground/60">
                  <span>{message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}</span>
                </div>
              </div>

            </div>
          </div>
        ))}

        {isTyping && (
          <div className={cn(
            "group relative border-b border-border/5"
          )}>
            <div className="w-full max-w-[880px] mx-auto px-6 py-6">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }} />
                <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
              </div>
            </div>
          </div>
        )}
        
        {/* Scroll target */}
        <div ref={messagesEndRef} className="h-0 w-0 overflow-hidden" />
      </div>
    </ScrollArea>
  )
})