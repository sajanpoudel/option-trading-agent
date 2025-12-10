"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Send, Bot, User, TrendingUp, BarChart3, PieChart, Activity, Brain, Eye, Zap, Clock, CheckCircle, AlertCircle, Loader2 } from "lucide-react"

interface Message {
  id: string
  type: "user" | "bot"
  content: string
  timestamp: Date
  suggestions?: string[]
  isStreaming?: boolean
  agentActivity?: AgentActivity[]
  buyConfirmation?: BuyConfirmation | null
}

interface BuyConfirmation {
  type: "single_option" | "multi_options"
  analysis: any
  budget: number
  symbol?: string
  requiresConfirmation: boolean
}

interface AgentActivity {
  name: string
  status: 'waiting' | 'active' | 'completed' | 'error'
  icon: any
  description: string
  startTime?: Date
  completionTime?: Date
}

interface ChatInterfaceProps {
  selectedStock: string | null
  onAnalysisRequest: (stock: string, query: string) => void
  onStockSelect: (stock: string) => void
}

// Helper function to extract budget from message
const extractBudgetFromMessage = (message: string): number => {
  const budgetMatch = message.match(/budget.*?(\d+)/i) || message.match(/(\d+).*?budget/i) || message.match(/\$(\d+)/i)
  return budgetMatch ? parseInt(budgetMatch[1]) : 0
}

export function ChatInterface({ selectedStock, onAnalysisRequest, onStockSelect }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "bot",
      content: `Hi! I'm your AI trading assistant. ${selectedStock ? `I see you're interested in ${selectedStock}.` : "Ask me about any stock or market analysis."} What would you like to analyze?`,
      timestamp: new Date(),
      suggestions: [
        "Analyze AAPL technical indicators",
        "Show TSLA options strategies",
        "NVDA risk assessment",
        "Market sentiment for SPY",
      ],
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [activeAgents, setActiveAgents] = useState<AgentActivity[]>([])
  const [currentProcessingMessage, setCurrentProcessingMessage] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return

    // Extract stock symbol from message if present
    const stockMatch = content.match(/\b([A-Z]{1,5})\b/)
    if (stockMatch && !selectedStock) {
      onStockSelect(stockMatch[1])
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsTyping(true)

    try {
      // Call real backend API instead of mock response
      const response = await fetch('http://localhost:8080/api/v1/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          selectedStock: stockMatch?.[1] || selectedStock
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const chatResponse = await response.json()
      
      // Check if this is a buy confirmation request
      let buyConfirmation: BuyConfirmation | null = null
      if (chatResponse.intent === "OPTIONS_BUYING" || chatResponse.intent === "PORTFOLIO_BUYING") {
        buyConfirmation = {
          type: chatResponse.intent === "OPTIONS_BUYING" ? "single_option" : "multi_options",
          analysis: chatResponse.data || {},
          budget: extractBudgetFromMessage(content),
          symbol: chatResponse.symbol,
          requiresConfirmation: true
        } as BuyConfirmation
      }
      
      // Create agent activity from the response
      if (chatResponse.agents_triggered && chatResponse.agents_triggered.length > 0) {
        const agentIcons = {
          'technical': TrendingUp,
          'sentiment': Brain,
          'flow': Zap,
          'history': Clock,
          'risk': AlertCircle,
          'education': Eye,
          'buy': Activity
        }
        
        const agentDescriptions = {
          'technical': 'Analyzing technical indicators and chart patterns',
          'sentiment': 'Gathering market sentiment and news analysis',
          'flow': 'Monitoring options flow and unusual activity',
          'history': 'Studying historical patterns and trends',
          'risk': 'Assessing risk metrics and position sizing',
          'education': 'Generating educational content and explanations',
          'buy': 'Preparing trading recommendations and execution plan'
        }
        
        const newActiveAgents: AgentActivity[] = chatResponse.agents_triggered.map((agentName: string) => ({
          name: agentName,
          status: 'completed' as const,
          icon: agentIcons[agentName as keyof typeof agentIcons] || Activity,
          description: agentDescriptions[agentName as keyof typeof agentDescriptions] || 'Processing analysis',
          completionTime: new Date()
        }))
        
        setActiveAgents(newActiveAgents)
      }
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: chatResponse.response,
        timestamp: new Date(),
        suggestions: chatResponse.suggestions,
        agentActivity: activeAgents,
        buyConfirmation: buyConfirmation
      }
      
      setMessages((prev) => [...prev, botMessage])
      setIsTyping(false)
      setCurrentProcessingMessage(null)
      
      // Clear agents after a delay
      setTimeout(() => {
        setActiveAgents([])
      }, 3000)

      // Execute actions if provided
      if (chatResponse.actions) {
        if (chatResponse.actions.analyzeStock) {
          setTimeout(() => {
            onAnalysisRequest(chatResponse.actions.analyzeStock, content)
          }, 1000)
        }
      }
      
      // Also trigger analysis if a stock symbol is detected in the message
      const additionalStockMatch = content.match(/\b([A-Z]{2,5})\b/)
      if (additionalStockMatch && !chatResponse.actions?.analyzeStock) {
        setTimeout(() => {
          onAnalysisRequest(additionalStockMatch[1], content)
        }, 1000)
      }
    } catch (error) {
      console.error('Error calling chat API:', error)
      
      // Fallback to mock response if API fails
      const detectedStock = stockMatch?.[1] || selectedStock || "MARKET"
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: `I'm having trouble connecting to the AI system. Let me try to analyze ${detectedStock} with cached data...`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, botMessage])
      setIsTyping(false)

      // Trigger analysis view
      setTimeout(() => {
        onAnalysisRequest(detectedStock, content)
      }, 1000)
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion)
  }

  const handleBuyConfirmation = async (confirmation: BuyConfirmation | null, confirmed: boolean) => {
    if (!confirmation) return

    try {
      const executionMessage = confirmed 
        ? `Executing ${confirmation.type === "single_option" ? "option purchase" : "portfolio purchase"} for $${confirmation.budget}...`
        : `Purchase cancelled.`

      // Add user confirmation message
      const userMessage: Message = {
        id: Date.now().toString(),
        type: "user",
        content: confirmed ? "✓ Confirm Purchase" : "✗ Cancel Purchase",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMessage])

      if (confirmed) {
        setIsTyping(true)

        // Call execution API
        const response = await fetch('http://localhost:8080/api/v1/options/execute', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            type: confirmation.type,
            analysis: confirmation.analysis,
            budget: confirmation.budget,
            symbol: confirmation.symbol,
            confirmed: true
          })
        })

        if (!response.ok) {
          throw new Error(`Execution failed: ${response.status}`)
        }

        const executionResult = await response.json()
        
        const executionMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: "bot",
          content: `🎉 **Purchase Executed Successfully!**\n\n${executionResult.confirmation_message || 'Options purchase completed.'}\n\n**Order ID**: ${executionResult.order_id || 'N/A'}\n**Total Cost**: $${executionResult.total_cost || confirmation.budget}`,
          timestamp: new Date(),
        }
        
        setMessages((prev) => [...prev, executionMessage])
      } else {
        const cancelMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: "bot", 
          content: "Purchase cancelled. Feel free to ask for another analysis anytime!",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, cancelMessage])
      }

      setIsTyping(false)
      
    } catch (error) {
      console.error('Buy confirmation error:', error)
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: `❌ **Execution Failed**\n\nThere was an issue executing your purchase: ${error}`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
      setIsTyping(false)
    }
  }

  const quickActions = [
    { icon: TrendingUp, label: "Technical Analysis", query: "Show me technical analysis with charts and indicators" },
    {
      icon: BarChart3,
      label: "Options Strategies",
      query: "What are the best options strategies for current market conditions?",
    },
    { icon: PieChart, label: "Risk Assessment", query: "Analyze risk metrics and Greeks for portfolio management" },
    { icon: Activity, label: "Live Monitoring", query: "Set up real-time monitoring with alerts and signals" },
  ]

  return (
    <div className="h-full flex flex-col bg-card">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-foreground">AI Assistant</h2>
            {selectedStock && <p className="text-sm text-muted-foreground">Analyzing {selectedStock}</p>}
          </div>
          <Badge variant="default" className="text-xs">
            <Bot className="h-3 w-3 mr-1" />
            Online
          </Badge>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="p-4 border-b border-border">
        <div className="grid grid-cols-2 gap-2">
          {quickActions.map((action, index) => (
            <Button
              key={index}
              variant="outline"
              size="sm"
              className="h-auto p-3 flex flex-col items-center gap-1 text-xs bg-transparent"
              onClick={() => handleSendMessage(action.query)}
            >
              <action.icon className="h-4 w-4" />
              <span className="text-center leading-tight">{action.label}</span>
            </Button>
          ))}
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-lg p-3 text-sm ${
                  message.type === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                }`}
              >
                <div className="flex items-start gap-2">
                  {message.type === "bot" && <Bot className="h-3 w-3 mt-0.5 flex-shrink-0" />}
                  {message.type === "user" && <User className="h-3 w-3 mt-0.5 flex-shrink-0" />}
                  <div className="flex-1">
                    <p className="text-sm">{message.content}</p>
                    <p className="text-xs opacity-70 mt-1">{message.timestamp.toLocaleTimeString()}</p>
                  </div>
                </div>

                {message.suggestions && (
                  <div className="mt-3 space-y-2">
                    <p className="text-xs opacity-70">Try asking:</p>
                    <div className="flex flex-wrap gap-1">
                      {message.suggestions.map((suggestion, index) => (
                        <Button
                          key={index}
                          variant="secondary"
                          size="sm"
                          className="text-xs h-6 px-2"
                          onClick={() => handleSuggestionClick(suggestion)}
                        >
                          {suggestion}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {message.buyConfirmation && (
                  <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="h-4 w-4 text-blue-600" />
                      <span className="font-medium text-blue-900">
                        {message.buyConfirmation.type === "single_option" ? "Option Purchase" : "Portfolio Purchase"}
                      </span>
                    </div>
                    <div className="text-sm text-blue-800 mb-3">
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>Budget: ${message.buyConfirmation.budget}</div>
                        {message.buyConfirmation.symbol && <div>Symbol: {message.buyConfirmation.symbol}</div>}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button 
                        size="sm" 
                        className="bg-green-600 hover:bg-green-700 text-white text-xs px-3 py-1"
                        onClick={() => handleBuyConfirmation(message.buyConfirmation || null, true)}
                        disabled={isTyping}
                      >
                        {isTyping ? (
                          <span className="flex items-center gap-1">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Executing...
                          </span>
                        ) : (
                          <>✓ Confirm Purchase</>
                        )}
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="border-red-300 text-red-600 hover:bg-red-50 text-xs px-3 py-1"
                        onClick={() => handleBuyConfirmation(message.buyConfirmation || null, false)}
                        disabled={isTyping}
                      >
                        ✗ Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-lg p-3 max-w-[85%]">
                <div className="flex items-center gap-2">
                  <Bot className="h-3 w-3" />
                  <div className="flex space-x-1">
                    <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce"></div>
                    <div
                      className="w-1.5 h-1.5 bg-current rounded-full animate-bounce"
                      style={{ animationDelay: "0.1s" }}
                    ></div>
                    <div
                      className="w-1.5 h-1.5 bg-current rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Agent Activity Display */}
          {activeAgents.length > 0 && (
            <div className="flex justify-start">
              <div className="bg-muted/50 rounded-lg p-3 max-w-[85%] border border-border/20">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-3 w-3 text-primary" />
                  <span className="text-xs font-medium text-muted-foreground">AI Agents Executed</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {activeAgents.map((agent, index) => {
                    const IconComponent = agent.icon
                    return (
                      <div
                        key={index}
                        className="flex items-center gap-1 px-2 py-1 bg-primary/10 rounded text-xs"
                        title={agent.description}
                      >
                        <IconComponent className="h-3 w-3 text-primary" />
                        <span className="text-primary font-medium capitalize">{agent.name}</span>
                        <CheckCircle className="h-3 w-3 text-green-500" />
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={`Ask about ${selectedStock || "any stock"}...`}
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage(inputValue)}
              className="flex-1 text-sm"
            />
            <Button size="sm" onClick={() => handleSendMessage(inputValue)} disabled={!inputValue.trim()}>
              <Send className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
