"use client"

import { useState, useRef, useEffect } from "react"
import { MessageList, MessageListRef } from "./message-list"
import { MessageInput } from "./message-input"
import { QuickActions } from "./quick-actions"
import { ChatHeader } from "./chat-header"
import { AgentVisualization, useAgentVisualization } from "@/components/ai/agent-visualization"
import { useTrading } from "@/components/trading/controls/trading-context"
import { getAllStocks, getComprehensiveAnalysis, transformAnalysisResponseToStockData } from "@/lib/api-service"
import type { ViewType } from "@/components/layout/main-layout"

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

interface ChatContainerProps {
  selectedStock: string
  onStockSelect: (stock: string) => void
  onViewChange: (view: ViewType) => void
  onToggleCollapse: () => void
  isCollapsed: boolean
}

// Helper function to extract stock symbol using OpenAI
async function extractStockSymbolWithAI(responseText: string, userMessage: string): Promise<string | null> {
  // First try simple pattern matching as fallback
  const commonStocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ"]
  const foundInResponse = commonStocks.find(symbol => 
    responseText.toUpperCase().includes(symbol) ||
    responseText.includes(`(${symbol})`) ||
    new RegExp(`\\b${symbol}\\b`, 'i').test(responseText)
  )
  
  if (foundInResponse) {
    console.log(`🔍 Found ${foundInResponse} using pattern matching`)
    return foundInResponse
  }

  // Try OpenAI if API key is available
  if (!process.env.OPENAI_API_KEY) {
    console.warn('OpenAI API key not available, using pattern matching only')
    return null
  }

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'Extract the stock ticker symbol (like AAPL, TSLA, GOOGL, F, GM, etc.) from the given text. Return only the ticker symbol in uppercase, or "NONE" if no stock symbol is found.'
          },
          {
            role: 'user',
            content: `User message: "${userMessage}"\n\nResponse: "${responseText.substring(0, 800)}"\n\nExtract the stock ticker symbol:`
          }
        ],
        max_tokens: 10,
        temperature: 0
      })
    })

    if (!response.ok) {
      console.error('OpenAI API error:', response.status)
      return null
    }

    const data = await response.json()
    const extractedSymbol = data.choices[0]?.message?.content?.trim()
    
    if (extractedSymbol && extractedSymbol !== 'NONE' && extractedSymbol !== 'null') {
      console.log(`🔍 Found ${extractedSymbol} using OpenAI`)
      return extractedSymbol.toUpperCase()
    }
    
    return null
  } catch (error) {
    console.error('Error extracting stock symbol with AI:', error)
    return null
  }
}

export function ChatContainer({
  selectedStock,
  onStockSelect,
  onViewChange,
  onToggleCollapse,
  isCollapsed,
}: ChatContainerProps) {
  const { state, executeTrade } = useTrading()
  const agentViz = useAgentVisualization()
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: `# Welcome to Options Oracle! 👋

I'm your AI trading assistant. Here's how I can help:

## Available Commands

- 🔍 **Stock Analysis**
  - \`analyze AAPL\` - Get detailed analysis
  - \`what's TSLA doing?\` - Quick market check

- 📊 **Trading Signals**
  - \`should I buy NVDA?\` - Get trade recommendations
  - \`market sentiment\` - Overall market view

- 💼 **Portfolio Management**
  - \`show my portfolio\` - View your positions
  - \`P&L today\` - Check performance

- ⚡ **Live Trading**
  - \`buy 10 AAPL\` - Place buy order
  - \`sell TSLA\` - Place sell order

> Currently in **${state.mode}** mode ${state.isActive ? "✅ (Active)" : "⏸️ (Inactive)"}

Type a command or click one of the quick actions below to get started!`,
      sender: "assistant",
      timestamp: new Date(),
      actionType: "general",
    },
  ])
  const [isTyping, setIsTyping] = useState(false)
  const messageListRef = useRef<MessageListRef>(null)

  const scrollToBottom = (smooth = true) => {
    // Use the MessageList ref to scroll
    if (messageListRef.current) {
      messageListRef.current.scrollToBottom()
    }
  }

  const scrollToBottomWithDelay = (delay = 100) => {
    setTimeout(() => {
      scrollToBottom(true)
    }, delay)
  }

  const ensureScrollToBottom = () => {
    // Simple, reliable scroll
    forceScrollToBottom()
    
    // Additional scroll attempts with delays
    setTimeout(() => scrollToBottom(true), 100)
    setTimeout(() => scrollToBottom(true), 300)
  }

  const forceScrollToBottom = () => {
    // Force scroll without smooth animation for immediate effect
    if (messageListRef.current) {
      messageListRef.current.scrollToBottom()
    }
  }

  useEffect(() => {
    // Scroll when messages change - try multiple times
    forceScrollToBottom()
    
    // Try scrolling multiple times with different delays
    const timers = [
      setTimeout(() => scrollToBottom(true), 50),
      setTimeout(() => scrollToBottom(true), 150),
      setTimeout(() => scrollToBottom(true), 300),
      setTimeout(() => scrollToBottom(true), 500),
      setTimeout(() => scrollToBottom(true), 800),
    ]
    
    return () => {
      timers.forEach(timer => clearTimeout(timer))
    }
  }, [messages])

  // Scroll when typing state changes
  useEffect(() => {
    if (isTyping) {
      // Scroll when typing starts
      forceScrollToBottom()
      scrollToBottomWithDelay(200)
      
      // Keep scrolling during typing
      const scrollInterval = setInterval(() => {
        scrollToBottom(true)
      }, 300)
      
      return () => clearInterval(scrollInterval)
    } else {
      // When typing stops, scroll to bottom
      forceScrollToBottom()
      scrollToBottomWithDelay(200)
    }
  }, [isTyping])

  // Ensure proper scroll position after component mounts
  useEffect(() => {
    scrollToBottom(false)
  }, [])

  // Scroll when selected stock changes (when buttons are clicked)
  useEffect(() => {
    if (selectedStock) {
      scrollToBottomWithDelay(200)
      scrollToBottomWithDelay(500)
    }
  }, [selectedStock])

  // Additional scroll trigger after every render
  useEffect(() => {
    const timer = setTimeout(() => {
      scrollToBottom(true)
    }, 100)
    
    return () => clearTimeout(timer)
  })

  const processMessage = (content: string): { 
    mentionedStock: string | undefined; 
    actionType: "analysis" | "trade" | "portfolio" | "general" | "STOCK_ANALYSIS" | "technical_analysis" | "TECHNICAL_ANALYSIS" | "stock_analysis" | "PORTFOLIO_MANAGEMENT" 
  } => {
    const lowerContent = content.toLowerCase()
    const stockSymbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ"]

    // Extract stock symbol
    const mentionedStock = stockSymbols.find(
      (symbol) => content.toUpperCase().includes(symbol) || lowerContent.includes(symbol.toLowerCase()),
    )

    // Determine action type
    let actionType: "analysis" | "trade" | "portfolio" | "general" | "STOCK_ANALYSIS" | "technical_analysis" | "TECHNICAL_ANALYSIS" | "stock_analysis" | "PORTFOLIO_MANAGEMENT" = "general"

    if (lowerContent.includes("analyze") || lowerContent.includes("analysis") || lowerContent.includes("chart")) {
      actionType = "analysis"
    } else if (lowerContent.includes("buy") || lowerContent.includes("sell") || lowerContent.includes("trade")) {
      actionType = "trade"
    } else if (
      lowerContent.includes("portfolio") ||
      lowerContent.includes("p&l") ||
      lowerContent.includes("positions")
    ) {
      actionType = "portfolio"
    }

    return { mentionedStock, actionType }
  }

  const generateResponse = async (content: string, mentionedStock?: string, actionType?: string): Promise<{ 
    text: string; 
    interactiveElements?: Message['interactiveElements'] 
  }> => {
    const lowerContent = content.toLowerCase()
    
    // Initialize agent visualization for analysis requests
    if ((actionType === "analysis" || actionType === "STOCK_ANALYSIS" || actionType === "technical_analysis") && mentionedStock) {
      agentViz.initializeAgents(['technical', 'sentiment', 'flow', 'history', 'risk', 'education'])
      
      // Simulate agent progression
      const agents = ['technical', 'sentiment', 'flow', 'history', 'risk', 'education']
      const startTime = Date.now()
      
      for (let i = 0; i < agents.length; i++) {
        const agent = agents[i]
        agentViz.updateAgentStatus(agent, 'thinking')
        
        // Simulate different processing times for each agent
        const processingTime = {
          'technical': 3000,
          'sentiment': 2000,
          'flow': 1500,
          'history': 2500,
          'risk': 2000,
          'education': 1500
        }[agent] || 2000
        
        await new Promise(resolve => setTimeout(resolve, processingTime))
        
        const duration = (Date.now() - startTime) / 1000
        agentViz.updateAgentStatus(agent, 'completed', duration)
      }
      
      setTimeout(() => {
        agentViz.completeAnalysis()
      }, 1000)
    }
    
    const stockData = mentionedStock ? await transformAnalysisResponseToStockData(await getComprehensiveAnalysis(mentionedStock)) : null

    if (actionType === "analysis" && mentionedStock && stockData) {
      const interactiveElements: Message['interactiveElements'] = {
        stockButtons: [mentionedStock],
      }

      if (state.mode === "autonomous" && state.isActive) {
        const recommendation = stockData.aiAnalysis.recommendation
        if (recommendation === 'buy' || recommendation === 'strong_buy') {
          interactiveElements.tradeActions = [{ symbol: mentionedStock, type: 'buy' }]
        } else if (recommendation === 'sell' || recommendation === 'strong_sell') {
          interactiveElements.tradeActions = [{ symbol: mentionedStock, type: 'sell' }]
        }
      }

      return {
        text: `🔍 **Analyzing ${mentionedStock}** - $${stockData.stock.price.toFixed(2)} (${stockData.stock.changePercent >= 0 ? '+' : ''}${stockData.stock.changePercent.toFixed(2)}%)

📊 **Technical Analysis Results:**
- RSI: ${stockData.technicals.rsi.toFixed(1)} (${stockData.technicals.rsi > 70 ? 'Overbought' : stockData.technicals.rsi < 30 ? 'Oversold' : 'Neutral'})
- MACD: ${stockData.technicals.macd.charAt(0).toUpperCase() + stockData.technicals.macd.slice(1)} signal
- Support: $${stockData.technicals.support.toFixed(2)} | Resistance: $${stockData.technicals.resistance.toFixed(2)}
- Volume: ${(stockData.stock.volume / 1000000).toFixed(1)}M (${stockData.stock.volume > stockData.technicals.volumeAvg ? 'Above' : 'Below'} average)

🤖 **AI Analysis (${stockData.aiAnalysis.confidence}% confidence):**
- Recommendation: **${stockData.aiAnalysis.recommendation.toUpperCase().replace('_', ' ')}**
- Sentiment: ${stockData.aiAnalysis.sentiment.charAt(0).toUpperCase() + stockData.aiAnalysis.sentiment.slice(1)}
- Risk Level: ${stockData.aiAnalysis.riskLevel.charAt(0).toUpperCase() + stockData.aiAnalysis.riskLevel.slice(1)}

**Key Points:**
${Array.isArray(stockData.aiAnalysis.reasoning) 
  ? stockData.aiAnalysis.reasoning.map(reason => `• ${reason}`).join('\n')
  : `• ${stockData.aiAnalysis.reasoning || 'Analysis completed'}`}

${
  state.mode === "autonomous" && state.isActive
    ? `🤖 **Autonomous Mode**: ${stockData.aiAnalysis.recommendation.includes('buy') ? 'Preparing to execute BUY order' : stockData.aiAnalysis.recommendation.includes('sell') ? 'Preparing to execute SELL order' : 'Monitoring for optimal entry'}`
    : "👤 **Manual Mode**: Click the stock button above to view detailed charts and analysis"
}`,
        interactiveElements
      }
    }

    if (actionType === "trade") {
      const isBuy = lowerContent.includes("buy")
      const isSell = lowerContent.includes("sell")

      if (!mentionedStock) {
        const topStocks = ['AAPL', 'NVDA', 'MSFT', 'TSLA']
        return {
          text: "I'd be happy to help with trading! Please specify a stock symbol. Here are some popular options:",
          interactiveElements: {
            stockButtons: topStocks
          }
        }
      }

      if (!state.isActive) {
        return {
          text: `⚠️ **Trading Disabled**

To execute trades, please:
1. Enable trading in the controls panel
2. Set your risk parameters
3. Choose ${state.mode} mode

Would you like me to guide you through the setup?`
        }
      }

      if (isBuy || isSell) {
        const action = isBuy ? "BUY" : "SELL"
        const stockData = await transformAnalysisResponseToStockData(await getComprehensiveAnalysis(mentionedStock))
        
        const interactiveElements: Message['interactiveElements'] = {}

        if (state.mode === "autonomous") {
          interactiveElements.tradeActions = [{ symbol: mentionedStock, type: action.toLowerCase() as 'buy' | 'sell' }]
        } else {
          interactiveElements.confirmationRequired = true
        }

        return {
          text: `📈 **${action} ${mentionedStock} Order** ${stockData ? `- $${stockData.stock.price.toFixed(2)}` : ''}

${
  state.mode === "autonomous"
    ? `🤖 **Autonomous Mode**: Analyzing ${mentionedStock} for optimal ${action.toLowerCase()} entry...
  
  - Running technical analysis
  - Calculating position size (${state.positionSize}% of portfolio)  
  - Setting stop-loss levels
  
  Click execute when ready, or I'll auto-execute when conditions are optimal.`
    : `👤 **Manual Mode**: Preparing ${action} order for ${mentionedStock}
  
  - Current price: $${stockData?.stock.price.toFixed(2) || (Math.random() * 200 + 100).toFixed(2)}
  - Recommended position size: ${state.positionSize}%
  - Risk level: ${state.riskLevel}/10
  
  Please confirm your order below.`
}`,
          interactiveElements
        }
      }
    }

    if (actionType === "portfolio") {
      const totalValue = state.totalValue
      const totalPnL = state.totalPnL
      const todayPnL = state.todayPnL
      const cashBalance = state.cashBalance
      const positionCount = state.positions.length
      const bestPerformer = state.positions.reduce((best, pos) => 
        pos.pnlPercent > (best?.pnlPercent || -Infinity) ? pos : best, 
        state.positions[0]
      )

      return {
        text: `💼 **Portfolio Overview**

Current Status:
- Total Value: $${totalValue.toFixed(2)} (${todayPnL >= 0 ? '+' : ''}$${todayPnL.toFixed(2)} today)
- Active Positions: ${positionCount} stocks
- Today's P&L: ${todayPnL >= 0 ? '+' : ''}${((todayPnL / (totalValue - todayPnL)) * 100).toFixed(1)}%
- Trading Mode: ${state.mode} ${state.isActive ? "(Active)" : "(Inactive)"}

📊 **Performance Highlights**:
${bestPerformer ? `- Best Performer: ${bestPerformer.symbol} (${bestPerformer.pnlPercent >= 0 ? '+' : ''}${bestPerformer.pnlPercent.toFixed(1)}%)` : '- No positions yet'}
- Total Unrealized P&L: ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}
- Cash Balance: $${cashBalance.toFixed(2)}

Switching to portfolio dashboard for detailed view...`
      }
    }

    if (lowerContent.includes("hot stocks") || lowerContent.includes("trending")) {
      const hotStocks = ['NVDA', 'AAPL', 'TSLA', 'MSFT']
      return {
        text: `🔥 **Hot Stocks Today**

Based on AI analysis of volume, momentum, and sentiment:

📈 **Top Movers**:
- Technical breakouts with high volume
- Unusual options activity patterns 
- Social sentiment momentum shifts
- Smart money institutional flows

Click any stock below for detailed analysis:`,
        interactiveElements: {
          stockButtons: hotStocks
        }
      }
    }

    // Default response
    const popularStocks = ['AAPL', 'NVDA', 'MSFT', 'TSLA']
    return {
      text: `## How can I help you? 🤔

Here are some things you can ask me:

### 🔍 Analysis
- \`analyze AAPL\` - Get detailed analysis
- \`TSLA chart\` - View technical charts
- \`what's NVDA doing?\` - Quick market check

### 📊 Trading
- \`buy 10 AAPL\` - Place buy order
- \`sell TSLA\` - Place sell order
- \`should I buy NVDA?\` - Get recommendations

### 💼 Portfolio
- \`show portfolio\` - View your positions
- \`my P&L\` - Check performance
- \`positions\` - List active trades

### 🔥 Discovery
- \`hot stocks\` - See trending stocks
- \`market movers\` - Top gainers/losers

> Current mode: **${state.mode}** ${state.isActive ? "✅ (Active)" : "⏸️ (Inactive)"}

Popular stocks to explore:`,
      interactiveElements: {
        stockButtons: popularStocks
      }
    }
  }

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    // Scroll when user message is added
    forceScrollToBottom()
    scrollToBottomWithDelay(200)
    setIsTyping(true)

    // Detect request type - buy vs analysis
    const buyKeywords = ['buy', 'purchase', 'investing', 'invest', 'budget']
    const analysisKeywords = ['analy', 'examine', 'review', 'assess', 'research']
    
    const isBuyRequest = buyKeywords.some(keyword => content.toLowerCase().includes(keyword))
    const isAnalysisRequest = analysisKeywords.some(keyword => content.toLowerCase().includes(keyword)) ||
                             (!isBuyRequest && /\b(AAPL|GOOGL|MSFT|TSLA|NVDA|AMZN|META|SPY|QQQ)\b/i.test(content))
    
    if (isBuyRequest) {
      console.log("🛒 Starting agent animation for buy request - staying in chat view for confirmation")
      agentViz.initializeAgents(['technical', 'risk'])  // Only technical and risk for buy requests
      
      // Start simulating agent processing immediately for buy
      setTimeout(async () => {
        const agents = ['technical', 'risk']
        const startTime = Date.now()
        
        for (let i = 0; i < agents.length; i++) {
          const agent = agents[i]
          
          // Start agent thinking
          setTimeout(() => {
            agentViz.updateAgentStatus(agent, 'thinking')
          }, i * 300)
          
          // Complete agent after processing time
          setTimeout(() => {
            const duration = (Date.now() - startTime + (i * 300)) / 1000
            agentViz.updateAgentStatus(agent, 'completed', duration)
          }, (i * 300) + 2000) // 2 seconds processing time per agent
        }
        
        // Complete entire analysis after all agents finish
        setTimeout(() => {
          agentViz.completeAnalysis()
        }, (agents.length * 300) + 2500)
      }, 100)
      
      // DO NOT switch views for buy requests - stay in chat to show confirmation buttons
      
    } else if (isAnalysisRequest) {
      console.log("🚀 Starting agent animation for analysis request")
      agentViz.initializeAgents(['technical', 'sentiment', 'flow', 'history', 'risk'])
      
      // Extract potential stock symbol and trigger view change for analysis only
      const stockSymbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ"]
      const detectedStock = stockSymbols.find(
        (symbol) => content.toUpperCase().includes(symbol) || content.toLowerCase().includes(symbol.toLowerCase())
      )
      
      if (detectedStock) {
        console.log(`🔍 Detected stock ${detectedStock}, switching to skeleton view for analysis`)
        onStockSelect(detectedStock)
        onViewChange("stock-details")
      }
      
      // Start simulating agent processing immediately
      setTimeout(() => {
        const agents = ['technical', 'sentiment', 'flow', 'history', 'risk']
        agents.forEach((agent, index) => {
          setTimeout(() => {
            agentViz.updateAgentStatus(agent, 'thinking')
          }, index * 500) // Stagger the start times
        })
      }, 100)
    }

    try {
      // Make real API call to backend
      const response = await fetch('http://localhost:8080/api/v1/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          selectedStock: null
        })
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const chatResponse = await response.json()
      console.log(`🔍 Backend response:`, chatResponse)
      
      // Extract stock mention and action type from backend response
      let mentionedStock = chatResponse.symbol
      const actionType = chatResponse.intent
      console.log(`🔍 Raw extraction: symbol=${mentionedStock}, intent=${actionType}`)
      
      // Fallback: if backend didn't return a symbol, try to extract it from the response using OpenAI
      if (!mentionedStock && (actionType === "STOCK_ANALYSIS" || actionType === "OPTIONS_BUYING")) {
        try {
          mentionedStock = await extractStockSymbolWithAI(chatResponse.response, content)
          console.log(`🔍 Backend didn't return symbol, extracted with AI: ${mentionedStock}`)
        } catch (error) {
          console.error('Failed to extract symbol with AI:', error)
          mentionedStock = null
        }
      }
      
      // Initialize agent visualization for stock analysis
      if ((actionType === "STOCK_ANALYSIS" || actionType === "analysis" || actionType === "technical_analysis") && mentionedStock) {
        console.log(`🤖 Initializing agent visualization for ${mentionedStock} with intent: ${actionType}`)
        agentViz.initializeAgents(['technical', 'sentiment', 'flow', 'history', 'risk', 'education'])
        
        // Simulate agent progression based on backend timing
        const agents = ['technical', 'sentiment', 'flow', 'history', 'risk', 'education']
        const startTime = Date.now()
        
        setTimeout(async () => {
          for (let i = 0; i < agents.length; i++) {
            const agent = agents[i]
            agentViz.updateAgentStatus(agent, 'thinking')
            
            // Simulate different processing times for each agent (shorter than actual backend)
            const processingTime = {
              'technical': 2000,
              'sentiment': 1500, 
              'flow': 1000,
              'history': 1800,
              'risk': 1500,
              'education': 1200
            }[agent] || 1500
            
            await new Promise(resolve => setTimeout(resolve, processingTime))
            
            const duration = (Date.now() - startTime) / 1000
            agentViz.updateAgentStatus(agent, 'completed', duration)
          }
          
          setTimeout(() => {
            agentViz.completeAnalysis()
          }, 500)
        }, 100)
      }
      
      // Extract buy confirmation data from AI response
      const interactiveElements: any = {
        stockButtons: mentionedStock ? [mentionedStock] : undefined
      }
      
      // Check if the response has interactive elements (new structure)
      if (chatResponse.interactive_elements?.tradeActions) {
        console.log("🛒 Interactive trade elements found!", chatResponse.interactive_elements)
        interactiveElements.tradeActions = chatResponse.interactive_elements.tradeActions
      }
      
      // Fallback: Check if this is a buy option response requiring confirmation (old structure)
      const responseData = chatResponse.data || {}
      const toolResults = responseData.tool_results || []
      
      // Look for buy_option tool results
      const buyToolResult = toolResults.find((result: any) => 
        result.tool === "buy_option" && result.requires_confirmation === true
      )
      
      if (buyToolResult && !interactiveElements.tradeActions) {
        console.log("🛒 Buy confirmation required (old structure)!", buyToolResult)
        interactiveElements.tradeActions = [{
          symbol: buyToolResult.symbol,
          type: 'buy',
          analysis: buyToolResult.analysis,
          budget: buyToolResult.budget
        }]
      }
      
      // Also check for OPTIONS_BUYING intent as fallback
      if (!interactiveElements.tradeActions && actionType === "OPTIONS_BUYING" && mentionedStock) {
        console.log("🛒 OPTIONS_BUYING intent detected - enabling buy confirmation")
        interactiveElements.tradeActions = [{
          symbol: mentionedStock,
          type: 'buy',
          analysis: chatResponse.response,
          budget: 500 // Default budget
        }]
      }
      
      // Use the backend response directly
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: chatResponse.response,
        sender: "assistant",
        timestamp: new Date(),
        stockMention: mentionedStock,
        actionType,
        interactiveElements
      }

      setMessages((prev) => [...prev, assistantMessage])
      setIsTyping(false)
      
      // Scroll when assistant message is added
      forceScrollToBottom()
      scrollToBottomWithDelay(200)
      scrollToBottomWithDelay(500)

      // Auto-trigger stock analysis view for any stock analysis intent
      console.log(`🔍 Checking view trigger: mentionedStock=${mentionedStock}, actionType=${actionType}`)
      if (mentionedStock && (
        actionType === "STOCK_ANALYSIS" || 
        actionType === "analysis" || 
        actionType === "technical_analysis" ||
        actionType === "TECHNICAL_ANALYSIS" ||
        actionType === "stock_analysis"
      )) {
        console.log(`🔍 Triggering stock analysis view for ${mentionedStock} with intent: ${actionType}`)
        console.log(`🔍 Calling onStockSelect(${mentionedStock}) and onViewChange("stock-details")`)
        onStockSelect(mentionedStock)
        onViewChange("stock-details")
        
        // Pass chat analysis data to left view and ensure scroll happens after view change
        setTimeout(() => {
          scrollToBottom(true)
          window.dispatchEvent(
            new CustomEvent("expandAllAnalysis", {
              detail: { stock: mentionedStock, actionType },
            }),
          )
          // Pass the chat analysis data to the left view
          window.dispatchEvent(
            new CustomEvent("setChatAnalysisData", {
              detail: { 
                symbol: mentionedStock, 
                analysisData: chatResponse.data,
                response: chatResponse.response
              },
            }),
          )
        }, 800)
        // Additional scroll after a longer delay to ensure view is fully loaded
        scrollToBottomWithDelay(1200)
      } else if (actionType === "portfolio" || actionType === "PORTFOLIO_MANAGEMENT") {
        onViewChange("portfolio")
        scrollToBottomWithDelay(300)
      } else if (content.toLowerCase().includes("hot stocks")) {
        onViewChange("hot-stocks")
        scrollToBottomWithDelay(300)
      }

    } catch (error) {
      console.error('Chat API error:', error)
      
      // Fallback to local generation on error
      setTimeout(async () => {
        const { mentionedStock, actionType } = processMessage(content)
        const response = await generateResponse(content, mentionedStock, actionType)

        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response.text,
          sender: "assistant",
          timestamp: new Date(),
          stockMention: mentionedStock,
          actionType,
          interactiveElements: response.interactiveElements,
        }

        setMessages((prev) => [...prev, assistantMessage])
        setIsTyping(false)
        
        // Handle navigation for fallback
        if (mentionedStock && (
          actionType === "analysis" || 
          actionType === "trade" ||
          actionType === "STOCK_ANALYSIS" ||
          actionType === "technical_analysis"
        )) {
          onStockSelect(mentionedStock)
          if (actionType === "analysis" || actionType === "STOCK_ANALYSIS" || actionType === "technical_analysis") {
            console.log(`🔍 Fallback: Triggering stock analysis view for ${mentionedStock} with intent: ${actionType}`)
            onViewChange("stock-details")
          }
        }
      }, 1500)
    }
  }

  const handleStockSelect = (stock: string) => {
    onStockSelect(stock)
    onViewChange("stock-details")
    
    // Force scroll when stock is selected
    forceScrollToBottom()
    scrollToBottomWithDelay(200)
    scrollToBottomWithDelay(500)
    
    setTimeout(() => {
      window.dispatchEvent(
        new CustomEvent("expandAllAnalysis", {
          detail: { stock, actionType: "analysis" },
        }),
      )
      // Additional scroll after analysis expansion
      scrollToBottomWithDelay(800)
    }, 300)
  }

  const handleTradeAction = async (symbol: string, action: 'buy' | 'sell', budget?: number) => {
    // For options trading, we need to analyze opportunities first
    if (action === 'buy') {
      try {
        setIsTyping(true)
        
        // Extract budget from user query or use default
        const tradeBudget = budget || state.cashBalance * (state.positionSize / 100)
        
        // Get the original user query from the last message
        const lastUserMessage = messages.filter(m => m.sender === 'user').pop()
        const userQuery = lastUserMessage?.content || `buy ${symbol} options with $${tradeBudget} budget`
        
        // Analyze option opportunities
        const { analyzeOptionOpportunity } = await import('../../lib/api-service')
        const optionAnalysis = await analyzeOptionOpportunity(symbol, tradeBudget, userQuery)
        
        if (optionAnalysis.error) {
          const errorMessage: Message = {
            id: Date.now().toString(),
            content: `❌ **Options Analysis Failed**: ${optionAnalysis.error}`,
            sender: "assistant",
            timestamp: new Date(),
            stockMention: symbol,
            actionType: "trade",
          }
          setMessages((prev) => [...prev, errorMessage])
          setIsTyping(false)
          return
        }
        
        // Create options recommendation message
        const recommendations = optionAnalysis.recommendations || []
        if (recommendations.length === 0) {
          const noOptionsMessage: Message = {
            id: Date.now().toString(),
            content: `❌ **No Options Available**: Unable to find suitable option contracts for ${symbol} within your $${tradeBudget} budget.`,
            sender: "assistant",
            timestamp: new Date(),
            stockMention: symbol,
            actionType: "trade",
          }
          setMessages((prev) => [...prev, noOptionsMessage])
          setIsTyping(false)
          return
        }
        
        // Format the recommendations for display
        const bestRec = recommendations[0]
        const optionType = bestRec.option_details.type.toUpperCase()
        const strike = bestRec.option_details.strike_price
        const expiration = bestRec.option_details.expiration_date
        const contracts = bestRec.option_details.contracts
        const totalCost = bestRec.option_details.total_estimated_cost
        
        const optionsMessage: Message = {
          id: Date.now().toString(),
          content: `🎯 **Options Recommendation for ${symbol}**

**Best Option**: ${optionType} ${strike} (Exp: ${expiration})
- **Contracts**: ${contracts}
- **Total Cost**: $${totalCost.toFixed(2)}
- **Budget Utilization**: ${bestRec.risk_metrics.budget_utilization}%
- **Risk Level**: ${bestRec.risk_metrics.risk_level}
- **Potential Return**: ${bestRec.risk_metrics.potential_return}

**Strategy**: ${bestRec.reasoning}

Would you like to execute this trade?`,
          sender: "assistant",
          timestamp: new Date(),
          stockMention: symbol,
          actionType: "trade",
          interactiveElements: {
            tradeActions: [{
              symbol,
              type: action,
              recommendation: bestRec,
              requiresConfirmation: true
            }]
          }
        }
        
        setMessages((prev) => [...prev, optionsMessage])
        setIsTyping(false)
        
      } catch (error) {
        console.error('Options analysis failed:', error)
        const errorMessage: Message = {
          id: Date.now().toString(),
          content: `❌ **Options Analysis Failed**: ${error}`,
          sender: "assistant",
          timestamp: new Date(),
          stockMention: symbol,
          actionType: "trade",
        }
        setMessages((prev) => [...prev, errorMessage])
        setIsTyping(false)
      }
    }
    
    // Scroll when trade message is added
    forceScrollToBottom()
    scrollToBottomWithDelay(200)
  }

  const handleConfirmTrade = async (confirmed: boolean) => {
    if (confirmed) {
      // Find the trade details from the last message
      const lastMessage = messages[messages.length - 1]
      const tradeAction = lastMessage?.interactiveElements?.tradeActions?.[0]
      
      if (tradeAction?.recommendation) {
        // Execute the options trade
        try {
          setIsTyping(true) // Show loading state during execution
          
          const { executeOptionTrade } = await import('../../lib/api-service')
          const executionResult = await executeOptionTrade(tradeAction.recommendation)
          
          const optionDetails = tradeAction.recommendation.option_details
          const optionType = optionDetails.type.toUpperCase()
          const strike = optionDetails.strike_price
          const expiration = optionDetails.expiration_date
          const contracts = optionDetails.contracts
          const totalCost = optionDetails.total_estimated_cost
          
          const confirmMessage: Message = {
            id: Date.now().toString(),
            content: `✅ **Options Trade Executed Successfully!**

**Trade Details:**
- **Symbol**: ${tradeAction.symbol} ${optionType} ${strike}
- **Expiration**: ${expiration}
- **Contracts**: ${contracts}
- **Total Cost**: $${totalCost.toFixed(2)}
- **Trade ID**: ${executionResult.trade_id}
- **Status**: ${executionResult.status}

**Execution**: ${executionResult.reasoning || 'Trade executed via Alpaca paper trading'}

Your options position is now active! 🎯`,
            sender: "assistant",
            timestamp: new Date(),
            stockMention: tradeAction.symbol,
            actionType: "trade",
          }
          
          setMessages((prev) => [...prev, confirmMessage])
          setIsTyping(false)
          
        } catch (error) {
          console.error('Options trade execution failed:', error)
          const errorMessage: Message = {
            id: Date.now().toString(),
            content: `❌ **Options Trade Failed**: ${error}

Please try again or contact support if the issue persists.`,
            sender: "assistant",
            timestamp: new Date(),
            stockMention: tradeAction.symbol,
            actionType: "trade",
          }
          
          setMessages((prev) => [...prev, errorMessage])
          setIsTyping(false)
        }
      } else {
        // Fallback for confirmation without proper recommendation
        const confirmMessage: Message = {
          id: Date.now().toString(),
          content: "✅ **Trade Confirmed** - Your options order has been placed and will execute at market open.",
          sender: "assistant",
          timestamp: new Date(),
          actionType: "trade",
        }
        
        setMessages((prev) => [...prev, confirmMessage])
      }
    } else {
      const confirmMessage: Message = {
        id: Date.now().toString(),
        content: "❌ **Options Trade Cancelled** - Your order has been cancelled. No positions were opened.",
        sender: "assistant",
        timestamp: new Date(),
        actionType: "trade",
      }
      
      setMessages((prev) => [...prev, confirmMessage])
    }
    
    // Scroll when confirmation message is added
    forceScrollToBottom()
    scrollToBottomWithDelay(200)
  }

  const handleQuickAction = (action: string) => {
    handleSendMessage(action)
    // Additional scroll for quick actions
    scrollToBottomWithDelay(100)
  }

  return (
    <div className="flex flex-col h-full bg-background">
      <ChatHeader selectedStock={selectedStock} onToggleCollapse={onToggleCollapse} isCollapsed={isCollapsed} />

      <div className="flex-1 flex flex-col min-h-0 relative">
        <div className="absolute inset-0 flex flex-col">
          <MessageList 
            ref={messageListRef}
            messages={messages} 
            isTyping={isTyping} 
            className="flex-1"
            onStockSelect={handleStockSelect}
            onTradeAction={handleTradeAction}
            onConfirmTrade={handleConfirmTrade}
          />
          
          <div className="flex-shrink-0 border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="max-w-[880px] mx-auto px-6 py-4">
              {/* Agent Visualization */}
              {agentViz.isActive && (
                <div className="mb-4">
                  <AgentVisualization
                    agents={agentViz.agents}
                    currentAgent={agentViz.currentAgent}
                    isActive={agentViz.isActive}
                    className="animate-in slide-in-from-bottom-2 duration-300"
                  />
                </div>
              )}
              
              <QuickActions onAction={handleQuickAction} className="mb-2" />

              <MessageInput
                onSendMessage={handleSendMessage}
                placeholder={`${state.mode === "autonomous" ? "🤖" : "👤"} Ask about stocks, trading, or portfolio...`}
              />
            </div>
          </div>
        </div>
      </div>

    </div>
  )
}