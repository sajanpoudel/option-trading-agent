"use client"

import { useState, useEffect } from "react"
import { getHotStocks, type HotStock } from "@/lib/api-service"
import { StockGrid } from "./stock-grid"
import { StockFilters } from "./stock-filters"
import { MarketStats } from "./market-stats"
import { HotStocksHeader } from "./hot-stocks-header"

// Local interface for the component (matches StockGrid expectations)
interface LocalHotStock {
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  volume: string
  sparklineData: { value: number }[]
  aiScore: number
  signals: string[]
  trending: boolean
}

interface HotStocksContainerProps {
  onStockSelect: (stock: string) => void
}

export function HotStocksContainer({ onStockSelect }: HotStocksContainerProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [sortBy, setSortBy] = useState<"change" | "volume">("change")
  const [hotStocks, setHotStocks] = useState<LocalHotStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Load real hot stocks data from API
  useEffect(() => {
    const loadHotStocks = async () => {
      try {
        setLoading(true)
        setError(null)
        
        const response = await getHotStocks()
        const stocks: LocalHotStock[] = response.stocks.map(stock => ({
          symbol: stock.symbol,
          name: stock.name,
          price: stock.price,
          change: stock.change,
          changePercent: stock.changePercent,
          volume: stock.volume.toString(), // Convert to string for display
          sparklineData: stock.sparklineData || Array.from({ length: 20 }, (_, i) => ({
            value: stock.price + Math.sin(i * 0.3) * (stock.changePercent / 10) + Math.random() * 5
          })),
          aiScore: stock.aiScore,
          signals: stock.signals || ["Analysis Pending"],
          trending: stock.trending || false,
        }))
        
        setHotStocks(stocks)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load hot stocks')
        console.error('Error loading hot stocks:', err)
      } finally {
        setLoading(false)
      }
    }

    loadHotStocks()
  }, [])

  const filteredAndSortedStocks = hotStocks
    .filter(
      (stock) =>
        stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        stock.name.toLowerCase().includes(searchQuery.toLowerCase()),
    )
    .sort((a, b) => {
      switch (sortBy) {
        case "change":
          return b.changePercent - a.changePercent
        case "volume":
          return Number.parseFloat(b.volume) - Number.parseFloat(a.volume)
        default:
          return 0
      }
    })

  // Show loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <HotStocksHeader />
        <main className="container mx-auto px-4 sm:px-6 py-6 space-y-8">
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading hot stocks...</p>
          </div>
        </main>
      </div>
    )
  }

  // Show error state
  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <HotStocksHeader />
        <main className="container mx-auto px-4 sm:px-6 py-6 space-y-8">
          <div className="text-center py-12">
            <div className="text-red-500 mb-4">
              <svg className="h-12 w-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2">Failed to load hot stocks</h3>
            <p className="text-muted-foreground mb-4">{error}</p>
            <button 
              onClick={() => window.location.reload()} 
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Retry
            </button>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <HotStocksHeader />

      <main className="container mx-auto px-4 sm:px-6 py-6 space-y-8">
        <StockFilters
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          sortBy={sortBy}
          onSortChange={setSortBy}
          totalStocks={filteredAndSortedStocks.length}
          trendingCount={filteredAndSortedStocks.filter((s) => s.trending).length}
        />

        <StockGrid stocks={filteredAndSortedStocks} onStockSelect={onStockSelect} />

        <MarketStats />
      </main>
    </div>
  )
}
