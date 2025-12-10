"use client"

import { useState, useEffect } from "react"
import { getHotStocks, getPortfolio } from "@/lib/api-service"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { LineChart, Line, ResponsiveContainer } from "recharts"
import { TrendingUp, TrendingDown, Eye, Bell, Filter, BarChart3, Menu, X } from "lucide-react"

interface WatchlistStock {
  symbol: string
  price: number
  change: number
  changePercent: number
  sparklineData: { value: number }[]
  trending: boolean
}

interface NotificationSetting {
  id: string
  label: string
  enabled: boolean
}

export function DashboardSidebar({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
  const [watchlist, setWatchlist] = useState<WatchlistStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [dataSourceFilters, setDataSourceFilters] = useState({
    alpaca: true,
    stocktwits: true,
    jigsawstack: true,
  })

  const [notifications, setNotifications] = useState<NotificationSetting[]>([
    { id: "signals", label: "New Signals", enabled: true },
    { id: "risk", label: "Risk Breaches", enabled: true },
    { id: "volatility", label: "High Volatility", enabled: false },
    { id: "earnings", label: "Earnings Alerts", enabled: true },
  ])

  const [quickStats, setQuickStats] = useState({
    signalWinRate: 0,
    portfolioPnL: 0,
    maxDrawdown: 0,
    activeSignals: 0,
  })

  // Load real data from API
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        setError(null)
        
        // Load hot stocks for watchlist
        const hotStocksResponse = await getHotStocks()
        const stocks: WatchlistStock[] = hotStocksResponse.stocks.slice(0, 5).map(stock => ({
          symbol: stock.symbol,
          price: stock.price,
          change: stock.change,
          changePercent: stock.changePercent,
          sparklineData: stock.sparklineData || Array.from({ length: 10 }, () => ({ 
            value: stock.price + (Math.random() - 0.5) * 10 
          })),
          trending: stock.trending || false,
        }))
        setWatchlist(stocks)

        // Load portfolio data for quick stats
        const portfolio = await getPortfolio()
        setQuickStats({
          signalWinRate: 75.0, // This would come from backend analytics
          portfolioPnL: portfolio.totalPnL,
          maxDrawdown: portfolio.totalPnL < 0 ? portfolio.totalPnL : 0,
          activeSignals: portfolio.positions.length,
        })
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data')
        console.error('Error loading sidebar data:', err)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  const toggleDataSource = (source: keyof typeof dataSourceFilters) => {
    setDataSourceFilters((prev) => ({ ...prev, [source]: !prev[source] }))
  }

  const toggleNotification = (id: string) => {
    setNotifications((prev) => prev.map((notif) => (notif.id === id ? { ...notif, enabled: !notif.enabled } : notif)))
  }

  const getPriceChangeColor = (change: number) => {
    return change >= 0 ? "text-green-400" : "text-red-400"
  }

  const getPriceChangeIcon = (change: number) => {
    return change >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />
  }

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={onToggle} />}

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 h-full w-80 bg-sidebar border-r border-sidebar-border z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 lg:static lg:z-auto`}
      >
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
            <h2 className="font-semibold text-sidebar-foreground">Market Overview</h2>
            <Button variant="ghost" size="icon" onClick={onToggle} className="lg:hidden">
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Sidebar Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Watchlist */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-sm text-sidebar-foreground">Trending Stocks</h3>
                <Badge variant="secondary" className="text-xs">
                  Live
                </Badge>
              </div>
              <div className="space-y-2">
                {loading ? (
                  // Loading skeleton
                  Array.from({ length: 5 }).map((_, i) => (
                    <Card key={i} className="p-3 animate-pulse">
                      <div className="flex items-center justify-between mb-2">
                        <div className="h-4 w-12 bg-gray-300 rounded"></div>
                        <div className="h-6 w-6 bg-gray-300 rounded"></div>
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="h-4 w-16 bg-gray-300 rounded mb-1"></div>
                          <div className="h-3 w-20 bg-gray-300 rounded"></div>
                        </div>
                        <div className="w-16 h-8 bg-gray-300 rounded"></div>
                      </div>
                    </Card>
                  ))
                ) : error ? (
                  <Card className="p-3 text-center">
                    <p className="text-sm text-red-500">Failed to load stocks</p>
                  </Card>
                ) : (
                  watchlist.map((stock) => (
                  <Card key={stock.symbol} className="p-3 hover:bg-sidebar-accent cursor-pointer transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm text-sidebar-foreground">{stock.symbol}</span>
                        {stock.trending && (
                          <Badge variant="outline" className="text-xs px-1 py-0">
                            Hot
                          </Badge>
                        )}
                      </div>
                      <Button variant="ghost" size="icon" className="h-6 w-6">
                        <Eye className="h-3 w-3" />
                      </Button>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium text-sidebar-foreground">${stock.price.toFixed(2)}</div>
                        <div className={`text-xs flex items-center gap-1 ${getPriceChangeColor(stock.change)}`}>
                          {getPriceChangeIcon(stock.change)}
                          {stock.change > 0 ? "+" : ""}
                          {stock.change.toFixed(2)} ({stock.changePercent > 0 ? "+" : ""}
                          {stock.changePercent.toFixed(1)}%)
                        </div>
                      </div>
                      <div className="w-16 h-8">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={stock.sparklineData}>
                            <Line
                              type="monotone"
                              dataKey="value"
                              stroke={stock.change >= 0 ? "#4CAF50" : "#F44336"}
                              strokeWidth={1}
                              dot={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </Card>
                  ))
                )}
              </div>
            </div>

            <div className="border-t border-sidebar-border my-4" />

            {/* Data Source Filters */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Filter className="h-4 w-4 text-sidebar-foreground" />
                <h3 className="font-medium text-sm text-sidebar-foreground">Data Sources</h3>
              </div>
              <div className="space-y-2">
                {Object.entries(dataSourceFilters).map(([source, enabled]) => (
                  <div key={source} className="flex items-center justify-between">
                    <span className="text-sm text-sidebar-foreground capitalize">{source}</span>
                    <Switch
                      checked={enabled}
                      onCheckedChange={() => toggleDataSource(source as keyof typeof dataSourceFilters)}
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-sidebar-border my-4" />

            {/* Notifications */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Bell className="h-4 w-4 text-sidebar-foreground" />
                <h3 className="font-medium text-sm text-sidebar-foreground">Notifications</h3>
              </div>
              <div className="space-y-2">
                {notifications.map((notification) => (
                  <div key={notification.id} className="flex items-center justify-between">
                    <span className="text-sm text-sidebar-foreground">{notification.label}</span>
                    <Switch
                      checked={notification.enabled}
                      onCheckedChange={() => toggleNotification(notification.id)}
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-sidebar-border my-4" />

            {/* Quick Stats */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-sidebar-foreground" />
                <h3 className="font-medium text-sm text-sidebar-foreground">Quick Stats</h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-sidebar-foreground">Signal Win Rate</span>
                  {loading ? (
                    <div className="h-5 w-12 bg-gray-300 rounded animate-pulse"></div>
                  ) : (
                    <Badge variant="default" className="text-xs">
                      {quickStats.signalWinRate}%
                    </Badge>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-sidebar-foreground">Portfolio P&L</span>
                  {loading ? (
                    <div className="h-4 w-16 bg-gray-300 rounded animate-pulse"></div>
                  ) : (
                    <span className={`text-sm font-medium ${getPriceChangeColor(quickStats.portfolioPnL)}`}>
                      ${quickStats.portfolioPnL.toFixed(0)}
                    </span>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-sidebar-foreground">Max Drawdown</span>
                  {loading ? (
                    <div className="h-4 w-16 bg-gray-300 rounded animate-pulse"></div>
                  ) : (
                    <span className="text-sm font-medium text-red-400">${quickStats.maxDrawdown.toFixed(0)}</span>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-sidebar-foreground">Active Signals</span>
                  {loading ? (
                    <div className="h-5 w-8 bg-gray-300 rounded animate-pulse"></div>
                  ) : (
                    <Badge variant="secondary" className="text-xs">
                      {quickStats.activeSignals}
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar Footer */}
          <div className="border-t border-sidebar-border p-4">
            <div className="flex items-center justify-between text-xs text-sidebar-foreground">
              <span>Last Update</span>
              <span>{new Date().toLocaleTimeString()}</span>
            </div>
            <div className="flex items-center justify-between text-xs text-sidebar-foreground mt-1">
              <span>Data Sources</span>
              <div className="flex gap-1">
                <Badge variant="outline" className="text-xs px-1 py-0">
                  Alpaca
                </Badge>
                <Badge variant="outline" className="text-xs px-1 py-0">
                  StockTwits
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

// Mobile sidebar toggle button
export function SidebarToggle({ onClick }: { onClick: () => void }) {
  return (
    <Button variant="ghost" size="icon" onClick={onClick} className="lg:hidden">
      <Menu className="h-4 w-4" />
    </Button>
  )
}
