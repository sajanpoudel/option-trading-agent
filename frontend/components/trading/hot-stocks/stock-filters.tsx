"use client"

import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Search, Flame, Filter } from "lucide-react"

interface StockFiltersProps {
  searchQuery: string
  onSearchChange: (query: string) => void
  sortBy: "change" | "volume"
  onSortChange: (sort: "change" | "volume") => void
  totalStocks: number
  trendingCount: number
}

export function StockFilters({
  searchQuery,
  onSearchChange,
  sortBy,
  onSortChange,
  totalStocks,
  trendingCount,
}: StockFiltersProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <Flame className="h-5 w-5 text-primary" />
        <h2 className="text-xl font-semibold">Hot Stocks</h2>
        <Badge variant="secondary" className="text-xs">
          {trendingCount} Trending
        </Badge>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search stocks..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Sort */}
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Select value={sortBy} onValueChange={onSortChange}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="change">Price Change</SelectItem>
              <SelectItem value="volume">Volume</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="text-sm text-muted-foreground">Showing {totalStocks} stocks</div>
    </div>
  )
}
