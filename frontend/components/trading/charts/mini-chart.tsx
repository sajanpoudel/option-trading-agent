"use client"

import { useEffect, useRef } from "react"

interface MiniChartProps {
  symbol: string
  width?: string | number
  height?: string | number
  theme?: "light" | "dark"
}

export function MiniChart({ symbol, width = "100%", height = 200, theme = "light" }: MiniChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
    script.type = "text/javascript"
    script.async = true
    script.innerHTML = JSON.stringify({
      symbol: symbol,
      width: width,
      height: height,
      locale: "en",
      dateRange: "12M",
      colorTheme: theme,
      trendLineColor: "rgba(41, 98, 255, 1)",
      underLineColor: "rgba(41, 98, 255, 0.3)",
      underLineBottomColor: "rgba(41, 98, 255, 0)",
      isTransparent: false,
      autosize: false,
      largeChartUrl: "",
    })

    containerRef.current.innerHTML = ""
    containerRef.current.appendChild(script)

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ""
      }
    }
  }, [symbol, width, height, theme])

  return (
    <div
      ref={containerRef}
      className="tradingview-widget-container rounded-lg overflow-hidden"
      style={{ width, height }}
    />
  )
}
