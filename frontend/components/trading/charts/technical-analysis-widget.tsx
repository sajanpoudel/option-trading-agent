"use client"

import { useEffect, useRef } from "react"

interface TechnicalAnalysisWidgetProps {
  symbol: string
  width?: string | number
  height?: string | number
  theme?: "light" | "dark"
}

export function TechnicalAnalysisWidget({
  symbol,
  width = "100%",
  height = 400,
  theme = "light",
}: TechnicalAnalysisWidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js"
    script.type = "text/javascript"
    script.async = true
    script.innerHTML = JSON.stringify({
      interval: "1m",
      width: width,
      isTransparent: false,
      height: height,
      symbol: symbol,
      showIntervalTabs: true,
      locale: "en",
      colorTheme: theme,
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
      className="tradingview-widget-container rounded-lg overflow-hidden border"
      style={{ width, height }}
    />
  )
}
