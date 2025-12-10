"use client"

import { useEffect, useRef } from "react"

interface TradingViewChartProps {
  symbol: string
  height?: string | number
  theme?: "light" | "dark"
  style?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
  interval?: "1" | "3" | "5" | "15" | "30" | "60" | "120" | "180" | "240" | "D" | "W" | "M"
  allow_symbol_change?: boolean
  hide_side_toolbar?: boolean
  hide_top_toolbar?: boolean
  save_image?: boolean
  studies?: string[]
  width?: string | number
  locale?: string
}

export function TradingViewChart({
  symbol,
  height = 600,
  theme = "dark",
  style = "1",
  interval = "D",
  allow_symbol_change = true,
  hide_side_toolbar = false,
  hide_top_toolbar = false,
  save_image = true,
  studies = [],
  width = "100%",
  locale = "en"
}: TradingViewChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
    script.type = "text/javascript"
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: symbol,
      interval: interval,
      timezone: "America/New_York", // US market timezone
      theme: theme,
      style: style,
      locale: locale,
      toolbar_bg: theme === "dark" ? "#1e1e1e" : "#f1f3f6",
      enable_publishing: false,
      allow_symbol_change: allow_symbol_change,
      hide_side_toolbar: hide_side_toolbar,
      hide_top_toolbar: hide_top_toolbar,
      save_image: save_image,
      studies: [], // No indicators - clean line chart only
      container_id: `tradingview_${symbol}_${Date.now()}`,
      width: width,
      height: height,
      // Additional configuration for better integration
      withdateranges: true,
      range: "1D",
      hide_legend: false,
      hide_volume: true, // Hide volume for cleaner look
      scalePosition: "right",
      scaleMode: "Normal",
      fontFamily: "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
      fontSize: "12",
      noTimeScale: false,
      valuesTracking: "1",
      changeMode: "price-and-percent",
      chartType: "line", // Clean line chart
      // Exclude after-hours and premarket data
      session: "regular", // Only regular trading hours
      extendedHours: false, // No extended hours
      // Trading hours: 9:30 AM - 4:00 PM EST
      "trading_hours": {
        "regular": {
          "start": "09:30",
          "end": "16:00"
        }
      },
      gridLineColor: theme === "dark" ? "rgba(255, 255, 255, 0.06)" : "rgba(240, 243, 250, 0.06)",
      scaleFontColor: theme === "dark" ? "rgba(255, 255, 255, 0.7)" : "rgba(120, 123, 134, 1)",
      belowLineFillColorGrowing: "rgba(41, 98, 255, 0.12)",
      belowLineFillColorFalling: "rgba(41, 98, 255, 0.12)",
      belowLineFillColorGrowingBottom: "rgba(41, 98, 255, 0)",
      belowLineFillColorFallingBottom: "rgba(41, 98, 255, 0)",
      symbolActiveColor: "rgba(41, 98, 255, 0.12)",
      // Better sizing and layout
      overrides: {
        "paneProperties.background": theme === "dark" ? "#1e1e1e" : "#ffffff",
        "paneProperties.vertGridProperties.color": theme === "dark" ? "rgba(255, 255, 255, 0.06)" : "rgba(240, 243, 250, 0.06)",
        "paneProperties.horzGridProperties.color": theme === "dark" ? "rgba(255, 255, 255, 0.06)" : "rgba(240, 243, 250, 0.06)",
        "symbolWatermarkProperties.transparency": 90,
        "scalesProperties.textColor": theme === "dark" ? "rgba(255, 255, 255, 0.7)" : "rgba(120, 123, 134, 1)",
        // Disable drawing tools and overlays for clean line chart
        "drawingToolbar.show": false,
        "paneProperties.legendProperties.showSeriesTitle": false,
        "paneProperties.legendProperties.showSeriesOHLC": false,
        // Trading hours configuration
        "tradingSession": "regular",
        "session": "regular",
        "extendedHours": false,
        "showExtendedHours": false,
      }
    })

    containerRef.current.innerHTML = ""
    containerRef.current.appendChild(script)

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ""
      }
    }
  }, [symbol, height, theme, style, interval, allow_symbol_change, hide_side_toolbar, hide_top_toolbar, save_image, studies, width, locale])

  return (
    <div
      ref={containerRef}
      className="tradingview-widget-container rounded-lg overflow-hidden border"
      style={{ width, height }}
    />
  )
}
