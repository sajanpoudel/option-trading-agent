"use client"

import type React from "react"

import { useRef, useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Cable as Cube, RotateCcw, ZoomIn, ZoomOut } from "lucide-react"

interface GreeksData {
  delta: number
  gamma: number
  theta: number
  vega: number
}

export function PayoffSurface3D({ selectedStock }: { selectedStock: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [priceRange, setPriceRange] = useState([150, 250])
  const [daysToExpiry, setDaysToExpiry] = useState([30])
  const [greeks, setGreeks] = useState<GreeksData>({
    delta: 0.65,
    gamma: 0.08,
    theta: -0.12,
    vega: 0.25,
  })
  const [rotation, setRotation] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [isDragging, setIsDragging] = useState(false)
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 })

  // Mock options portfolio data
  const portfolioPositions = [
    { type: "Call", strike: 180, quantity: 5, premium: 3.2 },
    { type: "Put", strike: 170, quantity: -3, premium: 2.1 },
    { type: "Call", strike: 190, quantity: -2, premium: 1.8 },
  ]

  // Calculate payoff for given stock price and time
  const calculatePayoff = (stockPrice: number, timeValue: number): number => {
    let totalPayoff = 0

    portfolioPositions.forEach((position) => {
      const { type, strike, quantity, premium } = position
      let optionValue = 0

      if (type === "Call") {
        const intrinsicValue = Math.max(0, stockPrice - strike)
        const timeDecay = Math.max(0, premium * (timeValue / 30) * 0.7) // Simplified time decay
        optionValue = intrinsicValue + timeDecay
      } else {
        const intrinsicValue = Math.max(0, strike - stockPrice)
        const timeDecay = Math.max(0, premium * (timeValue / 30) * 0.7)
        optionValue = intrinsicValue + timeDecay
      }

      totalPayoff += (optionValue - premium) * quantity * 100 // 100 shares per contract
    })

    return totalPayoff
  }

  // Generate 3D surface data
  const generateSurfaceData = () => {
    const data = []
    const priceStep = (priceRange[1] - priceRange[0]) / 20
    const timeStep = 30 / 15 // 30 days divided into 15 steps

    for (let i = 0; i <= 20; i++) {
      for (let j = 0; j <= 15; j++) {
        const price = priceRange[0] + i * priceStep
        const time = j * timeStep
        const payoff = calculatePayoff(price, time)

        data.push({
          x: (i - 10) * 20, // Center around 0
          y: (j - 7.5) * 20, // Center around 0
          z: payoff / 100, // Scale down for visualization
          price,
          time,
          payoff,
        })
      }
    }
    return data
  }

  // Simple 3D projection
  const project3D = (x: number, y: number, z: number, canvas: HTMLCanvasElement) => {
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const scale = zoom * 2

    // Apply rotation
    const cosX = Math.cos(rotation.x)
    const sinX = Math.sin(rotation.x)
    const cosY = Math.cos(rotation.y)
    const sinY = Math.sin(rotation.y)

    // Rotate around Y axis
    const x1 = x * cosY - z * sinY
    const z1 = x * sinY + z * cosY

    // Rotate around X axis
    const y1 = y * cosX - z1 * sinX
    const z2 = y * sinX + z1 * cosX

    // Project to 2D
    const perspective = 300 / (300 + z2)
    return {
      x: centerX + x1 * scale * perspective,
      y: centerY + y1 * scale * perspective,
      z: z2,
    }
  }

  // Render the 3D surface
  const render3DSurface = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = "#0a0a0a"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const surfaceData = generateSurfaceData()

    // Draw wireframe
    ctx.strokeStyle = "#ff5722"
    ctx.lineWidth = 1
    ctx.globalAlpha = 0.8

    // Draw grid lines
    for (let i = 0; i <= 20; i++) {
      ctx.beginPath()
      for (let j = 0; j <= 15; j++) {
        const point = surfaceData[i * 16 + j]
        const projected = project3D(point.x, point.y, point.z, canvas)

        if (j === 0) {
          ctx.moveTo(projected.x, projected.y)
        } else {
          ctx.lineTo(projected.x, projected.y)
        }
      }
      ctx.stroke()
    }

    for (let j = 0; j <= 15; j++) {
      ctx.beginPath()
      for (let i = 0; i <= 20; i++) {
        const point = surfaceData[i * 16 + j]
        const projected = project3D(point.x, point.y, point.z, canvas)

        if (i === 0) {
          ctx.moveTo(projected.x, projected.y)
        } else {
          ctx.lineTo(projected.x, projected.y)
        }
      }
      ctx.stroke()
    }

    // Draw axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 2
    ctx.globalAlpha = 0.5

    // X axis (Price)
    ctx.beginPath()
    const xStart = project3D(-200, 0, 0, canvas)
    const xEnd = project3D(200, 0, 0, canvas)
    ctx.moveTo(xStart.x, xStart.y)
    ctx.lineTo(xEnd.x, xEnd.y)
    ctx.stroke()

    // Y axis (Time)
    ctx.beginPath()
    const yStart = project3D(0, -150, 0, canvas)
    const yEnd = project3D(0, 150, 0, canvas)
    ctx.moveTo(yStart.x, yStart.y)
    ctx.lineTo(yEnd.x, yEnd.y)
    ctx.stroke()

    // Z axis (Payoff)
    ctx.beginPath()
    const zStart = project3D(0, 0, -100, canvas)
    const zEnd = project3D(0, 0, 100, canvas)
    ctx.moveTo(zStart.x, zStart.y)
    ctx.lineTo(zEnd.x, zEnd.y)
    ctx.stroke()

    ctx.globalAlpha = 1
  }

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render3DSurface()
      animationRef.current = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [rotation, zoom, priceRange, daysToExpiry])

  // Mouse interaction handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return

    const deltaX = e.clientX - lastMouse.x
    const deltaY = e.clientY - lastMouse.y

    setRotation((prev) => ({
      x: prev.x + deltaY * 0.01,
      y: prev.y + deltaX * 0.01,
    }))

    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const resetView = () => {
    setRotation({ x: 0, y: 0 })
    setZoom(1)
  }

  // Update Greeks based on current portfolio
  useEffect(() => {
    // Simplified Greeks calculation
    const currentPrice = (priceRange[0] + priceRange[1]) / 2
    const newGreeks = {
      delta: 0.65 + (Math.random() - 0.5) * 0.1,
      gamma: 0.08 + (Math.random() - 0.5) * 0.02,
      theta: -0.12 + (Math.random() - 0.5) * 0.04,
      vega: 0.25 + (Math.random() - 0.5) * 0.05,
    }
    setGreeks(newGreeks)
  }, [priceRange, selectedStock])

  return (
    <Card className="col-span-12 md:col-span-6 lg:col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cube className="h-5 w-5" />
          3D Payoff Surface
          <Badge variant="outline" className="ml-auto">
            {selectedStock}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 3D Canvas */}
        <div className="relative border rounded-lg bg-black overflow-hidden">
          <canvas
            ref={canvasRef}
            width={300}
            height={200}
            className="w-full cursor-grab active:cursor-grabbing"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />

          {/* Controls Overlay */}
          <div className="absolute top-2 right-2 flex gap-1">
            <Button variant="ghost" size="icon" className="h-6 w-6 bg-black/50" onClick={resetView}>
              <RotateCcw className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 bg-black/50"
              onClick={() => setZoom((prev) => Math.min(prev * 1.2, 3))}
            >
              <ZoomIn className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 bg-black/50"
              onClick={() => setZoom((prev) => Math.max(prev / 1.2, 0.5))}
            >
              <ZoomOut className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Controls */}
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-2 block">
              Price Range: ${priceRange[0]} - ${priceRange[1]}
            </label>
            <Slider value={priceRange} onValueChange={setPriceRange} min={100} max={300} step={5} className="w-full" />
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Days to Expiry: {daysToExpiry[0]}</label>
            <Slider value={daysToExpiry} onValueChange={setDaysToExpiry} min={1} max={60} step={1} className="w-full" />
          </div>
        </div>

        {/* Portfolio Greeks */}
        <div className="border-t pt-3">
          <h4 className="font-medium text-sm mb-2">Portfolio Greeks</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span>Delta:</span>
              <span className={greeks.delta > 0 ? "text-green-400" : "text-red-400"}>{greeks.delta.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span>Gamma:</span>
              <span className="text-muted-foreground">{greeks.gamma.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span>Theta:</span>
              <span className="text-red-400">{greeks.theta.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span>Vega:</span>
              <span className="text-muted-foreground">{greeks.vega.toFixed(3)}</span>
            </div>
          </div>
        </div>

        {/* Portfolio Summary */}
        <div className="text-xs text-muted-foreground">
          <p>Interactive 3D visualization of portfolio payoff across stock prices and time decay.</p>
          <p className="mt-1">Drag to rotate • Scroll to zoom • Use controls to adjust parameters</p>
        </div>
      </CardContent>
    </Card>
  )
}
