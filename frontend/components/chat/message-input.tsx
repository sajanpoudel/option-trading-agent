"use client"

import { useState, useRef, useEffect, type KeyboardEvent, type ChangeEvent } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

interface MessageInputProps {
  onSendMessage: (message: string) => void
  placeholder?: string
  className?: string
}

export function MessageInput({ onSendMessage, placeholder, className }: MessageInputProps) {
  const [message, setMessage] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [rows, setRows] = useState(1)

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current
    if (!textarea) return

    // Create a shadow div to measure text
    const measureDiv = document.createElement('div')
    measureDiv.style.cssText = window.getComputedStyle(textarea).cssText
    measureDiv.style.height = 'auto'
    measureDiv.style.position = 'absolute'
    measureDiv.style.visibility = 'hidden'
    measureDiv.style.width = textarea.offsetWidth + 'px'
    measureDiv.style.whiteSpace = 'pre-wrap'
    measureDiv.style.wordBreak = 'break-word'
    measureDiv.style.overflow = 'hidden'
    measureDiv.innerText = textarea.value || textarea.placeholder

    document.body.appendChild(measureDiv)
    
    const minHeight = 44
    const maxHeight = 200
    const padding = 24 // total vertical padding
    const lineHeight = parseInt(window.getComputedStyle(textarea).lineHeight) || 20

    // Get the height needed for the content
    const contentHeight = measureDiv.offsetHeight + padding
    document.body.removeChild(measureDiv)

    // Calculate the optimal height
    const optimalHeight = Math.max(minHeight, Math.min(contentHeight, maxHeight))
    
    // Only update if height actually needs to change
    if (textarea.style.height !== optimalHeight + 'px') {
      textarea.style.height = optimalHeight + 'px'
      
      // Update rows
      const newRows = Math.min(Math.ceil((optimalHeight - padding) / lineHeight), 10)
      if (newRows !== rows) {
        setRows(newRows)
      }
    }
  }

  // Debounce the height adjustment
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      adjustTextareaHeight()
    }, 10) // Small delay for smoother updates
    return () => clearTimeout(timeoutId)
  }, [message])

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value)
  }

  const handleSend = () => {
    if (message.trim()) {
      onSendMessage(message.trim())
      setMessage("")
      // Reset height after sending
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
      setRows(1)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className={cn("relative", className)}>
      <div className="group relative flex items-end gap-2 rounded-xl bg-muted/50 backdrop-blur-sm ring-1 ring-border/50 p-1 shadow-sm transition-all duration-200 hover:shadow-md hover:translate-y-[-1px] focus-within:shadow-md focus-within:translate-y-[-1px]">
        <Textarea
          ref={textareaRef}
          value={message}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Type your message..."}
          className={cn(
            "min-h-[44px] resize-none bg-transparent border-0",
            "focus-visible:ring-0 focus-visible:ring-offset-0 p-2",
            "scrollbar-thin scrollbar-thumb-primary/10 hover:scrollbar-thumb-primary/20",
            "will-change-[height] motion-reduce:transition-none"
          )}
          style={{
            height: '44px', // Start with min-height
            overflow: rows >= 10 ? 'auto' : 'hidden',
            transition: 'height 150ms cubic-bezier(0.4, 0, 0.2, 1)'
          }}
          rows={rows}
        />
        <Button 
          onClick={handleSend} 
          disabled={!message.trim()} 
          size="icon"
          className="h-[34px] w-[34px] rounded-xl bg-primary hover:bg-primary/90 transition-colors"
        >
          <svg 
            className="w-4 h-4 text-primary-foreground" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
            style={{ transform: "rotate(45deg)" }}
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" 
            />
          </svg>
        </Button>
      </div>

      {/* Keyboard shortcut hint */}
      <div className="absolute right-2 bottom-[-20px] text-[10px] text-muted-foreground">
        Press Enter to send, Shift + Enter for new line
      </div>
    </div>
  )
}
