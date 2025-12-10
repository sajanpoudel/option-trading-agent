"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeRaw from "rehype-raw"
import rehypeSanitize from "rehype-sanitize"
import { cn } from "@/lib/utils"

interface MarkdownRendererProps {
  content: string
  className?: string
}

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  return (
    <div className={cn("prose dark:prose-invert max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        components={{
        // Override default element styling
        h1: ({ className, ...props }) => (
          <h1 className={cn("text-2xl font-bold mt-6 mb-4 text-foreground", className)} {...props} />
        ),
        h2: ({ className, ...props }) => (
          <h2 className={cn("text-xl font-bold mt-5 mb-3 text-foreground", className)} {...props} />
        ),
        h3: ({ className, ...props }) => (
          <h3 className={cn("text-lg font-bold mt-4 mb-2 text-foreground", className)} {...props} />
        ),
        p: ({ className, ...props }) => (
          <p className={cn("leading-7 mb-3 text-muted-foreground", className)} {...props} />
        ),
        ul: ({ className, ...props }) => (
          <ul className={cn("list-disc list-inside mb-3 text-muted-foreground", className)} {...props} />
        ),
        ol: ({ className, ...props }) => (
          <ol className={cn("list-decimal list-inside mb-3 text-muted-foreground", className)} {...props} />
        ),
        li: ({ className, ...props }) => (
          <li className={cn("mt-1", className)} {...props} />
        ),
        blockquote: ({ className, ...props }) => (
          <blockquote
            className={cn(
              "mt-3 mb-3 border-l-2 border-border pl-4 italic text-muted-foreground",
              className
            )}
            {...props}
          />
        ),
        a: ({ className, ...props }) => (
          <a
            className={cn("text-primary hover:underline cursor-pointer", className)}
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          />
        ),
        code: ({ className, ...props }) => (
          <code
            className={cn(
              "relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm",
              className
            )}
            {...props}
          />
        ),
        pre: ({ className, ...props }) => (
          <pre
            className={cn(
              "mt-3 mb-3 overflow-x-auto rounded-lg bg-muted p-4 font-mono text-sm",
              className
            )}
            {...props}
          />
        ),
        img: ({ className, alt, ...props }) => (
          <img
            className={cn("rounded-lg border border-border", className)}
            alt={alt}
            {...props}
          />
        ),
        table: ({ className, ...props }) => (
          <div className="my-3 overflow-x-auto">
            <table
              className={cn("min-w-full divide-y divide-border", className)}
              {...props}
            />
          </div>
        ),
        th: ({ className, ...props }) => (
          <th
            className={cn(
              "px-3 py-2 text-left text-sm font-semibold text-foreground bg-muted",
              className
            )}
            {...props}
          />
        ),
        td: ({ className, ...props }) => (
          <td
            className={cn("px-3 py-2 text-left text-sm text-muted-foreground", className)}
            {...props}
          />
        ),
      }}
    >
      {content}
      </ReactMarkdown>
    </div>
  )
}
