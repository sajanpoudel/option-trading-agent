"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { BookOpen, Clock, User, RefreshCw, Play, X } from "lucide-react"
import { getEducationalContent, getEducationalContentForStock, type EducationalContent as EducationalContentType } from "@/lib/api-service"

interface EducationalContentProps {
  topic?: string
  difficulty?: 'beginner' | 'intermediate' | 'advanced'
  content_type?: 'lesson' | 'quiz' | 'interactive' | 'video'
  className?: string
  symbol?: string
  analysisData?: any
}

export function EducationalContent({ 
  topic = 'options_basics', 
  difficulty = 'beginner',
  content_type = 'lesson',
  className,
  symbol,
  analysisData
}: EducationalContentProps) {
  const [contentList, setContentList] = useState<EducationalContentType[]>([])
  const [currentContent, setCurrentContent] = useState<EducationalContentType | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showVideoModal, setShowVideoModal] = useState(false)

  const fetchContent = async () => {
    try {
      setIsLoading(true)
      setError(null)
      
      let contentList: EducationalContentType[]
      
      // If symbol is provided, get stock-specific educational content
      if (symbol && analysisData) {
        try {
          const stockContent = await getEducationalContentForStock(symbol, analysisData)
          contentList = [stockContent]
        } catch (error) {
          console.warn('Failed to get stock-specific content, falling back to general content:', error)
          contentList = await getEducationalContent(topic, difficulty, content_type, 1)
        }
      } else {
        contentList = await getEducationalContent(topic, difficulty, content_type, 1)
      }
      
      setContentList(contentList)
      if (contentList.length > 0) {
        setCurrentContent(contentList[0])
      }
    } catch (err) {
      console.error('Failed to fetch educational content:', err)
      setError('Failed to load educational content. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchContent()
  }, [topic, difficulty, content_type, symbol, analysisData])

  const getDifficultyColor = (level: string) => {
    switch (level) {
      case 'beginner': return 'bg-green-500/10 text-green-500 border-green-500/20'
      case 'intermediate': return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
      case 'advanced': return 'bg-red-500/10 text-red-500 border-red-500/20'
      default: return 'bg-gray-500/10 text-gray-500 border-gray-500/20'
    }
  }

  const renderContent = (content: EducationalContentType) => {
    return (
      <div className="space-y-6">
        {/* Introduction */}
        {content.content.introduction && (
          <div className="prose prose-sm max-w-none">
            <p className="text-muted-foreground leading-relaxed">
              {content.content.introduction}
            </p>
          </div>
        )}
        
        {/* Stock-Specific Analysis (if available) */}
        {content.content.stock_specific && (
          <div className="p-4 rounded-lg bg-blue-500/5 border border-blue-500/20">
            <h3 className="text-lg font-semibold mb-3 text-blue-600">Current Analysis for {content.content.stock_specific.symbol}</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Signal:</span>
                <span className="font-medium">{content.content.stock_specific.current_signal.toUpperCase()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Confidence:</span>
                <span className="font-medium">{content.content.stock_specific.confidence}%</span>
              </div>
              {content.content.stock_specific.key_factors && (
                <div>
                  <span className="text-sm text-muted-foreground block mb-1">Key Factors:</span>
                  <ul className="space-y-1">
                    {content.content.stock_specific.key_factors.map((factor, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                        <span className="text-blue-500 mt-1">•</span>
                        <span>{factor}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Key Concepts */}
        {content.content.key_concepts && content.content.key_concepts.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Key Concepts</h3>
            <ul className="space-y-2">
              {content.content.key_concepts.map((concept, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span className="text-sm text-muted-foreground">{concept}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Greeks (if applicable) */}
        {content.content.greeks && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Options Greeks</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(content.content.greeks).map(([greek, description]) => (
                <div key={greek} className="p-3 rounded-lg border bg-muted/50">
                  <div className="font-medium text-sm capitalize mb-1">{greek}</div>
                  <div className="text-xs text-muted-foreground">{description}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Practical Examples */}
        {content.content.practical_examples && content.content.practical_examples.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Practical Examples</h3>
            <div className="space-y-3">
              {content.content.practical_examples.map((example, index) => (
                <div key={index} className="p-4 rounded-lg bg-green-500/5 border border-green-500/20">
                  <div className="font-medium text-sm text-green-600 mb-2">{example.scenario}</div>
                  <div className="text-sm text-muted-foreground">{example.explanation}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Examples */}
        {content.content.examples && content.content.examples.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Examples</h3>
            <div className="space-y-3">
              {content.content.examples.map((example, index) => (
                <div key={index} className="p-4 rounded-lg bg-blue-500/5 border border-blue-500/20">
                  <div className="font-medium text-sm text-blue-600 mb-2">{example.scenario}</div>
                  <div className="text-sm text-muted-foreground">{example.explanation}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Practical Tips */}
        {content.content.practical_tips && content.content.practical_tips.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Practical Tips</h3>
            <ul className="space-y-2">
              {content.content.practical_tips.map((tip, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">💡</span>
                  <span className="text-sm text-muted-foreground">{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Learning Objectives */}
        {content.learning_objectives && content.learning_objectives.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Learning Objectives</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {content.learning_objectives.map((objective, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <span className="text-green-500">✓</span>
                  <span className="text-muted-foreground">{objective}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tags */}
        {content.tags && content.tags.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3">Related Topics</h3>
            <div className="flex flex-wrap gap-2">
              {content.tags.map((tag, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="h-6 w-48 bg-muted rounded animate-pulse" />
            <div className="h-5 w-20 bg-muted rounded animate-pulse" />
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-4 w-full bg-muted rounded animate-pulse" />
          <div className="h-4 w-3/4 bg-muted rounded animate-pulse" />
          <div className="h-32 w-full bg-muted rounded animate-pulse" />
          <div className="grid grid-cols-2 gap-4">
            <div className="h-20 w-full bg-muted rounded animate-pulse" />
            <div className="h-20 w-full bg-muted rounded animate-pulse" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Educational Content
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground mb-4">{error}</p>
          <Button onClick={fetchContent} variant="outline" className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Try Again
          </Button>
        </CardContent>
      </Card>
    )
  }

  if (!currentContent) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Educational Content
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground">No educational content available for this topic.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            {currentContent.title}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge 
              variant="outline" 
              className={getDifficultyColor(currentContent.difficulty)}
            >
              {currentContent.difficulty}
            </Badge>
          </div>
        </div>
        
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            {currentContent.estimated_duration_minutes} min
          </div>
          <div className="flex items-center gap-1">
            <User className="h-4 w-4" />
            {currentContent.content_type}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {renderContent(currentContent)}
        
        {/* Generate Video Button */}
        <div className="mt-6 pt-4 border-t">
          <div className="text-center">
            <Button 
              onClick={() => setShowVideoModal(true)}
              className="gap-2"
              size="sm"
            >
              <Play className="h-4 w-4" />
              Generate Video
            </Button>
          </div>
        </div>
      </CardContent>
      
      {/* Video Modal */}
      <Dialog open={showVideoModal} onOpenChange={setShowVideoModal}>
        <DialogContent className="max-w-4xl max-h-[90vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between">
              Educational Video Tutorial
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowVideoModal(false)}
                className="h-6 w-6 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>
          <div className="w-full h-[60vh] bg-black rounded-lg overflow-hidden">
            <video
              controls
              className="w-full h-full object-contain"
              autoPlay
              muted
            >
              <source src="/videoplayback.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
