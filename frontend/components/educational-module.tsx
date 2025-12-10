"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { BookOpen, ChevronDown, ChevronRight, HelpCircle, Play, CheckCircle, XCircle, Lightbulb, X } from "lucide-react"
import { getEducationalContentForStock, generateQuiz, getTradingGlossary, type EducationalContent, type Quiz, type Glossary } from "@/lib/api-service"

interface EducationalModuleProps {
  selectedStock: string
  analysisData?: any
}

interface LegacyQuizQuestion {
  id: string
  question: string
  options: string[]
  correctAnswer: number
  explanation: string
}

export function EducationalModule({ selectedStock, analysisData }: EducationalModuleProps) {
  const [currentContent, setCurrentContent] = useState<EducationalContent | null>(null)
  const [glossaryData, setGlossaryData] = useState<Glossary | null>(null)
  const [isLoadingGlossary, setIsLoadingGlossary] = useState(false)
  const [expandedSections, setExpandedSections] = useState<string[]>(["current-signal"])
  const [currentQuiz, setCurrentQuiz] = useState<LegacyQuizQuestion | null>(null)
  const [isLoadingQuiz, setIsLoadingQuiz] = useState(false)
  const [quizAnswer, setQuizAnswer] = useState<number | null>(null)
  const [showQuizResult, setShowQuizResult] = useState(false)
  const [learningProgress, setLearningProgress] = useState(65)
  const [showVideoModal, setShowVideoModal] = useState(false)

  // Load educational content from backend
  const loadEducationalContent = async () => {
    try {
      const content = await getEducationalContentForStock(selectedStock, analysisData)
      setCurrentContent(content)
    } catch (error) {
      console.error('Failed to load educational content:', error)
      // Fallback to mock content
      setCurrentContent({
        content_id: `fallback_${selectedStock}`,
        title: `Understanding ${selectedStock} Analysis`,
        topic: 'stock_analysis',
        difficulty: 'beginner',
        content_type: 'lesson',
        estimated_duration_minutes: 10,
        content: {
          introduction: `Learn about the current analysis for ${selectedStock} and key trading concepts.`,
          key_concepts: [
            'Technical indicators and market signals',
            'Risk management principles',
            'Options trading fundamentals',
            'Market sentiment analysis'
          ]
        },
        prerequisites: [],
        learning_objectives: [
          'Understand current market analysis',
          'Learn key trading concepts',
          'Apply risk management principles'
        ],
        tags: ['analysis', 'trading', selectedStock.toLowerCase()]
      })
    }
  }
  
  // Load glossary data
  const loadGlossary = async () => {
    try {
      setIsLoadingGlossary(true)
      const glossary = await getTradingGlossary()
      setGlossaryData(glossary)
    } catch (error) {
      console.error('Failed to load glossary:', error)
    } finally {
      setIsLoadingGlossary(false)
    }
  }

  // Legacy quiz questions for fallback
  const fallbackQuizQuestions: LegacyQuizQuestion[] = [
    {
      id: "1",
      question: "What does bullish RSI divergence indicate?",
      options: [
        "Price will continue falling",
        "Potential trend reversal to the upside",
        "High volatility ahead",
        "Options are overpriced",
      ],
      correctAnswer: 1,
      explanation:
        "Bullish RSI divergence occurs when price makes lower lows while RSI makes higher lows, suggesting weakening downward momentum and potential reversal.",
    },
    {
      id: "2",
      question: "Which Greek measures an option's sensitivity to time decay?",
      options: ["Delta", "Gamma", "Theta", "Vega"],
      correctAnswer: 2,
      explanation:
        "Theta measures how much an option's price decreases as time passes, representing time decay. It's typically negative for long options positions.",
    },
    {
      id: "3",
      question: "What does unusual call volume typically suggest?",
      options: [
        "Random market noise",
        "Bearish sentiment",
        "Potential bullish positioning by informed traders",
        "High volatility only",
      ],
      correctAnswer: 2,
      explanation:
        "Unusual call volume often indicates that informed traders or institutions are positioning for potential upward price movement.",
    },
  ]

  useEffect(() => {
    loadEducationalContent()
    loadGlossary()
  }, [selectedStock, analysisData])

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => (prev.includes(section) ? prev.filter((s) => s !== section) : [...prev, section]))
  }

  const startQuiz = async () => {
    try {
      setIsLoadingQuiz(true)
      // Try to generate AI quiz first
      const quiz = await generateQuiz('options_basics', 'beginner', 1)
      if (quiz.questions && quiz.questions.length > 0) {
        const question = quiz.questions[0]
        setCurrentQuiz({
          id: question.id,
          question: question.question,
          options: question.options || [],
          correctAnswer: question.options?.findIndex(opt => opt === question.correct_answer) || 0,
          explanation: question.explanation
        })
      } else {
        // Fallback to local questions
        const randomQuestion = fallbackQuizQuestions[Math.floor(Math.random() * fallbackQuizQuestions.length)]
        setCurrentQuiz(randomQuestion)
      }
    } catch (error) {
      console.error('Failed to generate quiz:', error)
      // Fallback to local questions
      const randomQuestion = fallbackQuizQuestions[Math.floor(Math.random() * fallbackQuizQuestions.length)]
      setCurrentQuiz(randomQuestion)
    } finally {
      setIsLoadingQuiz(false)
      setQuizAnswer(null)
      setShowQuizResult(false)
    }
  }

  const submitQuizAnswer = () => {
    if (quizAnswer !== null) {
      setShowQuizResult(true)
      if (quizAnswer === currentQuiz?.correctAnswer) {
        setLearningProgress((prev) => Math.min(prev + 5, 100))
      }
    }
  }

  // Get glossary terms from API or fallback to content
  const glossaryTerms = glossaryData ? 
    Object.values(glossaryData.glossary).map(term => ({
      term: term.term,
      definition: term.definition
    })) : 
    (currentContent?.content.key_concepts?.map(concept => ({ 
      term: concept, 
      definition: `Learn more about ${concept}` 
    })) || [])

  return (
    <Card className="col-span-12 md:col-span-6 lg:col-span-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          Educational Module
          <Badge variant="outline" className="ml-auto">
            Progress: {learningProgress}%
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Learning Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Learning Progress</span>
            <span>{learningProgress}%</span>
          </div>
          <Progress value={learningProgress} className="w-full" />
        </div>

        {/* Current Signal Explanation */}
        <Collapsible open={expandedSections.includes("current-signal")}>
          <CollapsibleTrigger
            className="flex items-center justify-between w-full p-2 hover:bg-accent rounded-lg"
            onClick={() => toggleSection("current-signal")}
          >
            <span className="font-medium text-sm">Current Signal Explanation</span>
            {expandedSections.includes("current-signal") ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2 p-2 text-sm text-muted-foreground space-y-2">
            {currentContent && (
              <>
                <h4 className="font-medium text-foreground">{currentContent.title}</h4>
                <div className="space-y-2">
                  {currentContent.content.introduction && (
                    <p>{currentContent.content.introduction}</p>
                  )}
                  
                  {/* Stock-specific analysis if available */}
                  {currentContent.content.stock_specific && (
                    <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
                      <div className="font-medium text-blue-800 mb-2">Current Analysis</div>
                      <div className="text-sm space-y-1">
                        <div>Signal: <span className="font-medium">{currentContent.content.stock_specific.current_signal}</span></div>
                        <div>Confidence: <span className="font-medium">{currentContent.content.stock_specific.confidence}%</span></div>
                      </div>
                    </div>
                  )}
                  
                  {/* Key concepts */}
                  {currentContent.content.key_concepts && (
                    <div>
                      <div className="font-medium mb-2">Key Concepts:</div>
                      <ul className="space-y-1">
                        {currentContent.content.key_concepts.map((concept, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <Lightbulb className="h-3 w-3 mt-1 text-yellow-500 flex-shrink-0" />
                            <span>{concept}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </>
            )}
          </CollapsibleContent>
        </Collapsible>

        {/* Interactive Glossary */}
        <Collapsible open={expandedSections.includes("glossary")}>
          <CollapsibleTrigger
            className="flex items-center justify-between w-full p-2 hover:bg-accent rounded-lg"
            onClick={() => toggleSection("glossary")}
          >
            <span className="font-medium text-sm">Interactive Glossary</span>
            {expandedSections.includes("glossary") ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2 space-y-2">
            {isLoadingGlossary ? (
              <div className="text-center py-4">
                <div className="animate-spin h-4 w-4 border-2 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
                <p className="text-sm text-muted-foreground">Loading glossary...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-2">
                {glossaryTerms.slice(0, 8).map((term, index) => (
                  <Dialog key={index}>
                    <DialogTrigger asChild>
                      <Button
                        variant="ghost"
                        className="justify-start text-left h-auto p-2 text-sm text-primary hover:bg-accent"
                      >
                        <HelpCircle className="h-3 w-3 mr-2 flex-shrink-0" />
                        {term.term}
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-md">
                      <DialogHeader>
                        <DialogTitle>{term.term}</DialogTitle>
                      </DialogHeader>
                      <p className="text-sm text-muted-foreground">{term.definition}</p>
                      {glossaryData && glossaryData.glossary[term.term.toLowerCase().replace(' ', '_')] && (
                        <div className="mt-3">
                          <p className="text-sm font-medium mb-1">Example:</p>
                          <p className="text-sm text-muted-foreground">
                            {glossaryData.glossary[term.term.toLowerCase().replace(' ', '_')].example}
                          </p>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                ))}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>

        {/* Quiz Section */}
        <Collapsible open={expandedSections.includes("quiz")}>
          <CollapsibleTrigger
            className="flex items-center justify-between w-full p-2 hover:bg-accent rounded-lg"
            onClick={() => toggleSection("quiz")}
          >
            <span className="font-medium text-sm">Knowledge Quiz</span>
            {expandedSections.includes("quiz") ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2 space-y-3">
            {!currentQuiz ? (
              <div className="text-center py-4">
                <p className="text-sm text-muted-foreground mb-3">Test your understanding with a quick quiz!</p>
                <Button onClick={startQuiz} disabled={isLoadingQuiz} className="gap-2">
                  <Play className="h-4 w-4" />
                  {isLoadingQuiz ? 'Generating Quiz...' : 'Start Quiz'}
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="p-3 border rounded-lg">
                  <h4 className="font-medium text-sm mb-3">{currentQuiz.question}</h4>
                  <div className="space-y-2">
                    {currentQuiz.options.map((option, index) => (
                      <Button
                        key={index}
                        variant={quizAnswer === index ? "default" : "outline"}
                        className="w-full justify-start text-left h-auto p-2 text-sm"
                        onClick={() => setQuizAnswer(index)}
                        disabled={showQuizResult}
                      >
                        {String.fromCharCode(65 + index)}. {option}
                      </Button>
                    ))}
                  </div>
                </div>

                {!showQuizResult ? (
                  <Button onClick={submitQuizAnswer} disabled={quizAnswer === null} className="w-full">
                    Submit Answer
                  </Button>
                ) : (
                  <div className="space-y-2">
                    <div
                      className={`flex items-center gap-2 p-2 rounded-lg ${
                        quizAnswer === currentQuiz.correctAnswer
                          ? "bg-green-500/20 text-green-400"
                          : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {quizAnswer === currentQuiz.correctAnswer ? (
                        <CheckCircle className="h-4 w-4" />
                      ) : (
                        <XCircle className="h-4 w-4" />
                      )}
                      <span className="text-sm font-medium">
                        {quizAnswer === currentQuiz.correctAnswer ? "Correct!" : "Incorrect"}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">{currentQuiz.explanation}</p>
                    <Button onClick={startQuiz} variant="outline" className="w-full bg-transparent">
                      Try Another Question
                    </Button>
                  </div>
                )}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>

        {/* Related Concepts */}
        <div className="border-t pt-3">
          <h4 className="font-medium text-sm mb-2">Related Topics</h4>
          <div className="flex flex-wrap gap-2">
            {currentContent?.tags?.map((tag, index) => (
              <Badge key={index} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            )) || [
              <Badge key="analysis" variant="secondary" className="text-xs">Analysis</Badge>,
              <Badge key="trading" variant="secondary" className="text-xs">Trading</Badge>,
              <Badge key="options" variant="secondary" className="text-xs">Options</Badge>
            ]}
          </div>
        </div>

        {/* Video Section */}
        <div className="border-t pt-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-sm">Video Tutorials</h4>
            <Badge variant="outline" className="text-xs bg-green-500/10 text-green-600 border-green-500/20">
              Available
            </Badge>
          </div>
          <div className="bg-muted rounded-lg p-4 text-center">
            <Play className="h-8 w-8 mx-auto mb-2 text-primary" />
            <p className="text-xs text-muted-foreground mb-3">Watch our educational video tutorial</p>
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
      </CardContent>
    </Card>
  )
}
