"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  Brain, 
  TrendingUp, 
  Eye, 
  Clock, 
  Shield, 
  BookOpen,
  Activity,
  CheckCircle,
  Loader2,
  Zap
} from "lucide-react"
import { cn } from "@/lib/utils"

export interface Agent {
  id: string
  name: string
  icon: React.ComponentType<{ className?: string }>
  color: string
  description: string
  weight: number
  status: 'pending' | 'thinking' | 'completed' | 'error'
  duration?: number
  result?: any
}

interface AgentVisualizationProps {
  agents: Agent[]
  currentAgent?: string
  isActive: boolean
  className?: string
}

const agentConfigs = {
  'technical': {
    name: 'Technical Analysis Agent',
    icon: TrendingUp,
    color: 'slate',
    description: 'Analyzing market indicators and chart patterns'
  },
  'sentiment': {
    name: 'Sentiment Analysis Agent',
    icon: Eye,
    color: 'gray',
    description: 'Processing social media and news sentiment'
  },
  'flow': {
    name: 'Options Flow Agent',
    icon: Activity,
    color: 'neutral',
    description: 'Tracking unusual options activity'
  },
  'history': {
    name: 'Historical Patterns Agent',
    icon: Clock,
    color: 'stone',
    description: 'Identifying historical price patterns'
  },
  'risk': {
    name: 'Risk Management Agent',
    icon: Shield,
    color: 'zinc',
    description: 'Calculating risk metrics and position sizing'
  },
  'education': {
    name: 'Education Agent',
    icon: BookOpen,
    color: 'slate',
    description: 'Generating educational insights'
  }
}

export function AgentVisualization({ 
  agents, 
  currentAgent, 
  isActive, 
  className 
}: AgentVisualizationProps) {
  const [visibleAgents, setVisibleAgents] = useState<Agent[]>([])
  const [completedCount, setCompletedCount] = useState(0)

  useEffect(() => {
    if (isActive) {
      // Show all agents immediately in single row
      setVisibleAgents(agents)
    } else {
      setVisibleAgents([])
      setCompletedCount(0)
    }
  }, [agents, isActive])

  useEffect(() => {
    const completed = agents.filter(a => a.status === 'completed').length
    setCompletedCount(completed)
  }, [agents])

  if (!isActive && visibleAgents.length === 0) {
    return null
  }

  const getStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'thinking':
        return <Loader2 className="w-4 h-4 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-4 h-4" />
      case 'error':
        return <Activity className="w-4 h-4" />
      default:
        return <Brain className="w-4 h-4 opacity-50" />
    }
  }

  const getColorClasses = (color: string, status: Agent['status']) => {
    const isActive = status === 'thinking'
    const isCompleted = status === 'completed'
    
    const colors = {
      slate: {
        bg: isActive ? 'bg-slate-500/20' : isCompleted ? 'bg-slate-500/10' : 'bg-muted/50',
        border: isActive ? 'border-slate-500/50' : isCompleted ? 'border-slate-500/30' : 'border-border',
        text: isActive ? 'text-slate-400' : isCompleted ? 'text-slate-300' : 'text-muted-foreground',
        accent: 'bg-slate-500'
      },
      gray: {
        bg: isActive ? 'bg-gray-500/20' : isCompleted ? 'bg-gray-500/10' : 'bg-muted/50',
        border: isActive ? 'border-gray-500/50' : isCompleted ? 'border-gray-500/30' : 'border-border',
        text: isActive ? 'text-gray-400' : isCompleted ? 'text-gray-300' : 'text-muted-foreground',
        accent: 'bg-gray-500'
      },
      neutral: {
        bg: isActive ? 'bg-neutral-500/20' : isCompleted ? 'bg-neutral-500/10' : 'bg-muted/50',
        border: isActive ? 'border-neutral-500/50' : isCompleted ? 'border-neutral-500/30' : 'border-border',
        text: isActive ? 'text-neutral-400' : isCompleted ? 'text-neutral-300' : 'text-muted-foreground',
        accent: 'bg-neutral-500'
      },
      stone: {
        bg: isActive ? 'bg-stone-500/20' : isCompleted ? 'bg-stone-500/10' : 'bg-muted/50',
        border: isActive ? 'border-stone-500/50' : isCompleted ? 'border-stone-500/30' : 'border-border',
        text: isActive ? 'text-stone-400' : isCompleted ? 'text-stone-300' : 'text-muted-foreground',
        accent: 'bg-stone-500'
      },
      zinc: {
        bg: isActive ? 'bg-zinc-500/20' : isCompleted ? 'bg-zinc-500/10' : 'bg-muted/50',
        border: isActive ? 'border-zinc-500/50' : isCompleted ? 'border-zinc-500/30' : 'border-border',
        text: isActive ? 'text-zinc-400' : isCompleted ? 'text-zinc-300' : 'text-muted-foreground',
        accent: 'bg-zinc-500'
      }
    }
    
    return colors[color as keyof typeof colors] || colors.slate
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={cn("space-y-4", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <motion.div
            animate={{ rotate: isActive ? 360 : 0 }}
            transition={{ duration: 2, repeat: isActive ? Infinity : 0, ease: "linear" }}
          >
            <Brain className="w-5 h-5 text-primary" />
          </motion.div>
          <h3 className="text-sm font-medium">AI Agent Analysis</h3>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {completedCount}/{agents.length} Complete
          </Badge>
          {isActive && (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <Zap className="w-4 h-4 text-yellow-500" />
            </motion.div>
          )}
        </div>
      </div>

      {/* Agent Cards - Single Row Layout */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        <AnimatePresence>
          {visibleAgents.map((agent, index) => {
            const colorClasses = getColorClasses(agent.color, agent.status)
            const isCurrentAgent = currentAgent === agent.id
            
            return (
              <motion.div
                key={agent.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ 
                  opacity: 1, 
                  scale: isCurrentAgent ? 1.05 : 1,
                }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ 
                  duration: 0.3,
                  scale: { duration: 0.2 }
                }}
                className="relative flex-shrink-0"
              >
                <Card className={cn(
                  "transition-all duration-300 w-20", // Much smaller width
                  colorClasses.bg,
                  colorClasses.border,
                  isCurrentAgent && "ring-2 ring-primary/20"
                )}>
                  <CardContent className="p-1.5"> {/* Smaller padding */}
                    <div className="flex flex-col items-center gap-1"> {/* Smaller gaps */}
                      {/* Agent Icon */}
                      <div className={cn(
                        "flex items-center justify-center w-6 h-6 rounded-full transition-all duration-300", // Smaller icon container
                        agent.status === 'thinking' ? colorClasses.accent : 'bg-muted',
                        agent.status === 'thinking' && "animate-pulse"
                      )}>
                        <agent.icon className={cn(
                          "w-3 h-3 transition-colors duration-300", // Smaller icon
                          agent.status === 'thinking' ? "text-white" : colorClasses.text
                        )} />
                      </div>

                      {/* Agent Info */}
                      <div className="flex flex-col items-center text-center">
                        <h4 className={cn(
                          "text-xs font-medium transition-colors duration-300 truncate w-full", // Smaller text, truncate
                          colorClasses.text
                        )}>
                          {agent.name.split(' ')[0].substring(0, 4)} {/* First 4 chars only */}
                        </h4>
                        
                        {/* Status & Duration - Combined */}
                        <div className="flex items-center gap-0.5">
                          <div className={cn(
                            "transition-colors duration-300",
                            colorClasses.text
                          )}>
                            {getStatusIcon(agent.status)}
                          </div>
                          {agent.duration && agent.status === 'completed' && (
                            <span className="text-xs text-muted-foreground">
                              {agent.duration}s
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Thinking Animation */}
                    {agent.status === 'thinking' && (
                      <motion.div
                        className="mt-1 flex items-center justify-center"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <div className="flex gap-0.5">
                          {[0, 1, 2].map((i) => (
                            <motion.div
                              key={i}
                              className={cn("w-0.5 h-0.5 rounded-full", colorClasses.accent)}
                              animate={{
                                scale: [1, 1.2, 1],
                                opacity: [0.5, 1, 0.5]
                              }}
                              transition={{
                                duration: 1,
                                repeat: Infinity,
                                delay: i * 0.2
                              }}
                            />
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </CardContent>
                </Card>

                {/* Progress Bar */}
                {agent.status === 'thinking' && (
                  <motion.div
                    className={cn(
                      "absolute bottom-0 left-0 h-0.5 rounded-full",
                      colorClasses.accent
                    )}
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  />
                )}
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {/* Progress Summary */}
      {completedCount === agents.length && completedCount > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="p-3 rounded-lg bg-green-500/10 border border-green-500/20"
        >
          <div className="flex items-center gap-2 text-green-400">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm font-medium">
              Analysis Complete! All agents have finished processing.
            </span>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

// Hook to manage agent states during analysis
export function useAgentVisualization() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [currentAgent, setCurrentAgent] = useState<string>()
  const [isActive, setIsActive] = useState(false)

  const initializeAgents = (agentIds: string[]) => {
    const initialAgents = agentIds.map(id => ({
      id,
      ...agentConfigs[id as keyof typeof agentConfigs],
      weight: getAgentWeight(id),
      status: 'pending' as const
    }))
    setAgents(initialAgents)
    setIsActive(true)
  }

  const updateAgentStatus = (agentId: string, status: Agent['status'], duration?: number) => {
    setAgents(prev => prev.map(agent => 
      agent.id === agentId 
        ? { ...agent, status, duration }
        : agent
    ))
    
    if (status === 'thinking') {
      setCurrentAgent(agentId)
    } else if (status === 'completed') {
      setCurrentAgent(undefined)
    }
  }

  const completeAnalysis = () => {
    setIsActive(false)
    setCurrentAgent(undefined)
  }

  const resetVisualization = () => {
    setAgents([])
    setCurrentAgent(undefined)
    setIsActive(false)
  }

  return {
    agents,
    currentAgent,
    isActive,
    initializeAgents,
    updateAgentStatus,
    completeAnalysis,
    resetVisualization
  }
}

// Helper function to get agent weights
function getAgentWeight(agentId: string): number {
  const weights = {
    'technical': 60,
    'sentiment': 10,
    'flow': 10,
    'history': 20,
    'risk': 15,
    'education': 5
  }
  return weights[agentId as keyof typeof weights] || 10
}