import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { ScrollArea } from "@/components/ui/ScrollArea"
import { Badge } from "@/components/ui/Badge"
import { Terminal, Clock, Activity, FileText, Brain } from "lucide-react"
import { cn } from "@/lib/utils"

interface ReasoningStep {
  agent: string
  step: string
  content: string
  timestamp: string
  meta?: any
}

interface ReasoningLogViewerProps {
  logs: ReasoningStep[]
  className?: string
}

export function ReasoningLogViewer({ logs, className }: ReasoningLogViewerProps) {
  if (!logs || logs.length === 0) return null

  const getAgentIcon = (agent: string) => {
    if (!agent) return <Terminal className="h-4 w-4" />
    switch (agent.toLowerCase()) {
      case 'vision': return <Activity className="h-4 w-4" />
      case 'clinical': return <FileText className="h-4 w-4" />
      case 'report': return <Brain className="h-4 w-4" />
      case 'orchestrator': return <Terminal className="h-4 w-4" />
      default: return <Terminal className="h-4 w-4" />
    }
  }

  const getAgentColor = (agent: string) => {
    if (!agent) return "bg-slate-500/10 text-slate-500 border-slate-500/20"
    switch (agent.toLowerCase()) {
      case 'vision': return "bg-blue-500/10 text-blue-500 border-blue-500/20"
      case 'clinical': return "bg-green-500/10 text-green-500 border-green-500/20"
      case 'report': return "bg-purple-500/10 text-purple-500 border-purple-500/20"
      case 'orchestrator': return "bg-orange-500/10 text-orange-500 border-orange-500/20"
      default: return "bg-slate-500/10 text-slate-500 border-slate-500/20"
    }
  }

  return (
    <Card className={cn("border-primary/20", className)}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Terminal className="h-5 w-5 text-primary" />
          <CardTitle className="text-lg">AI 추론 로그</CardTitle>
        </div>
        <CardDescription>
          멀티 에이전트 시스템의 분석 과정을 투명하게 공개합니다.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] w-full rounded-md border p-4 bg-black/95">
          <div className="space-y-4">
            {logs.map((log, index) => (
              <div key={index} className="flex gap-3 text-sm animate-in slide-in-from-left-2 fade-in duration-300" style={{ animationDelay: `${index * 50}ms` }}>
                <div className="flex flex-col items-center gap-1 mt-0.5">
                  <div className={cn("p-1.5 rounded-full border", getAgentColor(log.agent))}>
                    {getAgentIcon(log.agent)}
                  </div>
                  {index < logs.length - 1 && (
                    <div className="w-px h-full bg-border/50 my-1" />
                  )}
                </div>
                <div className="flex-1 space-y-1 pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={cn("text-xs font-medium px-2 py-0.5 rounded-full border", getAgentColor(log.agent))}>
                        {(log.agent || 'UNKNOWN').toUpperCase()}
                      </span>
                      <span className="font-medium text-slate-200">{log.step}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </div>
                  </div>
                  <p className="text-slate-400 leading-relaxed font-mono text-xs bg-white/5 p-2 rounded-md border border-white/10">
                    {log.content}
                  </p>
                  {log.meta && Object.keys(log.meta).length > 0 && (
                    <div className="mt-2">
                       <details className="text-xs text-muted-foreground cursor-pointer">
                         <summary className="hover:text-primary transition-colors">메타데이터 보기</summary>
                         <pre className="mt-2 p-2 bg-black rounded border border-white/10 overflow-x-auto">
                           {JSON.stringify(log.meta, null, 2)}
                         </pre>
                       </details>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
