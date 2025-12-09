"use client"

import * as React from "react"
import { Send, Bot, User, Sparkles, Microscope, Zap } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { cn } from "@/lib/utils"
import { useAnalysisStore } from "@/store/analysisStore"

interface Message {
    id: string
    role: "agent" | "user"
    content: string
    timestamp: Date
    isVlm?: boolean
}

interface ChatInterfaceProps {
    initialMessages?: Message[]
    className?: string
}

// VLM trigger keywords for detecting VLM requests
const VLM_KEYWORDS = [
    "vlm", "gpt-4v", "gpt4v", "비전", "vision",
    "정밀 분석", "정밀분석", "다시 분석", "재분석",
    "영상 분석", "영상분석", "이미지 분석", "세컨드 오피니언",
    "더 정밀하게", "자세히 분석", "gpt로 분석"
]

function isVlmRequest(message: string): boolean {
    const lower = message.toLowerCase()
    return VLM_KEYWORDS.some(keyword => lower.includes(keyword))
}

// Simple markdown-like formatting for VLM responses
function formatResponse(content: string): React.ReactNode {
    // Split by lines and process
    const lines = content.split('\n')

    return lines.map((line, idx) => {
        // Bold: **text**
        let formatted: React.ReactNode = line

        // Headers with emoji
        if (line.startsWith('**') && line.endsWith('**')) {
            return (
                <div key={idx} className="font-bold text-primary mt-2 mb-1">
                    {line.replace(/\*\*/g, '')}
                </div>
            )
        }

        // Bullet points
        if (line.trim().match(/^[•\-\d+\.]\s/)) {
            return (
                <div key={idx} className="ml-2 text-muted-foreground">
                    {line}
                </div>
            )
        }

        // Score line highlighting
        if (line.includes('UPDRS') && line.includes('점')) {
            return (
                <div key={idx} className="font-semibold text-lg my-2 p-2 bg-primary/10 rounded">
                    {line}
                </div>
            )
        }

        // Match indicator
        if (line.includes('✅')) {
            return <div key={idx} className="text-green-600 mt-2">{line}</div>
        }
        if (line.includes('⚠️')) {
            return <div key={idx} className="text-yellow-600 mt-2">{line}</div>
        }
        if (line.includes('🔴')) {
            return <div key={idx} className="text-red-600 mt-2">{line}</div>
        }

        // Empty line
        if (!line.trim()) {
            return <div key={idx} className="h-2" />
        }

        return <div key={idx}>{line}</div>
    })
}

export function ChatInterface({ initialMessages = [], className }: ChatInterfaceProps) {
    const [messages, setMessages] = React.useState<Message[]>(initialMessages)
    const [inputValue, setInputValue] = React.useState("")
    const [isLoading, setIsLoading] = React.useState(false)
    const [isVlmLoading, setIsVlmLoading] = React.useState(false)
    const scrollRef = React.useRef<HTMLDivElement>(null)

    // Get analysis result from Zustand store
    const analysisResult = useAnalysisStore((state) => state.result)

    React.useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight
        }
    }, [messages])

    const handleSendMessage = async (e?: React.FormEvent) => {
        e?.preventDefault()
        if (!inputValue.trim() || isLoading) return

        const userMessage = inputValue.trim()
        const isVlm = isVlmRequest(userMessage)

        const newMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: userMessage,
            timestamp: new Date(),
        }

        setMessages((prev) => [...prev, newMessage])
        setInputValue("")
        setIsLoading(true)
        if (isVlm) setIsVlmLoading(true)

        try {
            // Use analysis result from Zustand store as context
            const context = analysisResult || null

            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage,
                    context: context
                }),
            })

            if (!response.ok) throw new Error('Network response was not ok')

            const data = await response.json()
            const wasVlmAnalysis = data.vlm_analysis?.performed

            const agentResponse: Message = {
                id: (Date.now() + 1).toString(),
                role: "agent",
                content: data.response || "죄송합니다. 답변을 생성할 수 없습니다.",
                timestamp: new Date(),
                isVlm: wasVlmAnalysis,
            }
            setMessages((prev) => [...prev, agentResponse])
        } catch (error) {
            console.error('Chat error:', error)
            const errorResponse: Message = {
                id: (Date.now() + 1).toString(),
                role: "agent",
                content: "죄송합니다. 서버 연결에 실패했습니다.",
                timestamp: new Date(),
            }
            setMessages((prev) => [...prev, errorResponse])
        } finally {
            setIsLoading(false)
            setIsVlmLoading(false)
        }
    }

    const handleQuickAction = (action: string) => {
        setInputValue(action)
    }

    // Check if we have analysis context
    const hasContext = !!(analysisResult && (analysisResult.metrics || analysisResult.updrs_score))

    return (
        <div className={cn("flex flex-col h-full", className)}>
            {/* Header */}
            <div className="p-4 border-b border-border flex items-center gap-2 bg-card/50">
                <div className="p-2 rounded-full bg-primary/20 text-primary">
                    <Bot className="h-5 w-5" />
                </div>
                <div className="flex-1">
                    <h3 className="font-semibold text-sm">HawkEye 어시스턴트</h3>
                    <p className="text-xs text-muted-foreground">언제든지 도와드립니다</p>
                </div>
                {hasContext && (
                    <div className="flex items-center gap-1 px-2 py-1 bg-green-500/10 text-green-600 rounded-full text-xs">
                        <Sparkles className="h-3 w-3" />
                        <span>환자 데이터 연동</span>
                    </div>
                )}
            </div>

            {/* Context indicator */}
            {hasContext && (
                <div className="px-4 py-2 bg-primary/5 border-b border-border text-xs text-muted-foreground">
                    <span className="font-medium text-primary">분석 데이터:</span>{" "}
                    {analysisResult?.video_type === "finger_tapping" ? "손가락 태핑" :
                     analysisResult?.video_type === "gait" ? "보행 분석" : analysisResult?.video_type}
                    {analysisResult?.updrs_score && (
                        <span> • UPDRS: {analysisResult.updrs_score.score}점 ({analysisResult.updrs_score.severity})</span>
                    )}
                </div>
            )}

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-muted-foreground text-sm py-6">
                        <Bot className="h-10 w-10 mx-auto mb-2 opacity-50" />
                        <p>검사 결과에 대해 무엇이든 물어보세요.</p>
                        {!hasContext && (
                            <p className="text-xs mt-2 text-yellow-600">
                                (영상 분석 후 더 정확한 답변을 받을 수 있습니다)
                            </p>
                        )}

                        {/* Quick Actions */}
                        {hasContext && (
                            <div className="mt-6 space-y-2">
                                <p className="text-xs font-medium text-muted-foreground mb-2">빠른 질문</p>
                                <div className="flex flex-wrap justify-center gap-2">
                                    <button
                                        onClick={() => handleQuickAction("이 결과를 설명해주세요")}
                                        className="px-3 py-1.5 text-xs bg-secondary hover:bg-secondary/80 rounded-full transition-colors"
                                    >
                                        결과 설명
                                    </button>
                                    <button
                                        onClick={() => handleQuickAction("점수가 의미하는 바가 뭔가요?")}
                                        className="px-3 py-1.5 text-xs bg-secondary hover:bg-secondary/80 rounded-full transition-colors"
                                    >
                                        점수 해석
                                    </button>
                                    <button
                                        onClick={() => handleQuickAction("VLM 정밀 분석해줘")}
                                        className="px-3 py-1.5 text-xs bg-violet-500/10 text-violet-600 hover:bg-violet-500/20 rounded-full transition-colors flex items-center gap-1"
                                    >
                                        <Microscope className="h-3 w-3" />
                                        VLM 정밀분석
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* VLM Loading State */}
                {isVlmLoading && (
                    <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-violet-500/20 text-violet-600">
                            <Microscope className="h-4 w-4" />
                        </div>
                        <div className="p-4 rounded-lg bg-violet-500/5 border border-violet-500/20 flex-1">
                            <div className="flex items-center gap-2 text-violet-600 text-sm font-medium mb-2">
                                <div className="h-4 w-4 border-2 border-violet-300 border-t-violet-600 rounded-full animate-spin" />
                                VLM 정밀 분석 중...
                            </div>
                            <div className="text-xs text-muted-foreground space-y-1">
                                <p>GPT-4V가 영상 프레임을 분석하고 있습니다.</p>
                                <p className="text-violet-500">약 10-20초 소요됩니다.</p>
                            </div>
                        </div>
                    </div>
                )}

                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={cn(
                            "flex gap-3 max-w-[95%]",
                            msg.role === "user" ? "ml-auto flex-row-reverse" : ""
                        )}
                    >
                        <div className={cn(
                            "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                            msg.role === "agent"
                                ? msg.isVlm
                                    ? "bg-violet-500/20 text-violet-600"
                                    : "bg-primary/20 text-primary"
                                : "bg-secondary text-secondary-foreground"
                        )}>
                            {msg.role === "agent"
                                ? msg.isVlm
                                    ? <Microscope className="h-4 w-4" />
                                    : <Bot className="h-4 w-4" />
                                : <User className="h-4 w-4" />
                            }
                        </div>
                        <div className={cn(
                            "p-3 rounded-lg text-sm",
                            msg.role === "agent"
                                ? msg.isVlm
                                    ? "bg-violet-500/5 border border-violet-500/20 text-card-foreground"
                                    : "bg-card border border-border text-card-foreground"
                                : "bg-primary text-primary-foreground"
                        )}>
                            {msg.isVlm && (
                                <div className="flex items-center gap-1 text-violet-600 text-xs font-medium mb-2 pb-2 border-b border-violet-500/20">
                                    <Zap className="h-3 w-3" />
                                    VLM 정밀 분석 결과
                                </div>
                            )}
                            <div className={msg.isVlm ? "vlm-response" : ""}>
                                {msg.isVlm ? formatResponse(msg.content) : msg.content}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-border bg-card/50">
                {hasContext && messages.length > 0 && (
                    <div className="flex gap-2 mb-2 overflow-x-auto pb-2">
                        <button
                            onClick={() => handleQuickAction("VLM 정밀 분석해줘")}
                            disabled={isLoading}
                            className="px-2 py-1 text-xs bg-violet-500/10 text-violet-600 hover:bg-violet-500/20 rounded-full transition-colors flex items-center gap-1 whitespace-nowrap disabled:opacity-50"
                        >
                            <Microscope className="h-3 w-3" />
                            VLM 분석
                        </button>
                        <button
                            onClick={() => handleQuickAction("더 자세히 설명해줘")}
                            disabled={isLoading}
                            className="px-2 py-1 text-xs bg-secondary hover:bg-secondary/80 rounded-full transition-colors whitespace-nowrap disabled:opacity-50"
                        >
                            더 자세히
                        </button>
                    </div>
                )}
                <form onSubmit={handleSendMessage} className="flex gap-2">
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder={
                            isVlmLoading
                                ? "VLM 분석 중..."
                                : isLoading
                                ? "답변 생성 중..."
                                : "메시지를 입력하세요..."
                        }
                        disabled={isLoading}
                        className="flex-1 bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
                    />
                    <Button type="submit" size="icon" disabled={!inputValue.trim() || isLoading}>
                        {isLoading ? (
                            <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        ) : (
                            <Send className="h-4 w-4" />
                        )}
                    </Button>
                </form>
            </div>
        </div>
    )
}
