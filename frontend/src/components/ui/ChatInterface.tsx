"use client"

import * as React from "react"
import { Send, Bot, User } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { cn } from "@/lib/utils"

interface Message {
    id: string
    role: "agent" | "user"
    content: string
    timestamp: Date
}

interface ChatInterfaceProps {
    initialMessages?: Message[]
    className?: string
}

export function ChatInterface({ initialMessages = [], className }: ChatInterfaceProps) {
    const [messages, setMessages] = React.useState<Message[]>(initialMessages)
    const [inputValue, setInputValue] = React.useState("")
    const scrollRef = React.useRef<HTMLDivElement>(null)

    React.useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight
        }
    }, [messages])

    const handleSendMessage = async (e?: React.FormEvent) => {
        e?.preventDefault()
        if (!inputValue.trim()) return

        const newMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: inputValue,
            timestamp: new Date(),
        }

        setMessages((prev) => [...prev, newMessage])
        setInputValue("")

        try {
            // Get analysis result from sessionStorage for context
            const storedResult = sessionStorage.getItem('analysisResult')
            const context = storedResult ? JSON.parse(storedResult) : null

            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: inputValue,
                    context: context
                }),
            })

            if (!response.ok) throw new Error('Network response was not ok')

            const data = await response.json()

            const agentResponse: Message = {
                id: (Date.now() + 1).toString(),
                role: "agent",
                content: data.response || "죄송합니다. 답변을 생성할 수 없습니다.",
                timestamp: new Date(),
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
        }
    }

    return (
        <div className={cn("flex flex-col h-full", className)}>
            {/* Header */}
            <div className="p-4 border-b border-border flex items-center gap-2 bg-card/50">
                <div className="p-2 rounded-full bg-primary/20 text-primary">
                    <Bot className="h-5 w-5" />
                </div>
                <div>
                    <h3 className="font-semibold text-sm">HawkEye 어시스턴트</h3>
                    <p className="text-xs text-muted-foreground">언제든지 도와드립니다</p>
                </div>
            </div>

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-muted-foreground text-sm py-10">
                        <Bot className="h-10 w-10 mx-auto mb-2 opacity-50" />
                        <p>검사 결과에 대해 무엇이든 물어보세요.</p>
                    </div>
                )}
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={cn(
                            "flex gap-3 max-w-[90%]",
                            msg.role === "user" ? "ml-auto flex-row-reverse" : ""
                        )}
                    >
                        <div className={cn(
                            "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                            msg.role === "agent" ? "bg-primary/20 text-primary" : "bg-secondary text-secondary-foreground"
                        )}>
                            {msg.role === "agent" ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
                        </div>
                        <div className={cn(
                            "p-3 rounded-lg text-sm",
                            msg.role === "agent"
                                ? "bg-card border border-border text-card-foreground"
                                : "bg-primary text-primary-foreground"
                        )}>
                            {msg.content}
                        </div>
                    </div>
                ))}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-border bg-card/50">
                <form onSubmit={handleSendMessage} className="flex gap-2">
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder="메시지를 입력하세요..."
                        className="flex-1 bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                    <Button type="submit" size="icon" disabled={!inputValue.trim()}>
                        <Send className="h-4 w-4" />
                    </Button>
                </form>
            </div>
        </div>
    )
}
