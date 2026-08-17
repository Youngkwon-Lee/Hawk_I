"use client"

import * as React from "react"
import { BookOpen, ChevronLeft, Headphones, MessageCircle, Video } from "lucide-react"
import { ChatInterface } from "@/components/ui/ChatInterface"

export function EvidencePanel() {
    const [open, setOpen] = React.useState(false)

    return (
        <div className="relative flex h-full items-center justify-center">
            <button onClick={() => setOpen(true)} className="flex h-32 w-full flex-col items-center justify-center gap-2 rounded-l-xl border border-border bg-card/70 text-sm text-foreground shadow-sm transition-colors hover:bg-accent" aria-label="도움말 열기">
                <ChevronLeft className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                <span className="[writing-mode:vertical-rl] tracking-[0.18em]">도움말 열기</span>
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground"><Headphones className="h-4 w-4" aria-hidden="true" /></span>
            </button>

            {open && (
                <div className="fixed right-4 top-20 z-50 flex h-[min(42rem,calc(100vh-6rem))] w-80 flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl">
                    <div className="flex items-center gap-3 border-b border-border px-4 py-4">
                        <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-primary"><MessageCircle className="h-4 w-4" aria-hidden="true" /></span>
                        <div className="min-w-0 flex-1"><p className="text-sm font-semibold">HawkEye 어시스턴트</p><p className="mt-1 text-xs text-muted-foreground">검사와 업로드를 도와드립니다.</p></div>
                        <button onClick={() => setOpen(false)} className="text-xs text-muted-foreground hover:text-foreground">닫기</button>
                    </div>
                    <div className="divide-y divide-border">
                        <button className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm hover:bg-accent"><BookOpen className="h-4 w-4 text-primary" aria-hidden="true" /><span className="flex-1">분석 가이드 보기</span><ChevronLeft className="h-4 w-4 rotate-180 text-muted-foreground" aria-hidden="true" /></button>
                        <button className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm hover:bg-accent"><Video className="h-4 w-4 text-primary" aria-hidden="true" /><span className="flex-1">영상 촬영 팁</span><ChevronLeft className="h-4 w-4 rotate-180 text-muted-foreground" aria-hidden="true" /></button>
                    </div>
                    <div className="min-h-0 flex-1 border-t border-border"><ChatInterface initialMessages={[{ id: "1", role: "agent", content: "검사를 시작할 준비가 되면 알려주세요.", timestamp: new Date() }]} /></div>
                </div>
            )}
        </div>
    )
}
