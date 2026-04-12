"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Brain, Lightbulb, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react"
import { cn } from "@/lib/utils"

interface AIInterpretationProps {
    summary: string
    explanation: string
    recommendations: string[]
    className?: string
    defaultExpanded?: boolean
}

export function AIInterpretation({
    summary,
    explanation,
    recommendations,
    className,
    defaultExpanded = false
}: AIInterpretationProps) {
    const [isExpanded, setIsExpanded] = React.useState(defaultExpanded)

    // Get first sentence of summary for collapsed preview
    const previewText = summary?.split(/[.!?]/)[0] + '...' || '분석 결과를 확인하세요'

    return (
        <Card className={cn("bg-gradient-to-br from-violet-500/10 to-purple-500/10 border-violet-500/20", className)}>
            <CardHeader
                className="cursor-pointer select-none"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Brain className="h-5 w-5 text-violet-500" />
                        <CardTitle className="text-lg">AI 해석</CardTitle>
                    </div>
                    <div className="flex items-center gap-2">
                        {!isExpanded && (
                            <span className="text-xs text-muted-foreground max-w-[200px] truncate hidden sm:inline">
                                {previewText}
                            </span>
                        )}
                        {isExpanded ? (
                            <ChevronUp className="h-5 w-5 text-muted-foreground" />
                        ) : (
                            <ChevronDown className="h-5 w-5 text-muted-foreground" />
                        )}
                    </div>
                </div>
            </CardHeader>

            {isExpanded && (
                <CardContent className="space-y-6 animate-in slide-in-from-top-2 fade-in duration-200">
                    {/* Summary */}
                    <div className="p-4 rounded-lg bg-card/50 border border-violet-500/20">
                        <div className="flex items-start gap-3">
                            <CheckCircle2 className="h-5 w-5 text-violet-500 mt-0.5 flex-shrink-0" />
                            <div>
                                <p className="font-medium text-sm text-muted-foreground mb-1">요약</p>
                                <p className="text-base leading-relaxed">{summary}</p>
                            </div>
                        </div>
                    </div>

                    {/* Explanation */}
                    <div>
                        <p className="font-medium text-sm text-muted-foreground mb-2">상세 설명</p>
                        <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap">
                            {explanation}
                        </p>
                    </div>

                    {/* Recommendations */}
                    {recommendations && recommendations.length > 0 && (
                        <div>
                            <div className="flex items-center gap-2 mb-3">
                                <Lightbulb className="h-4 w-4 text-yellow-500" />
                                <p className="font-medium text-sm">추천 사항</p>
                            </div>
                            <ul className="space-y-2">
                                {recommendations.map((rec, index) => (
                                    <li key={index} className="flex items-start gap-3">
                                        <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-violet-500/20 text-violet-500 text-xs font-medium flex-shrink-0 mt-0.5">
                                            {index + 1}
                                        </span>
                                        <span className="text-sm text-muted-foreground leading-relaxed">
                                            {rec}
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* AI Badge */}
                    <div className="pt-3 border-t border-violet-500/10">
                        <p className="text-xs text-muted-foreground flex items-center gap-1.5">
                            <Brain className="h-3 w-3" />
                            GPT-4로 생성된 해석입니다. 전문의 상담을 대체하지 않습니다.
                        </p>
                    </div>
                </CardContent>
            )}
        </Card>
    )
}
