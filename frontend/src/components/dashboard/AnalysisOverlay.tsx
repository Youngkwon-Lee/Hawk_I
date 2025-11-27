"use client"

import * as React from "react"
import { useState, useEffect } from 'react'
import { Card, CardContent } from "@/components/ui/Card"
import { Activity, UploadCloud, Eye, Stethoscope, FileText, CheckCircle2, Circle } from 'lucide-react'
import { getAnalysisResult } from "@/lib/services/api"
import { cn } from "@/lib/utils"

interface AnalysisOverlayProps {
    isUploading: boolean
    uploadProgress: number
    videoId: string | null
    onComplete: (result: any) => void
    onError: (error: string) => void
}

export function AnalysisOverlay({
    isUploading,
    uploadProgress,
    videoId,
    onComplete,
    onError
}: AnalysisOverlayProps) {
    const [progress, setProgress] = useState(0)
    const [activeStep, setActiveStep] = useState(0) // 0: Upload, 1: Vision, 2: Clinical, 3: Report
    const [currentReasoning, setCurrentReasoning] = useState("시스템 초기화 중...")

    // Steps configuration - Multi-Agent Architecture
    const steps = [
        { id: 'upload', label: 'Upload', icon: UploadCloud },
        { id: 'vision', label: 'Vision Agent', icon: Eye },
        { id: 'clinical', label: 'Clinical Agent', icon: Stethoscope },
        { id: 'report', label: 'Report Agent', icon: FileText },
    ]

    const updateReasoning = (message: string) => {
        setCurrentReasoning(message)
    }

    // Handle Upload Phase
    useEffect(() => {
        if (isUploading) {
            setActiveStep(0)
            setProgress(uploadProgress * 0.2) // Upload is first 20%

            if (uploadProgress < 30) updateReasoning("영상 데이터 업로드 세션 시작...")
            else if (uploadProgress < 70) updateReasoning("데이터 패킷 전송 및 무결성 확인...")
            else updateReasoning("업로드 완료. 분석 큐 등록 중...")
        }
    }, [isUploading, uploadProgress])

    // Handle Analysis Phase
    useEffect(() => {
        if (!videoId) return

        let pollCount = 0
        const MAX_POLLS = 120

        // Simulated progress for analysis phase
        const progressInterval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 95) return prev
                return prev + (Math.random() * 1.5)
            })
        }, 1000)

        const pollProgress = async () => {
            try {
                const response = await fetch(`http://localhost:5000/api/analysis/progress/${videoId}`)
                if (!response.ok) throw new Error('Failed to fetch progress')

                const data = await response.json()
                pollCount++

                if (data.steps) {
                    // Vision Agent Phase (ROI + Skeleton + Task Classification)
                    if (data.steps.roi_detection?.status === 'in_progress' ||
                        data.steps.skeleton?.status === 'in_progress' ||
                        (data.steps.roi_detection?.status === 'completed' && data.steps.skeleton?.status !== 'completed')) {
                        setActiveStep(1)
                        const msgs = [
                            "Vision Agent: ROI 감지 및 모션 분석 중...",
                            "Vision Agent: MediaPipe 스켈레톤 추출 중...",
                            "Vision Agent: Task 유형 분류 및 신뢰도 계산..."
                        ]
                        updateReasoning(msgs[Math.floor(Math.random() * msgs.length)])
                    }
                    // Clinical Agent Phase (Heatmap + Metrics + UPDRS)
                    else if (data.steps.heatmap?.status === 'in_progress' ||
                             data.steps.updrs_calculation?.status === 'in_progress' ||
                             (data.steps.skeleton?.status === 'completed' && data.steps.updrs_calculation?.status !== 'completed')) {
                        setActiveStep(2)
                        const msgs = [
                            "Clinical Agent: 운동학적 메트릭 계산 중...",
                            "Clinical Agent: UPDRS 점수 추정 중...",
                            "Clinical Agent: 속도, 진폭, 불규칙성 분석..."
                        ]
                        updateReasoning(msgs[Math.floor(Math.random() * msgs.length)])
                    }
                    // Report Agent Phase (AI Interpretation)
                    else if (data.steps.ai_interpretation?.status === 'in_progress' ||
                             (data.steps.updrs_calculation?.status === 'completed' && data.steps.ai_interpretation?.status !== 'completed')) {
                        setActiveStep(3)
                        const msgs = [
                            "Report Agent: GPT-4 기반 임상 해석 생성 중...",
                            "Report Agent: 환자용/의료진용 리포트 작성...",
                            "Report Agent: 권장사항 및 추천 운동 생성..."
                        ]
                        updateReasoning(msgs[Math.floor(Math.random() * msgs.length)])
                    }
                }

                if (data.status === 'completed') {
                    clearInterval(progressInterval)
                    setProgress(100)
                    setActiveStep(4) // All done (past last step)
                    updateReasoning("모든 에이전트 분석 완료! 결과 페이지로 이동합니다.")

                    try {
                        const result = await getAnalysisResult(videoId)
                        onComplete(result)
                        return true
                    } catch (err) {
                        console.error('Failed to fetch result:', err)
                        onError('결과를 가져오는데 실패했습니다.')
                        return true
                    }
                }

                if (pollCount >= MAX_POLLS) return true
                return false
            } catch (error) {
                console.error('Poll error:', error)
                return false
            }
        }

        const pollInterval = setInterval(async () => {
            const isComplete = await pollProgress()
            if (isComplete) clearInterval(pollInterval)
        }, 2000)

        pollProgress()

        return () => {
            clearInterval(pollInterval)
            clearInterval(progressInterval)
        }
    }, [videoId])

    return (
        <div className="fixed inset-0 bg-background/95 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-4 animate-in fade-in duration-300">
            <div className="w-full max-w-4xl mx-auto flex flex-col items-center">

                {/* Header */}
                <div className="text-center space-y-2 mb-16">
                    <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/50">
                        HAWKEYE AI ANALYSIS
                    </h1>
                    <p className="text-muted-foreground">
                        정밀 영상 분석 프로세스가 진행 중입니다
                    </p>
                </div>

                {/* Node Pipeline Visualization */}
                <div className="w-full mb-16 relative">
                    {/* Connecting Line Background */}
                    <div className="absolute top-1/2 left-0 w-full h-1 bg-secondary -translate-y-1/2 z-0" />

                    {/* Active Progress Line */}
                    <div
                        className="absolute top-1/2 left-0 h-1 bg-primary -translate-y-1/2 z-0 transition-all duration-1000 ease-out"
                        style={{ width: `${Math.min((activeStep / (steps.length - 1)) * 100, 100)}%` }}
                    />

                    <div className="relative z-10 flex justify-between w-full px-4">
                        {steps.map((step, index) => {
                            const isActive = index === activeStep
                            const isCompleted = index < activeStep
                            const Icon = step.icon

                            return (
                                <div key={step.id} className="flex flex-col items-center gap-3">
                                    <div
                                        className={cn(
                                            "w-14 h-14 rounded-full flex items-center justify-center border-4 transition-all duration-500 bg-background",
                                            isActive ? "border-primary text-primary scale-110 shadow-[0_0_20px_rgba(var(--primary),0.4)]" :
                                                isCompleted ? "border-primary bg-primary text-primary-foreground" :
                                                    "border-secondary text-muted-foreground"
                                        )}
                                    >
                                        {isCompleted ? (
                                            <CheckCircle2 className="w-6 h-6" />
                                        ) : (
                                            <Icon className={cn("w-6 h-6", isActive && "animate-pulse")} />
                                        )}
                                    </div>
                                    <span className={cn(
                                        "text-sm font-medium transition-colors duration-300",
                                        isActive ? "text-primary" :
                                            isCompleted ? "text-foreground" :
                                                "text-muted-foreground"
                                    )}>
                                        {step.label}
                                    </span>
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Reasoning / Status Display */}
                <Card className="w-full max-w-xl border-primary/20 bg-card/50 backdrop-blur shadow-lg">
                    <CardContent className="p-6 flex items-center gap-4">
                        <div className="relative flex-shrink-0">
                            <div className="w-3 h-3 bg-primary rounded-full animate-ping absolute inset-0" />
                            <div className="w-3 h-3 bg-primary rounded-full relative" />
                        </div>
                        <div className="space-y-1">
                            <p className="text-xs font-mono text-primary/70 uppercase tracking-wider">
                                Current Process
                            </p>
                            <p className="text-lg font-medium animate-in fade-in slide-in-from-bottom-1 duration-300 key={currentReasoning}">
                                {currentReasoning}
                            </p>
                        </div>
                    </CardContent>
                </Card>

                {/* Overall Progress Bar */}
                <div className="w-full max-w-xl mt-8 space-y-2">
                    <div className="flex justify-between text-xs font-mono text-muted-foreground">
                        <span>TOTAL PROGRESS</span>
                        <span>{Math.round(progress)}%</span>
                    </div>
                    <div className="h-1 bg-secondary rounded-full overflow-hidden">
                        <div
                            className="h-full bg-primary transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>

            </div>
        </div>
    )
}
