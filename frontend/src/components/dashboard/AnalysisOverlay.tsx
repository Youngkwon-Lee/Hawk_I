"use client"

import * as React from "react"
import { useState, useEffect, useRef } from 'react'
import { Card, CardContent } from "@/components/ui/Card"
import { Activity, UploadCloud, Eye, Stethoscope, FileText, CheckCircle2, Sparkles, Zap } from 'lucide-react'
import { getAnalysisResult } from "@/lib/services/api"
import { cn } from "@/lib/utils"

interface AnalysisOverlayProps {
    isUploading: boolean
    uploadProgress: number
    videoId: string | null
    onComplete: (result: any) => void
    onError: (error: string) => void
}

interface LogEntry {
    timestamp: string
    message: string
    type: 'info' | 'success' | 'processing'
}

export function AnalysisOverlay({
    isUploading,
    uploadProgress,
    videoId,
    onComplete,
    onError
}: AnalysisOverlayProps) {
    const [progress, setProgress] = useState(0)
    const [activeStep, setActiveStep] = useState(0)
    const [currentReasoning, setCurrentReasoning] = useState("시스템 초기화 중...")
    const [logs, setLogs] = useState<LogEntry[]>([])
    const [startTime] = useState(Date.now())
    const [estimatedTime, setEstimatedTime] = useState<string | null>(null)
    const [frameInfo, setFrameInfo] = useState<string | null>(null)
    const logsEndRef = useRef<HTMLDivElement>(null)

    // Steps configuration with descriptions
    const steps = [
        {
            id: 'upload',
            label: 'Upload',
            icon: UploadCloud,
            description: '영상 데이터 전송'
        },
        {
            id: 'vision',
            label: 'Vision Agent',
            icon: Eye,
            description: '신체 움직임 추출'
        },
        {
            id: 'clinical',
            label: 'Clinical Agent',
            icon: Stethoscope,
            description: '의학적 지표 계산'
        },
        {
            id: 'report',
            label: 'Report Agent',
            icon: FileText,
            description: 'AI 해석 리포트'
        },
    ]

    const addLog = (message: string, type: LogEntry['type'] = 'info') => {
        const now = new Date()
        const timestamp = now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        setLogs(prev => [...prev.slice(-7), { timestamp, message, type }])
    }

    const updateReasoning = (message: string) => {
        setCurrentReasoning(message)
    }

    // Calculate estimated time
    useEffect(() => {
        if (progress > 5 && progress < 95) {
            const elapsed = (Date.now() - startTime) / 1000
            const estimated = (elapsed / progress) * (100 - progress)
            if (estimated < 60) {
                setEstimatedTime(`~${Math.round(estimated)}초 남음`)
            } else {
                setEstimatedTime(`~${Math.round(estimated / 60)}분 남음`)
            }
        } else if (progress >= 95) {
            setEstimatedTime("거의 완료...")
        }
    }, [progress, startTime])

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [logs])

    // Handle Upload Phase
    useEffect(() => {
        if (isUploading) {
            setActiveStep(0)
            setProgress(uploadProgress * 0.2)

            if (uploadProgress < 30) {
                updateReasoning("영상 데이터 업로드 세션 시작...")
                addLog("업로드 세션 초기화", 'info')
            } else if (uploadProgress < 70) {
                updateReasoning("데이터 패킷 전송 및 무결성 확인...")
                if (uploadProgress === 50) addLog(`데이터 전송 중... ${Math.round(uploadProgress)}%`, 'processing')
            } else {
                updateReasoning("업로드 완료. 분석 큐 등록 중...")
                addLog("업로드 완료, 분석 대기열 등록", 'success')
            }
        }
    }, [isUploading, uploadProgress])

    // Handle Analysis Phase
    useEffect(() => {
        if (!videoId) return

        let pollCount = 0
        const MAX_POLLS = 120

        addLog(`분석 시작: ${videoId.slice(0, 8)}...`, 'info')

        const progressInterval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 95) return prev
                return prev + (Math.random() * 1.2)
            })
        }, 1000)

        const pollProgress = async () => {
            try {
                const response = await fetch(`http://localhost:5000/api/analysis/progress/${videoId}`)
                if (!response.ok) throw new Error('Failed to fetch progress')

                const data = await response.json()
                pollCount++

                if (data.steps) {
                    // Vision Agent Phase
                    if (data.steps.roi_detection?.status === 'in_progress' ||
                        data.steps.skeleton?.status === 'in_progress' ||
                        (data.steps.roi_detection?.status === 'completed' && data.steps.skeleton?.status !== 'completed')) {
                        setActiveStep(1)

                        if (data.steps.roi_detection?.status === 'in_progress') {
                            updateReasoning("Vision Agent: ROI 감지 및 모션 영역 분석...")
                            if (pollCount % 3 === 0) addLog("관심 영역(ROI) 스캔 중...", 'processing')
                        } else {
                            updateReasoning("Vision Agent: MediaPipe 스켈레톤 추출 중...")
                            setFrameInfo(`프레임 처리 중... (${Math.round(progress * 3)}개 완료)`)
                            if (pollCount % 2 === 0) addLog(`랜드마크 추출: ${Math.round(progress * 2)}개 포인트`, 'processing')
                        }
                    }
                    // Clinical Agent Phase
                    else if (data.steps.heatmap?.status === 'in_progress' ||
                             data.steps.updrs_calculation?.status === 'in_progress' ||
                             (data.steps.skeleton?.status === 'completed' && data.steps.updrs_calculation?.status !== 'completed')) {
                        setActiveStep(2)
                        setFrameInfo(null)

                        if (data.steps.heatmap?.status === 'in_progress') {
                            updateReasoning("Clinical Agent: 히트맵 시각화 생성...")
                            addLog("모션 히트맵 렌더링 중...", 'processing')
                        } else {
                            updateReasoning("Clinical Agent: UPDRS 점수 산출 중...")
                            if (pollCount % 2 === 0) addLog("운동학적 메트릭 계산 중...", 'processing')
                        }
                    }
                    // Report Agent Phase
                    else if (data.steps.ai_interpretation?.status === 'in_progress' ||
                             (data.steps.updrs_calculation?.status === 'completed' && data.steps.ai_interpretation?.status !== 'completed')) {
                        setActiveStep(3)

                        const msgs = [
                            "Report Agent: GPT-4 기반 임상 해석 생성...",
                            "Report Agent: 권장사항 및 운동 처방 작성...",
                            "Report Agent: 최종 리포트 컴파일..."
                        ]
                        updateReasoning(msgs[pollCount % msgs.length])
                        if (pollCount % 3 === 0) addLog("AI 해석 모델 추론 중...", 'processing')
                    }
                }

                if (data.status === 'completed') {
                    clearInterval(progressInterval)
                    setProgress(100)
                    setActiveStep(4)
                    updateReasoning("모든 에이전트 분석 완료!")
                    addLog("분석 완료 - 결과 페이지로 이동", 'success')

                    try {
                        const result = await getAnalysisResult(videoId)
                        setTimeout(() => onComplete(result), 800)
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
        <div className="fixed inset-0 bg-slate-950/98 backdrop-blur-md z-50 flex flex-col items-center justify-center p-4 animate-in fade-in duration-500">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl animate-pulse" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
                <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%239C92AC%22%20fill-opacity%3D%220.03%22%3E%3Cpath%20d%3D%22M36%2034v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6%2034v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6%204V0H4v4H0v2h4v4h2V6h4V4H6z%22%2F%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E')] opacity-50" />
            </div>

            <div className="relative w-full max-w-5xl mx-auto flex flex-col items-center">

                {/* Header */}
                <div className="text-center space-y-3 mb-12">
                    <div className="flex items-center justify-center gap-3 mb-2">
                        <Sparkles className="w-6 h-6 text-violet-400 animate-pulse" />
                        <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-blue-400 to-cyan-400">
                            HAWKEYE AI ANALYSIS
                        </h1>
                        <Sparkles className="w-6 h-6 text-cyan-400 animate-pulse" style={{ animationDelay: '0.5s' }} />
                    </div>
                    <p className="text-slate-400 text-lg">
                        Multi-Agent 파이프라인이 영상을 정밀 분석하고 있습니다
                    </p>
                </div>

                {/* Pipeline Stepper */}
                <div className="w-full mb-12 relative px-8">
                    {/* Background Line */}
                    <div className="absolute top-8 left-12 right-12 h-1 bg-slate-800 rounded-full" />

                    {/* Animated Progress Line */}
                    <div
                        className="absolute top-8 left-12 h-1 rounded-full transition-all duration-1000 ease-out overflow-hidden"
                        style={{ width: `calc(${Math.min((activeStep / (steps.length - 1)) * 100, 100)}% - 48px)` }}
                    >
                        <div className="absolute inset-0 bg-gradient-to-r from-violet-500 via-blue-500 to-cyan-500" />
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
                    </div>

                    <div className="relative z-10 flex justify-between">
                        {steps.map((step, index) => {
                            const isActive = index === activeStep
                            const isCompleted = index < activeStep
                            const Icon = step.icon

                            return (
                                <div key={step.id} className="flex flex-col items-center gap-3 group">
                                    {/* Node */}
                                    <div className="relative">
                                        {/* Glow Effect */}
                                        {isActive && (
                                            <div className="absolute inset-0 w-16 h-16 -m-1 bg-violet-500/30 rounded-full blur-xl animate-pulse" />
                                        )}
                                        <div
                                            className={cn(
                                                "relative w-16 h-16 rounded-full flex items-center justify-center border-2 transition-all duration-500",
                                                isActive ? "border-violet-400 bg-slate-900 scale-110 shadow-[0_0_30px_rgba(139,92,246,0.5)]" :
                                                    isCompleted ? "border-emerald-400 bg-emerald-500/20" :
                                                        "border-slate-700 bg-slate-900/50"
                                            )}
                                        >
                                            {isCompleted ? (
                                                <CheckCircle2 className="w-7 h-7 text-emerald-400" />
                                            ) : (
                                                <Icon className={cn(
                                                    "w-7 h-7 transition-all",
                                                    isActive ? "text-violet-400 animate-pulse" : "text-slate-500"
                                                )} />
                                            )}

                                            {/* Active Indicator Ring */}
                                            {isActive && (
                                                <div className="absolute inset-0 rounded-full border-2 border-violet-400 animate-ping opacity-50" />
                                            )}
                                        </div>
                                    </div>

                                    {/* Label */}
                                    <div className="text-center">
                                        <span className={cn(
                                            "text-sm font-semibold transition-colors duration-300",
                                            isActive ? "text-violet-400" :
                                                isCompleted ? "text-emerald-400" :
                                                    "text-slate-500"
                                        )}>
                                            {step.label}
                                        </span>
                                        <p className={cn(
                                            "text-xs mt-0.5 transition-colors",
                                            isActive ? "text-slate-300" : "text-slate-600"
                                        )}>
                                            {step.description}
                                        </p>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Main Content Grid */}
                <div className="w-full grid md:grid-cols-2 gap-6">
                    {/* Current Process Card */}
                    <Card className="border-slate-800 bg-slate-900/50 backdrop-blur shadow-2xl overflow-hidden">
                        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-violet-500 via-blue-500 to-cyan-500" />
                        <CardContent className="p-6">
                            <div className="flex items-start gap-4">
                                <div className="relative flex-shrink-0 mt-1">
                                    <div className="w-3 h-3 bg-violet-500 rounded-full" />
                                    <div className="absolute inset-0 w-3 h-3 bg-violet-500 rounded-full animate-ping" />
                                </div>
                                <div className="space-y-2 flex-1">
                                    <p className="text-xs font-mono text-violet-400/80 uppercase tracking-widest flex items-center gap-2">
                                        <Zap className="w-3 h-3" />
                                        Current Process
                                    </p>
                                    <p className="text-xl font-medium text-white leading-relaxed">
                                        {currentReasoning}
                                    </p>
                                    {frameInfo && (
                                        <p className="text-sm text-slate-400 font-mono bg-slate-800/50 px-3 py-1.5 rounded-lg inline-block">
                                            {frameInfo}
                                        </p>
                                    )}
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Live Logs Card */}
                    <Card className="border-slate-800 bg-slate-900/50 backdrop-blur shadow-2xl">
                        <CardContent className="p-4">
                            <p className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                <Activity className="w-3 h-3" />
                                Live Log
                            </p>
                            <div className="h-32 overflow-y-auto space-y-1.5 font-mono text-xs scrollbar-thin scrollbar-thumb-slate-700">
                                {logs.length === 0 ? (
                                    <p className="text-slate-600">대기 중...</p>
                                ) : (
                                    logs.map((log, i) => (
                                        <div
                                            key={i}
                                            className={cn(
                                                "flex gap-2 animate-in fade-in slide-in-from-left-2 duration-300",
                                                log.type === 'success' ? "text-emerald-400" :
                                                    log.type === 'processing' ? "text-blue-400" : "text-slate-400"
                                            )}
                                        >
                                            <span className="text-slate-600 flex-shrink-0">[{log.timestamp}]</span>
                                            <span>{log.message}</span>
                                        </div>
                                    ))
                                )}
                                <div ref={logsEndRef} />
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Progress Bar */}
                <div className="w-full mt-8 space-y-3">
                    <div className="flex justify-between items-center text-sm">
                        <span className="font-mono text-slate-500 uppercase tracking-wider">Total Progress</span>
                        <div className="flex items-center gap-4">
                            {estimatedTime && (
                                <span className="text-slate-400 text-sm">{estimatedTime}</span>
                            )}
                            <span className="font-mono text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-cyan-400">
                                {Math.round(progress)}%
                            </span>
                        </div>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                            className="h-full rounded-full transition-all duration-500 ease-out relative overflow-hidden"
                            style={{ width: `${progress}%` }}
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-violet-500 via-blue-500 to-cyan-500" />
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
                        </div>
                    </div>
                </div>

            </div>

            {/* CSS for shimmer animation */}
            <style jsx global>{`
                @keyframes shimmer {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }
                .animate-shimmer {
                    animation: shimmer 2s infinite;
                }
            `}</style>
        </div>
    )
}
