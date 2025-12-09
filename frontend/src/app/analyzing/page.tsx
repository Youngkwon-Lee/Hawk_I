"use client"

import { getAnalysisResult } from "@/lib/services/api"
import { useAnalysisStore } from "@/store/analysisStore"

import * as React from "react"
import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Activity, CheckCircle2, Loader2 } from 'lucide-react'

type AnalysisStep = {
  id: string
  label: string
  status: 'pending' | 'in_progress' | 'completed' | 'error'
  resultUrl?: string
}

export default function AnalyzingPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const videoId = searchParams.get('videoId')

  // Zustand store actions
  const { setResult, setAnalyzing, setError, clearResult } = useAnalysisStore()

  // Simple progress state
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState("시스템 초기화 중...")
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    setLogs(prev => {
      // Keep last 5 logs
      const newLogs = [...prev, message]
      if (newLogs.length > 5) return newLogs.slice(newLogs.length - 5)
      return newLogs
    })
  }

  useEffect(() => {
    if (!videoId) return

    // Clear previous result when starting new analysis
    clearResult()

    let pollCount = 0
    const MAX_POLLS = 120 // 4 minutes max

    // Simulated progress for better UX while polling
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev // Cap at 90% until complete
        return prev + 1
      })
    }, 500)

    const pollProgress = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/analysis/progress/${videoId}`)
        if (!response.ok) throw new Error('Failed to fetch progress')

        const data = await response.json()
        pollCount++

        // Update status message based on backend status
        if (data.steps) {
          if (data.steps.roi_detection?.status === 'in_progress') {
            setStatusMessage("관심 영역(ROI) 스캔 중...")
            if (Math.random() > 0.7) addLog("> ROI 감지 알고리즘 실행...")
          }
          else if (data.steps.skeleton?.status === 'in_progress') {
            setStatusMessage("스켈레톤 구조 추출 중...")
            if (Math.random() > 0.7) addLog("> MediaPipe 랜드마크 추출 중...")
          }
          else if (data.steps.heatmap?.status === 'in_progress') {
            setStatusMessage("운동 히트맵 생성 중...")
            if (Math.random() > 0.7) addLog("> 움직임 밀도 분석 중...")
          }
          else if (data.steps.ai_interpretation?.status === 'in_progress') {
            setStatusMessage("AI 종합 분석 수행 중...")
            if (Math.random() > 0.7) addLog("> LLM 추론 엔진 가동...")
          }
        }

        if (data.status === 'completed') {
          clearInterval(progressInterval)
          setProgress(100)
          setStatusMessage("분석 완료. 리포트 생성 중...")
          addLog("> 분석 데이터 저장 완료.")
          addLog("> 결과 페이지로 이동합니다.")

          try {
            const result = await getAnalysisResult(videoId)
            // Use Zustand store instead of sessionStorage
            setResult(result)
            setTimeout(() => {
              router.push(`/result?analysisId=${videoId}`)
            }, 1000)
            return true
          } catch (err) {
            console.error('Failed to fetch result:', err)
            setError('분석 결과를 가져오는데 실패했습니다.')
            setTimeout(() => {
              router.push(`/result?analysisId=${videoId}&error=fetch_failed`)
            }, 1000)
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
  }, [videoId, router])

  return (
    <PageLayout agentPanel={<ChatInterface initialMessages={[{
      id: "1",
      role: "agent",
      content: "영상을 정밀 분석하고 있습니다. 잠시만 기다려주세요.",
      timestamp: new Date()
    }]} />}>
      <div className="flex flex-col items-center justify-center min-h-[70vh] w-full max-w-3xl mx-auto px-4">

        {/* Main Visual */}
        <div className="relative mb-12">
          {/* Outer Ring */}
          <div className="absolute inset-0 rounded-full border-2 border-primary/20 animate-[spin_3s_linear_infinite]" />
          <div className="absolute inset-[-10px] rounded-full border border-primary/10 animate-[spin_5s_linear_infinite_reverse]" />

          {/* Inner Pulse */}
          <div className="relative w-32 h-32 rounded-full bg-primary/5 flex items-center justify-center backdrop-blur-sm border border-primary/20">
            <Activity className="h-12 w-12 text-primary animate-pulse" />

            {/* Scanning Line */}
            <div className="absolute inset-0 w-full h-1 bg-primary/50 blur-sm animate-[scan_2s_ease-in-out_infinite]" />
          </div>
        </div>

        {/* Status Text */}
        <div className="text-center space-y-2 mb-8">
          <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/50">
            AI 영상 분석 중
          </h1>
          <p className="text-lg text-muted-foreground font-medium animate-pulse">
            {statusMessage}
          </p>
        </div>

        {/* Terminal / Log View */}
        <Card className="w-full bg-black/95 border-primary/20 shadow-2xl overflow-hidden mb-8">
          <CardContent className="p-6 font-mono text-sm">
            <div className="flex items-center gap-2 border-b border-primary/20 pb-2 mb-4">
              <div className="w-3 h-3 rounded-full bg-red-500/50" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
              <div className="w-3 h-3 rounded-full bg-green-500/50" />
              <span className="ml-2 text-xs text-primary/50">HAWKEYE_CORE_SYSTEM</span>
            </div>
            <div className="space-y-2 h-32 flex flex-col justify-end">
              {logs.map((log, i) => (
                <div key={i} className="text-green-500/80 animate-in slide-in-from-left-2 fade-in duration-300">
                  {log}
                </div>
              ))}
              <div className="flex items-center gap-2 text-primary">
                <span className="animate-pulse">_</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Progress Bar */}
        <div className="w-full space-y-2">
          <div className="flex justify-between text-xs font-mono text-primary/70">
            <span>PROGRESS</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-1 bg-primary/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary shadow-[0_0_10px_rgba(var(--primary),0.5)] transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <style jsx global>{`
            @keyframes scan {
                0% { transform: translateY(-60px); opacity: 0; }
                50% { opacity: 1; }
                100% { transform: translateY(60px); opacity: 0; }
            }
        `}</style>
      </div>
    </PageLayout>
  )
}
