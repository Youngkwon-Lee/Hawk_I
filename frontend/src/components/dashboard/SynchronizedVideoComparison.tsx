"use client"

import * as React from "react"
import { Expand, Pause, Play, RotateCcw } from "lucide-react"

import { Button } from "@/components/ui/Button"
import { cn } from "@/lib/utils"

interface SynchronizedVideoComparisonProps {
    originalSrc: string
    analysisSrc: string
    className?: string
    autoPlay?: boolean
    loop?: boolean
    originalLabel?: string
    analysisLabel?: string
}

function formatTime(value: number): string {
    if (!Number.isFinite(value)) return "00:00.0"
    const minutes = Math.floor(value / 60)
    const seconds = value - minutes * 60
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(1).padStart(4, "0")}`
}

export function SynchronizedVideoComparison({
    originalSrc,
    analysisSrc,
    className,
    autoPlay = false,
    loop = false,
    originalLabel = "원본 영상",
    analysisLabel = "관절 추적 결과",
}: SynchronizedVideoComparisonProps) {
    const containerRef = React.useRef<HTMLDivElement>(null)
    const originalRef = React.useRef<HTMLVideoElement>(null)
    const analysisRef = React.useRef<HTMLVideoElement>(null)
    const originalReadyRef = React.useRef(false)
    const analysisReadyRef = React.useRef(false)
    const [isPlaying, setIsPlaying] = React.useState(false)
    const [currentTime, setCurrentTime] = React.useState(0)
    const [duration, setDuration] = React.useState(0)
    const [readyCount, setReadyCount] = React.useState(0)
    const [error, setError] = React.useState<string | null>(null)

    const syncAnalysis = React.useCallback((force = false) => {
        const original = originalRef.current
        const analysis = analysisRef.current
        if (!original || !analysis) return
        if (force || Math.abs(analysis.currentTime - original.currentTime) > 0.08) {
            analysis.currentTime = original.currentTime
        }
    }, [])

    const playBoth = React.useCallback(async () => {
        const original = originalRef.current
        const analysis = analysisRef.current
        if (!original || !analysis) return

        setError(null)
        syncAnalysis(true)
        try {
            await Promise.all([original.play(), analysis.play()])
            setIsPlaying(true)
        } catch {
            setIsPlaying(false)
            setError("재생을 시작하려면 화면을 한 번 눌러 주세요.")
        }
    }, [syncAnalysis])

    const pauseBoth = React.useCallback(() => {
        originalRef.current?.pause()
        analysisRef.current?.pause()
        setIsPlaying(false)
    }, [])

    React.useEffect(() => {
        if (autoPlay && readyCount >= 2) void playBoth()
    }, [autoPlay, playBoth, readyCount])

    React.useEffect(() => {
        if (!isPlaying) return
        let frameId = 0
        const tick = () => {
            syncAnalysis()
            frameId = requestAnimationFrame(tick)
        }
        frameId = requestAnimationFrame(tick)
        return () => cancelAnimationFrame(frameId)
    }, [isPlaying, syncAnalysis])

    const handleSeek = (value: number) => {
        const nextTime = Math.min(Math.max(value, 0), duration || 0)
        if (originalRef.current) originalRef.current.currentTime = nextTime
        if (analysisRef.current) analysisRef.current.currentTime = nextTime
        setCurrentTime(nextTime)
    }

    const handleEnded = () => {
        if (loop) {
            handleSeek(0)
            void playBoth()
            return
        }
        setIsPlaying(false)
    }

    const handleFullscreen = async () => {
        if (!containerRef.current?.requestFullscreen) return
        await containerRef.current.requestFullscreen()
    }

    const markReady = (source: "original" | "analysis") => {
        const target = source === "original" ? originalReadyRef : analysisReadyRef
        if (target.current) return
        target.current = true
        setReadyCount(count => count + 1)
    }

    return (
        <section
            ref={containerRef}
            className={cn("overflow-hidden rounded-2xl border border-border bg-card shadow-sm", className)}
            aria-label="원본 영상과 관절 추적 결과 동기 비교"
        >
            <div className="grid gap-px bg-border lg:grid-cols-2">
                <VideoPanel label={originalLabel} badge="공개 촬영 데모">
                    <video
                        ref={originalRef}
                        src={originalSrc}
                        className="h-[clamp(22rem,43vw,35rem)] w-full bg-[#081017] object-contain"
                        muted
                        playsInline
                        preload="auto"
                        onCanPlay={() => markReady("original")}
                        onLoadedMetadata={event => setDuration(event.currentTarget.duration)}
                        onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime)}
                        onEnded={handleEnded}
                        onError={() => setError("원본 영상을 불러오지 못했습니다.")}
                        aria-label={originalLabel}
                    />
                </VideoPanel>

                <VideoPanel label={analysisLabel} badge="실제 포즈 추정">
                    <video
                        ref={analysisRef}
                        src={analysisSrc}
                        className="h-[clamp(22rem,43vw,35rem)] w-full bg-[#081017] object-contain"
                        muted
                        playsInline
                        preload="auto"
                        onCanPlay={() => markReady("analysis")}
                        onError={() => setError("분석 영상을 불러오지 못했습니다.")}
                        aria-label={analysisLabel}
                    />
                </VideoPanel>
            </div>

            <div className="space-y-3 border-t border-border bg-card px-4 py-3 sm:px-5">
                <div className="flex items-center gap-3">
                    <Button
                        type="button"
                        size="icon"
                        className="h-10 w-10 shrink-0 rounded-full"
                        onClick={isPlaying ? pauseBoth : () => void playBoth()}
                        aria-label={isPlaying ? "두 영상 일시정지" : "두 영상 재생"}
                    >
                        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
                    </Button>
                    <span className="w-24 shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                        {formatTime(currentTime)} / {formatTime(duration)}
                    </span>
                    <input
                        type="range"
                        min={0}
                        max={duration || 0}
                        step={0.01}
                        value={Math.min(currentTime, duration || 0)}
                        onChange={event => handleSeek(Number(event.target.value))}
                        className="h-2 min-w-0 flex-1 cursor-pointer accent-primary"
                        aria-label="영상 재생 위치"
                    />
                    <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-9 w-9 shrink-0"
                        onClick={() => handleSeek(0)}
                        aria-label="처음부터 보기"
                    >
                        <RotateCcw className="h-4 w-4" />
                    </Button>
                    <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="hidden h-9 w-9 shrink-0 sm:inline-flex"
                        onClick={() => void handleFullscreen()}
                        aria-label="전체 화면"
                    >
                        <Expand className="h-4 w-4" />
                    </Button>
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                    <span>두 화면은 동일한 프레임을 함께 재생합니다.</span>
                    <span className="font-medium text-primary">시간 오차 ±0.08초 이내 보정</span>
                </div>
                {error && <p className="text-xs text-destructive" role="status">{error}</p>}
            </div>
        </section>
    )
}

function VideoPanel({ label, badge, children }: { label: string; badge: string; children: React.ReactNode }) {
    return (
        <div className="relative bg-[#081017]">
            <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-3 bg-gradient-to-b from-black/75 to-transparent px-4 pb-8 pt-3 text-white">
                <span className="text-sm font-semibold tracking-tight">{label}</span>
                <span className="rounded-full border border-white/20 bg-black/35 px-2.5 py-1 text-[0.68rem] font-medium text-white/85 backdrop-blur-sm">
                    {badge}
                </span>
            </div>
            {children}
        </div>
    )
}
