"use client"

import * as React from "react"
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
    Cell
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { AlertCircle, CheckCircle2, Info } from "lucide-react"

interface GaitCycleData {
    step: number
    duration: number  // milliseconds
    stancePhase: number  // % of cycle
    swingPhase: number   // % of cycle
    strideLength: number // meters
    confidence?: number  // 0-1
}

// Backend V2 format
interface GaitCycleAnalysisData {
    summary?: {
        total_cycles: number
        num_cycles_left: number
        num_cycles_right: number
        analysis_duration_sec: number
        overall_confidence: number
        detection_method: string
        camera_view: string
        is_partial: boolean
        partial_reason: string
    }
    timing?: {
        cycle_time_mean_sec: number
        cycle_time_std_sec: number
        cycle_time_cv_percent: number
    }
    cycles?: {
        left: Array<{
            cycle_number: number
            confidence: number
            durations_sec: { stance: number; swing: number; total: number }
            phase_percent: { stance: number; swing: number }
            spatial_meters: { step_length: number }
        }>
        right: Array<{
            cycle_number: number
            confidence: number
            durations_sec: { stance: number; swing: number; total: number }
            phase_percent: { stance: number; swing: number }
            spatial_meters: { step_length: number }
        }>
    }
    events?: Array<{
        frame: number
        time: number
        type: string
        side: string
        confidence: number
        method: string
    }>
}

interface GaitCycleChartProps {
    data?: GaitCycleData[]
    analysisData?: GaitCycleAnalysisData  // New V2 format
    normalCycleTime?: number
    className?: string
}

// Convert backend V2 format to chart format
function convertAnalysisToChartData(analysisData: GaitCycleAnalysisData): GaitCycleData[] {
    const cycles = analysisData.cycles
    if (!cycles) return []

    const allCycles = [
        ...cycles.left.map((c, i) => ({ ...c, side: 'L', originalIndex: i })),
        ...cycles.right.map((c, i) => ({ ...c, side: 'R', originalIndex: i }))
    ].sort((a, b) => a.cycle_number - b.cycle_number)

    return allCycles.map((cycle, index) => ({
        step: index + 1,
        duration: Math.round(cycle.durations_sec.total * 1000), // seconds to ms
        stancePhase: cycle.phase_percent.stance,
        swingPhase: cycle.phase_percent.swing,
        strideLength: cycle.spatial_meters.step_length,
        confidence: cycle.confidence
    }))
}

// Mock data for demo - 12 steps
const MOCK_GAIT_CYCLE_DATA: GaitCycleData[] = [
    { step: 1, duration: 1050, stancePhase: 62, swingPhase: 38, strideLength: 0.72 },
    { step: 2, duration: 980, stancePhase: 60, swingPhase: 40, strideLength: 0.68 },
    { step: 3, duration: 1020, stancePhase: 61, swingPhase: 39, strideLength: 0.70 },
    { step: 4, duration: 1100, stancePhase: 65, swingPhase: 35, strideLength: 0.65 },
    { step: 5, duration: 950, stancePhase: 58, swingPhase: 42, strideLength: 0.74 },
    { step: 6, duration: 1080, stancePhase: 64, swingPhase: 36, strideLength: 0.66 },
    { step: 7, duration: 1150, stancePhase: 67, swingPhase: 33, strideLength: 0.62 },
    { step: 8, duration: 990, stancePhase: 59, swingPhase: 41, strideLength: 0.71 },
    { step: 9, duration: 1030, stancePhase: 61, swingPhase: 39, strideLength: 0.69 },
    { step: 10, duration: 1070, stancePhase: 63, swingPhase: 37, strideLength: 0.67 },
    { step: 11, duration: 1120, stancePhase: 66, swingPhase: 34, strideLength: 0.63 },
    { step: 12, duration: 1000, stancePhase: 60, swingPhase: 40, strideLength: 0.70 },
]

export function GaitCycleChart({ data, analysisData, normalCycleTime = 1000, className }: GaitCycleChartProps) {
    // Handle V2 analysis data
    const summary = analysisData?.summary
    const isPartial = summary?.is_partial || false
    const confidence = summary?.overall_confidence ?? 0
    const detectionMethod = summary?.detection_method ?? 'unknown'

    // Convert analysis data to chart format if available
    let chartData: GaitCycleData[]
    let isUsingMockData = false

    if (analysisData && summary && summary.total_cycles > 0) {
        chartData = convertAnalysisToChartData(analysisData)
        if (chartData.length === 0) {
            chartData = MOCK_GAIT_CYCLE_DATA
            isUsingMockData = true
        }
    } else if (data && data.length > 0) {
        chartData = data
    } else {
        chartData = MOCK_GAIT_CYCLE_DATA
        isUsingMockData = true
    }

    // Calculate statistics
    const avgDuration = chartData.reduce((acc, d) => acc + d.duration, 0) / chartData.length
    const variability = Math.sqrt(
        chartData.reduce((acc, d) => acc + Math.pow(d.duration - avgDuration, 2), 0) / chartData.length
    )
    const cv = (variability / avgDuration) * 100

    const getVariabilityStatus = (cv: number) => {
        if (cv < 5) return { label: "안정적", variant: "default" as const, color: "text-green-400" }
        if (cv < 10) return { label: "약간 불규칙", variant: "secondary" as const, color: "text-yellow-400" }
        return { label: "불규칙", variant: "destructive" as const, color: "text-red-400" }
    }

    const status = getVariabilityStatus(cv)

    // Confidence status
    const getConfidenceStatus = (conf: number) => {
        if (conf >= 0.7) return { label: "높음", icon: CheckCircle2, color: "text-green-400" }
        if (conf >= 0.4) return { label: "보통", icon: Info, color: "text-yellow-400" }
        return { label: "낮음", icon: AlertCircle, color: "text-red-400" }
    }

    const confidenceStatus = getConfidenceStatus(confidence)

    // Color based on deviation from normal
    const getBarColor = (duration: number, itemConfidence?: number) => {
        // If low confidence, show gray
        if (itemConfidence !== undefined && itemConfidence < 0.3) {
            return "#6b7280" // gray
        }
        const deviation = Math.abs(duration - normalCycleTime) / normalCycleTime * 100
        if (deviation < 5) return "#22c55e" // green
        if (deviation < 15) return "#eab308" // yellow
        return "#ef4444" // red
    }

    // Show partial analysis message
    if (isPartial && summary?.partial_reason) {
        return (
            <Card className={className}>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="text-sm">보행 주기 분석</CardTitle>
                            <CardDescription>각 걸음의 주기 시간 분포</CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                        <AlertCircle className="h-12 w-12 mb-4 text-yellow-500" />
                        <p className="text-center text-sm font-medium">부분 분석</p>
                        <p className="text-center text-xs mt-2">{summary.partial_reason}</p>
                        {analysisData?.events && analysisData.events.length > 0 && (
                            <p className="text-center text-xs mt-2 text-blue-400">
                                감지된 이벤트: {analysisData.events.length}개
                            </p>
                        )}
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card className={className}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-sm">보행 주기 분석</CardTitle>
                        <CardDescription>각 걸음의 주기 시간 분포 (ms)</CardDescription>
                    </div>
                    <div className="text-right">
                        <Badge variant={status.variant}>{status.label}</Badge>
                        <p className="text-xs text-muted-foreground mt-1">
                            변동계수: {cv.toFixed(1)}%
                        </p>
                        {!isUsingMockData && confidence > 0 && (
                            <div className={`flex items-center justify-end gap-1 mt-1 ${confidenceStatus.color}`}>
                                <confidenceStatus.icon className="h-3 w-3" />
                                <span className="text-xs">신뢰도 {(confidence * 100).toFixed(0)}%</span>
                            </div>
                        )}
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                            dataKey="step"
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '걸음 번호', position: 'bottom', fontSize: 11, fill: '#9ca3af' }}
                        />
                        <YAxis
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '주기 (ms)', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#9ca3af' }}
                            domain={[
                                Math.max(600, Math.min(...chartData.map(d => d.duration)) - 100),
                                Math.min(1600, Math.max(...chartData.map(d => d.duration)) + 100)
                            ]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1f2937',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value: number, name: string, props: { payload: GaitCycleData }) => {
                                if (name === 'duration') {
                                    const items: [string, string][] = [[`${value} ms`, '주기 시간']]
                                    if (props.payload.confidence !== undefined) {
                                        items.push([`${(props.payload.confidence * 100).toFixed(0)}%`, '신뢰도'])
                                    }
                                    return items[0]
                                }
                                return [value, name]
                            }}
                        />
                        <ReferenceLine
                            y={normalCycleTime}
                            stroke="#3b82f6"
                            strokeDasharray="5 5"
                            label={{ value: '정상 기준', fill: '#3b82f6', fontSize: 10 }}
                        />
                        <Bar dataKey="duration" radius={[4, 4, 0, 0]}>
                            {chartData.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={getBarColor(entry.duration, entry.confidence)}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>

                {/* Summary Stats */}
                <div className="mt-4 grid grid-cols-4 gap-2 text-xs">
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">평균 주기</div>
                        <div className="font-bold text-blue-400">{avgDuration.toFixed(0)} ms</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">표준편차</div>
                        <div className="font-bold text-purple-400">{variability.toFixed(0)} ms</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">최소</div>
                        <div className="font-bold text-green-400">{Math.min(...chartData.map(d => d.duration))} ms</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">최대</div>
                        <div className="font-bold text-red-400">{Math.max(...chartData.map(d => d.duration))} ms</div>
                    </div>
                </div>

                {/* Detection info */}
                {!isUsingMockData && detectionMethod !== 'unknown' && (
                    <div className="mt-2 text-xs text-center text-muted-foreground">
                        감지 방법: {detectionMethod === 'ensemble' ? '앙상블 (다중 방법)' : detectionMethod}
                    </div>
                )}

                {isUsingMockData && (
                    <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>
                )}
            </CardContent>
        </Card>
    )
}
