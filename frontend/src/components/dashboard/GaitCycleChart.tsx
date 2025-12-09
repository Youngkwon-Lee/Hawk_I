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

interface GaitCycleData {
    step: number
    duration: number  // milliseconds
    stancePhase: number  // % of cycle
    swingPhase: number   // % of cycle
    strideLength: number // meters
}

interface GaitCycleChartProps {
    data?: GaitCycleData[]
    normalCycleTime?: number
    className?: string
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

export function GaitCycleChart({ data, normalCycleTime = 1000, className }: GaitCycleChartProps) {
    const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data : MOCK_GAIT_CYCLE_DATA

    // Calculate statistics
    const avgDuration = chartData.reduce((acc, d) => acc + d.duration, 0) / chartData.length
    const variability = Math.sqrt(
        chartData.reduce((acc, d) => acc + Math.pow(d.duration - avgDuration, 2), 0) / chartData.length
    )
    const cv = (variability / avgDuration) * 100 // Coefficient of variation

    const getVariabilityStatus = (cv: number) => {
        if (cv < 5) return { label: "안정적", variant: "default" as const }
        if (cv < 10) return { label: "약간 불규칙", variant: "secondary" as const }
        return { label: "불규칙", variant: "destructive" as const }
    }

    const status = getVariabilityStatus(cv)

    // Color based on deviation from normal
    const getBarColor = (duration: number) => {
        const deviation = Math.abs(duration - normalCycleTime) / normalCycleTime * 100
        if (deviation < 5) return "#22c55e" // green
        if (deviation < 15) return "#eab308" // yellow
        return "#ef4444" // red
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
                            domain={[800, 1200]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1f2937',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value: number, name: string) => {
                                if (name === 'duration') return [`${value} ms`, '주기 시간']
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
                                <Cell key={`cell-${index}`} fill={getBarColor(entry.duration)} />
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
                {isUsingMockData && <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>}
            </CardContent>
        </Card>
    )
}
