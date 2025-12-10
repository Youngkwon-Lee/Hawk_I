"use client"

import * as React from "react"
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Legend
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { TrendingDown, TrendingUp, Minus } from "lucide-react"

interface SpeedData {
    time: number
    speed?: number
    // Finger tapping specific fields
    leftSpeed?: number
    rightSpeed?: number
    avgSpeed?: number
    normalLow?: number
    normalHigh?: number
}

interface SpeedProfileChartProps {
    data?: SpeedData[]
    className?: string
    taskType?: "gait" | "finger"
}

// Mock data - shows speed variation over ~5 second walking test (PD4T 기준: 0.55-0.95)
const MOCK_SPEED_DATA: SpeedData[] = Array.from({ length: 50 }, (_, i) => {
    const time = i * 0.1
    // Simulate: start slow, accelerate, maintain, then slow down slightly (fatigue)
    let baseSpeed = 0.75  // PD4T normal mean
    if (time < 1) baseSpeed = 0.5 + time * 0.25 // Acceleration
    else if (time < 4) baseSpeed = 0.72 + Math.sin(time * 2) * 0.08 // Steady with variation
    else baseSpeed = 0.75 - (time - 4) * 0.03 // Slight fatigue

    return {
        time: parseFloat(time.toFixed(1)),
        speed: parseFloat((baseSpeed + Math.random() * 0.08 - 0.04).toFixed(3)),
        normalLow: 0.55,
        normalHigh: 0.95
    }
})

export function SpeedProfileChart({ data, className, taskType = "gait" }: SpeedProfileChartProps) {
    const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data : MOCK_SPEED_DATA
    const isFinger = taskType === "finger"

    // Normalize data - finger tapping uses avgSpeed, gait uses speed
    const normalizedData = chartData.map(d => ({
        ...d,
        speed: d.speed ?? d.avgSpeed ?? 0
    }))

    // Calculate statistics
    const speeds = normalizedData.map(d => d.speed).filter(s => !isNaN(s) && s !== undefined)
    const avgSpeed = speeds.length > 0 ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0
    const maxSpeed = speeds.length > 0 ? Math.max(...speeds) : 0
    const minSpeed = speeds.length > 0 ? Math.min(...speeds) : 0

    // Calculate trend (first half vs second half)
    const halfIndex = Math.floor(speeds.length / 2)
    const firstHalfAvg = halfIndex > 0 ? speeds.slice(0, halfIndex).reduce((a, b) => a + b, 0) / halfIndex : 0
    const secondHalfAvg = speeds.length - halfIndex > 0 ? speeds.slice(halfIndex).reduce((a, b) => a + b, 0) / (speeds.length - halfIndex) : 0
    const trend = firstHalfAvg > 0 ? ((secondHalfAvg - firstHalfAvg) / firstHalfAvg) * 100 : 0

    const getTrendInfo = (trend: number) => {
        if (trend < -5) return {
            label: "감속 경향",
            icon: <TrendingDown className="h-4 w-4" />,
            color: "text-red-400",
            description: "후반부 속도 저하 (피로 가능성)"
        }
        if (trend > 5) return {
            label: "가속 경향",
            icon: <TrendingUp className="h-4 w-4" />,
            color: "text-green-400",
            description: "후반부 속도 증가"
        }
        return {
            label: "일정 속도",
            icon: <Minus className="h-4 w-4" />,
            color: "text-blue-400",
            description: isFinger ? "안정적인 탭핑 속도 유지" : "안정적인 보행 속도 유지"
        }
    }

    // Task-specific labels
    const title = isFinger ? "탭핑 속도 프로파일" : "보행 속도 프로파일"
    const description = isFinger ? "시간에 따른 탭핑 속도 변화" : "시간에 따른 순간 보행 속도 변화"
    const speedLabel = isFinger ? "탭핑 속도" : "속도 (m/s)"
    const speedUnit = isFinger ? "" : "m/s"

    const trendInfo = getTrendInfo(trend)

    return (
        <Card className={className}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-sm">{title}</CardTitle>
                        <CardDescription>{description}</CardDescription>
                    </div>
                    <div className="text-right">
                        <div className={`flex items-center gap-1 ${trendInfo.color}`}>
                            {trendInfo.icon}
                            <span className="text-sm font-medium">{trendInfo.label}</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                            {trendInfo.description}
                        </p>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={normalizedData}>
                        <defs>
                            <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="normalGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#22c55e" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                            dataKey="time"
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '시간 (초)', position: 'bottom', fontSize: 11, fill: '#9ca3af' }}
                        />
                        <YAxis
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: speedLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#9ca3af' }}
                            domain={isFinger ? ['auto', 'auto'] : [0, 1.5]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1f2937',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value: number, name: string) => {
                                if (name === 'speed') return [
                                    isFinger ? value.toFixed(2) : `${value.toFixed(2)} m/s`,
                                    isFinger ? '순간 탭핑 속도' : '순간 보행 속도'
                                ]
                                return [value, name]
                            }}
                        />

                        {/* Normal range band - only for gait (PD4T: 0.55-0.95) */}
                        {!isFinger && (
                            <>
                                <ReferenceLine y={0.55} stroke="#22c55e" strokeDasharray="5 5" />
                                <ReferenceLine y={0.95} stroke="#22c55e" strokeDasharray="5 5" />
                            </>
                        )}

                        {/* Average line */}
                        {avgSpeed > 0 && (
                            <ReferenceLine
                                y={avgSpeed}
                                stroke="#f97316"
                                strokeDasharray="3 3"
                                label={{ value: `평균: ${avgSpeed.toFixed(2)}`, fill: '#f97316', fontSize: 10 }}
                            />
                        )}

                        <Area
                            type="monotone"
                            dataKey="speed"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            fill="url(#speedGradient)"
                        />
                    </AreaChart>
                </ResponsiveContainer>

                {/* Summary Stats */}
                <div className="mt-4 grid grid-cols-4 gap-2 text-xs">
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">평균 속도</div>
                        <div className="font-bold text-blue-400">{avgSpeed.toFixed(2)}{speedUnit && ` ${speedUnit}`}</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">최대</div>
                        <div className="font-bold text-green-400">{maxSpeed.toFixed(2)}{speedUnit && ` ${speedUnit}`}</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">최소</div>
                        <div className="font-bold text-red-400">{minSpeed.toFixed(2)}{speedUnit && ` ${speedUnit}`}</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">변화율</div>
                        <div className={`font-bold ${trendInfo.color}`}>
                            {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
                        </div>
                    </div>
                </div>
                {isUsingMockData && <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>}
            </CardContent>
        </Card>
    )
}
