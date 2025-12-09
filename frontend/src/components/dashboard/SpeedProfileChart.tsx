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
    speed: number
    normalLow?: number
    normalHigh?: number
}

interface SpeedProfileChartProps {
    data?: SpeedData[]
    className?: string
}

// Mock data - shows speed variation over ~5 second walking test
const MOCK_SPEED_DATA: SpeedData[] = Array.from({ length: 50 }, (_, i) => {
    const time = i * 0.1
    // Simulate: start slow, accelerate, maintain, then slow down slightly (fatigue)
    let baseSpeed = 0.8
    if (time < 1) baseSpeed = 0.5 + time * 0.3 // Acceleration
    else if (time < 4) baseSpeed = 0.75 + Math.sin(time * 2) * 0.1 // Steady with variation
    else baseSpeed = 0.8 - (time - 4) * 0.05 // Slight fatigue

    return {
        time: parseFloat(time.toFixed(1)),
        speed: parseFloat((baseSpeed + Math.random() * 0.1 - 0.05).toFixed(3)),
        normalLow: 0.8,
        normalHigh: 1.2
    }
})

export function SpeedProfileChart({ data, className }: SpeedProfileChartProps) {
    const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data : MOCK_SPEED_DATA

    // Calculate statistics
    const speeds = chartData.map(d => d.speed)
    const avgSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length
    const maxSpeed = Math.max(...speeds)
    const minSpeed = Math.min(...speeds)

    // Calculate trend (first half vs second half)
    const halfIndex = Math.floor(speeds.length / 2)
    const firstHalfAvg = speeds.slice(0, halfIndex).reduce((a, b) => a + b, 0) / halfIndex
    const secondHalfAvg = speeds.slice(halfIndex).reduce((a, b) => a + b, 0) / (speeds.length - halfIndex)
    const trend = ((secondHalfAvg - firstHalfAvg) / firstHalfAvg) * 100

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
            description: "안정적인 보행 속도 유지"
        }
    }

    const trendInfo = getTrendInfo(trend)

    return (
        <Card className={className}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-sm">보행 속도 프로파일</CardTitle>
                        <CardDescription>시간에 따른 순간 보행 속도 변화</CardDescription>
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
                    <AreaChart data={chartData}>
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
                            label={{ value: '속도 (m/s)', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#9ca3af' }}
                            domain={[0, 1.5]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1f2937',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value: number, name: string) => {
                                if (name === 'speed') return [`${value.toFixed(2)} m/s`, '순간 속도']
                                return [value, name]
                            }}
                        />

                        {/* Normal range band */}
                        <ReferenceLine y={0.8} stroke="#22c55e" strokeDasharray="5 5" />
                        <ReferenceLine y={1.2} stroke="#22c55e" strokeDasharray="5 5" />

                        {/* Average line */}
                        <ReferenceLine
                            y={avgSpeed}
                            stroke="#f97316"
                            strokeDasharray="3 3"
                            label={{ value: `평균: ${avgSpeed.toFixed(2)}`, fill: '#f97316', fontSize: 10 }}
                        />

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
                        <div className="font-bold text-blue-400">{avgSpeed.toFixed(2)} m/s</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">최대</div>
                        <div className="font-bold text-green-400">{maxSpeed.toFixed(2)} m/s</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">최소</div>
                        <div className="font-bold text-red-400">{minSpeed.toFixed(2)} m/s</div>
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
