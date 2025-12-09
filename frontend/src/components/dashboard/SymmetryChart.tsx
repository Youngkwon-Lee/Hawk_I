"use client"

import * as React from "react"
import {
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
    Legend,
    ResponsiveContainer,
    Tooltip
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"

interface SymmetryData {
    metric: string
    left: number
    right: number
    normal: number
}

interface SymmetryChartProps {
    data?: SymmetryData[]
    asymmetryScore?: number
    className?: string
}

// Mock data for demo
const MOCK_SYMMETRY_DATA: SymmetryData[] = [
    { metric: "팔 흔들기", left: 85, right: 72, normal: 100 },
    { metric: "보폭", left: 92, right: 88, normal: 100 },
    { metric: "발 높이", left: 78, right: 65, normal: 100 },
    { metric: "무릎 굴곡", left: 88, right: 82, normal: 100 },
    { metric: "엉덩이 굴곡", left: 90, right: 85, normal: 100 },
    { metric: "발목 각도", left: 82, right: 78, normal: 100 },
]

export function SymmetryChart({ data, asymmetryScore, className }: SymmetryChartProps) {
    const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data : MOCK_SYMMETRY_DATA

    // Calculate asymmetry if not provided
    const calculatedAsymmetry = asymmetryScore ?? chartData.reduce((acc, item) => {
        return acc + Math.abs(item.left - item.right)
    }, 0) / chartData.length

    const getAsymmetryStatus = (score: number) => {
        if (score < 5) return { label: "정상", variant: "default" as const, color: "text-green-400" }
        if (score < 15) return { label: "경미한 비대칭", variant: "secondary" as const, color: "text-yellow-400" }
        if (score < 25) return { label: "중등도 비대칭", variant: "destructive" as const, color: "text-orange-400" }
        return { label: "심한 비대칭", variant: "destructive" as const, color: "text-red-400" }
    }

    const status = getAsymmetryStatus(calculatedAsymmetry)

    return (
        <Card className={className}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-sm">좌우 대칭성 분석</CardTitle>
                        <CardDescription>왼쪽/오른쪽 운동 패턴 비교 (정규화 %)</CardDescription>
                    </div>
                    <div className="text-right">
                        <Badge variant={status.variant}>{status.label}</Badge>
                        <p className={`text-lg font-bold mt-1 ${status.color}`}>
                            {calculatedAsymmetry.toFixed(1)}% 차이
                        </p>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={chartData}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis
                            dataKey="metric"
                            tick={{ fontSize: 11, fill: '#9ca3af' }}
                        />
                        <PolarRadiusAxis
                            angle={30}
                            domain={[0, 100]}
                            tick={{ fontSize: 9, fill: '#6b7280' }}
                        />
                        <Radar
                            name="왼쪽"
                            dataKey="left"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.4}
                        />
                        <Radar
                            name="오른쪽"
                            dataKey="right"
                            stroke="#f97316"
                            fill="#f97316"
                            fillOpacity={0.4}
                        />
                        <Legend />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1f2937',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value: number) => [`${value.toFixed(0)}%`]}
                        />
                    </RadarChart>
                </ResponsiveContainer>

                {/* Detailed breakdown */}
                <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
                    {chartData.map((item, idx) => {
                        const diff = Math.abs(item.left - item.right)
                        const diffColor = diff < 5 ? 'text-green-400' : diff < 15 ? 'text-yellow-400' : 'text-red-400'
                        return (
                            <div key={idx} className="bg-slate-800/50 rounded p-2 text-center">
                                <div className="text-muted-foreground">{item.metric}</div>
                                <div className={`font-medium ${diffColor}`}>
                                    {diff.toFixed(0)}% 차이
                                </div>
                            </div>
                        )
                    })}
                </div>
                {isUsingMockData && <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>}
            </CardContent>
        </Card>
    )
}
