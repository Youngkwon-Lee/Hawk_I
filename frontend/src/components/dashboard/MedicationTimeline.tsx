"use client"

import * as React from "react"
import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { LineChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Clock, Activity, AlertCircle, CheckCircle2, Pill } from 'lucide-react'
import { cn } from "@/lib/utils"

interface TimelineData {
    patient_id: string
    timeline: Array<{
        time: string
        hour: number
        motor_score: number
        medication_effect: number
        state: "ON" | "OFF"
        tremor_intensity: number
        rigidity: number
        bradykinesia: number
    }>
    pattern: {
        on_periods: Array<{ state: string; start_hour: number; end_hour: number }>
        off_periods: Array<{ state: string; start_hour: number; end_hour: number }>
        avg_motor_score: number
        on_avg_score: number
        off_avg_score: number
        total_on_hours: number
        total_off_hours: number
    }
    recommendations: {
        optimal_exercise_times: string[]
        best_exercise_hour: string
        next_medication_time: string
        avoid_activities: string[]
    }
}

interface MedicationTimelineProps {
    patientId: string
}

export function MedicationTimeline({ patientId }: MedicationTimelineProps) {
    const [timelineData, setTimelineData] = useState<TimelineData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const fetchTimeline = async () => {
            try {
                const response = await fetch(`http://localhost:5000/api/timeline/${patientId}`)
                if (!response.ok) throw new Error('Failed to fetch timeline')

                const data = await response.json()
                setTimelineData(data.data)
            } catch (err) {
                console.error('Timeline fetch error:', err)
                setError(err instanceof Error ? err.message : 'Unknown error')
            } finally {
                setLoading(false)
            }
        }

        fetchTimeline()
    }, [patientId])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
        )
    }

    if (error || !timelineData) {
        return (
            <Card className="border-red-200 bg-red-50/50">
                <CardContent className="p-6">
                    <div className="flex items-center gap-2 text-red-900">
                        <AlertCircle className="h-5 w-5" />
                        <p>타임라인 데이터를 불러올 수 없습니다: {error}</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    const { timeline, pattern, recommendations } = timelineData

    return (
        <div className="space-y-6">
            {/* Header Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-muted-foreground">평균 운동 점수</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{pattern.avg_motor_score}</div>
                        <p className="text-xs text-muted-foreground mt-1">
                            ON: {pattern.on_avg_score} / OFF: {pattern.off_avg_score}
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-muted-foreground">약물 효과 시간</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-green-600">{pattern.total_on_hours}시간</div>
                        <p className="text-xs text-muted-foreground mt-1">
                            OFF: {pattern.total_off_hours}시간
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium text-muted-foreground">다음 복용 시간</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-center gap-2">
                            <Pill className="h-5 w-5 text-primary" />
                            <div className="text-xl font-bold">{recommendations.next_medication_time}</div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Timeline Chart */}
            <Card>
                <CardHeader>
                    <CardTitle>24시간 운동 능력 추이</CardTitle>
                    <CardDescription>
                        시간대별 운동 점수 및 약물 효과 (낮을수록 좋음)
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={timeline} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="hour"
                                tickFormatter={(hour) => `${hour}:00`}
                                label={{ value: '시간', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                                yAxisId="left"
                                label={{ value: '운동 점수', angle: -90, position: 'insideLeft' }}
                            />
                            <YAxis
                                yAxisId="right"
                                orientation="right"
                                domain={[0, 1]}
                                label={{ value: '약물 효과', angle: 90, position: 'insideRight' }}
                            />
                            <Tooltip
                                content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        const data = payload[0].payload
                                        return (
                                            <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
                                                <p className="font-semibold">{data.hour}:00</p>
                                                <p className="text-sm">상태: <span className={cn(
                                                    "font-medium",
                                                    data.state === "ON" ? "text-green-600" : "text-red-600"
                                                )}>{data.state}</span></p>
                                                <p className="text-sm">운동 점수: {data.motor_score}</p>
                                                <p className="text-sm">약물 효과: {(data.medication_effect * 100).toFixed(0)}%</p>
                                                <p className="text-sm">떨림: {data.tremor_intensity}</p>
                                                <p className="text-sm">경직: {data.rigidity}</p>
                                            </div>
                                        )
                                    }
                                    return null
                                }}
                            />
                            <Legend />

                            {/* Background areas for ON/OFF periods */}
                            {pattern.on_periods.map((period, i) => (
                                <ReferenceLine
                                    key={`on-${i}`}
                                    x={period.start_hour}
                                    stroke="green"
                                    strokeDasharray="3 3"
                                    opacity={0.3}
                                />
                            ))}

                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="motor_score"
                                stroke="#ef4444"
                                strokeWidth={2}
                                name="운동 점수"
                                dot={{ r: 3 }}
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="medication_effect"
                                stroke="#22c55e"
                                strokeWidth={2}
                                name="약물 효과"
                                dot={{ r: 3 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            {/* Recommendations */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="border-green-200 bg-green-50/50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-green-900">
                            <CheckCircle2 className="h-5 w-5" />
                            권장 활동 시간
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <div>
                            <p className="text-sm font-medium text-green-900 mb-2">최적 운동 시간:</p>
                            <div className="flex items-center gap-2 text-lg font-bold text-green-700">
                                <Activity className="h-5 w-5" />
                                {recommendations.best_exercise_hour}
                            </div>
                        </div>
                        <div>
                            <p className="text-sm font-medium text-green-900 mb-2">권장 시간대:</p>
                            <ul className="space-y-1">
                                {recommendations.optimal_exercise_times.map((time, i) => (
                                    <li key={i} className="text-sm text-green-800 flex items-center gap-2">
                                        <Clock className="h-4 w-4" />
                                        {time}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-red-200 bg-red-50/50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-red-900">
                            <AlertCircle className="h-5 w-5" />
                            주의 시간대
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <p className="text-sm text-red-900">다음 시간대는 약물 효과가 낮아 격렬한 활동을 피하세요:</p>
                        <ul className="space-y-1">
                            {recommendations.avoid_activities.length > 0 ? (
                                recommendations.avoid_activities.map((time, i) => (
                                    <li key={i} className="text-sm text-red-800 flex items-center gap-2">
                                        <Clock className="h-4 w-4" />
                                        {time}
                                    </li>
                                ))
                            ) : (
                                <li className="text-sm text-muted-foreground">없음</li>
                            )}
                        </ul>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
