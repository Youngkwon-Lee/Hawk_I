"use client"

import * as React from "react"
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
    Area,
    ComposedChart
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"

interface JointAngleData {
    frame: number
    time: number
    leftKnee: number
    rightKnee: number
    leftHip: number
    rightHip: number
}

interface JointAngleChartProps {
    data?: JointAngleData[]
    className?: string
}

// Mock data for demo
const MOCK_JOINT_ANGLE_DATA: JointAngleData[] = Array.from({ length: 100 }, (_, i) => {
    const time = i * 0.033 // ~30fps
    const phase = (i / 30) * Math.PI * 2 // One gait cycle per 30 frames
    return {
        frame: i,
        time: parseFloat(time.toFixed(2)),
        leftKnee: 15 + 45 * Math.sin(phase) + Math.random() * 3,
        rightKnee: 15 + 45 * Math.sin(phase + Math.PI) + Math.random() * 3,
        leftHip: 10 + 25 * Math.sin(phase + 0.5) + Math.random() * 2,
        rightHip: 10 + 25 * Math.sin(phase + Math.PI + 0.5) + Math.random() * 2,
    }
})

export function JointAngleChart({ data, className }: JointAngleChartProps) {
    const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data : MOCK_JOINT_ANGLE_DATA

    return (
        <Card className={className}>
            <CardHeader>
                <CardTitle className="text-sm">관절 각도 변화</CardTitle>
                <CardDescription>시간에 따른 무릎/엉덩이 굴곡각 변화 (단위: 도)</CardDescription>
            </CardHeader>
            <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                            dataKey="time"
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '시간 (초)', position: 'bottom', fontSize: 11, fill: '#9ca3af' }}
                        />
                        <YAxis
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '각도 (°)', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#9ca3af' }}
                            domain={[0, 80]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1f2937',
                                border: '1px solid #374151',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value: number, name: string) => [
                                `${value.toFixed(1)}°`,
                                name === 'leftKnee' ? '왼쪽 무릎' :
                                    name === 'rightKnee' ? '오른쪽 무릎' :
                                        name === 'leftHip' ? '왼쪽 엉덩이' : '오른쪽 엉덩이'
                            ]}
                        />
                        <Legend
                            formatter={(value) =>
                                value === 'leftKnee' ? '왼쪽 무릎' :
                                    value === 'rightKnee' ? '오른쪽 무릎' :
                                        value === 'leftHip' ? '왼쪽 엉덩이' : '오른쪽 엉덩이'
                            }
                        />
                        {/* Normal range reference */}
                        <ReferenceLine y={60} stroke="#ef4444" strokeDasharray="5 5" label={{ value: '정상 최대', fill: '#ef4444', fontSize: 10 }} />

                        <Line type="monotone" dataKey="leftKnee" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="rightKnee" stroke="#60a5fa" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="leftHip" stroke="#10b981" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="rightHip" stroke="#6ee7b7" strokeWidth={2} dot={false} />
                    </ComposedChart>
                </ResponsiveContainer>
                {isUsingMockData && <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>}
            </CardContent>
        </Card>
    )
}
