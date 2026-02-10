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
    ReferenceLine
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
    const phase = (i / 30) * Math.PI * 2
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
    const [activeTab, setActiveTab] = React.useState<'knee' | 'hip'>('knee')
    const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data : MOCK_JOINT_ANGLE_DATA

    // Calculate statistics
    const kneeStats = React.useMemo(() => {
        const leftKnees = chartData.map(d => d.leftKnee)
        const rightKnees = chartData.map(d => d.rightKnee)
        return {
            leftROM: Math.max(...leftKnees) - Math.min(...leftKnees),
            rightROM: Math.max(...rightKnees) - Math.min(...rightKnees),
            asymmetry: Math.abs(
                (leftKnees.reduce((a, b) => a + b, 0) / leftKnees.length) -
                (rightKnees.reduce((a, b) => a + b, 0) / rightKnees.length)
            )
        }
    }, [chartData])

    const hipStats = React.useMemo(() => {
        const leftHips = chartData.map(d => d.leftHip)
        const rightHips = chartData.map(d => d.rightHip)
        return {
            leftROM: Math.max(...leftHips) - Math.min(...leftHips),
            rightROM: Math.max(...rightHips) - Math.min(...rightHips),
            asymmetry: Math.abs(
                (leftHips.reduce((a, b) => a + b, 0) / leftHips.length) -
                (rightHips.reduce((a, b) => a + b, 0) / rightHips.length)
            )
        }
    }, [chartData])

    const getAsymmetryStatus = (asymmetry: number) => {
        if (asymmetry < 5) return { label: "정상", color: "text-green-400" }
        if (asymmetry < 10) return { label: "경미", color: "text-yellow-400" }
        return { label: "비대칭", color: "text-red-400" }
    }

    const stats = activeTab === 'knee' ? kneeStats : hipStats

    return (
        <Card className={className}>
            <CardHeader className="pb-2">
                <CardTitle className="text-sm">관절 각도 변화</CardTitle>
                <CardDescription>시간에 따른 관절 굴곡각 변화</CardDescription>
            </CardHeader>
            <CardContent>
                {/* Tab Buttons */}
                <div className="flex gap-1 mb-4">
                    <button
                        onClick={() => setActiveTab('knee')}
                        className={`flex-1 py-2 px-3 text-sm rounded-md transition-colors ${
                            activeTab === 'knee'
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                    >
                        무릎 (Knee)
                    </button>
                    <button
                        onClick={() => setActiveTab('hip')}
                        className={`flex-1 py-2 px-3 text-sm rounded-md transition-colors ${
                            activeTab === 'hip'
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                    >
                        엉덩이 (Hip)
                    </button>
                </div>

                {/* Chart */}
                <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                            dataKey="time"
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '시간 (초)', position: 'bottom', fontSize: 10, fill: '#9ca3af' }}
                        />
                        <YAxis
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            label={{ value: '각도 (°)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#9ca3af' }}
                            domain={activeTab === 'knee' ? [0, 80] : [0, 50]}
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
                                name.includes('left') || name.includes('Left') ? '왼쪽' : '오른쪽'
                            ]}
                        />
                        <Legend
                            formatter={(value) => value.includes('left') || value.includes('Left') ? '왼쪽' : '오른쪽'}
                        />
                        <ReferenceLine
                            y={activeTab === 'knee' ? 60 : 35}
                            stroke="#ef4444"
                            strokeDasharray="5 5"
                        />
                        {activeTab === 'knee' ? (
                            <>
                                <Line type="monotone" dataKey="leftKnee" stroke="#3b82f6" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="rightKnee" stroke="#f97316" strokeWidth={2} dot={false} />
                            </>
                        ) : (
                            <>
                                <Line type="monotone" dataKey="leftHip" stroke="#10b981" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="rightHip" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                            </>
                        )}
                    </LineChart>
                </ResponsiveContainer>

                {/* Stats */}
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">왼쪽 ROM</div>
                        <div className={`font-bold ${activeTab === 'knee' ? 'text-blue-400' : 'text-emerald-400'}`}>
                            {stats.leftROM.toFixed(1)}°
                        </div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">오른쪽 ROM</div>
                        <div className={`font-bold ${activeTab === 'knee' ? 'text-orange-400' : 'text-violet-400'}`}>
                            {stats.rightROM.toFixed(1)}°
                        </div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                        <div className="text-muted-foreground">좌우차</div>
                        <div className={`font-bold ${getAsymmetryStatus(stats.asymmetry).color}`}>
                            {stats.asymmetry.toFixed(1)}° ({getAsymmetryStatus(stats.asymmetry).label})
                        </div>
                    </div>
                </div>

                {isUsingMockData && <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>}
            </CardContent>
        </Card>
    )
}
