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
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"

interface PopulationStats {
    task_type: string
    total_samples: number
    metrics: {
        [key: string]: {
            [scoreGroup: string]: {
                mean: number
                std: number
                min?: number
                max?: number
                median?: number
                count?: number
            }
        }
    }
    display_names: {
        [key: string]: string
    }
    score_distribution?: {
        [key: string]: number
    }
}

interface PopulationComparisonProps {
    taskType: string
    patientScore?: number
    patientMetrics?: Record<string, number>
}

// Map frontend metric keys to backend population stats keys
const METRIC_MAPPING: Record<string, string> = {
    // Finger Tapping
    'tapping_speed': 'tapping_freq',
    'amplitude_mean': 'amplitude_ratio',
    'rhythm_variability': 'velocity_variability',
    'fatigue_rate': 'fatigue_index',
    'hesitation_count': 'hesitation_rate',
    'amplitude_std': 'amplitude_variability',
    // Gait
    'velocity_mean': 'walking_speed',
    'speed': 'walking_speed',
    'stride_length': 'stride_length',
    'step_length': 'stride_length',
    'cadence': 'cadence',
    'step_frequency': 'cadence',
    'stride_variability': 'stride_variability',
    'cv_stride': 'stride_variability',
    'arm_swing_asymmetry': 'arm_swing_asymmetry',
    'asymmetry': 'arm_swing_asymmetry'
}

export function PopulationComparison({ taskType, patientScore, patientMetrics }: PopulationComparisonProps) {
    const [stats, setStats] = React.useState<PopulationStats | null>(null)
    const [loading, setLoading] = React.useState(true)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        const fetchStats = async () => {
            try {
                setLoading(true)
                const normalizedType = taskType === 'finger' ? 'finger_tapping' : taskType
                const response = await fetch(`http://localhost:5000/api/population-stats/${normalizedType}`)

                if (!response.ok) {
                    throw new Error('Failed to fetch population statistics')
                }

                const result = await response.json()
                if (result.success) {
                    setStats(result.data)
                } else {
                    throw new Error(result.error || 'Unknown error')
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load statistics')
            } finally {
                setLoading(false)
            }
        }

        fetchStats()
    }, [taskType])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
        )
    }

    if (error || !stats) {
        return (
            <Card>
                <CardContent className="p-6">
                    <p className="text-muted-foreground text-center">
                        통계 데이터를 불러올 수 없습니다: {error}
                    </p>
                </CardContent>
            </Card>
        )
    }

    // Prepare bar chart data
    const barChartData = Object.entries(stats.metrics).map(([metricKey, scoreGroups]) => {
        const displayName = stats.display_names[metricKey] || metricKey
        return {
            name: displayName,
            metricKey,
            '정상 (Score 0)': scoreGroups.score_0?.mean || 0,
            '정상 std': scoreGroups.score_0?.std || 0,
            '중등 (Score 3-4)': scoreGroups.score_3_4?.mean || 0,
            '중등 std': scoreGroups.score_3_4?.std || 0,
        }
    })

    // Prepare radar chart data (normalized to 0-1 scale)
    const radarChartData = Object.entries(stats.metrics).map(([metricKey, scoreGroups]) => {
        const displayName = stats.display_names[metricKey] || metricKey
        const score0 = scoreGroups.score_0?.mean || 0
        const score34 = scoreGroups.score_3_4?.mean || 0
        const maxVal = Math.max(score0, score34, 0.001)

        // Get patient value if available
        let patientValue = 0
        if (patientMetrics) {
            // Find frontend key that maps to this backend metricKey
            const mappedKey = Object.entries(METRIC_MAPPING).find(([k, v]) => v === metricKey)?.[0]
            if (mappedKey && patientMetrics[mappedKey] !== undefined) {
                patientValue = patientMetrics[mappedKey]
            }
            // Also try direct key match (if metric keys are same)
            if (patientValue === 0 && patientMetrics[metricKey] !== undefined) {
                patientValue = patientMetrics[metricKey]
            }
            // Try all patient metric keys to find a match
            if (patientValue === 0) {
                for (const [pKey, pVal] of Object.entries(patientMetrics)) {
                    if (METRIC_MAPPING[pKey] === metricKey && typeof pVal === 'number') {
                        patientValue = pVal
                        break
                    }
                }
            }
        }

        return {
            metric: displayName.replace(/\([^)]*\)/g, '').trim(), // Remove units for radar
            metricKey,
            '정상': score0 / maxVal,
            '중등': score34 / maxVal,
            '환자': patientValue > 0 ? patientValue / maxVal : undefined,
            rawScore0: score0,
            rawScore34: score34,
            rawPatient: patientValue
        }
    })

    // Determine patient's score group
    const getScoreGroup = (score: number | undefined) => {
        if (score === undefined) return null
        if (score === 0) return 'score_0'
        if (score >= 3) return 'score_3_4'
        if (score === 1) return 'score_1'
        return 'score_2'
    }

    const patientScoreGroup = getScoreGroup(patientScore)

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">정상군 비교 분석</h2>
                    <p className="text-sm text-muted-foreground">
                        {stats.total_samples > 0
                            ? `${stats.total_samples}명의 데이터 기반 통계`
                            : '기본 참조 통계'}
                    </p>
                </div>
                {patientScore !== undefined && (
                    <Badge
                        variant={patientScore === 0 ? "default" : patientScore >= 3 ? "destructive" : "secondary"}
                    >
                        환자 점수: {patientScore}점
                    </Badge>
                )}
            </div>

            {/* Score Distribution */}
            {stats.score_distribution && (
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm">점수 분포</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex gap-4 flex-wrap">
                            {Object.entries(stats.score_distribution).map(([group, count]) => {
                                const label = group === 'score_0' ? '0점 (정상)'
                                    : group === 'score_1' ? '1점 (경미)'
                                        : group === 'score_2' ? '2점 (경도)'
                                            : '3-4점 (중등~중증)'
                                const isPatientGroup = patientScoreGroup === group

                                return (
                                    <div
                                        key={group}
                                        className={`px-3 py-2 rounded-lg border ${isPatientGroup
                                                ? 'border-primary bg-primary/10'
                                                : 'border-border'
                                            }`}
                                    >
                                        <div className="text-lg font-semibold">{count}명</div>
                                        <div className="text-xs text-muted-foreground">{label}</div>
                                    </div>
                                )
                            })}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Charts Grid */}
            <div className="grid md:grid-cols-2 gap-6">
                {/* Bar Chart */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">지표별 비교 (Bar Chart)</CardTitle>
                        <CardDescription>정상군 vs 중등도 증상군 평균 비교</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart
                                data={barChartData}
                                layout="vertical"
                                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis type="number" />
                                <YAxis
                                    dataKey="name"
                                    type="category"
                                    tick={{ fontSize: 11 }}
                                    width={95}
                                />
                                <Tooltip
                                    formatter={(value: number, name: string) => [
                                        value.toFixed(2),
                                        name
                                    ]}
                                />
                                <Legend />
                                <Bar
                                    dataKey="정상 (Score 0)"
                                    fill="#4ade80"
                                    radius={[0, 4, 4, 0]}
                                />
                                <Bar
                                    dataKey="중등 (Score 3-4)"
                                    fill="#f87171"
                                    radius={[0, 4, 4, 0]}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Radar Chart */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">다차원 비교 (Radar Chart)</CardTitle>
                        <CardDescription>정규화된 지표 비교 (0-1 스케일)</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={350}>
                            <RadarChart data={radarChartData}>
                                <PolarGrid />
                                <PolarAngleAxis
                                    dataKey="metric"
                                    tick={{ fontSize: 10 }}
                                />
                                <PolarRadiusAxis
                                    angle={30}
                                    domain={[0, 1]}
                                    tick={{ fontSize: 9, fill: '#6b7280' }}
                                    stroke="#374151"
                                    axisLine={false}
                                />
                                <Radar
                                    name="정상 (Score 0)"
                                    dataKey="정상"
                                    stroke="#4ade80"
                                    fill="#4ade80"
                                    fillOpacity={0.3}
                                />
                                <Radar
                                    name="중등 (Score 3-4)"
                                    dataKey="중등"
                                    stroke="#f87171"
                                    fill="#f87171"
                                    fillOpacity={0.3}
                                />
                                {patientMetrics && (
                                    <Radar
                                        name="환자"
                                        dataKey="환자"
                                        stroke="#3b82f6"
                                        fill="#3b82f6"
                                        fillOpacity={0.5}
                                    />
                                )}
                                <Legend />
                                <Tooltip
                                    formatter={(value: number, name: string, props: any) => {
                                        const rawKey = name === '정상 (Score 0)' ? 'rawScore0'
                                            : name === '중등 (Score 3-4)' ? 'rawScore34' : 'rawPatient'
                                        const rawValue = props.payload[rawKey]
                                        return [rawValue?.toFixed(2) || 'N/A', name]
                                    }}
                                />
                            </RadarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>

            {/* Detailed Metrics Table */}
            <Card>
                <CardHeader>
                    <CardTitle className="text-sm">상세 지표 비교</CardTitle>
                    <CardDescription>각 지표별 정상군과 중등도 증상군의 통계 비교</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b">
                                    <th className="text-left py-2 px-3 font-medium">지표</th>
                                    <th className="text-center py-2 px-3 font-medium text-green-600">정상 (Score 0)</th>
                                    <th className="text-center py-2 px-3 font-medium text-red-500">중등 (Score 3-4)</th>
                                    <th className="text-center py-2 px-3 font-medium">변화율</th>
                                    {patientMetrics && (
                                        <th className="text-center py-2 px-3 font-medium text-blue-500">환자</th>
                                    )}
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(stats.metrics).map(([metricKey, scoreGroups]) => {
                                    const displayName = stats.display_names[metricKey] || metricKey
                                    const score0Mean = scoreGroups.score_0?.mean || 0
                                    const score0Std = scoreGroups.score_0?.std || 0
                                    const score34Mean = scoreGroups.score_3_4?.mean || 0
                                    const score34Std = scoreGroups.score_3_4?.std || 0

                                    const changePercent = score0Mean > 0
                                        ? ((score34Mean - score0Mean) / score0Mean * 100)
                                        : 0

                                    // Get patient value
                                    let patientValue: number | undefined
                                    if (patientMetrics) {
                                        // Find frontend key that maps to this backend metricKey
                                        const mappedKey = Object.entries(METRIC_MAPPING)
                                            .find(([k, v]) => v === metricKey)?.[0]
                                        if (mappedKey && patientMetrics[mappedKey] !== undefined) {
                                            patientValue = patientMetrics[mappedKey]
                                        }
                                        // Try direct key match
                                        if (patientValue === undefined && patientMetrics[metricKey] !== undefined) {
                                            patientValue = patientMetrics[metricKey]
                                        }
                                        // Try all patient metric keys
                                        if (patientValue === undefined) {
                                            for (const [pKey, pVal] of Object.entries(patientMetrics)) {
                                                if (METRIC_MAPPING[pKey] === metricKey && typeof pVal === 'number') {
                                                    patientValue = pVal
                                                    break
                                                }
                                            }
                                        }
                                    }

                                    return (
                                        <tr key={metricKey} className="border-b last:border-0">
                                            <td className="py-2 px-3">{displayName}</td>
                                            <td className="text-center py-2 px-3">
                                                <span className="font-medium">{score0Mean.toFixed(2)}</span>
                                                <span className="text-muted-foreground text-xs ml-1">
                                                    ±{score0Std.toFixed(2)}
                                                </span>
                                            </td>
                                            <td className="text-center py-2 px-3">
                                                <span className="font-medium">{score34Mean.toFixed(2)}</span>
                                                <span className="text-muted-foreground text-xs ml-1">
                                                    ±{score34Std.toFixed(2)}
                                                </span>
                                            </td>
                                            <td className={`text-center py-2 px-3 font-medium ${changePercent > 0 ? 'text-red-500' : 'text-green-600'
                                                }`}>
                                                {changePercent > 0 ? '+' : ''}{changePercent.toFixed(1)}%
                                            </td>
                                            {patientMetrics && (
                                                <td className="text-center py-2 px-3 font-medium text-blue-500">
                                                    {patientValue !== undefined ? patientValue.toFixed(2) : '-'}
                                                </td>
                                            )}
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>

            {/* Interpretation */}
            <Card className="bg-slate-800 border-slate-700">
                <CardContent className="p-4">
                    <h3 className="font-medium text-blue-300 mb-2">해석 가이드</h3>
                    <ul className="text-sm text-slate-300 space-y-1">
                        <li>• <strong className="text-blue-400">태핑 빈도, 진폭 비율</strong>: 높을수록 좋음 (정상군이 더 높음)</li>
                        <li>• <strong className="text-blue-400">변동성, 멈춤율</strong>: 낮을수록 좋음 (정상군이 더 낮음)</li>
                        <li>• 환자 점수가 정상군 범위에 가까울수록 운동 기능이 양호함을 의미합니다</li>
                    </ul>
                </CardContent>
            </Card>
        </div>
    )
}
