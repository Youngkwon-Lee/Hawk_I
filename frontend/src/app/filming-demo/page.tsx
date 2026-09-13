"use client"

import * as React from "react"
import { Activity, Clock3, Gauge, ScanLine, ShieldCheck } from "lucide-react"

import { JointAngleChart } from "@/components/dashboard/JointAngleChart"
import { SpeedProfileChart } from "@/components/dashboard/SpeedProfileChart"
import { SynchronizedVideoComparison } from "@/components/dashboard/SynchronizedVideoComparison"
import { Card, CardContent } from "@/components/ui/Card"

interface JointAnglePoint {
    frame: number
    time: number
    leftKnee: number
    rightKnee: number
    leftHip: number
    rightHip: number
}

interface SpeedPoint {
    time: number
    speed: number
    normalLow?: number
    normalHigh?: number
}

interface FilmingDemoData {
    source: string
    duration: number
    fps: number
    detection_rate: number
    visualization_data: {
        joint_angles: JointAnglePoint[]
        speed_profile: SpeedPoint[]
    }
}

function range(values: number[]): number {
    if (values.length === 0) return 0
    return Math.max(...values) - Math.min(...values)
}

export default function FilmingDemoPage() {
    const [data, setData] = React.useState<FilmingDemoData | null>(null)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        fetch("/demo/movement-analysis.json")
            .then(response => {
                if (!response.ok) throw new Error(`촬영 데모 데이터 오류 (${response.status})`)
                return response.json() as Promise<FilmingDemoData>
            })
            .then(setData)
            .catch(reason => setError(reason instanceof Error ? reason.message : "데이터를 불러오지 못했습니다."))
    }, [])

    const metrics = React.useMemo(() => {
        const angles = data?.visualization_data.joint_angles ?? []
        const speeds = data?.visualization_data.speed_profile.map(point => point.speed) ?? []
        return {
            leftKneeRom: range(angles.map(point => point.leftKnee)),
            rightKneeRom: range(angles.map(point => point.rightKnee)),
            averageSpeed: speeds.length > 0 ? speeds.reduce((sum, value) => sum + value, 0) / speeds.length : 0,
        }
    }, [data])

    return (
        <main className="min-h-screen bg-background text-foreground">
            <div className="mx-auto max-w-[94rem] px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
                <header className="mb-6 flex flex-col gap-5 border-b border-border pb-6 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                        <div className="mb-3 flex items-center gap-2.5 text-primary">
                            <Activity className="h-6 w-6" aria-hidden="true" />
                            <span className="text-sm font-semibold tracking-tight">HawkEye PD</span>
                            <span className="h-3 w-px bg-border" aria-hidden="true" />
                            <span className="text-xs font-medium text-muted-foreground">Movement Analysis</span>
                        </div>
                        <h1 className="max-w-3xl text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">
                            원본 영상과 관절 추적 결과
                        </h1>
                        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
                            같은 5초 구간에서 관절 위치, 각도, 동작 속도와 시간 변화를 함께 확인합니다.
                        </p>
                    </div>
                    <div className="inline-flex w-fit items-center gap-2 rounded-full border border-primary/25 bg-primary/10 px-3 py-2 text-xs font-semibold text-primary">
                        <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                        비환자 공개 영상 · 촬영 안전 모드
                    </div>
                </header>

                {error ? (
                    <div className="rounded-2xl border border-destructive/30 bg-destructive/5 p-8 text-sm text-destructive" role="alert">
                        {error}
                    </div>
                ) : !data ? (
                    <div className="flex min-h-[34rem] items-center justify-center rounded-2xl border border-border bg-card text-sm text-muted-foreground">
                        분석 결과를 불러오는 중입니다.
                    </div>
                ) : (
                    <>
                        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
                            <SynchronizedVideoComparison
                                originalSrc="/demo/movement-original.mp4"
                                analysisSrc="/demo/movement-analysis.mp4"
                                autoPlay
                                loop
                            />

                            <aside className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1" aria-label="촬영용 핵심 분석값">
                                <MetricCard
                                    icon={ScanLine}
                                    label="무릎 관절 가동범위"
                                    value={`${metrics.leftKneeRom.toFixed(1)}° / ${metrics.rightKneeRom.toFixed(1)}°`}
                                    detail="왼쪽 / 오른쪽 · 영상에서 직접 계산"
                                />
                                <MetricCard
                                    icon={Gauge}
                                    label="평균 동작 속도"
                                    value={metrics.averageSpeed.toFixed(3)}
                                    unit="정규화/초"
                                    detail="영상 좌표 기준 · 실제 추출값"
                                />
                                <MetricCard
                                    icon={Clock3}
                                    label="분석 시간"
                                    value={data.duration.toFixed(1)}
                                    unit="초"
                                    detail={`${data.visualization_data.joint_angles.length} 프레임 · ${data.fps} FPS`}
                                />
                                <Card className="border-primary/20 bg-primary/5 sm:col-span-2 xl:col-span-1">
                                    <CardContent className="p-5">
                                        <div className="flex items-center gap-2 text-sm font-semibold text-primary">
                                            <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                                            촬영 표시 기준
                                        </div>
                                        <ul className="mt-3 space-y-2 text-xs leading-5 text-muted-foreground">
                                            <li>• 환자 정보와 계정 정보 없음</li>
                                            <li>• 분석 추적률 {(data.detection_rate * 100).toFixed(0)}%</li>
                                            <li>• 임상 판정이 아닌 기술 시연</li>
                                        </ul>
                                    </CardContent>
                                </Card>
                            </aside>
                        </div>

                        <section className="mt-6 grid gap-6 lg:grid-cols-2" aria-label="관절 각도와 속도 상세 그래프">
                            <JointAngleChart data={data.visualization_data.joint_angles} />
                            <SpeedProfileChart
                                data={data.visualization_data.speed_profile}
                                title="동작 속도 변화"
                                description="시간에 따른 영상 좌표 기반 이동 속도"
                                speedUnit="정규화/초"
                                showReferenceRange={false}
                            />
                        </section>

                        <footer className="mt-5 flex flex-col gap-1 border-t border-border pt-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
                            <span>포즈 추정 결과는 동일 영상에서 생성했으며 임상 진단값이 아닙니다.</span>
                            <span>Pexels 공개 영상 9616883 · 촬영 데모 전용</span>
                        </footer>
                    </>
                )}
            </div>
        </main>
    )
}

function MetricCard({
    icon: Icon,
    label,
    value,
    unit,
    detail,
}: {
    icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>
    label: string
    value: string
    unit?: string
    detail: string
}) {
    return (
        <Card>
            <CardContent className="p-5">
                <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <Icon className="h-4 w-4" aria-hidden={true} />
                    </span>
                    {label}
                </div>
                <div className="mt-4 flex items-baseline gap-2">
                    <strong className="text-3xl font-semibold tracking-[-0.04em] text-foreground">{value}</strong>
                    {unit && <span className="text-sm font-medium text-muted-foreground">{unit}</span>}
                </div>
                <p className="mt-2 text-xs leading-5 text-muted-foreground">{detail}</p>
            </CardContent>
        </Card>
    )
}
