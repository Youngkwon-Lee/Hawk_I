"use client"

import * as React from "react"
import Link from "next/link"
import { AlertCircle, Activity, Clock, Pill, ShieldCheck } from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card"
import {
    buildMedicationObservationSummary,
    getUnifiedTimeline,
    type MedicationEvent,
    type TimelineItem,
} from "@/lib/services/timeline"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"

interface MedicationTimelineProps {
    subjectPersonId?: string
    subjectDisplayName?: string
}

function formatObservedAt(value: string | null): string {
    if (!value) return "시각 미상"
    const parsed = new Date(value)
    if (Number.isNaN(parsed.getTime())) return value
    return parsed.toLocaleString("ko-KR")
}

function formatDose(event: MedicationEvent): string | null {
    if (event.dose_mg === null || event.dose_mg === undefined) return null
    return `${event.dose_mg}${event.dose_unit || "mg"}`
}

export function MedicationTimeline({ subjectPersonId, subjectDisplayName }: MedicationTimelineProps) {
    const [items, setItems] = React.useState<TimelineItem[]>([])
    const [medications, setMedications] = React.useState<MedicationEvent[]>([])
    const [loading, setLoading] = React.useState(Boolean(subjectPersonId))
    const [requiresLogin, setRequiresLogin] = React.useState(false)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        let active = true

        async function loadTimeline() {
            if (!subjectPersonId) {
                setLoading(false)
                setItems([])
                setMedications([])
                return
            }

            setLoading(true)
            setError(null)
            setRequiresLogin(false)

            const supabase = getSupabaseBrowserClient()
            if (!supabase) {
                if (active) {
                    setError("이 배포에는 Supabase 로그인이 설정되어 있지 않습니다.")
                    setLoading(false)
                }
                return
            }

            const { data, error: sessionError } = await supabase.auth.getSession()
            if (!active) return
            if (sessionError || !data.session?.access_token) {
                setRequiresLogin(true)
                setLoading(false)
                return
            }

            try {
                const response = await getUnifiedTimeline(subjectPersonId, data.session.access_token)
                if (!active) return
                setItems(response.items || [])
                setMedications(response.medications || [])
                if (!response.enabled) {
                    setError(response.reason || "공통 임상 타임라인이 설정되어 있지 않습니다.")
                }
            } catch (timelineError) {
                if (!active) return
                setError(timelineError instanceof Error ? timelineError.message : "타임라인을 불러오지 못했습니다.")
            } finally {
                if (active) setLoading(false)
            }
        }

        void loadTimeline()
        return () => {
            active = false
        }
    }, [subjectPersonId])

    if (!subjectPersonId) {
        return (
            <Card className="border-amber-200 bg-amber-50/50">
                <CardContent className="flex gap-3 p-6 text-amber-900">
                    <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                    <div>
                        <p className="font-medium">환자 기록과 연결되지 않은 분석입니다.</p>
                        <p className="mt-1 text-sm">physio_app 환자를 선택해 새로 분석하면 복약 기록과 평가 결과를 함께 확인할 수 있습니다.</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    if (loading) {
        return (
            <div className="flex h-64 items-center justify-center" aria-label="복약 타임라인 불러오는 중">
                <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-primary" />
            </div>
        )
    }

    if (requiresLogin) {
        return (
            <Card className="border-sky-200 bg-sky-50/50">
                <CardContent className="flex gap-3 p-6 text-sky-950">
                    <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0" />
                    <div>
                        <p className="font-medium">임상 기록을 보려면 로그인이 필요합니다.</p>
                        <p className="mt-1 text-sm">History에서 physio_app 계정으로 로그인한 뒤 이 결과로 돌아오세요.</p>
                        <Link href="/history" className="mt-3 inline-block text-sm font-semibold underline underline-offset-4">
                            History에서 로그인
                        </Link>
                    </div>
                </CardContent>
            </Card>
        )
    }

    if (error) {
        return (
            <Card className="border-red-200 bg-red-50/50">
                <CardContent className="flex gap-3 p-6 text-red-900">
                    <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                    <div>
                        <p className="font-medium">복약 타임라인을 불러오지 못했습니다.</p>
                        <p className="mt-1 text-sm">{error}</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    const medicationObservations = items.filter((item) => item.has_medication_context)
    const summary = buildMedicationObservationSummary(items)
    const displayName = subjectDisplayName || "선택 환자"

    if (medications.length === 0 && medicationObservations.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>복약과 평가 기록</CardTitle>
                    <CardDescription>{displayName} · 공통 Supabase 임상 타임라인</CardDescription>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-muted-foreground">연결된 복약 기록 또는 복약 맥락이 포함된 평가 결과가 아직 없습니다.</p>
                </CardContent>
            </Card>
        )
    }

    return (
        <div className="space-y-6">
            <Card className="border-sky-200 bg-sky-50/40">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <ShieldCheck className="h-5 w-5 text-sky-700" />
                        실제 임상 기록 기반 타임라인
                    </CardTitle>
                    <CardDescription>{displayName} · 환자 보고 복약과 측정된 평가 결과만 표시</CardDescription>
                </CardHeader>
                <CardContent className="text-sm text-sky-950">
                    이 화면은 ON/OFF 상태, 약효, 다음 복용 시각 또는 인과관계를 추정하지 않습니다. 복약 변경과 임상 해석은 의료진 검토가 필요합니다.
                </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="flex items-center gap-2 text-base">
                            <Pill className="h-5 w-5 text-amber-600" />
                            환자 보고 복약
                        </CardTitle>
                        <CardDescription>ParkiCheck/physio_app에 저장된 medication_statements</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {medications.length === 0 ? (
                            <p className="text-sm text-muted-foreground">저장된 복약 기록이 없습니다.</p>
                        ) : medications.slice(0, 8).map((event) => {
                            const dose = formatDose(event)
                            return (
                                <div key={event.event_id || `${event.medication_code}-${event.observed_at}`} className="rounded-lg border p-3">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className="font-semibold">{event.medication_display || event.medication_code || "약물명 미입력"}</span>
                                        {dose && <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-900">{dose}</span>}
                                        <span className="ml-auto text-xs text-muted-foreground">{event.app_source === "parkicheck" ? "ParkiCheck 환자 보고" : "physio_app 기록"}</span>
                                    </div>
                                    <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
                                        <Clock className="h-3.5 w-3.5" />
                                        {formatObservedAt(event.observed_at)}
                                    </div>
                                </div>
                            )
                        })}
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="flex items-center gap-2 text-base">
                            <Activity className="h-5 w-5 text-indigo-600" />
                            복약 맥락이 있는 평가
                        </CardTitle>
                        <CardDescription>복약 정보를 함께 보고한 실제 관찰 기록</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {medicationObservations.length === 0 ? (
                            <p className="text-sm text-muted-foreground">복약 맥락이 연결된 평가 결과가 없습니다.</p>
                        ) : medicationObservations.slice(0, 8).map((item, index) => (
                            <div key={item.fhir_id || `${item.code}-${item.observed_at}-${index}`} className="rounded-lg border p-3">
                                <div className="flex flex-wrap items-center gap-2">
                                    <span className="font-semibold">{item.code || "평가"}</span>
                                    {typeof item.score === "number" && <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs text-indigo-900">측정 점수 {item.score}</span>}
                                    <span className="ml-auto text-xs text-muted-foreground">{item.app_source === "parkicheck" ? "ParkiCheck" : "Hawk I"}</span>
                                </div>
                                <p className="mt-2 text-sm text-muted-foreground">
                                    {item.medication_name || "약물명 미입력"}
                                    {item.medication_dose_mg !== null ? ` · ${item.medication_dose_mg}mg` : ""}
                                    {item.hours_after_reported_dose !== null ? ` · 보고 복약 ${item.hours_after_reported_dose}시간 후` : ""}
                                </p>
                                <p className="mt-1 text-xs text-muted-foreground">{formatObservedAt(item.observed_at)}</p>
                            </div>
                        ))}
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>반복 관찰 비교</CardTitle>
                    <CardDescription>같은 평가·약물·용량 조건의 기록이 2회 이상일 때만 계산</CardDescription>
                </CardHeader>
                <CardContent>
                    {summary.available ? (
                        <div className="space-y-2">
                            <p className="text-lg font-semibold">
                                {summary.medicationName}{summary.doseMg !== null ? ` ${summary.doseMg}mg` : ""} · {summary.code || "평가"}
                            </p>
                            <p className="text-sm">
                                첫 기록 {summary.firstScore} → 최근 기록 {summary.latestScore} (관찰된 점수 변화 {summary.observedScoreChange && summary.observedScoreChange > 0 ? "+" : ""}{summary.observedScoreChange})
                            </p>
                            <p className="text-xs text-muted-foreground">{summary.observationCount}개 기록의 단순 비교이며 약효·인과관계·ON/OFF 상태를 의미하지 않습니다.</p>
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground">비교 가능한 동일 조건 기록이 2회 미만입니다. 임의의 약효 추정은 표시하지 않습니다.</p>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
