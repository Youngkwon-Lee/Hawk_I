"use client"

import { Pill } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card"
import type { MedicationContext, MedicationTiming } from "@/lib/services/api"

interface MedicationTimelineProps {
    patientId: string
    medicationContext?: MedicationContext | null
    medicationTiming?: MedicationTiming | null
}

function formatReportedDose(context: MedicationContext): string {
    const name = context.medication || "약물명 미입력"
    const dose = context.dose_mg == null ? "용량 미입력" : `${context.dose_mg} mg`
    return `${name} · ${dose}`
}

function formatTiming(timing: MedicationTiming): string {
    const hours = timing.hours_after_reported_dose
    if (hours == null) return "복용 후 경과시간 확인 필요"
    return `환자 보고 복용 후 ${hours}시간에 검사`
}

function formatReportedTime(value?: string): string {
    if (!value) return "복용 시각 미입력"
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return "복용 시각 확인 필요"
    return new Intl.DateTimeFormat("ko-KR", {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(date)
}

export function MedicationTimeline({ patientId, medicationContext, medicationTiming }: MedicationTimelineProps) {
    if (!medicationContext?.available || !medicationTiming?.available) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Pill className="h-5 w-5 text-primary" />
                        약물 타임라인 데이터 없음
                    </CardTitle>
                    <CardDescription>
                        검증된 복용 기록과 반복 운동 평가가 연결된 뒤 실제 시간 관계를 표시합니다.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm text-muted-foreground">
                    <p>환자 세션: {patientId}</p>
                    <p>현재는 복용 시간, ON/OFF 상태, 다음 복용 시간 또는 활동 권고를 추정하지 않습니다.</p>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Pill className="h-5 w-5 text-primary" />
                    복약–검사 시간 관계
                </CardTitle>
                <CardDescription>
                    ParkiCheck에서 동의 후 전달된 환자 보고 기록입니다.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
                <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-lg border p-3">
                        <p className="text-xs text-muted-foreground">환자 보고 복약</p>
                        <p className="mt-1 font-medium">{formatReportedDose(medicationContext)}</p>
                    </div>
                    <div className="rounded-lg border p-3">
                        <p className="text-xs text-muted-foreground">보고된 복용 시각</p>
                        <p className="mt-1 font-medium">{formatReportedTime(medicationContext.taken_at)}</p>
                    </div>
                    <div className="rounded-lg border p-3">
                        <p className="text-xs text-muted-foreground">검사와의 관계</p>
                        <p className="mt-1 font-medium">{formatTiming(medicationTiming)}</p>
                    </div>
                </div>
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-muted-foreground">
                    단일 검사에서 확인되는 것은 시간 관계뿐입니다. 이 결과만으로 약효, ON/OFF 상태 또는 복약 변경 필요성을 판단하지 않습니다. 동일 과제를 반복 측정한 뒤 의료진이 함께 검토해야 합니다.
                </div>
                <p className="text-xs text-muted-foreground">분석 세션: {patientId}</p>
            </CardContent>
        </Card>
    )
}
