"use client"

import { Pill } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card"

interface MedicationTimelineProps {
    patientId: string
}

export function MedicationTimeline({ patientId }: MedicationTimelineProps) {
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
