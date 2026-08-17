"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Copy, Check, FileText, ClipboardList, ShieldCheck } from "lucide-react"

interface SOAPNoteProps {
    taskType: string
    metrics?: Record<string, number>
    performabilityAssessment?: {
        status: "performable" | "uncertain" | "non_performable_or_near_impossible" | "unscorable_due_to_tracking"
        summary: string
    } | null
    scoreAdvisory?: {
        level: "standard" | "review_recommended" | "reference_only"
        summary: string
    } | null
    updrsScore?: {
        score?: number
        total_score?: number
        severity: string
        confidence?: number
    }
    aiInterpretation?: string
    patientId?: string
    analysisDate?: string
}

// UPDRS item mapping
const UPDRS_ITEMS: Record<string, string> = {
    finger_tapping: '3.4 Finger Tapping',
    // hand_movement: '3.5 Hand Movements',  // Not implemented yet
    gait: '3.10 Gait',
    // leg_agility: '3.8 Leg Agility'  // Not implemented yet
}

// Medical terminology for severity
const SEVERITY_TERMS: Record<string, string> = {
    'Normal': 'Normal motor function',
    'Slight': 'Minimal bradykinesia',
    'Mild': 'Mild bradykinesia with amplitude decrement',
    'Moderate': 'Moderate bradykinesia with fatigue and hesitations',
    'Severe': 'Severe bradykinesia with marked motor impairment'
}

export function SOAPNote({ taskType, metrics, performabilityAssessment, scoreAdvisory, updrsScore, analysisDate }: SOAPNoteProps) {
    const [copiedTarget, setCopiedTarget] = React.useState<"note" | "objective" | "assessment" | null>(null)
    const [format, setFormat] = React.useState<'full' | 'compact'>('compact')

    const normalizedType = taskType?.includes('finger') || taskType?.includes('tapping')
        ? 'finger_tapping'
        : taskType?.includes('gait') ? 'gait' : taskType

    const updrsItem = UPDRS_ITEMS[normalizedType] || 'Motor Assessment'
    const date = analysisDate ? new Date(analysisDate).toLocaleDateString('ko-KR') : new Date().toLocaleDateString('ko-KR')

    // Generate Objective section - concise 3 lines
    const generateObjective = (): string => {
        const lines: string[] = []

        // Line 1: UPDRS Score
        if (updrsScore) {
            const rawScore = updrsScore.total_score ?? updrsScore.score
            const scoreValue = rawScore !== undefined ? Math.round(rawScore) : '-'
            lines.push(`UPDRS-III ${updrsItem}: ${scoreValue}/4`)
        }

        // Line 2-3: Top 2 key metrics only
        if (metrics) {
            const keyMetrics: string[] = []

            if (normalizedType === 'finger_tapping') {
                // Finger tapping: speed and fatigue are most important
                if (metrics.tapping_speed !== undefined) {
                    keyMetrics.push(`Tap ${metrics.tapping_speed.toFixed(1)}Hz`)
                }
                if (metrics.amplitude_mean !== undefined) {
                    keyMetrics.push(`Amp ${metrics.amplitude_mean.toFixed(2)}`)
                }
                if (metrics.fatigue_rate !== undefined) {
                    keyMetrics.push(`Fatigue ${metrics.fatigue_rate.toFixed(0)}%`)
                }
            } else {
                // Gait: velocity and stride are most important
                if (metrics.velocity_mean !== undefined) {
                    keyMetrics.push(`Vel ${metrics.velocity_mean.toFixed(2)}m/s`)
                }
                if (metrics.stride_length !== undefined) {
                    keyMetrics.push(`Stride ${metrics.stride_length.toFixed(2)}m`)
                }
                if (metrics.cadence !== undefined) {
                    keyMetrics.push(`Cad ${metrics.cadence.toFixed(0)}/min`)
                }
            }

            if (keyMetrics.length > 0) {
                lines.push(keyMetrics.slice(0, 3).join(', '))
            }
        }

        return lines.join('\n')
    }

    // Generate Assessment section - concise 2 lines
    const generateAssessment = (): string => {
        const lines: string[] = []

        // Line 1: Severity
        if (updrsScore) {
            const severityTerm = SEVERITY_TERMS[updrsScore.severity] || updrsScore.severity
            lines.push(severityTerm)
        }

        if (normalizedType === 'finger_tapping' && performabilityAssessment) {
            const statusMap: Record<string, string> = {
                performable: 'task performable',
                uncertain: 'performability borderline',
                non_performable_or_near_impossible: 'task near-impossible',
                unscorable_due_to_tracking: 'video unscorable',
            }
            lines.push(statusMap[performabilityAssessment.status] || performabilityAssessment.status)
        }

        // Line 2: Key findings (max 2)
        if (metrics) {
            const findings: string[] = []

            if (normalizedType === 'finger_tapping') {
                if (metrics.fatigue_rate && metrics.fatigue_rate > 20) findings.push('fatigue(+)')
                if (metrics.rhythm_variability && metrics.rhythm_variability > 10) findings.push('rhythm var(+)')
                if (metrics.hesitation_count && metrics.hesitation_count > 2) findings.push('hesitation(+)')
            } else {
                if (metrics.stride_variability && metrics.stride_variability > 5) findings.push('gait var(+)')
                if (metrics.arm_swing_asymmetry && metrics.arm_swing_asymmetry > 10) findings.push('arm asym(+)')
            }

            if (findings.length > 0) {
                lines.push(findings.slice(0, 2).join(', '))
            }
        }

        return lines.join('\n')
    }

    // Generate full SOAP note
    const generateSOAPNote = (): string => {
        const objective = generateObjective()
        const assessment = generateAssessment()
        const taskName = normalizedType === 'finger_tapping' ? 'FT' : 'Gait'

        if (format === 'compact') {
            // Compact single-line format for EMR
            const rawScore = updrsScore?.total_score ?? updrsScore?.score
            const score = rawScore !== undefined ? Math.round(rawScore) : '-'
            return `[${taskName}] UPDRS ${score}/4. ${objective.split('\n').slice(1).join(', ')}. ${assessment.split('\n')[0]}`
        }

        // Full format - still concise
        return `[${date}] ${taskName} Assessment
[O] ${objective.replace(/\n/g, ' | ')}
[A] ${assessment.replace(/\n/g, ' | ')}${scoreAdvisory ? ` | ${scoreAdvisory.summary}` : ''}`
    }

    const copyText = async (text: string, target: "note" | "objective" | "assessment") => {
        await navigator.clipboard.writeText(text)
        setCopiedTarget(target)
        window.setTimeout(() => setCopiedTarget((current) => current === target ? null : current), 2000)
    }

    const soapNote = generateSOAPNote()
    const objective = generateObjective()
    const assessment = generateAssessment()
    const formatLabel = format === 'compact' ? '한 줄 요약' : '상세 SOAP 기록'

    return (
        <div className="space-y-5">
            <section className="rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-card to-card p-5 md:p-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                        <div className="flex flex-wrap items-center gap-2">
                            <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary">
                                <ClipboardList className="h-3.5 w-3.5" />
                                SOAP 기록 초안
                            </span>
                            <span className="text-xs text-muted-foreground">EMR 복사용</span>
                        </div>
                        <h2 className="mt-3 text-xl font-semibold tracking-tight md:text-2xl">기록할 내용을 확인하고 복사하세요</h2>
                        <p className="mt-1 text-sm text-muted-foreground">자동 생성된 연구용 초안입니다. EMR 반영 전 담당자가 내용을 검토해야 합니다.</p>
                    </div>
                    <div className="inline-flex rounded-xl border border-border bg-background p-1" role="group" aria-label="SOAP 기록 형식">
                        <Button
                            variant={format === 'compact' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => setFormat('compact')}
                            aria-pressed={format === 'compact'}
                        >
                            한 줄 요약
                        </Button>
                        <Button
                            variant={format === 'full' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => setFormat('full')}
                            aria-pressed={format === 'full'}
                        >
                            상세 기록
                        </Button>
                    </div>
                </div>
            </section>

            <Card className="overflow-hidden border-primary/25 shadow-sm">
                <CardHeader className="border-b border-border/70 bg-muted/30 pb-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                            <CardTitle className="flex items-center gap-2 text-base">
                                <FileText className="h-4 w-4 text-primary" />
                                {formatLabel}
                            </CardTitle>
                            <p className="mt-1 text-xs text-muted-foreground">
                                {format === 'compact' ? 'EMR에 바로 붙여 넣을 수 있는 짧은 기록입니다.' : '객관적 관찰과 평가 초안을 함께 복사합니다.'}
                            </p>
                        </div>
                        <Button variant="default" size="sm" onClick={() => copyText(soapNote, 'note')} className="gap-2">
                            {copiedTarget === 'note' ? <><Check className="h-4 w-4" />복사됨</> : <><Copy className="h-4 w-4" />기록 복사</>}
                        </Button>
                    </div>
                </CardHeader>
                <CardContent className="space-y-3 pt-5">
                    <pre className="whitespace-pre-wrap break-words rounded-xl border border-border bg-background p-4 font-mono text-sm leading-6 text-foreground">
                        {soapNote}
                    </pre>
                    <div className="flex items-start gap-2 text-xs text-muted-foreground">
                        <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                        복사한 초안은 환자 상태와 영상 품질을 확인한 뒤 EMR에 반영하세요.
                    </div>
                </CardContent>
            </Card>

            <section aria-label="SOAP 항목별 검토">
                <div className="mb-3">
                    <h3 className="text-base font-semibold">항목별 검토</h3>
                    <p className="mt-1 text-xs text-muted-foreground">필요한 부분만 따로 복사하거나, 최종 기록 전에 문구를 확인할 수 있습니다.</p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                    <Card className="border-border/80">
                        <CardHeader className="pb-3">
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <span className="rounded-md bg-primary/10 px-2 py-1 text-xs font-bold text-primary">O</span>
                                    <CardTitle className="mt-2 text-base">Objective</CardTitle>
                                    <p className="mt-1 text-xs text-muted-foreground">객관적 측정값</p>
                                </div>
                                <Button variant="outline" size="sm" className="gap-1.5" onClick={() => copyText(objective, 'objective')}>
                                    {copiedTarget === 'objective' ? <Check className="h-3.5 w-3.5 text-primary" /> : <Copy className="h-3.5 w-3.5" />}
                                    {copiedTarget === 'objective' ? '복사됨' : '복사'}
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <pre className="min-h-24 whitespace-pre-wrap break-words rounded-lg bg-muted/70 p-3 font-mono text-xs leading-5 text-foreground">{objective}</pre>
                        </CardContent>
                    </Card>

                    <Card className="border-border/80">
                        <CardHeader className="pb-3">
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <span className="rounded-md bg-amber-500/10 px-2 py-1 text-xs font-bold text-amber-700 dark:text-amber-400">A</span>
                                    <CardTitle className="mt-2 text-base">Assessment</CardTitle>
                                    <p className="mt-1 text-xs text-muted-foreground">자동 생성된 해석 초안</p>
                                </div>
                                <Button variant="outline" size="sm" className="gap-1.5" onClick={() => copyText(assessment, 'assessment')}>
                                    {copiedTarget === 'assessment' ? <Check className="h-3.5 w-3.5 text-primary" /> : <Copy className="h-3.5 w-3.5" />}
                                    {copiedTarget === 'assessment' ? '복사됨' : '복사'}
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <pre className="min-h-24 whitespace-pre-wrap break-words rounded-lg bg-muted/70 p-3 font-mono text-xs leading-5 text-foreground">{assessment}</pre>
                        </CardContent>
                    </Card>
                </div>
            </section>

        </div>
    )
}
