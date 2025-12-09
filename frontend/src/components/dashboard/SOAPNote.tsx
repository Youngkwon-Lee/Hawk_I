"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Copy, Check, FileText, ClipboardList } from "lucide-react"

interface SOAPNoteProps {
    taskType: string
    metrics?: Record<string, number>
    updrsScore?: {
        score: number
        severity: string
        confidence?: number
    }
    aiInterpretation?: string
    patientId?: string
    analysisDate?: string
}

// Reference ranges for metrics (mean ± std from normal population)
const REFERENCE_RANGES: Record<string, Record<string, { mean: number; std: number; unit: string; name: string }>> = {
    finger_tapping: {
        tapping_speed: { mean: 2.4, std: 0.6, unit: 'Hz', name: 'Tapping frequency' },
        amplitude_mean: { mean: 1.5, std: 0.3, unit: '', name: 'Amplitude ratio' },
        rhythm_variability: { mean: 4.0, std: 4.2, unit: '%', name: 'Rhythm CV' },
        fatigue_rate: { mean: 10, std: 8, unit: '%', name: 'Fatigue index' },
        hesitation_count: { mean: 1.0, std: 1.0, unit: '', name: 'Hesitations' },
    },
    gait: {
        velocity_mean: { mean: 1.1, std: 0.2, unit: 'm/s', name: 'Gait velocity' },
        stride_length: { mean: 1.3, std: 0.15, unit: 'm', name: 'Stride length' },
        cadence: { mean: 110, std: 10, unit: 'steps/min', name: 'Cadence' },
        stride_variability: { mean: 3.0, std: 1.5, unit: '%', name: 'Stride CV' },
        arm_swing_asymmetry: { mean: 5, std: 3, unit: '%', name: 'Arm swing asymmetry' },
    }
}

// UPDRS item mapping
const UPDRS_ITEMS: Record<string, string> = {
    finger_tapping: '3.4 Finger Tapping',
    hand_movement: '3.5 Hand Movements',
    gait: '3.10 Gait',
    leg_agility: '3.8 Leg Agility'
}

// Medical terminology for severity
const SEVERITY_TERMS: Record<string, string> = {
    'Normal': 'Normal motor function',
    'Slight': 'Minimal bradykinesia',
    'Mild': 'Mild bradykinesia with amplitude decrement',
    'Moderate': 'Moderate bradykinesia with fatigue and hesitations',
    'Severe': 'Severe bradykinesia with marked motor impairment'
}

export function SOAPNote({ taskType, metrics, updrsScore, aiInterpretation, patientId, analysisDate }: SOAPNoteProps) {
    const [copied, setCopied] = React.useState(false)
    const [format, setFormat] = React.useState<'full' | 'compact'>('compact')

    const normalizedType = taskType?.includes('finger') || taskType?.includes('tapping')
        ? 'finger_tapping'
        : taskType?.includes('gait') ? 'gait' : taskType

    const refs = REFERENCE_RANGES[normalizedType] || {}
    const updrsItem = UPDRS_ITEMS[normalizedType] || 'Motor Assessment'
    const date = analysisDate ? new Date(analysisDate).toLocaleDateString('ko-KR') : new Date().toLocaleDateString('ko-KR')

    // Generate Objective section
    const generateObjective = (): string => {
        const lines: string[] = []

        if (metrics) {
            // Key metrics with reference ranges
            for (const [key, ref] of Object.entries(refs)) {
                const value = metrics[key]
                if (value !== undefined) {
                    const refRange = `(ref: ${ref.mean}±${ref.std})`
                    const status = Math.abs(value - ref.mean) > ref.std * 1.5 ? ' *' : ''
                    lines.push(`${ref.name}: ${value.toFixed(2)}${ref.unit} ${refRange}${status}`)
                }
            }

            // Add any other metrics not in reference
            for (const [key, value] of Object.entries(metrics)) {
                if (!refs[key] && typeof value === 'number') {
                    const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                    lines.push(`${formattedKey}: ${value.toFixed(2)}`)
                }
            }
        }

        // UPDRS Score
        if (updrsScore) {
            lines.push(`MDS-UPDRS Part III ${updrsItem}: ${updrsScore.score}/4`)
            if (updrsScore.confidence) {
                lines.push(`Model confidence: ${(updrsScore.confidence * 100).toFixed(0)}%`)
            }
        }

        return lines.join('\n')
    }

    // Generate Assessment section
    const generateAssessment = (): string => {
        const lines: string[] = []

        if (updrsScore) {
            // Severity interpretation
            const severityTerm = SEVERITY_TERMS[updrsScore.severity] || updrsScore.severity
            lines.push(severityTerm)

            // Score-specific findings
            if (updrsScore.score === 0) {
                lines.push('No evidence of parkinsonian motor signs')
            } else if (updrsScore.score === 1) {
                lines.push('Subtle slowing, may be normal for age')
            } else if (updrsScore.score === 2) {
                lines.push('Definite slowing with intermittent hesitation')
            } else if (updrsScore.score >= 3) {
                lines.push('Significant motor impairment requiring intervention')
            }
        }

        // Add key findings from metrics
        if (metrics) {
            const findings: string[] = []

            // Finger tapping specific
            if (normalizedType === 'finger_tapping') {
                if (metrics.fatigue_rate && metrics.fatigue_rate > 20) {
                    findings.push('Notable fatigue pattern')
                }
                if (metrics.rhythm_variability && metrics.rhythm_variability > 10) {
                    findings.push('Increased rhythm variability')
                }
                if (metrics.hesitation_count && metrics.hesitation_count > 2) {
                    findings.push('Frequent hesitations observed')
                }
            }

            // Gait specific
            if (normalizedType === 'gait') {
                if (metrics.stride_variability && metrics.stride_variability > 5) {
                    findings.push('Increased gait variability')
                }
                if (metrics.arm_swing_asymmetry && metrics.arm_swing_asymmetry > 10) {
                    findings.push('Asymmetric arm swing')
                }
            }

            if (findings.length > 0) {
                lines.push(findings.join('; '))
            }
        }

        return lines.join('\n')
    }

    // Generate full SOAP note
    const generateSOAPNote = (): string => {
        const objective = generateObjective()
        const assessment = generateAssessment()

        if (format === 'compact') {
            // Compact single-line format for EMR
            const taskName = normalizedType === 'finger_tapping' ? 'Finger Tapping' : 'Gait'
            const score = updrsScore?.score ?? '-'
            const severity = updrsScore?.severity ?? '-'
            return `[${taskName}] UPDRS ${score}/4 (${severity}). ${assessment.split('\n')[0]}`
        }

        // Full format
        return `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOTOR ASSESSMENT - ${date}
Task: ${normalizedType === 'finger_tapping' ? 'Finger Tapping' : 'Gait Analysis'}
${patientId ? `Patient ID: ${patientId}` : ''}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[O] OBJECTIVE
${objective}

[A] ASSESSMENT
${assessment}

[P] PLAN
- Continue monitoring
- Follow-up assessment recommended

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by HawkEye AI Motor Analysis
`
    }

    const handleCopy = async () => {
        const note = generateSOAPNote()
        await navigator.clipboard.writeText(note)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    const soapNote = generateSOAPNote()

    return (
        <div className="space-y-4">
            {/* Format Toggle */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <ClipboardList className="h-5 w-5" />
                        SOAP Note
                    </h2>
                    <p className="text-sm text-muted-foreground">의무기록용 복사 형식</p>
                </div>
                <div className="flex gap-2">
                    <Button
                        variant={format === 'compact' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setFormat('compact')}
                    >
                        간략
                    </Button>
                    <Button
                        variant={format === 'full' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setFormat('full')}
                    >
                        전체
                    </Button>
                </div>
            </div>

            {/* Note Preview */}
            <Card className="bg-slate-900 border-slate-700">
                <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                        <CardTitle className="text-sm flex items-center gap-2">
                            <FileText className="h-4 w-4" />
                            {format === 'compact' ? '간략 형식' : '전체 형식'}
                        </CardTitle>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleCopy}
                            className="gap-2"
                        >
                            {copied ? (
                                <>
                                    <Check className="h-4 w-4 text-green-500" />
                                    복사됨
                                </>
                            ) : (
                                <>
                                    <Copy className="h-4 w-4" />
                                    복사
                                </>
                            )}
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    <pre className="whitespace-pre-wrap font-mono text-sm text-slate-300 bg-slate-950 p-4 rounded-lg overflow-x-auto">
                        {soapNote}
                    </pre>
                </CardContent>
            </Card>

            {/* Quick Copy Sections */}
            <div className="grid md:grid-cols-2 gap-4">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm">[O] Objective</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <pre className="whitespace-pre-wrap font-mono text-xs text-muted-foreground bg-secondary/50 p-3 rounded">
                            {generateObjective()}
                        </pre>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="mt-2 w-full"
                            onClick={async () => {
                                await navigator.clipboard.writeText(generateObjective())
                            }}
                        >
                            <Copy className="h-3 w-3 mr-1" />
                            Objective만 복사
                        </Button>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm">[A] Assessment</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <pre className="whitespace-pre-wrap font-mono text-xs text-muted-foreground bg-secondary/50 p-3 rounded">
                            {generateAssessment()}
                        </pre>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="mt-2 w-full"
                            onClick={async () => {
                                await navigator.clipboard.writeText(generateAssessment())
                            }}
                        >
                            <Copy className="h-3 w-3 mr-1" />
                            Assessment만 복사
                        </Button>
                    </CardContent>
                </Card>
            </div>

            {/* Usage Guide */}
            <Card className="bg-blue-950/30 border-blue-900/50">
                <CardContent className="p-4">
                    <h3 className="font-medium text-blue-300 mb-2">사용 가이드</h3>
                    <ul className="text-sm text-blue-200/80 space-y-1">
                        <li>• <strong>간략 형식</strong>: EMR 한 줄 기록용</li>
                        <li>• <strong>전체 형식</strong>: 상세 의무기록용</li>
                        <li>• <strong>*</strong> 표시: 정상 범위 벗어남 (1.5 SD 이상)</li>
                        <li>• ref: 정상군 평균±표준편차</li>
                    </ul>
                </CardContent>
            </Card>
        </div>
    )
}
