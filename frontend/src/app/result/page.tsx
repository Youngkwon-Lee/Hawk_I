"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"
import { useAnalysisStore } from "@/store/analysisStore"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Badge } from "@/components/ui/Badge"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { SummaryCard } from "@/components/dashboard/SummaryCard"
import { MetricsTable, MetricRow } from "@/components/dashboard/MetricsTable"
import { VideoPlayer, type Marker } from "@/components/dashboard/VideoPlayer"
import { AIInterpretation } from "@/components/dashboard/AIInterpretation"
import { MedicationTimeline } from "@/components/dashboard/MedicationTimeline"
import { PopulationComparison } from "@/components/dashboard/PopulationComparison"
import {
    Activity,
    AlertTriangle,
    BarChart3,
    Brain,
    CheckCircle2,
    ChevronRight,
    CircleSlash,
    Database,
    Download,
    FileText,
    HelpCircle,
    LayoutDashboard,
    Pill,
    PlayCircle,
    Share2,
    TableProperties,
    UsersRound,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { apiUrl, establishMediaSession, getAnalysisResult, type AnalysisResult, type FingerPerformabilityAssessment, type FingerTappingMetrics, type GaitMetrics, type TimelineEvent } from "@/lib/services/api"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"
import { ReasoningLogViewer } from "@/components/dashboard/ReasoningLogViewer"
import { JointAngleChart } from "@/components/dashboard/JointAngleChart"
import { SymmetryChart } from "@/components/dashboard/SymmetryChart"
import { SpeedProfileChart } from "@/components/dashboard/SpeedProfileChart"
import { SOAPNote } from "@/components/dashboard/SOAPNote"

const DEBUG_LOGS = process.env.NODE_ENV !== "production"

// Mock Data - Gait (PD4T 기준)
const GAIT_METRICS: MetricRow[] = [
    { label: "보행 속도", value: "0.75 m/s", unit: "", change: "-2%", status: "good", normalRange: "0.55-0.95" },
    { label: "보행률 (Cadence)", value: "140", unit: "steps/min", change: "+1%", status: "good", normalRange: "120-160" },
    { label: "보폭 길이", value: "0.33", unit: "(정규화)", change: "-8%", status: "good", normalRange: "0.25-0.45" },
    { label: "팔 흔들기 비대칭", value: "12", unit: "%", change: "+5%", status: "good", normalRange: "<20" },
]

// Mock Data - Finger
const FINGER_METRICS: MetricRow[] = [
    { label: "태핑 속도", value: "3.2 Hz", unit: "", change: "-5%", status: "warning", normalRange: "3.0-6.0" },
    { label: "진폭 (Amplitude)", value: "4.5 cm", unit: "", change: "-10%", status: "bad", normalRange: ">0.8" },
    { label: "주저함", value: "3", unit: "회", change: "+1", status: "warning", normalRange: "≤2" },
    { label: "피로율", value: "12", unit: "%", change: "+3%", status: "bad", normalRange: "<20" },
]

function getPerformabilityPresentation(assessment: FingerPerformabilityAssessment) {
    switch (assessment.status) {
        case "performable":
            return {
                label: "수행 가능",
                icon: CheckCircle2,
                badgeVariant: "default" as const,
                cardClassName: "border-emerald-500/20 bg-emerald-500/5",
                iconClassName: "text-emerald-500",
                note: null,
            }
        case "non_performable_or_near_impossible":
            return {
                label: "수행 곤란",
                icon: CircleSlash,
                badgeVariant: "destructive" as const,
                cardClassName: "border-rose-500/20 bg-rose-500/5",
                iconClassName: "text-rose-500",
                note: "자동 점수는 참고용으로만 해석하고, 수기 검토를 우선 권장합니다.",
            }
        case "unscorable_due_to_tracking":
            return {
                label: "영상 판독 불가",
                icon: AlertTriangle,
                badgeVariant: "destructive" as const,
                cardClassName: "border-amber-500/20 bg-amber-500/5",
                iconClassName: "text-amber-500",
                note: "손 추적 또는 영상 품질 문제로 자동 점수 신뢰도가 낮습니다.",
            }
        default:
            return {
                label: "판정 보류",
                icon: HelpCircle,
                badgeVariant: "outline" as const,
                cardClassName: "border-sky-500/20 bg-sky-500/5",
                iconClassName: "text-sky-500",
                note: "일부 tapping signal은 있지만 경계 케이스여서 수기 확인을 권장합니다.",
            }
    }
}

function formatPerformabilityTrigger(trigger: string): string {
    const triggerMap: Record<string, string> = {
        very_slow_tapping_speed: "매우 느린 태핑 속도",
        slow_tapping_speed: "느린 태핑 속도",
        low_peak_velocity: "낮은 최대 속도",
        reduced_peak_velocity: "감소된 최대 속도",
        high_rhythm_variability: "큰 리듬 변동성",
        moderate_post_onset_pause: "중등도 pause",
        long_post_onset_pause: "긴 pause",
        halt_present: "halt 존재",
        multiple_halts: "반복 halt",
        high_second_half_variability: "후반부 변동성 증가",
        strong_velocity_drop: "강한 속도 저하",
        low_detection_rate: "낮은 추적률",
        almost_no_detected_taps: "탭 검출 부족",
        no_velocity_signal: "속도 신호 부족",
        no_effective_motor_signal: "유효한 움직임 신호 부족",
    }
    return triggerMap[trigger] || trigger.replaceAll("_", " ")
}

// Helper function to convert backend metrics to frontend display format
function convertFingerMetricsToRows(metrics: FingerTappingMetrics): MetricRow[] {
    // Determine status based on clinical normal ranges
    const getTappingSpeedStatus = (speed: number) => {
        if (speed >= 3.0 && speed <= 6.0) return "good"
        if ((speed >= 2.0 && speed < 3.0) || (speed > 6.0 && speed <= 7.0)) return "warning"
        return "bad"
    }

    const getAmplitudeStatus = (amplitude: number) => {
        // Normalized by index finger length (dimensionless)
        if (amplitude > 0.8) return "good"
        if (amplitude >= 0.4) return "warning"
        return "bad"
    }

    const getHesitationStatus = (hesitation: number) => {
        if (hesitation <= 2) return "good"
        if (hesitation <= 5) return "warning"
        return "bad"
    }

    const getFatigueStatus = (fatigue: number) => {
        if (fatigue < 20) return "good"
        if (fatigue < 40) return "warning"
        return "bad"
    }

    return [
        {
            label: "태핑 속도",
            value: metrics.tapping_speed.toFixed(2),
            unit: "Hz",
            normalRange: "3.0-6.0",
            status: getTappingSpeedStatus(metrics.tapping_speed)
        },
        {
            label: "진폭 (Amplitude)",
            value: metrics.amplitude_mean.toFixed(2),
            unit: "×finger",
            normalRange: ">0.8",
            status: getAmplitudeStatus(metrics.amplitude_mean)
        },
        {
            label: "주저함",
            value: metrics.hesitation_count.toString(),
            unit: "회",
            normalRange: "≤2",
            status: getHesitationStatus(metrics.hesitation_count)
        },
        {
            label: "피로율",
            value: metrics.fatigue_rate.toFixed(1),
            unit: "%",
            normalRange: "<20",
            status: getFatigueStatus(metrics.fatigue_rate)
        },
        {
            label: "총 탭 수",
            value: metrics.total_taps.toString(),
            unit: "",
            normalRange: "-",
            status: "neutral"
        },
    ]
}

function convertGaitMetricsToRows(metrics: GaitMetrics): MetricRow[] {
    // Status based on PD4T dataset (Score 0 = normal)
    // walking_speed: 0.76 ± 0.13, cadence: 140 ± 11, stride_length: 0.33 ± 0.05
    const getSpeedStatus = (speed: number) => {
        // PD4T normal: 0.76 ± 0.13 (range ~0.55-0.95)
        if (speed >= 0.55 && speed <= 0.95) return "good"
        if (speed >= 0.40 && speed < 0.55) return "warning"
        return "bad"
    }

    const getCadenceStatus = (cadence: number) => {
        // PD4T normal: 140 ± 11 (range ~120-160)
        if (cadence >= 120 && cadence <= 160) return "good"
        if ((cadence >= 100 && cadence < 120) || (cadence > 160 && cadence <= 180)) return "warning"
        return "bad"
    }

    const getStrideLengthStatus = (stride: number) => {
        // PD4T normal: 0.33 ± 0.05 (range ~0.25-0.45) - normalized units
        if (stride >= 0.25 && stride <= 0.45) return "good"
        if ((stride >= 0.18 && stride < 0.25) || (stride > 0.45 && stride <= 0.55)) return "warning"
        return "bad"
    }

    const getAsymmetryStatus = (asymmetry: number) => {
        // PD4T normal: 11.7 ± 8.4
        if (asymmetry < 20) return "good"
        if (asymmetry < 35) return "warning"
        return "bad"
    }

    return [
        {
            label: "보행 속도",
            value: metrics.walking_speed.toFixed(2),
            unit: "m/s",
            normalRange: "0.55-0.95",
            status: getSpeedStatus(metrics.walking_speed)
        },
        {
            label: "보행률 (Cadence)",
            value: metrics.cadence.toFixed(0),
            unit: "steps/min",
            normalRange: "120-160",
            status: getCadenceStatus(metrics.cadence)
        },
        {
            label: "보폭 길이",
            value: metrics.stride_length.toFixed(2),
            unit: "(정규화)",
            normalRange: "0.25-0.45",
            status: getStrideLengthStatus(metrics.stride_length)
        },
        {
            label: "팔 흔들기 비대칭",
            value: metrics.arm_swing_asymmetry.toFixed(1),
            unit: "%",
            normalRange: "<20",
            status: getAsymmetryStatus(metrics.arm_swing_asymmetry)
        },
    ]
}

export default function ResultPage() {
    return (
        <React.Suspense fallback={<div>Loading...</div>}>
            <ResultContent />
        </React.Suspense>
    )
}

function ResultContent() {
    const searchParams = useSearchParams()
    const [activeTab, setActiveTab] = React.useState("dashboard")
    const [showScoreDetails, setShowScoreDetails] = React.useState(false)

    // Zustand store for analysis state management
    const { result: analysisResult, setResult, error, setError, clearResult } = useAnalysisStore()
    const [isLoading, setIsLoading] = React.useState(false)
    const [protectedMediaReady, setProtectedMediaReady] = React.useState(false)

    // Fetch from API if id is present in URL and result doesn't match
    React.useEffect(() => {
        const urlId = searchParams.get("id") || searchParams.get("analysisId")

        // Check if we need to fetch: URL has id AND (no result OR result id doesn't match)
        const storedId = analysisResult?.id || analysisResult?.video_id
        const needsFetch = urlId && (!analysisResult || (storedId && storedId !== urlId))

        if (needsFetch) {
            // Clear old result if id mismatch
            if (storedId && storedId !== urlId) {
                clearResult()
            }

            setIsLoading(true)
            const loadResult = async () => {
                const supabase = getSupabaseBrowserClient()
                const { data } = supabase
                    ? await supabase.auth.getSession()
                    : { data: { session: null } }
                return getAnalysisResult(urlId, data.session?.access_token)
            }
            loadResult()
                .then(data => {
                    // Ensure result has id for future comparisons
                    const resultWithId = { ...data, id: urlId }
                    setResult(resultWithId)
                })
                .catch(err => {
                    console.error("Error fetching result:", err)
                    setError(err.message)
                })
                .finally(() => setIsLoading(false))
        }
    }, [searchParams, analysisResult, setResult, setError, clearResult])

    const isPatientLinked = Boolean(analysisResult?.physio_context?.subject_person_id)
    React.useEffect(() => {
        let cancelled = false
        if (!isPatientLinked) {
            setProtectedMediaReady(true)
            return () => { cancelled = true }
        }

        setProtectedMediaReady(false)
        const prepareMedia = async () => {
            const supabase = getSupabaseBrowserClient()
            const { data } = supabase
                ? await supabase.auth.getSession()
                : { data: { session: null } }
            const accessToken = data.session?.access_token
            if (!accessToken) throw new Error("Protected media requires sign-in")
            await establishMediaSession(accessToken)
            if (!cancelled) setProtectedMediaReady(true)
        }
        prepareMedia().catch((mediaError) => {
            if (!cancelled) {
                setError(mediaError instanceof Error ? mediaError.message : "Protected media unavailable")
            }
        })
        return () => { cancelled = true }
    }, [isPatientLinked, analysisResult?.id, analysisResult?.video_id, setError])

    React.useEffect(() => {
        if (!DEBUG_LOGS) return

        console.log('=== RESULT PAGE DEBUG ===')
        console.log('Analysis result:', analysisResult)
        console.log('Has metrics?', !!analysisResult?.metrics)
        console.log('Has UPDRS?', !!analysisResult?.updrs_score)
        console.log('Has AI interpretation?', !!analysisResult?.ai_interpretation)
        console.log('Video type:', analysisResult?.video_type)
    }, [analysisResult])



    // Determine type from sessionStorage or URL
    const type = analysisResult?.video_type || searchParams.get("type") || "gait"
    const isFinger = type === "finger_tapping" || type === "finger"  // hand_movement not implemented yet

    // Use real metrics from backend if available, otherwise fall back to mock
    let metrics: MetricRow[] = isFinger ? FINGER_METRICS : GAIT_METRICS
    if (analysisResult?.metrics) {
        if (isFinger && 'tapping_speed' in analysisResult.metrics) {
            metrics = convertFingerMetricsToRows(analysisResult.metrics as FingerTappingMetrics)
        } else if (!isFinger && 'walking_speed' in analysisResult.metrics) {
            metrics = convertGaitMetricsToRows(analysisResult.metrics as GaitMetrics)
        }
    }
    const title = isFinger ? "손가락 태핑 분석" : "보행 분석"
    const detectionMode = analysisResult?.auto_detected === false
        ? "직접 선택"
        : "AI 자동 감지"
    const detectionSummary = analysisResult
        ? analysisResult.auto_detected === false
            ? `• 분석 유형: ${title} · ${detectionMode}`
            : `• 분석 유형: ${title} · ${detectionMode} (신뢰도: ${(analysisResult.confidence * 100).toFixed(0)}%)`
        : null

    // Get UPDRS score from backend result
    // Backend returns total_score, not score
    const score = analysisResult?.updrs_score?.total_score?.toString() ||
                  analysisResult?.updrs_score?.score?.toString() ||
                  "N/A"
    const severity = analysisResult?.updrs_score?.severity || "Unknown"

    // Get individual scoring method results (Rule, ML, Ensemble)
    const scoringMethod = analysisResult?.updrs_score?.method || "rule"
    const ruleScore = analysisResult?.updrs_score?.details?.rule
    const mlScore = analysisResult?.updrs_score?.details?.ml
    const confidence = analysisResult?.updrs_score?.confidence
    const performabilityAssessment = isFinger ? analysisResult?.performability_assessment : null
    const scoreAdvisory = isFinger ? analysisResult?.score_advisory : null
    const supabaseObservation = analysisResult?.integrations?.supabase_observation
    const isParkiCheckDelegated = supabaseObservation?.delegated === true
    const subjectDisplayName = analysisResult?.physio_context?.subject_display_name
    const analysisTrace = analysisResult?.analysis_trace

    // Use original video with canvas overlay (Method A - Frontend Canvas Overlay)
    // Priority: original video URL (from API) > skeleton video URL (legacy) > sample video
    const skeletonData = analysisResult?.skeleton_data
    const hasKeypointsData = Array.isArray(skeletonData?.keypoints) &&
        skeletonData.keypoints.length > 0

    // Use skeleton video generated by backend (reliable approach)
    const mediaAssetUrl = (
        result: AnalysisResult | null | undefined,
        asset: string,
        fallback?: string,
    ): string | null => {
        if (!result || !fallback) return null
        const analysisId = result.id || result.video_id
        if (result.physio_context?.subject_person_id) {
            return protectedMediaReady && analysisId
                ? `/api/media/${encodeURIComponent(analysisId)}/${asset}`
                : null
        }
        return apiUrl(fallback)
    }
    const skeletonVideoUrl = mediaAssetUrl(
        analysisResult,
        "skeleton_video",
        skeletonData?.skeleton_video_url,
    )

    const videoSrc = skeletonVideoUrl || undefined

    // Keypoints data for canvas overlay
    // When using skeleton video (backend-generated), don't use canvas overlay (avoid double skeleton)
    const keypointsData = (hasKeypointsData && !skeletonVideoUrl)
        ? skeletonData?.keypoints
        : undefined
    const keypointsFps = skeletonData?.fps || 30

    React.useEffect(() => {
        if (!DEBUG_LOGS) return

        console.log('=== VIDEO DEBUG ===')
        console.log('Video source:', videoSrc)
        console.log('Skeleton video URL (from API):', skeletonVideoUrl)
    }, [videoSrc, skeletonVideoUrl])

    // Map backend events to UI markers
    const markers = React.useMemo<Marker[]>(() => {
        if (analysisResult?.events && analysisResult.events.length > 0) {
            return analysisResult.events.map((event: TimelineEvent) => {
                let type: Marker["type"] = "info"
                const lowerType = event.type.toLowerCase()

                if (lowerType.includes("freeze") || lowerType.includes("hesitation") || lowerType.includes("stop")) type = "warning"
                else if (lowerType.includes("turn")) type = "info"
                else if (lowerType.includes("good") || lowerType.includes("normal")) type = "good"
                else if (lowerType.includes("bad") || lowerType.includes("abnormal")) type = "warning"

                return {
                    time: event.timestamp,
                    label: event.description,
                    type: type
                }
            })
        }
        // Return empty array if no events - don't show mock data
        return []
    }, [analysisResult])

    // Convert severity to Korean
    const severityKorean = severity === "Normal" ? "정상" :
        severity === "Slight" ? "경미한 증상" :
            severity === "Mild" ? "경도 증상" :
                severity === "Moderate" ? "중등도 증상" :
                    severity === "Severe" ? "중증 증상" : "알 수 없음"

    const reviewSections = [
        { id: "dashboard", label: "대시보드 뷰", hint: "검토 개요", icon: LayoutDashboard },
        { id: "video", label: "영상 분석", hint: "원본·스켈레톤", icon: PlayCircle },
        { id: "raw", label: "원시 데이터", hint: "측정값 확인", icon: TableProperties },
        { id: "visualizations", label: "시각화 분석", hint: "패턴 보기", icon: BarChart3 },
        { id: "timeline", label: "약물 타임라인", hint: "복약 맥락", icon: Pill },
        { id: "comparison", label: "정상군 비교", hint: "참조 분포", icon: UsersRound },
        { id: "reasoning", label: "AI 추론 과정", hint: "근거 검토", icon: Brain },
        { id: "soap", label: "SOAP 노트", hint: "기록 초안", icon: FileText },
    ] as const

    const activeReviewSection = reviewSections.find((section) => section.id === activeTab) || reviewSections[0]

    if (isLoading) return <div className="flex items-center justify-center min-h-screen">Loading analysis result...</div>
    if (error) return <div className="flex items-center justify-center min-h-screen text-red-500">Error: {error}</div>

    return (
        <PageLayout agentPanel={<ChatInterface initialMessages={[
            {
                id: "1",
                role: "agent",
                content: `${title} 결과입니다. UPDRS 점수는 ${score}점으로 ${severityKorean}을 나타냅니다.`,
                timestamp: new Date()
            },
            ...(analysisResult?.ai_interpretation?.summary ? [{
                id: "2",
                role: "agent" as const,
                content: analysisResult.ai_interpretation.summary,
                timestamp: new Date()
            }] : [])
        ]} />}>
            <div className="space-y-6 pb-10">
                {/* Header */}
                <div className="rounded-2xl border border-border bg-card p-5 shadow-sm md:p-6">
                    <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                        <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                                <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary">연구용 분석 기록</span>
                                <span className="text-xs text-muted-foreground">전문의 검토 후 기록에 반영</span>
                            </div>
                            <h1 className="mt-3 text-2xl font-semibold tracking-tight md:text-3xl">{title} 검토</h1>
                            <p className="mt-1 text-sm text-muted-foreground">{detectionSummary || "분석 유형을 확인 중입니다."}</p>
                            {(subjectDisplayName || supabaseObservation?.enabled) && (
                                <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
                                    {subjectDisplayName && (
                                        <span className="rounded-md border border-border bg-background px-2.5 py-1 text-muted-foreground">
                                            대상: {subjectDisplayName}
                                        </span>
                                    )}
                                    {supabaseObservation?.enabled && (
                                        <span className={cn(
                                            "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1",
                                            supabaseObservation.saved || isParkiCheckDelegated
                                                ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400"
                                                : "border-yellow-500/20 bg-yellow-500/5 text-yellow-700 dark:text-yellow-400"
                                        )}>
                                            <Database className="h-4 w-4" />
                                            {supabaseObservation.saved
                                                ? "연구 기록 저장됨"
                                                : isParkiCheckDelegated
                                                    ? "ParkiCheck 기록 연동됨"
                                                    : "연구 기록 저장 대기"}
                                        </span>
                                    )}
                                </div>
                            )}
                            {analysisTrace && (
                                <details className="mt-4 text-xs text-muted-foreground">
                                    <summary className="cursor-pointer select-none hover:text-foreground">연구 추적 ID 보기</summary>
                                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
                                        {analysisTrace.analysis_id && <span>analysis: {analysisTrace.analysis_id}</span>}
                                        {analysisTrace.activity_session_id && <span>session: {analysisTrace.activity_session_id}</span>}
                                        {analysisTrace.observation_id && <span>observation: {analysisTrace.observation_id}</span>}
                                        {analysisTrace.observation_fhir_id && <span>FHIR: {analysisTrace.observation_fhir_id}</span>}
                                    </div>
                                </details>
                            )}
                        </div>
                        <div className="flex shrink-0 flex-wrap gap-2">
                            <Button variant="outline" size="sm" className="gap-2">
                                <Share2 className="h-4 w-4" /> 공유
                            </Button>
                            <Button variant="outline" size="sm" className="gap-2">
                                <Download className="h-4 w-4" /> PDF 내보내기
                            </Button>
                        </div>
                    </div>
                </div>

                {/* Summary Section */}
                <div className="grid gap-4 md:grid-cols-4">
                    <div className="md:col-span-1">
                        <SummaryCard
                            title="연구 추정 점수"
                            value={score}
                            subtext={`UPDRS (0-4) · ${severityKorean}`}
                            status={
                                severity === "Normal" ? "good" :
                                    severity === "Slight" ? "neutral" :
                                        severity === "Mild" ? "warning" :
                                            "bad"
                            }
                        />
                        {scoringMethod === "ensemble" && ruleScore !== undefined && mlScore !== undefined && (
                            <button
                                onClick={() => setShowScoreDetails(!showScoreDetails)}
                                className="w-full mt-1 text-xs text-blue-400 hover:text-blue-300 flex items-center justify-center gap-1"
                            >
                                {showScoreDetails ? "▲ 산출 방식 숨기기" : "▼ 산출 방식 보기"}
                            </button>
                        )}
                    </div>
                    {isFinger && analysisResult?.metrics && 'amplitude_mean' in analysisResult.metrics ? (
                        <>
                            <SummaryCard
                                title="진폭"
                                value={`${analysisResult.metrics.amplitude_mean.toFixed(2)}×`}
                                subtext="검지손가락 대비"
                                className="md:col-span-1"
                            />
                            <SummaryCard
                                title="태핑 속도"
                                value={`${analysisResult.metrics.tapping_speed.toFixed(2)} Hz`}
                                subtext={`총 ${analysisResult.metrics.total_taps}회 탭`}
                                className="md:col-span-1"
                            />
                            <SummaryCard
                                title="피로도"
                                value={`${analysisResult.metrics.fatigue_rate.toFixed(1)}%`}
                                subtext="피로율"
                                status={analysisResult.metrics.fatigue_rate > 20 ? "bad" : "neutral"}
                                className="md:col-span-1"
                            />
                        </>
                    ) : !isFinger && analysisResult?.metrics && 'walking_speed' in analysisResult.metrics ? (
                        <>
                            <SummaryCard
                                title="보폭 길이"
                                value={`${analysisResult.metrics.stride_length.toFixed(2)}m`}
                                className="md:col-span-1"
                            />
                            <SummaryCard
                                title="보행 속도"
                                value={`${analysisResult.metrics.walking_speed.toFixed(2)} m/s`}
                                subtext={`총 ${analysisResult.metrics.step_count}걸음`}
                                className="md:col-span-1"
                            />
                            <SummaryCard
                                title="보행률"
                                value={`${analysisResult.metrics.cadence.toFixed(0)}`}
                                subtext="steps/min"
                                className="md:col-span-1"
                            />
                        </>
                    ) : (
                        <>
                            <SummaryCard
                                title={isFinger ? "진폭" : "보폭 길이"}
                                value="-"
                                subtext="분석 데이터 없음"
                                status="neutral"
                                className="md:col-span-1"
                            />
                            <SummaryCard
                                title={isFinger ? "태핑 속도" : "보행 속도"}
                                value="-"
                                subtext="분석 데이터 없음"
                                className="md:col-span-1"
                            />
                            <SummaryCard
                                title={isFinger ? "피로도" : "보행률"}
                                value="-"
                                subtext="분석 데이터 없음"
                                className="md:col-span-1"
                            />
                        </>
                        )}
                </div>

                {isFinger && scoreAdvisory && (
                    <div
                        className={cn(
                            "rounded-lg border px-4 py-3 text-sm",
                            scoreAdvisory.level === "standard" && "border-emerald-500/20 bg-emerald-500/5",
                            scoreAdvisory.level === "review_recommended" && "border-sky-500/20 bg-sky-500/5",
                            scoreAdvisory.level === "reference_only" && "border-amber-500/20 bg-amber-500/5"
                        )}
                    >
                        <div className="flex flex-wrap items-center gap-2">
                            <Badge
                                variant={
                                    scoreAdvisory.level === "standard"
                                        ? "default"
                                        : scoreAdvisory.level === "review_recommended"
                                            ? "outline"
                                            : "destructive"
                                }
                            >
                                {scoreAdvisory.level === "standard"
                                    ? "표준 해석"
                                    : scoreAdvisory.level === "review_recommended"
                                        ? "검토 권장"
                                        : "참고용"}
                            </Badge>
                            <span className="text-muted-foreground">{scoreAdvisory.summary}</span>
                        </div>
                    </div>
                )}

                {/* Scoring Method Details - Show Rule/ML/Ensemble breakdown */}
                {showScoreDetails && scoringMethod === "ensemble" && ruleScore !== undefined && mlScore !== undefined && (
                    <Card className="bg-slate-900 border-slate-700 animate-in fade-in slide-in-from-top-2">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm flex items-center gap-2">
                                <Brain className="h-4 w-4 text-blue-400" />
                                점수 산출 방법 (Scoring Method)
                            </CardTitle>
                            <CardDescription>Rule-based와 ML 모델의 앙상블 결과</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-3 gap-4 text-center">
                                <div className="p-3 rounded-lg bg-slate-800 border border-slate-600">
                                    <div className="text-xs text-slate-400 mb-1">Rule-based</div>
                                    <div className="text-2xl font-bold text-orange-400">{typeof ruleScore === 'number' ? ruleScore.toFixed(1) : ruleScore}</div>
                                    <div className="text-xs text-slate-500">임상 규칙 기반</div>
                                </div>
                                <div className="p-3 rounded-lg bg-slate-800 border border-slate-600">
                                    <div className="text-xs text-slate-400 mb-1">ML Model</div>
                                    <div className="text-2xl font-bold text-purple-400">{typeof mlScore === 'number' ? mlScore.toFixed(1) : mlScore}</div>
                                    <div className="text-xs text-slate-500">머신러닝 예측</div>
                                </div>
                                <div className="p-3 rounded-lg bg-blue-900/50 border border-blue-500">
                                    <div className="text-xs text-blue-300 mb-1">Ensemble</div>
                                    <div className="text-2xl font-bold text-blue-400">{score}</div>
                                    <div className="text-xs text-blue-300/70">신뢰도: {confidence ? (confidence * 100).toFixed(0) : '-'}%</div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {isFinger && performabilityAssessment && (() => {
                    const presentation = getPerformabilityPresentation(performabilityAssessment)
                    const Icon = presentation.icon
                    const evidence = performabilityAssessment.evidence || {}
                    const triggerItems = (performabilityAssessment.triggers || []).slice(0, 4)

                    return (
                        <Card className={cn("border", presentation.cardClassName)}>
                            <CardHeader className="pb-3">
                                <div className="flex items-start justify-between gap-3">
                                    <div className="flex items-start gap-3">
                                        <div className="rounded-full bg-background/70 p-2">
                                            <Icon className={cn("h-5 w-5", presentation.iconClassName)} />
                                        </div>
                                        <div>
                                            <CardTitle className="text-base">Finger Performability Gate</CardTitle>
                                            <CardDescription className="mt-1">
                                                점수 예측 전에 이 finger tapping 과제가 실제로 수행 가능한지 선제적으로 판정합니다.
                                            </CardDescription>
                                        </div>
                                    </div>
                                    <Badge variant={presentation.badgeVariant}>{presentation.label}</Badge>
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                <p className="text-sm text-muted-foreground">{performabilityAssessment.summary}</p>

                                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                                    {typeof evidence.tapping_speed === "number" && (
                                        <div className="rounded-lg border border-border/60 bg-background/60 p-3">
                                            <div className="text-xs text-muted-foreground">Tapping speed</div>
                                            <div className="text-sm font-semibold">{evidence.tapping_speed.toFixed(2)} Hz</div>
                                        </div>
                                    )}
                                    {typeof evidence.peak_velocity_mean === "number" && (
                                        <div className="rounded-lg border border-border/60 bg-background/60 p-3">
                                            <div className="text-xs text-muted-foreground">Peak velocity</div>
                                            <div className="text-sm font-semibold">{evidence.peak_velocity_mean.toFixed(2)}</div>
                                        </div>
                                    )}
                                    {typeof evidence.halt_count === "number" && (
                                        <div className="rounded-lg border border-border/60 bg-background/60 p-3">
                                            <div className="text-xs text-muted-foreground">Halt count</div>
                                            <div className="text-sm font-semibold">{Math.round(evidence.halt_count)}</div>
                                        </div>
                                    )}
                                    {typeof evidence.post_onset_pause_ratio === "number" && (
                                        <div className="rounded-lg border border-border/60 bg-background/60 p-3">
                                            <div className="text-xs text-muted-foreground">Pause ratio</div>
                                            <div className="text-sm font-semibold">{evidence.post_onset_pause_ratio.toFixed(2)}</div>
                                        </div>
                                    )}
                                </div>

                                {triggerItems.length > 0 && (
                                    <div className="flex flex-wrap gap-2">
                                        {triggerItems.map((trigger) => (
                                            <Badge key={trigger} variant="outline" className="bg-background/60">
                                                {formatPerformabilityTrigger(trigger)}
                                            </Badge>
                                        ))}
                                    </div>
                                )}

                                {presentation.note && (
                                    <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-sm text-muted-foreground">
                                        {presentation.note}
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )
                })()}

                {/* Warning Banner - Dynamic based on analysis results */}
                {(() => {
                    const warnings: string[] = [];

                    if (isFinger) {
                        // Finger tapping warnings
                        const ftMetrics = analysisResult?.metrics as { decrement_ratio?: number; fatigue_index?: number } | undefined;
                        if (ftMetrics?.decrement_ratio && ftMetrics.decrement_ratio > 20) {
                            warnings.push(`진폭 감소율이 ${ftMetrics.decrement_ratio.toFixed(1)}%로 측정되었습니다.`);
                        }
                        if (ftMetrics?.fatigue_index && ftMetrics.fatigue_index > 0.3) {
                            warnings.push("후반부 피로 징후가 감지되었습니다.");
                        }
                    } else {
                        // Gait warnings (PD4T 기준)
                        const gaitMetrics = analysisResult?.metrics as { walking_speed?: number; cadence?: number; arm_swing_asymmetry?: number } | undefined;
                        if (gaitMetrics?.walking_speed && gaitMetrics.walking_speed < 0.55) {
                            warnings.push(`보행 속도가 ${gaitMetrics.walking_speed.toFixed(2)} m/s로 정상 범위(0.55-0.95)보다 낮습니다.`);
                        }
                        if (gaitMetrics?.cadence && gaitMetrics.cadence < 120) {
                            warnings.push(`보행률이 ${Math.round(gaitMetrics.cadence)} steps/min으로 정상 범위(120-160)보다 낮습니다.`);
                        }
                        if (gaitMetrics?.arm_swing_asymmetry && gaitMetrics.arm_swing_asymmetry > 20) {
                            warnings.push(`팔 흔들기 비대칭이 ${gaitMetrics.arm_swing_asymmetry.toFixed(1)}%로 기준치(20%)를 초과합니다.`);
                        }
                    }

                    // Only show warning banner if there are actual warnings
                    if (warnings.length === 0) return null;

                    return (
                        <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-4 flex items-start gap-3">
                            <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5 shrink-0" />
                            <div>
                                <h3 className="font-semibold text-yellow-500 text-sm">주의 필요</h3>
                                <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                                    {warnings.map((warning, idx) => (
                                        <li key={idx}>• {warning}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    );
                })()}

                {/* Review navigation */}
                <section aria-label="분석 검토 단계" className="rounded-2xl border border-border bg-card p-3 shadow-sm md:p-4">
                    <div className="mb-3 flex flex-wrap items-end justify-between gap-2 px-1">
                        <div>
                            <p className="text-sm font-semibold">검토 단계</p>
                            <p className="mt-0.5 text-xs text-muted-foreground">데이터 확인부터 기록 초안 작성까지 순서대로 검토합니다.</p>
                        </div>
                        <div className="inline-flex items-center gap-1.5 text-xs font-medium text-primary">
                            {activeReviewSection.label}
                            <ChevronRight className="h-3.5 w-3.5" />
                        </div>
                    </div>
                    <div role="tablist" aria-label="분석 결과 탐색" className="grid grid-cols-2 gap-2 sm:grid-cols-4 2xl:grid-cols-8">
                        {reviewSections.map((section) => {
                            const Icon = section.icon
                            const isActive = activeTab === section.id
                            return (
                                <button
                                    key={section.id}
                                    type="button"
                                    role="tab"
                                    aria-selected={isActive}
                                    onClick={() => setActiveTab(section.id)}
                                    className={cn(
                                        "group min-h-[72px] rounded-xl border px-3 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                                        isActive
                                            ? "border-primary bg-primary text-primary-foreground shadow-sm"
                                            : "border-border bg-background hover:border-primary/40 hover:bg-accent"
                                    )}
                                >
                                    <div className="flex items-center gap-2">
                                        <Icon className={cn("h-4 w-4 shrink-0", isActive ? "text-primary-foreground" : "text-primary")} />
                                        <span className="text-sm font-semibold leading-tight">{section.label}</span>
                                    </div>
                                    <span className={cn("mt-1.5 block text-xs", isActive ? "text-primary-foreground/80" : "text-muted-foreground")}>{section.hint}</span>
                                </button>
                            )
                        })}
                    </div>
                </section>

                {/* Tab Content */}
                {activeTab === "dashboard" && (
                    <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr] animate-in fade-in slide-in-from-bottom-2">
                        <Card className="h-full border-primary/20 bg-gradient-to-br from-primary/5 via-card to-card">
                            <CardHeader>
                                <CardTitle className="text-lg">이번 검사 검토 흐름</CardTitle>
                                <CardDescription>자동 생성 결과를 그대로 확정하지 않고, 근거를 확인한 뒤 기록 초안을 만듭니다.</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {[
                                    ["1", "영상과 추적 품질 확인", "영상 분석"],
                                    ["2", "측정값·패턴 확인", "시각화 분석"],
                                    ["3", "복약·정상군 맥락 확인", "약물 타임라인"],
                                    ["4", "AI 근거 검토 후 기록", "SOAP 노트"],
                                ].map(([step, label, destination]) => {
                                    const target = reviewSections.find((section) => section.label === destination)
                                    return (
                                        <button
                                            key={step}
                                            type="button"
                                            onClick={() => target && setActiveTab(target.id)}
                                            className="flex w-full items-center gap-3 rounded-xl border border-border/70 bg-background/80 p-3 text-left transition-colors hover:border-primary/40 hover:bg-accent"
                                        >
                                            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">{step}</span>
                                            <span className="min-w-0 flex-1 text-sm font-medium">{label}</span>
                                            <span className="hidden text-xs text-muted-foreground sm:block">{destination}</span>
                                            <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                                        </button>
                                    )
                                })}
                            </CardContent>
                        </Card>

                        {/* Right Col: Metrics Table */}
                        <div className="space-y-6">
                            <Card className="h-full">
                                <CardHeader>
                                    <CardTitle className="text-lg">핵심 측정값</CardTitle>
                                    <CardDescription>현재 영상에서 추출된 수치입니다. 최종 임상 평가는 담당자가 확인합니다.</CardDescription>
                                </CardHeader>
                                <CardContent className="p-0">
                                    <MetricsTable data={metrics} />
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                )}

                {activeTab === "video" && (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2">
                        <div className="grid md:grid-cols-3 gap-6">
                            <div className="md:col-span-2">
                                <VideoPlayer
                                    className="w-full shadow-2xl"
                                    videoSrc={videoSrc}
                                    keypointsData={keypointsData}
                                    keypointsFps={keypointsFps}
                                    taskType={isFinger ? "finger" : "gait"}
                                    markers={markers}
                                />
                            </div>
                            <div className="space-y-4">
                                {/* 감지된 이벤트 - 정확도 개선 후 다시 활성화 예정
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-sm">감지된 이벤트</CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-4">
                                        {markers.length > 0 ? (
                                            markers.map((m, i) => (
                                                <div key={i} className="flex items-center justify-between text-sm">
                                                    <span className={m.type === "warning" ? "text-yellow-500" : ""}>
                                                        {`00:0${Math.floor(m.time)} - ${m.label}`}
                                                    </span>
                                                    <Button size="sm" variant="ghost" className="h-6 text-xs">이동</Button>
                                                </div>
                                            ))
                                        ) : (
                                            <p className="text-sm text-muted-foreground">감지된 이벤트가 없습니다.</p>
                                        )}
                                    </CardContent>
                                </Card>
                                */}
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-sm">분석 부위</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-sm text-muted-foreground">
                                            {isFinger
                                                ? "손 관절(엄지, 검지)에 집중 분석. 떨림 분석 포함."
                                                : "하체 관절(엉덩이, 무릎, 발목)에 집중 분석. 상체 흔들림은 정상 범위 내."
                                            }
                                        </p>
                                    </CardContent>
                                </Card>
                            </div>
                        </div>

                        {/* Visualization Maps Gallery */}
                        {analysisResult?.visualization_maps && (
                            <div className="space-y-4">
                                <div>
                                    <h2 className="text-xl font-semibold">시각화 맵</h2>
                                    <p className="text-sm text-muted-foreground">
                                        모션 분석 과정에서 생성된 시각화 맵입니다.
                                    </p>
                                </div>
                                <div className="grid md:grid-cols-3 gap-4">
                                    {analysisResult.visualization_maps.heatmap_url && (
                                        <Card>
                                            <CardHeader>
                                                <CardTitle className="text-sm">Heatmap</CardTitle>
                                                <CardDescription className="text-xs">모션 강도 분포</CardDescription>
                                            </CardHeader>
                                            <CardContent>
                                                <img
                                                    src={mediaAssetUrl(analysisResult, "heatmap", analysisResult.visualization_maps.heatmap_url) || undefined}
                                                    alt="Heatmap"
                                                    className="w-full rounded-lg border"
                                                />
                                            </CardContent>
                                        </Card>
                                    )}
                                    {analysisResult.visualization_maps.temporal_map_url && (
                                        <Card>
                                            <CardHeader>
                                                <CardTitle className="text-sm">Temporal Map</CardTitle>
                                                <CardDescription className="text-xs">시간별 변화 추이</CardDescription>
                                            </CardHeader>
                                            <CardContent>
                                                <img
                                                    src={mediaAssetUrl(analysisResult, "temporal_map", analysisResult.visualization_maps.temporal_map_url) || undefined}
                                                    alt="Temporal Map"
                                                    className="w-full rounded-lg border"
                                                />
                                            </CardContent>
                                        </Card>
                                    )}

                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === "raw" && (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2">
                        <div className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
                            <span className="font-semibold text-foreground">원시 데이터 검토</span>
                            <span className="ml-2">자동 추출 값이며, 연구 분석의 재현성과 수기 검토를 위해 제공합니다.</span>
                        </div>
                        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-lg">추출 측정값</CardTitle>
                                    <CardDescription>화면 요약에 사용된 자동 추출 수치를 단위와 참조 범위로 확인합니다.</CardDescription>
                                </CardHeader>
                                <CardContent className="p-0">
                                    <MetricsTable data={metrics} />
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-lg">분석 메타데이터</CardTitle>
                                    <CardDescription>이 기록의 분석 유형과 연결 상태를 확인합니다.</CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-3 text-sm">
                                    <div className="flex items-center justify-between gap-3 rounded-lg border border-border/70 bg-background p-3">
                                        <span className="text-muted-foreground">분석 유형</span>
                                        <span className="font-medium">{title}</span>
                                    </div>
                                    <div className="flex items-center justify-between gap-3 rounded-lg border border-border/70 bg-background p-3">
                                        <span className="text-muted-foreground">선택 방식</span>
                                        <span className="font-medium">{detectionMode}</span>
                                    </div>
                                    <div className="flex items-center justify-between gap-3 rounded-lg border border-border/70 bg-background p-3">
                                        <span className="text-muted-foreground">연구 기록</span>
                                        <span className="font-medium">{supabaseObservation?.saved || isParkiCheckDelegated ? "연동됨" : "연동 대기"}</span>
                                    </div>
                                    <details className="rounded-lg border border-border/70 bg-background p-3">
                                        <summary className="cursor-pointer font-medium">구조화된 측정값 보기</summary>
                                        <pre className="mt-3 max-h-64 overflow-auto rounded-md bg-muted p-3 text-xs leading-5 text-muted-foreground">
                                            {JSON.stringify(analysisResult?.metrics ?? {}, null, 2)}
                                        </pre>
                                    </details>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                )}

                {activeTab === "visualizations" && (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2">
                        <div className="grid md:grid-cols-2 gap-6">
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm">히트맵 (Heatmap)</CardTitle>
                                    <CardDescription>움직임이 집중된 영역을 시각화합니다.</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center relative overflow-hidden">
                                        {analysisResult?.visualization_urls?.heatmap ? (
                                            <img
                                                src={mediaAssetUrl(analysisResult, "heatmap", analysisResult.visualization_urls.heatmap) || undefined}
                                                alt="Heatmap"
                                                className="object-contain w-full h-full"
                                            />
                                        ) : (
                                            <div className="text-center p-6">
                                                <Activity className="h-10 w-10 text-slate-300 mx-auto mb-2" />
                                                <p className="text-sm text-muted-foreground">히트맵 데이터가 없습니다.</p>
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-sm">시간적 흐름 (Temporal Map)</CardTitle>
                                    <CardDescription>시간에 따른 움직임 변화를 보여줍니다.</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center relative overflow-hidden">
                                        {analysisResult?.visualization_urls?.temporal_map ? (
                                            <img
                                                src={mediaAssetUrl(analysisResult, "temporal_map", analysisResult.visualization_urls.temporal_map) || undefined}
                                                alt="Temporal Map"
                                                className="object-contain w-full h-full"
                                            />
                                        ) : (
                                            <div className="text-center p-6">
                                                <Activity className="h-10 w-10 text-slate-300 mx-auto mb-2" />
                                                <p className="text-sm text-muted-foreground">시간 지도 데이터가 없습니다.</p>
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>


                        </div>

                        {/* Row 2: Chart-based visualizations - Task Type Specific */}
                        {isFinger ? (
                            <>
                                {/* Finger Tapping Charts - 진폭 분석과 리듬 분석만 표시 */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    {/* 탭핑 진폭 분석 */}
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="text-sm">탭핑 진폭 분석</CardTitle>
                                            <CardDescription>시간에 따른 탭핑 진폭 변화</CardDescription>
                                        </CardHeader>
                                        <CardContent>
                                            {analysisResult?.visualization_data?.joint_angles &&
                                             (analysisResult.visualization_data.joint_angles as unknown as Array<{leftAmplitude?: number, rightAmplitude?: number}>).length > 0 ? (
                                                <div className="space-y-4">
                                                    {(() => {
                                                        const ampData = analysisResult.visualization_data.joint_angles as unknown as Array<{time?: number, leftAmplitude?: number, rightAmplitude?: number, avgAmplitude?: number}>
                                                        const avgAmplitudes = ampData.filter(d => d.avgAmplitude != null).map(d => d.avgAmplitude!)
                                                        const avgAmp = avgAmplitudes.length > 0 ? avgAmplitudes.reduce((a, b) => a + b, 0) / avgAmplitudes.length : 0
                                                        const maxAmp = avgAmplitudes.length > 0 ? Math.max(...avgAmplitudes) : 0
                                                        const minAmp = avgAmplitudes.length > 0 ? Math.min(...avgAmplitudes) : 0
                                                        return (
                                                            <div className="grid grid-cols-3 gap-2 text-center">
                                                                <div className="bg-slate-800/50 rounded p-2">
                                                                    <div className="text-xs text-muted-foreground">평균 진폭</div>
                                                                    <div className="font-bold text-blue-400">{avgAmp.toFixed(1)}%</div>
                                                                </div>
                                                                <div className="bg-slate-800/50 rounded p-2">
                                                                    <div className="text-xs text-muted-foreground">최대</div>
                                                                    <div className="font-bold text-green-400">{maxAmp.toFixed(1)}%</div>
                                                                </div>
                                                                <div className="bg-slate-800/50 rounded p-2">
                                                                    <div className="text-xs text-muted-foreground">최소</div>
                                                                    <div className="font-bold text-red-400">{minAmp.toFixed(1)}%</div>
                                                                </div>
                                                            </div>
                                                        )
                                                    })()}
                                                    <p className="text-xs text-muted-foreground text-center">
                                                        {(analysisResult.visualization_data.joint_angles as unknown as Array<unknown>).length}개 샘플 분석됨
                                                    </p>
                                                </div>
                                            ) : (
                                                <div className="h-32 flex items-center justify-center">
                                                    <p className="text-sm text-muted-foreground">진폭 데이터 없음</p>
                                                </div>
                                            )}
                                        </CardContent>
                                    </Card>
                                    {/* 탭핑 리듬 분석 */}
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="text-sm">탭핑 리듬 분석</CardTitle>
                                            <CardDescription>탭핑 간격 및 규칙성</CardDescription>
                                        </CardHeader>
                                        <CardContent>
                                            {analysisResult?.visualization_data?.gait_cycles &&
                                             (analysisResult.visualization_data.gait_cycles as unknown as Array<{tap?: number, interval?: number}>).length > 0 ? (
                                                <div className="space-y-4">
                                                    {(() => {
                                                        const rhythmData = analysisResult.visualization_data.gait_cycles as unknown as Array<{tap?: number, interval?: number}>
                                                        const intervals = rhythmData.filter(d => d.interval != null && d.interval > 0).map(d => d.interval!)
                                                        const avgInterval = intervals.length > 0 ? intervals.reduce((a, b) => a + b, 0) / intervals.length : 0
                                                        return (
                                                            <>
                                                                <div className="text-center">
                                                                    <p className="text-3xl font-bold text-primary">{rhythmData.length}</p>
                                                                    <p className="text-sm text-muted-foreground">회 탭핑 감지</p>
                                                                </div>
                                                                <div className="grid grid-cols-2 gap-2 text-center text-xs">
                                                                    <div className="bg-slate-800/50 rounded p-2">
                                                                        <div className="text-muted-foreground">평균 간격</div>
                                                                        <div className="font-bold">{avgInterval.toFixed(0)}ms</div>
                                                                    </div>
                                                                    <div className="bg-slate-800/50 rounded p-2">
                                                                        <div className="text-muted-foreground">탭핑 속도</div>
                                                                        <div className="font-bold">{(1000 / avgInterval).toFixed(1)}Hz</div>
                                                                    </div>
                                                                </div>
                                                            </>
                                                        )
                                                    })()}
                                                </div>
                                            ) : (
                                                <div className="h-32 flex items-center justify-center">
                                                    <p className="text-sm text-muted-foreground">리듬 데이터 없음</p>
                                                </div>
                                            )}
                                        </CardContent>
                                    </Card>
                                </div>
                                {/* 탭핑 속도 프로파일 */}
                                <SpeedProfileChart
                                    data={analysisResult?.visualization_data?.speed_profile as unknown as Parameters<typeof SpeedProfileChart>[0]['data']}
                                    taskType="finger"
                                />
                            </>
                        ) : (
                            <>
                                {/* Gait Charts */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    <JointAngleChart data={analysisResult?.visualization_data?.joint_angles as unknown as Parameters<typeof JointAngleChart>[0]['data']} />
                                    <SymmetryChart data={analysisResult?.visualization_data?.symmetry as unknown as Parameters<typeof SymmetryChart>[0]['data']} />
                                </div>
                            </>
                        )}
                    </div>
                )}

                {activeTab === "timeline" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <MedicationTimeline
                            subjectPersonId={analysisResult?.physio_context?.subject_person_id}
                            subjectDisplayName={analysisResult?.physio_context?.subject_display_name}
                        />
                    </div>
                )}

                {activeTab === "comparison" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <PopulationComparison
                            taskType={type}
                            patientScore={analysisResult?.updrs_score?.total_score ?? analysisResult?.updrs_score?.score ?? undefined}
                            patientMetrics={analysisResult?.metrics as Record<string, number> | undefined}
                        />
                    </div>
                )}

                {activeTab === "reasoning" && (
                    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr] animate-in fade-in slide-in-from-bottom-2">
                        {analysisResult?.ai_interpretation ? (
                            <AIInterpretation
                                summary={analysisResult.ai_interpretation.summary}
                                explanation={analysisResult.ai_interpretation.explanation}
                                recommendations={analysisResult.ai_interpretation.recommendations}
                                defaultExpanded
                            />
                        ) : (
                            <Card className="border-dashed bg-muted/20">
                                <CardContent className="flex min-h-52 flex-col items-center justify-center p-6 text-center">
                                    <Brain className="h-8 w-8 text-muted-foreground" />
                                    <p className="mt-3 text-sm font-medium">AI 해석이 아직 생성되지 않았습니다.</p>
                                    <p className="mt-1 text-xs text-muted-foreground">자동 추출 지표와 영상은 다른 검토 단계에서 확인할 수 있습니다.</p>
                                </CardContent>
                            </Card>
                        )}
                        {(analysisResult?.reasoning_log?.length ?? 0) > 0 ? (
                            <ReasoningLogViewer logs={analysisResult?.reasoning_log || []} />
                        ) : (
                            <Card className="border-dashed bg-muted/20">
                                <CardContent className="flex min-h-52 flex-col items-center justify-center p-6 text-center">
                                    <FileText className="h-8 w-8 text-muted-foreground" />
                                    <p className="mt-3 text-sm font-medium">세부 추론 로그가 없습니다.</p>
                                    <p className="mt-1 text-xs text-muted-foreground">이 분석에서는 요약 근거만 제공됩니다.</p>
                                </CardContent>
                            </Card>
                        )}
                    </div>
                )}

                {activeTab === "soap" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <SOAPNote
                            taskType={type}
                            metrics={analysisResult?.metrics as Record<string, number> | undefined}
                            performabilityAssessment={performabilityAssessment}
                            scoreAdvisory={scoreAdvisory}
                            updrsScore={analysisResult?.updrs_score ?? undefined}
                            aiInterpretation={analysisResult?.ai_interpretation?.summary}
                            patientId={analysisResult?.patient_id}
                            analysisDate={new Date().toISOString()}
                        />
                    </div>
                )}
            </div>
        </PageLayout>
    )
}
