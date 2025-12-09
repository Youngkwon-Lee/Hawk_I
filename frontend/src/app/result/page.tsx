"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"
import { useAnalysisStore } from "@/store/analysisStore"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { SummaryCard } from "@/components/dashboard/SummaryCard"
import { MetricsTable, MetricRow } from "@/components/dashboard/MetricsTable"
import { TrendChart } from "@/components/dashboard/TrendChart"
import { VideoPlayer } from "@/components/dashboard/VideoPlayer"
import { AIInterpretation } from "@/components/dashboard/AIInterpretation"
import { MedicationTimeline } from "@/components/dashboard/MedicationTimeline"
import { PopulationComparison } from "@/components/dashboard/PopulationComparison"
import { AlertTriangle, Download, Share2, FileText, Activity, Brain, Users, ClipboardList } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AnalysisResult, FingerTappingMetrics, GaitMetrics, TimelineEvent } from "@/lib/services/api"
import { ReasoningLogViewer } from "@/components/dashboard/ReasoningLogViewer"
import { JointAngleChart } from "@/components/dashboard/JointAngleChart"
import { SymmetryChart } from "@/components/dashboard/SymmetryChart"
import { GaitCycleChart } from "@/components/dashboard/GaitCycleChart"
import { SpeedProfileChart } from "@/components/dashboard/SpeedProfileChart"
import { SOAPNote } from "@/components/dashboard/SOAPNote"

// Mock Data - Gait
const GAIT_METRICS: MetricRow[] = [
    { label: "보행 속도", value: "0.8 m/s", unit: "", change: "-2%", status: "neutral", normalRange: "0.8-1.2" },
    { label: "보행률 (Cadence)", value: "102", unit: "steps/min", change: "+1%", status: "good", normalRange: "100-120" },
    { label: "보폭 길이", value: "0.98", unit: "m", change: "-8%", status: "bad", normalRange: "0.6-0.8" },
    { label: "팔 흔들기 비대칭", value: "15", unit: "%", change: "+5%", status: "bad", normalRange: "<20" },
]

// Mock Data - Finger
const FINGER_METRICS: MetricRow[] = [
    { label: "태핑 속도", value: "3.2 Hz", unit: "", change: "-5%", status: "warning", normalRange: "3.0-6.0" },
    { label: "진폭 (Amplitude)", value: "4.5 cm", unit: "", change: "-10%", status: "bad", normalRange: ">0.8" },
    { label: "주저함", value: "3", unit: "회", change: "+1", status: "warning", normalRange: "≤2" },
    { label: "피로율", value: "12", unit: "%", change: "+3%", status: "bad", normalRange: "<20" },
]

// Mock trend data for demo (임시 데이터)
const MOCK_TREND_DATA = [
    { date: "8월", score: 1.2, stride: 1.1 },
    { date: "9월", score: 1.5, stride: 1.0 },
    { date: "10월", score: 1.3, stride: 0.95 },
    { date: "11월", score: 1.8, stride: 0.92 },
    { date: "12월", score: 1.6, stride: 0.98 },
]

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
    // Determine status based on clinical normal ranges
    const getSpeedStatus = (speed: number) => {
        if (speed >= 0.8 && speed <= 1.2) return "good"
        if (speed >= 0.6 && speed < 0.8) return "warning"
        return "bad"
    }

    const getCadenceStatus = (cadence: number) => {
        if (cadence >= 100 && cadence <= 120) return "good"
        if ((cadence >= 80 && cadence < 100) || (cadence > 120 && cadence <= 140)) return "warning"
        return "bad"
    }

    const getStrideLengthStatus = (stride: number) => {
        if (stride >= 0.6 && stride <= 0.8) return "good"
        if ((stride >= 0.5 && stride < 0.6) || (stride > 0.8 && stride <= 0.9)) return "warning"
        return "bad"
    }

    const getVariabilityStatus = (variability: number) => {
        if (variability < 10) return "good"
        if (variability < 20) return "warning"
        return "bad"
    }

    const getAsymmetryStatus = (asymmetry: number) => {
        if (asymmetry < 20) return "good"
        if (asymmetry < 40) return "warning"
        return "bad"
    }

    return [
        {
            label: "보행 속도",
            value: metrics.walking_speed.toFixed(2),
            unit: "m/s",
            normalRange: "0.8-1.2",
            status: getSpeedStatus(metrics.walking_speed)
        },
        {
            label: "보행률 (Cadence)",
            value: metrics.cadence.toFixed(0),
            unit: "steps/min",
            normalRange: "100-120",
            status: getCadenceStatus(metrics.cadence)
        },
        {
            label: "보폭 길이",
            value: metrics.stride_length.toFixed(2),
            unit: "m",
            normalRange: "0.6-0.8",
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

    // Fetch from API if id is present in URL and result doesn't match
    React.useEffect(() => {
        const urlId = searchParams.get("id") || searchParams.get("analysisId")

        // Check if we need to fetch: URL has id AND (no result OR result id doesn't match)
        const storedId = (analysisResult as any)?.id || (analysisResult as any)?.video_id
        const needsFetch = urlId && (!analysisResult || (storedId && storedId !== urlId))

        if (needsFetch) {
            // Clear old result if id mismatch
            if (storedId && storedId !== urlId) {
                clearResult()
            }

            setIsLoading(true)
            fetch(`http://localhost:5000/api/analysis/result/${urlId}`)
                .then(res => {
                    if (!res.ok) throw new Error("Failed to fetch result")
                    return res.json()
                })
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

    // Log for debugging
    React.useEffect(() => {
        console.log('=== RESULT PAGE DEBUG ===')
        console.log('Analysis result:', analysisResult)
        console.log('Has metrics?', !!analysisResult?.metrics)
        console.log('Has UPDRS?', !!analysisResult?.updrs_score)
        console.log('Has AI interpretation?', !!analysisResult?.ai_interpretation)
        console.log('Video type:', analysisResult?.video_type)
    }, [analysisResult])



    // Determine type from sessionStorage or URL
    const type = analysisResult?.video_type || searchParams.get("type") || "gait"
    const isFinger = type === "finger_tapping" || type === "finger" || type === "hand_movement"

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

    // Use actual uploaded video from backend if available
    const hasBackendVideo = !!analysisResult?.skeleton_data?.skeleton_video_url
    const videoSrc = hasBackendVideo
        ? `http://localhost:5000${analysisResult.skeleton_data.skeleton_video_url}`
        : (isFinger ? "/videos/finger_sample.mp4" : "/videos/12-104704.mp4")

    // Log video source for debugging
    React.useEffect(() => {
        console.log('=== VIDEO DEBUG ===')
        console.log('Has backend video:', hasBackendVideo)
        console.log('skeleton_video_url:', analysisResult?.skeleton_data?.skeleton_video_url)
        console.log('Final videoSrc:', videoSrc)
    }, [videoSrc, analysisResult, hasBackendVideo])

    // Map backend events to UI markers
    const markers = React.useMemo(() => {
        if (analysisResult?.events && analysisResult.events.length > 0) {
            return analysisResult.events.map((event: TimelineEvent) => {
                let type = "info"
                const lowerType = event.type.toLowerCase()

                if (lowerType.includes("freeze") || lowerType.includes("hesitation") || lowerType.includes("stop")) type = "warning"
                else if (lowerType.includes("turn")) type = "info"
                else if (lowerType.includes("good") || lowerType.includes("normal")) type = "good"
                else if (lowerType.includes("bad") || lowerType.includes("abnormal")) type = "bad"

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
            <div className="space-y-8 pb-10">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">분석 결과</h1>
                        <p className="text-muted-foreground mt-1">
                            {title}
                            {analysisResult && (
                                <span className="ml-2">
                                    • AI 감지: {analysisResult.video_type} (신뢰도: {(analysisResult.confidence * 100).toFixed(0)}%)
                                </span>
                            )}
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="gap-2">
                            <Share2 className="h-4 w-4" /> 공유
                        </Button>
                        <Button variant="outline" size="sm" className="gap-2">
                            <Download className="h-4 w-4" /> PDF 내보내기
                        </Button>
                    </div>
                </div>

                {/* Summary Section */}
                <div className="grid gap-4 md:grid-cols-4">
                    <div className="md:col-span-1">
                        <SummaryCard
                            title="추정 점수"
                            value={score}
                            subtext={`UPDRS (0-4) • ${severity}`}
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
                                subtext="평균 보폭"
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

                {/* AI Interpretation */}
                {analysisResult?.ai_interpretation && (
                    <AIInterpretation
                        summary={analysisResult.ai_interpretation.summary}
                        explanation={analysisResult.ai_interpretation.explanation}
                        recommendations={analysisResult.ai_interpretation.recommendations}
                    />
                )}

                {/* Warning Banner - Dynamic based on analysis results */}
                {(() => {
                    const warnings: string[] = [];

                    if (isFinger) {
                        // Finger tapping warnings
                        const metrics = analysisResult?.metrics;
                        if (metrics?.decrement_ratio && metrics.decrement_ratio > 20) {
                            warnings.push(`진폭 감소율이 ${metrics.decrement_ratio.toFixed(1)}%로 측정되었습니다.`);
                        }
                        if (metrics?.fatigue_index && metrics.fatigue_index > 0.3) {
                            warnings.push("후반부 피로 징후가 감지되었습니다.");
                        }
                    } else {
                        // Gait warnings
                        const metrics = analysisResult?.metrics;
                        if (metrics?.walking_speed && metrics.walking_speed < 0.8) {
                            warnings.push(`보행 속도가 ${metrics.walking_speed.toFixed(2)} m/s로 정상 범위(0.8-1.2)보다 낮습니다.`);
                        }
                        if (metrics?.cadence && metrics.cadence < 100) {
                            warnings.push(`보행률이 ${Math.round(metrics.cadence)} steps/min으로 정상 범위(100-120)보다 낮습니다.`);
                        }
                        if (metrics?.arm_swing_asymmetry && metrics.arm_swing_asymmetry > 20) {
                            warnings.push(`팔 흔들기 비대칭이 ${metrics.arm_swing_asymmetry.toFixed(1)}%로 기준치(20%)를 초과합니다.`);
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

                {/* Tabs Navigation */}
                <div className="border-b border-border">
                    <div className="flex gap-6">
                        <button
                            onClick={() => setActiveTab("dashboard")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "dashboard" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            대시보드 뷰
                        </button>
                        <button
                            onClick={() => setActiveTab("video")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "video" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            영상 분석
                        </button>
                        <button
                            onClick={() => setActiveTab("raw")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "raw" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            원시 데이터
                        </button>
                        <button
                            onClick={() => setActiveTab("visualizations")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "visualizations" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            시각화 분석
                        </button>
                        <button
                            onClick={() => setActiveTab("timeline")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "timeline" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            약물 타임라인
                        </button>
                        <button
                            onClick={() => setActiveTab("comparison")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "comparison" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            정상군 비교
                        </button>
                        <button
                            onClick={() => setActiveTab("reasoning")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2",
                                activeTab === "reasoning" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            AI 추론 과정
                        </button>
                        <button
                            onClick={() => setActiveTab("soap")}
                            className={cn(
                                "pb-3 text-sm font-medium transition-all border-b-2 flex items-center gap-1",
                                activeTab === "soap" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <ClipboardList className="h-4 w-4" />
                            SOAP 노트
                        </button>
                    </div>
                </div>

                {/* Tab Content */}
                {activeTab === "dashboard" && (
                    <div className="grid gap-6 md:grid-cols-2 animate-in fade-in slide-in-from-bottom-2">
                        {/* Left Col: Charts with mock trend data */}
                        <div className="space-y-6">
                            <div>
                                <TrendChart
                                    data={MOCK_TREND_DATA}
                                    title="점수 추세"
                                    description="최근 5개월 UPDRS 점수 변화"
                                    dataKey="score"
                                    color="#3b82f6"
                                />
                                <p className="text-xs text-muted-foreground text-center mt-1">(임시 데이터)</p>
                            </div>
                            <div>
                                <TrendChart
                                    data={MOCK_TREND_DATA}
                                    title={isFinger ? "진폭 추세" : "보폭 길이 추세"}
                                    description={isFinger ? "최근 5개월 진폭 변화" : "최근 5개월 보폭 변화"}
                                    dataKey="stride"
                                    color="#10b981"
                                />
                                <p className="text-xs text-muted-foreground text-center mt-1">(임시 데이터)</p>
                            </div>
                        </div>

                        {/* Right Col: Metrics Table */}
                        <div className="space-y-6">
                            <Card className="h-full">
                                <CardHeader>
                                    <CardTitle>상세 운동 분석</CardTitle>
                                    <CardDescription>운동 매개변수의 종합적인 분석 결과입니다.</CardDescription>
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
                                    taskType={isFinger ? "finger" : "gait"}
                                />
                            </div>
                            <div className="space-y-4">
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-sm">감지된 이벤트</CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-4">
                                        {markers.length > 0 ? (
                                            markers.map((m: any, i: number) => (
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
                                                    src={`http://localhost:5000${analysisResult.visualization_maps.heatmap_url}`}
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
                                                    src={`http://localhost:5000${analysisResult.visualization_maps.temporal_map_url}`}
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
                                                src={`http://localhost:5000${analysisResult.visualization_urls.heatmap}`}
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
                                                src={`http://localhost:5000${analysisResult.visualization_urls.temporal_map}`}
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

                        {/* Row 2: Chart-based visualizations */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <JointAngleChart data={analysisResult.visualization_data?.joint_angles} />
                            <SymmetryChart data={analysisResult.visualization_data?.symmetry} />
                        </div>

                        {/* Row 3: Gait cycle and speed profile */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <GaitCycleChart data={analysisResult.visualization_data?.gait_cycles} />
                            <SpeedProfileChart data={analysisResult.visualization_data?.speed_profile} />
                        </div>
                    </div>
                )}

                {activeTab === "timeline" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <MedicationTimeline patientId={analysisResult?.patient_id || "unknown"} />
                    </div>
                )}

                {activeTab === "comparison" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <PopulationComparison
                            taskType={type}
                            patientScore={analysisResult?.updrs_score?.total_score ?? analysisResult?.updrs_score?.score}
                            patientMetrics={analysisResult?.metrics as Record<string, number> | undefined}
                        />
                    </div>
                )}

                {activeTab === "reasoning" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <ReasoningLogViewer logs={analysisResult?.reasoning_log || []} />
                    </div>
                )}

                {activeTab === "soap" && (
                    <div className="animate-in fade-in slide-in-from-bottom-2">
                        <SOAPNote
                            taskType={type}
                            metrics={analysisResult?.metrics as Record<string, number> | undefined}
                            updrsScore={analysisResult?.updrs_score}
                            aiInterpretation={analysisResult?.ai_interpretation}
                            patientId={analysisResult?.patient_id}
                            analysisDate={new Date().toISOString()}
                        />
                    </div>
                )}
            </div>
        </PageLayout>
    )
}
