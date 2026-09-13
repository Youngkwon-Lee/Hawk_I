"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { Card, CardContent } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Upload, FileVideo, X, AlertTriangle, Loader2, ExternalLink, Footprints, Hand, ScanLine, ChevronRight, Circle, FileCheck2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"
import {
    analyzeVideoWithProgress,
    getPhysioSelf,
    getPhysioSubjects,
    type AnalysisResult,
    type PhysioAnalysisContext,
    type PhysioSubjectsResponse,
} from "@/lib/services/api"
import { useAnalysisStore } from "@/store/analysisStore"
import { AnalysisOverlay } from "@/components/dashboard/AnalysisOverlay"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"
import { EvidencePanel } from "@/components/test/EvidencePanel"
import { WorkspaceRail } from "@/components/test/WorkspaceRail"

const MAX_FILE_SIZE = 100 * 1024 * 1024 // 100MB
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime']
const DEBUG_LOGS = process.env.NODE_ENV !== "production"

export default function TestPage() {
    const router = useRouter()
    const { setResult, clearResult } = useAnalysisStore()
    const [selectedTest, setSelectedTest] = React.useState<string | null>(null)
    const [file, setFile] = React.useState<File | null>(null)
    const [fileError, setFileError] = React.useState<string>("")
    const [isAnalyzing, setIsAnalyzing] = React.useState(false)
    const [uploadProgress, setUploadProgress] = React.useState(0)
    const [analysisError, setAnalysisError] = React.useState<string>("")
    const [currentVideoId, setCurrentVideoId] = React.useState<string | null>(null)
    const [physioData, setPhysioData] = React.useState<PhysioSubjectsResponse | null>(null)
    const [selectedSubjectId, setSelectedSubjectId] = React.useState("")
    const [isLoadingPhysio, setIsLoadingPhysio] = React.useState(true)
    const [authReady, setAuthReady] = React.useState(false)
    const [accessToken, setAccessToken] = React.useState<string | null>(null)

    React.useEffect(() => {
        const supabase = getSupabaseBrowserClient()
        if (!supabase) {
            setAuthReady(true)
            return
        }

        let mounted = true
        void supabase.auth.getSession().then(({ data }) => {
            if (!mounted) return
            setAccessToken(data.session?.access_token ?? null)
            setAuthReady(true)
        })
        const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
            if (!mounted) return
            setAccessToken(session?.access_token ?? null)
            setAuthReady(true)
        })
        return () => {
            mounted = false
            listener.subscription.unsubscribe()
        }
    }, [])

    const loadPhysioSubjects = React.useCallback(async () => {
        if (!authReady) return
        setIsLoadingPhysio(true)
        if (!accessToken) {
            // Being signed out is not a failure: analysis still runs, only storage is skipped.
            setPhysioData({
                success: true,
                enabled: false,
                organization: null,
                subjects: [],
                reason: "signed_out",
            })
            setSelectedSubjectId("")
            setIsLoadingPhysio(false)
            return
        }
        try {
            const data = await getPhysioSubjects(accessToken)
            setPhysioData(data)
            if (data.enabled && data.subjects.length > 0) {
                setSelectedSubjectId((current) => {
                    if (data.subjects.some((subject) => subject.id === current)) {
                        return current
                    }
                    return data.default_subject_id || data.subjects[0].id
                })
            } else {
                setSelectedSubjectId("")
            }
        } catch {
            try {
                const selfData = await getPhysioSelf(accessToken)
                const selfSubjects: PhysioSubjectsResponse = {
                    success: selfData.success,
                    enabled: selfData.enabled,
                    organization: selfData.organization ?? null,
                    subjects: [selfData.subject],
                    default_subject_id: selfData.subject.id,
                    default_created_by_person_id: selfData.default_created_by_person_id,
                    default_performer_person_id: selfData.default_performer_person_id,
                    contract_version: selfData.contract_version,
                    persistence_owner: selfData.persistence_owner,
                }
                setPhysioData(selfSubjects)
                setSelectedSubjectId(selfData.subject.id)
            } catch {
                setPhysioData(null)
                setSelectedSubjectId("")
            }
        } finally {
            setIsLoadingPhysio(false)
        }
    }, [accessToken, authReady])

    React.useEffect(() => {
        if (authReady) void loadPhysioSubjects()
    }, [authReady, loadPhysioSubjects])

    const selectedSubject = React.useMemo(() => {
        return physioData?.subjects.find((subject) => subject.id === selectedSubjectId) ?? null
    }, [physioData, selectedSubjectId])

    const physioAnalysisContext = React.useMemo<PhysioAnalysisContext | undefined>(() => {
        if (!physioData?.enabled || !selectedSubject) return undefined

        return {
            subject_person_id: selectedSubject.id,
            organization_id: selectedSubject.organization_id,
            created_by_person_id: physioData.default_created_by_person_id ?? undefined,
            performer_person_id: physioData.default_performer_person_id ?? undefined,
            subject_display_name: selectedSubject.display_name,
            organization_display_name: physioData.organization?.display_name || physioData.organization?.name || undefined,
            contract_version: physioData.contract_version,
            persistence_owner: physioData.persistence_owner,
        }
    }, [physioData, selectedSubject])

    // Storage context is optional - the backend skips persistence when it is absent
    // instead of failing - so a missing subject must never disable the analysis itself.
    // Waiting on the lookup only makes sense once auth resolved; otherwise a stalled
    // Supabase session would leave the button dead for the rest of the page's life.
    const isPhysioLookupPending = authReady && isLoadingPhysio
    const isAnalysisDisabled = !file || isAnalyzing || isPhysioLookupPending

    const validateFile = (file: File): string | null => {
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            return `파일 크기는 ${MAX_FILE_SIZE / (1024 * 1024)}MB 이하여야 합니다`
        }

        // Check MIME type
        if (!ALLOWED_VIDEO_TYPES.includes(file.type)) {
            return "MP4, WebM, OGG, MOV 형식의 비디오 파일만 업로드 가능합니다"
        }

        return null
    }

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFileError("")
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0]
            const error = validateFile(selectedFile)

            if (error) {
                setFileError(error)
                return
            }

            setFile(selectedFile)
        }
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setFileError("")

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const selectedFile = e.dataTransfer.files[0]
            const error = validateFile(selectedFile)

            if (error) {
                setFileError(error)
                return
            }

            setFile(selectedFile)
        }
    }

    const handleStartAnalysis = async () => {
        if (!file) return
        if (isPhysioLookupPending) {
            setAnalysisError("physio_app 저장 대상을 확인 중입니다. 잠시 후 다시 시도해 주세요.")
            return
        }
        // No subject context: the analysis still runs, the result just is not persisted.

        // Clear previous result before starting new analysis
        clearResult()

        setIsAnalyzing(true)
        setAnalysisError("")
        setUploadProgress(0)
        setCurrentVideoId(null)

        try {
            const manualTestType = selectedTest === null ? undefined :
                selectedTest === "finger" ? "finger_tapping" : "gait"

            // Start upload and get videoId
            const result = await analyzeVideoWithProgress(
                file,
                physioAnalysisContext?.subject_person_id,
                (progress) => setUploadProgress(progress),
                manualTestType,
                "coral",
                physioAnalysisContext,
                accessToken ?? undefined
            )

            if (DEBUG_LOGS) {
                console.log("Upload complete, analysis started:", result)
            }
            setCurrentVideoId(result.id)

            // Note: We don't navigate anymore. The overlay handles the rest.

        } catch (error) {
            console.error("Upload failed:", error)
            setAnalysisError(error instanceof Error ? error.message : '비디오 업로드에 실패했습니다')
            setIsAnalyzing(false)
        }
    }

    const handleAnalysisComplete = (result: AnalysisResult) => {
        // Save result to Zustand store (auto-persisted to sessionStorage)
        setResult(result)
        // Navigate to result page
        router.push(`/result?analysisId=${result.id}`)
    }

    return (
        <PageLayout
            leftRail={
                <WorkspaceRail
                    selectedSubject={selectedSubject}
                    subjects={physioData?.subjects}
                    selectedSubjectId={selectedSubjectId}
                    onSelectSubject={setSelectedSubjectId}
                />
            }
            agentPanel={<EvidencePanel />}
            agentPanelWidth="w-16"
        >
            {isAnalyzing && (
                <AnalysisOverlay
                    isUploading={!currentVideoId}
                    uploadProgress={uploadProgress}
                    videoId={currentVideoId}
                    accessToken={accessToken ?? undefined}
                    onComplete={handleAnalysisComplete}
                    onError={(err) => {
                        setAnalysisError(err)
                        setIsAnalyzing(false)
                    }}
                />
            )}

            <div className="space-y-5 pb-24">
                <div className="border-b border-border pb-5">
                    <div>
                        <h1 className="mt-3 text-3xl font-semibold tracking-tight md:text-4xl">새 분석 시작</h1>
                        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">ParkiCheck 검사 결과를 연결하고 영상을 업로드해 분석을 시작하세요.</p>
                    </div>
                </div>

                <div className="hidden items-center gap-3 md:flex" aria-label="분석 진행 단계">
                    <ProgressStep number="1" label="검사 기록 연결" active />
                    <span className="h-px flex-1 bg-primary/70" />
                    <ProgressStep number="2" label="분석 유형 선택" />
                    <span className="h-px flex-1 bg-border" />
                    <ProgressStep number="3" label="비디오 업로드" />
                </div>

                {/* Step 1: ParkiCheck source record */}
                <div className="space-y-3 rounded-xl border border-border bg-card/35 p-4">
                    <div className="flex items-center justify-between gap-3">
                        <div>
                            <p className="text-xs font-medium uppercase tracking-[0.16em] text-primary">1. ParkiCheck 검사 기록</p>
                            <h2 className="mt-1 text-base font-semibold">검사는 ParkiCheck에서 진행해주세요.</h2>
                        </div>
                        <a href="https://finger-tap-fx.vercel.app/" target="_blank" rel="noreferrer" className="inline-flex shrink-0 items-center gap-1 rounded-md border border-primary/50 px-3 py-2 text-xs font-medium text-primary transition-colors hover:bg-primary/5">ParkiCheck에서 검사하기 <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" /></a>
                    </div>
                    <div className="flex flex-col gap-3 rounded-lg border border-primary/20 bg-primary/[0.03] p-4 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex items-start gap-3">
                            <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><FileCheck2 className="h-5 w-5" aria-hidden="true" /></div>
                            <div>
                                <p className="text-sm font-medium">검사 완료 후 결과 파일을 이곳에 업로드하면 분석이 시작됩니다.</p>
                                <p className="mt-1 text-xs leading-5 text-muted-foreground">검사 기록과 분석 결과는 환자 기록의 한 타임라인으로 연결됩니다.</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Step 2: Select Test Type (Optional) */}
                <div className="space-y-3 rounded-xl border border-border bg-card/35 p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-xs font-medium uppercase tracking-[0.16em] text-primary">2. 분석 유형 선택</p>
                            <h2 className="mt-1 text-base font-semibold">영상 유형을 선택하세요</h2>
                            <p className="mt-1 text-xs leading-5 text-muted-foreground">보행·손가락 영상은 유형을 직접 선택하면 오분류를 줄일 수 있습니다.</p>
                        </div>
                        {selectedTest && (
                            <Button variant="ghost" size="sm" onClick={() => setSelectedTest(null)}>
                                <X className="mr-1 h-4 w-4" />
                                자동 감지 사용
                            </Button>
                        )}
                    </div>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
                        <TestTypeCard
                            title="자동 감지"
                            description="구도가 불명확하거나 짧은 영상은 오분류할 수 있습니다."
                            icon={<ScanLine className="h-7 w-7" aria-hidden="true" />}
                            isSelected={selectedTest === null}
                            onClick={() => setSelectedTest(null)}
                        />
                        <TestTypeCard
                            title="손가락 태핑"
                            description="손가락의 속도와 리듬을 분석합니다."
                            icon={<Hand className="h-7 w-7" aria-hidden="true" />}
                            isSelected={selectedTest === "finger"}
                            onClick={() => setSelectedTest("finger")}
                        />
                        <TestTypeCard
                            title="보행 분석"
                            description="보행 패턴과 자세를 분석합니다."
                            icon={<Footprints className="h-7 w-7" aria-hidden="true" />}
                            isSelected={selectedTest === "gait"}
                            onClick={() => setSelectedTest("gait")}
                        />
                        <TestTypeCard
                            title="순차적 운동"
                            description="곧 제공될 예정입니다."
                            icon={<Circle className="h-7 w-7" aria-hidden="true" />}
                            isSelected={false}
                            onClick={() => undefined}
                            disabled
                        />
                    </div>
                </div>

                {/* Patient record connection is shown when a physio_app workspace is connected. */}
                {/* Step 3: Upload Video */}
                <div className="space-y-3 rounded-xl border border-border bg-card/35 p-4">
                    <div>
                        <p className="text-xs font-medium uppercase tracking-[0.16em] text-primary">3. 비디오 업로드</p>
                        <h2 className="mt-1 text-base font-semibold">환자의 움직임이 잘 보이는 영상을 업로드하세요</h2>
                    </div>
                    <div
                        className={cn(
                            "min-h-[18rem] border border-dashed rounded-xl p-8 text-center transition-colors",
                            file ? "border-primary/50 bg-primary/5" : "border-border hover:border-primary/50 hover:bg-accent/50"
                        )}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {!file ? (
                            <div className="flex flex-col items-center gap-4">
                                <div className="p-4 rounded-full bg-secondary">
                                    <Upload className="h-8 w-8 text-muted-foreground" />
                                </div>
                                <div>
                                    <p className="font-medium">비디오 파일을 이곳에 드래그하세요</p>
                                    <p className="text-sm text-muted-foreground mt-1">또는 클릭하여 파일 선택</p>
                                </div>
                                <input
                                    type="file"
                                    accept="video/*"
                                    className="hidden"
                                    id="video-upload"
                                    onChange={handleFileChange}
                                />
                                <Button variant="outline" onClick={() => document.getElementById("video-upload")?.click()}>
                                    파일 선택
                                </Button>
                            </div>
                        ) : (
                            <div className="flex items-center justify-between max-w-md mx-auto bg-card p-4 rounded-lg border border-border">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded bg-primary/10 text-primary">
                                        <FileVideo className="h-6 w-6" />
                                    </div>
                                    <div className="text-left">
                                        <p className="font-medium truncate max-w-[200px]">{file.name}</p>
                                        <p className="text-xs text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                                    </div>
                                </div>
                                <Button variant="ghost" size="icon" onClick={() => setFile(null)}>
                                    <X className="h-4 w-4" />
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* File Error Message */}
                    {fileError && (
                        <div className="mt-2 text-sm text-red-500 flex items-center gap-2">
                            <AlertTriangle className="h-4 w-4" />
                            {fileError}
                        </div>
                    )}
                </div>

                {/* Upload Error */}
                {analysisError && (
                    <Card className="border-red-200 bg-red-50/50">
                        <CardContent className="p-4">
                            <div className="flex items-center gap-2 text-red-900">
                                <AlertTriangle className="h-4 w-4" />
                                <p className="text-sm">{analysisError}</p>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Action Buttons */}
                <div className="fixed inset-x-0 bottom-0 z-30 flex justify-end gap-3 border-t border-border bg-background/95 px-4 py-3 backdrop-blur md:right-16 md:px-14">
                    <Button variant="ghost" disabled={isAnalyzing}>취소</Button>
                    <Button
                        size="lg"
                        disabled={isAnalysisDisabled}
                        onClick={handleStartAnalysis}
                        className="bg-[#f2675c] px-8 text-white shadow-sm hover:bg-[#e95a50]"
                    >
                        {isAnalyzing ? (
                            <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                분석 중...
                            </>
                        ) : isPhysioLookupPending ? (
                            <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                대상 확인 중...
                            </>
                        ) : (
                            <>분석 시작 <ChevronRight className="ml-1 h-4 w-4" aria-hidden="true" /></>
                        )}
                    </Button>
                </div>
            </div>
        </PageLayout>
    )
}

interface TestTypeCardProps {
    title: string
    description: string
    icon: React.ReactNode
    isSelected: boolean
    onClick: () => void
    disabled?: boolean
}

function TestTypeCard({ title, description, icon, isSelected, onClick, disabled = false }: TestTypeCardProps) {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={cn(
                "relative w-full overflow-hidden rounded-xl border p-4 text-left transition-all hover:border-primary/60 hover:bg-accent/40 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
                isSelected
                    ? "border-primary bg-primary/5 ring-1 ring-primary"
                    : "border-border bg-card hover:border-primary/50",
                disabled && "cursor-not-allowed opacity-45 hover:border-border hover:bg-card"
            )}
            aria-pressed={isSelected}
            aria-disabled={disabled}
        >
            <div className="flex items-start gap-3">
                <div className={cn("mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border", isSelected ? "border-primary/40 bg-primary/10 text-primary" : "border-border bg-background/50 text-muted-foreground")}>{icon}</div>
                <div className="min-w-0">
                    <h3 className={cn("font-semibold", isSelected ? "text-primary" : "text-foreground")}>{title}</h3>
                    <p className="mt-1 text-sm leading-5 text-muted-foreground">{description}</p>
                </div>
            </div>
        </button>
    )
}

function ProgressStep({ number, label, active = false }: { number: string; label: string; active?: boolean }) {
    return (
        <div className={cn("flex shrink-0 items-center gap-2 text-xs", active ? "text-primary" : "text-muted-foreground")}>
            <span className={cn("flex h-7 w-7 items-center justify-center rounded-full border text-xs font-semibold", active ? "border-primary bg-primary text-primary-foreground" : "border-border bg-card")}>{number}</span>
            <span className="whitespace-nowrap">{label}</span>
        </div>
    )
}
