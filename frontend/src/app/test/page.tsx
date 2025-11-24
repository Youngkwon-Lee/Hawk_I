"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Hand, Footprints, RotateCw, Upload, FileVideo, X, AlertTriangle, Loader2, CheckCircle2, Activity } from "lucide-react"
import { cn } from "@/lib/utils"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { analyzeVideoWithProgress, formatVideoType, getVideoTypeColor, type AnalysisResult, type AnalysisStartResponse } from "@/lib/services/api"

const MAX_FILE_SIZE = 100 * 1024 * 1024 // 100MB
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime']

import { AnalysisOverlay } from "@/components/dashboard/AnalysisOverlay"

export default function TestPage() {
    const router = useRouter()
    const [selectedTest, setSelectedTest] = React.useState<string | null>(null)
    const [file, setFile] = React.useState<File | null>(null)
    const [fileError, setFileError] = React.useState<string>("")
    const [isAnalyzing, setIsAnalyzing] = React.useState(false)
    const [uploadProgress, setUploadProgress] = React.useState(0)
    const [analysisError, setAnalysisError] = React.useState<string>("")
    const [currentVideoId, setCurrentVideoId] = React.useState<string | null>(null)

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
                undefined,
                (progress) => setUploadProgress(progress),
                manualTestType
            )

            console.log("Upload complete, analysis started:", result)
            setCurrentVideoId(result.id)

            // Note: We don't navigate anymore. The overlay handles the rest.

        } catch (error) {
            console.error("Upload failed:", error)
            setAnalysisError(error instanceof Error ? error.message : '비디오 업로드에 실패했습니다')
            setIsAnalyzing(false)
        }
    }

    const handleAnalysisComplete = (result: any) => {
        // Save result to session storage
        sessionStorage.setItem('analysisResult', JSON.stringify(result))
        // Navigate to result page
        router.push(`/result?analysisId=${result.id}`)
    }

    return (
        <PageLayout agentPanel={<ChatInterface initialMessages={[{
            id: "1",
            role: "agent",
            content: "안녕하세요! 검사를 시작하려면 테스트 유형을 선택해주세요. 녹화 방법에 대해 안내해 드릴 수 있습니다.",
            timestamp: new Date()
        }]} />}>
            {isAnalyzing && (
                <AnalysisOverlay
                    isUploading={!currentVideoId}
                    uploadProgress={uploadProgress}
                    videoId={currentVideoId}
                    onComplete={handleAnalysisComplete}
                    onError={(err) => {
                        setAnalysisError(err)
                        setIsAnalyzing(false)
                    }}
                />
            )}

            <div className="space-y-8">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">새 분석 시작</h1>
                    <p className="text-muted-foreground mt-2">테스트 유형을 선택하고 비디오를 업로드하세요. AI가 실제 움직임 유형을 자동으로 감지합니다.</p>
                </div>

                {/* Step 1: Select Test Type (Optional) */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h2 className="text-xl font-semibold">1. 분석 유형 선택 (선택사항)</h2>
                        {selectedTest && (
                            <Button variant="ghost" size="sm" onClick={() => setSelectedTest(null)}>
                                <X className="h-4 w-4 mr-1" />
                                자동 감지 사용
                            </Button>
                        )}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <TestTypeCard
                            title="자동 감지"
                            description="AI가 움직임 유형을 자동으로 식별합니다."
                            isSelected={selectedTest === null}
                            onClick={() => setSelectedTest(null)}
                        />
                        <TestTypeCard
                            title="손가락 태핑"
                            description="손가락의 속도와 리듬을 분석합니다."
                            isSelected={selectedTest === "finger"}
                            onClick={() => setSelectedTest("finger")}
                        />
                        <TestTypeCard
                            title="보행 분석"
                            description="보행 패턴과 자세를 분석합니다."
                            isSelected={selectedTest === "gait"}
                            onClick={() => setSelectedTest("gait")}
                        />
                    </div>
                </div>

                {/* Step 2: Upload Video */}
                <div className="space-y-4">
                    <h2 className="text-xl font-semibold">2. 비디오 업로드</h2>
                    <div
                        className={cn(
                            "border-2 border-dashed rounded-xl p-10 text-center transition-colors",
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
                <div className="flex justify-end gap-4 pt-4">
                    <Button variant="ghost" disabled={isAnalyzing}>취소</Button>
                    <Button
                        size="lg"
                        disabled={!file || isAnalyzing}
                        onClick={handleStartAnalysis}
                    >
                        {isAnalyzing ? (
                            <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                분석 중...
                            </>
                        ) : (
                            '분석 시작'
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
    isSelected: boolean
    onClick: () => void
}

function TestTypeCard({ title, description, isSelected, onClick }: TestTypeCardProps) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "w-full text-left relative overflow-hidden rounded-xl border p-4 transition-all hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
                isSelected
                    ? "border-primary bg-primary/5 ring-1 ring-primary"
                    : "border-border bg-card hover:border-primary/50"
            )}
            aria-pressed={isSelected}
        >
            <div className="flex flex-col gap-2">
                <h3 className={cn("font-semibold", isSelected ? "text-primary" : "text-foreground")}>{title}</h3>
                <p className="text-sm text-muted-foreground">{description}</p>
            </div>
        </button>
    )
}
