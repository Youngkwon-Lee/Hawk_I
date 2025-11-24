"use client"

import * as React from "react"
import { Play, Pause, Maximize2, Activity, Flame, AlertCircle, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { cn } from "@/lib/utils"

export interface Marker {
    time: number
    label: string
    type: "warning" | "info" | "good"
}

interface VideoPlayerProps {
    className?: string
    videoSrc?: string
    skeletonSrc?: string
    taskType?: "gait" | "finger"
    markers?: Marker[]
}

interface Keypoint {
    id: number
    x: number
    y: number
    score: number
}

interface FrameData {
    frame: number
    keypoints: Keypoint[]
}

export function VideoPlayer({
    className,
    videoSrc = "/videos/gait_sample.mp4",
    skeletonSrc,
    taskType = "gait",
    markers = []
}: VideoPlayerProps) {
    const [isPlaying, setIsPlaying] = React.useState(false)
    const [showSkeleton, setShowSkeleton] = React.useState(true)
    const [showHeatmap, setShowHeatmap] = React.useState(false)
    const [currentTime, setCurrentTime] = React.useState(0)
    const [duration, setDuration] = React.useState(0)
    const [skeletonData, setSkeletonData] = React.useState<FrameData[]>([])
    const [isLoadingData, setIsLoadingData] = React.useState(true)
    const [videoFps, setVideoFps] = React.useState(30) // Default to 30fps
    const [videoReady, setVideoReady] = React.useState(false)

    const videoRef = React.useRef<HTMLVideoElement>(null)
    const canvasRef = React.useRef<HTMLCanvasElement>(null)

    // Create indexed skeleton map for O(1) lookup
    const skeletonMap = React.useMemo(() => {
        const map = new Map<number, FrameData>()
        skeletonData.forEach(frame => map.set(frame.frame, frame))
        return map
    }, [skeletonData])

    // Load Skeleton Data (only if skeletonSrc is provided)
    React.useEffect(() => {
        if (!skeletonSrc) {
            console.log('No skeleton data source provided - using plain video player')
            setIsLoadingData(false)
            setShowSkeleton(false)
            return
        }

        setIsLoadingData(true)
        console.log('Loading skeleton data from:', skeletonSrc)
        fetch(skeletonSrc)
            .then(res => {
                if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`)
                return res.json()
            })
            .then(data => {
                console.log('Skeleton data loaded:', data?.length, 'frames')
                setSkeletonData(data)
                setIsLoadingData(false)
            })
            .catch(err => {
                console.error("Failed to load skeleton data", err)
                setIsLoadingData(false)
            })
    }, [skeletonSrc])

    // Draw Overlay
    React.useEffect(() => {
        if (!canvasRef.current || !videoRef.current) return

        const ctx = canvasRef.current.getContext("2d")
        if (!ctx) return

        let animationFrameId: number

        const draw = () => {
            if (!videoRef.current || !canvasRef.current || !ctx) return

            // Sync canvas size with video element
            const videoWidth = videoRef.current.clientWidth
            const videoHeight = videoRef.current.clientHeight

            if (canvasRef.current.width !== videoWidth || canvasRef.current.height !== videoHeight) {
                canvasRef.current.width = videoWidth
                canvasRef.current.height = videoHeight
                console.log('Canvas resized to:', videoWidth, 'x', videoHeight)
            }

            // Clear canvas
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)

            // Calculate current frame using actual FPS
            const currentFrame = Math.floor(videoRef.current.currentTime * videoFps)
            const frameData = skeletonMap.get(currentFrame)

            if (showSkeleton && frameData) {
                console.log('Drawing skeleton for frame:', currentFrame)
                ctx.lineWidth = 2.5
                ctx.lineCap = "round"
                ctx.lineJoin = "round"

                let connections: number[][] = []
                if (taskType === "gait") {
                    // Complete Pose connections including arms
                    connections = [
                        // Torso
                        [11, 12],   // Shoulders
                        [23, 24],   // Hips
                        [11, 23],   // Left side
                        [12, 24],   // Right side
                        // Legs
                        [23, 25],   // Left hip to knee
                        [24, 26],   // Right hip to knee
                        [25, 27],   // Left knee to ankle
                        [26, 28],   // Right knee to ankle
                        // Arms - ADDED
                        [11, 13],   // Left shoulder to elbow
                        [13, 15],   // Left elbow to wrist
                        [12, 14],   // Right shoulder to elbow
                        [14, 16],   // Right elbow to wrist
                    ]
                } else {
                    // Hand connections
                    connections = [
                        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
                        [0, 5], [5, 6], [6, 7], [7, 8], // Index
                        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
                        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
                        [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
                    ]
                }

                // Medical-grade colors: Professional blue with subtle glow
                ctx.shadowColor = "rgba(59, 130, 246, 0.5)" // Blue-500 with transparency
                ctx.shadowBlur = 8
                ctx.strokeStyle = "rgba(59, 130, 246, 0.9)" // Blue-500, high opacity

                connections.forEach(([start, end]) => {
                    const p1 = frameData.keypoints.find(k => k.id === start)
                    const p2 = frameData.keypoints.find(k => k.id === end)
                    if (p1 && p2) {
                        ctx.beginPath()
                        ctx.moveTo(p1.x * canvasRef.current!.width, p1.y * canvasRef.current!.height)
                        ctx.lineTo(p2.x * canvasRef.current!.width, p2.y * canvasRef.current!.height)
                        ctx.stroke()
                    }
                })
                ctx.shadowBlur = 0

                // Draw Joint Points - Medical blue color scheme
                frameData.keypoints.forEach(p => {
                    // Outer circle (white border)
                    ctx.beginPath()
                    ctx.fillStyle = "#ffffff"
                    ctx.arc(p.x * canvasRef.current!.width, p.y * canvasRef.current!.height, 5, 0, 2 * Math.PI)
                    ctx.fill()

                    // Inner circle (medical blue)
                    ctx.beginPath()
                    ctx.fillStyle = "rgba(37, 99, 235, 0.95)" // Blue-600
                    ctx.arc(p.x * canvasRef.current!.width, p.y * canvasRef.current!.height, 3, 0, 2 * Math.PI)
                    ctx.fill()
                })
            }

            if (showHeatmap && frameData) {
                const targets = taskType === "gait" ? [27, 28] : [4, 8]

                targets.forEach(id => {
                    const p = frameData.keypoints.find(k => k.id === id)
                    if (p) {
                        const gradient = ctx.createRadialGradient(
                            p.x * canvasRef.current!.width, p.y * canvasRef.current!.height, 5,
                            p.x * canvasRef.current!.width, p.y * canvasRef.current!.height, 50
                        )
                        gradient.addColorStop(0, "rgba(255, 50, 50, 0.8)")
                        gradient.addColorStop(1, "rgba(255, 50, 50, 0)")
                        ctx.fillStyle = gradient
                        ctx.fillRect(0, 0, canvasRef.current!.width, canvasRef.current!.height)
                    }
                })

                ctx.fillStyle = "white"
                ctx.font = "bold 16px sans-serif"
                ctx.shadowColor = "black"
                ctx.shadowBlur = 4
                const text = taskType === "gait" ? "Movement concentrated in ankles" : "Fine motor control analysis"
                ctx.fillText(text, 20, canvasRef.current.height - 80)
                ctx.shadowBlur = 0
            }

            // Continue drawing on every frame
            animationFrameId = requestAnimationFrame(draw)
        }

        // Start drawing
        draw()

        return () => {
            if (animationFrameId) cancelAnimationFrame(animationFrameId)
        }
    }, [showSkeleton, showHeatmap, currentTime, skeletonMap, taskType, videoFps, videoReady])

    const togglePlay = () => {
        if (videoRef.current) {
            if (videoRef.current.paused) {
                videoRef.current.play().catch(e => console.error("Play failed:", e))
            } else {
                videoRef.current.pause()
            }
        }
    }

    // Sync React state with Video state
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime)
        }
    }

    const handleLoadedMetadata = () => {
        if (videoRef.current) {
            setDuration(videoRef.current.duration)
            setVideoReady(true)

            // Try to get FPS from video (not always available in browser)
            // Default to 30 if not available
            console.log('Video loaded:', {
                duration: videoRef.current.duration,
                videoWidth: videoRef.current.videoWidth,
                videoHeight: videoRef.current.videoHeight,
                readyState: videoRef.current.readyState
            })
        }
    }

    const jumpTo = (time: number) => {
        if (videoRef.current) {
            videoRef.current.currentTime = time
            setCurrentTime(time)
        }
    }

    const formatTime = (time: number) => {
        const mins = Math.floor(time / 60)
        const secs = Math.floor(time % 60)
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }

    return (
        <div className={cn("relative rounded-xl overflow-hidden bg-black border border-border group", className)}>
            <div className="relative aspect-video bg-black flex items-center justify-center">
                <video
                    ref={videoRef}
                    src={videoSrc}
                    className="w-full h-full object-contain"
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={handleLoadedMetadata}
                    onPlay={onPlay}
                    onPause={onPause}
                    onEnded={onPause}
                    onError={(e) => {
                        console.error('Video error:', e)
                        const video = e.currentTarget
                        console.error('Video error details:', {
                            error: video.error,
                            networkState: video.networkState,
                            readyState: video.readyState,
                            src: video.src
                        })
                    }}
                    onCanPlayThrough={() => {
                        console.log('Video can play through')
                        setVideoReady(true)
                    }}
                    muted
                    playsInline
                    crossOrigin="anonymous"
                />
                {/* Only render Canvas if skeleton data source is provided */}
                {skeletonSrc && (
                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 pointer-events-none w-full h-full"
                    />
                )}

                {/* Play Overlay - Clickable */}
                {!isPlaying && (
                    <div
                        className="absolute inset-0 flex items-center justify-center bg-black/20 cursor-pointer hover:bg-black/30 transition-colors"
                        onClick={togglePlay}
                    >
                        <div className="p-4 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 shadow-xl">
                            <Play className="h-12 w-12 text-white fill-white" />
                        </div>
                    </div>
                )}

                {/* Loading State */}
                {isLoadingData && (
                    <div className="absolute top-4 right-4 bg-black/50 px-3 py-1 rounded-full flex items-center gap-2 text-xs text-white backdrop-blur-md">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        분석 데이터 로딩 중...
                    </div>
                )}

                {/* Timeline Markers Overlay */}
                <div className="absolute bottom-16 left-0 right-0 px-4 pointer-events-none h-8">
                    {markers.map((marker: Marker, i: number) => (
                        <div
                            key={i}
                            className="absolute bottom-0 flex flex-col items-center cursor-pointer pointer-events-auto group/marker transition-all hover:scale-110 z-10"
                            style={{ left: `${(marker.time / duration) * 100}%` }}
                            onClick={(e) => {
                                e.stopPropagation()
                                jumpTo(marker.time)
                            }}
                        >
                            <div className={cn(
                                "text-[10px] px-1.5 py-0.5 rounded opacity-0 group-hover/marker:opacity-100 transition-opacity mb-1 whitespace-nowrap font-medium shadow-lg",
                                marker.type === "warning" && "bg-red-500 text-white",
                                marker.type === "info" && "bg-blue-500 text-white",
                                marker.type === "good" && "bg-green-500 text-white"
                            )}>
                                {marker.label}
                            </div>
                            {marker.type === "warning" ? (
                                <AlertCircle className="w-5 h-5 text-red-500 fill-black stroke-2" />
                            ) : (
                                <div className={cn(
                                    "w-1 h-3 rounded-full shadow-sm",
                                    marker.type === "info" && "bg-blue-500",
                                    marker.type === "good" && "bg-green-500"
                                )} />
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Controls */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-4 space-y-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-20">
                {/* Progress Bar */}
                <div
                    className="w-full h-1 bg-white/20 rounded-full cursor-pointer relative group/progress"
                    onClick={(e) => {
                        e.stopPropagation()
                        const rect = e.currentTarget.getBoundingClientRect()
                        const pos = (e.clientX - rect.left) / rect.width
                        jumpTo(pos * duration)
                    }}
                >
                    <div
                        className="h-full bg-primary rounded-full relative"
                        style={{ width: `${(currentTime / duration) * 100}%` }}
                    >
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover/progress:opacity-100 shadow-sm" />
                    </div>
                </div>

                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <button onClick={togglePlay} className="text-white hover:text-primary transition-colors">
                            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                        </button>
                        <span className="text-xs text-white font-mono">
                            {formatTime(currentTime)} / {formatTime(duration)}
                        </span>
                    </div>

                    <div className="flex items-center gap-2">
                        {/* Skeleton and Heatmap buttons removed - using backend-generated skeleton overlay video */}
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-white hover:bg-white/10">
                            <Maximize2 className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    )
}
