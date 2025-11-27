/**
 * API Service for connecting to Flask Backend
 * Backend: http://localhost:5000
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

export interface FingerTappingMetrics {
  tapping_speed: number
  amplitude_mean: number
  amplitude_std: number
  rhythm_variability: number
  fatigue_rate: number
  hesitation_count: number
  total_taps: number
  duration: number
}

export interface GaitMetrics {
  walking_speed: number
  stride_length: number
  cadence: number
  stride_variability: number
  arm_swing_asymmetry: number
  step_count: number
  duration: number
}

export interface TimelineEvent {
  timestamp: number
  type: string
  description: string
}

export interface AnalysisResult {
  success: boolean
  video_type: string
  confidence: number
  auto_detected: boolean
  patient_id: string
  roi: {
    x: number
    y: number
    w: number
    h: number
  }
  motion_analysis: {
    body_part: string
    motion_area_ratio: number
    motion_pattern: string
  }
  reasoning: string
  reasoning_log?: {
    agent: string
    step: string
    content: string
    timestamp: string
    meta?: any
  }[]
  video_metadata: {
    width: number
    height: number
    fps: number
    duration: number
    total_frames: number
  }
  metrics?: FingerTappingMetrics | GaitMetrics | null
  skeleton_data?: {
    total_frames: number
    detection_rate: number
    mode: string
    skeleton_video_url?: string
  } | null
  updrs_score?: {
    score: number
    severity: string
  } | null
  ai_interpretation?: {
    summary: string
    explanation: string
    recommendations: string[]
  } | null
  visualization_maps?: {
    heatmap_url?: string
    temporal_map_url?: string
    attention_map_url?: string
    overlay_video_url?: string
  } | null
  visualization_urls?: {
    heatmap?: string
    temporal_map?: string
    attention_map?: string
  } | null
  events?: TimelineEvent[]
}

export interface HealthStatus {
  status: string
  service: string
  dependencies: {
    opencv: string
    mediapipe: string
    pytorch: string
    cuda_available: boolean
    cuda_device: string | null
  }
  capabilities: {
    roi_detection: boolean
    task_classification: boolean
    skeleton_extraction: boolean
    updrs_prediction: boolean
  }
}

/**
 * Check backend health status
 */
export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE_URL}/health`)

  if (!response.ok) {
    throw new Error('Backend health check failed')
  }

  return response.json()
}

/**
 * Analyze video with automatic task classification
 *
 * @param videoFile - Video file to analyze
 * @param patientId - Optional patient ID
 * @param manualTestType - Optional manual override for test type
 * @returns Analysis result with auto-detected video_type
 */
export async function analyzeVideo(
  videoFile: File,
  patientId?: string,
  manualTestType?: string
): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('video_file', videoFile)

  if (patientId) {
    formData.append('patient_id', patientId)
  }

  if (manualTestType) {
    formData.append('test_type', manualTestType)
  }

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Video analysis failed')
  }

  return response.json()
}

export interface AnalysisStartResponse {
  success: boolean
  message: string
  id: string
  status: string
}

/**
 * Upload video and start analysis (with progress callback)
 * Returns the video ID to track progress
 */
export async function analyzeVideoWithProgress(
  videoFile: File,
  patientId?: string,
  onProgress?: (progress: number) => void,
  manualTestType?: string
): Promise<AnalysisStartResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    // Progress tracking
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const percentComplete = (e.loaded / e.total) * 100
        onProgress(percentComplete)
      }
    })

    // Response handling
    xhr.addEventListener('load', () => {
      if (xhr.status === 200 || xhr.status === 202) {
        resolve(JSON.parse(xhr.responseText))
      } else {
        reject(new Error('Video upload failed'))
      }
    })

    xhr.addEventListener('error', () => {
      reject(new Error('Network error'))
    })

    // Send request
    const formData = new FormData()
    formData.append('video_file', videoFile)
    if (patientId) {
      formData.append('patient_id', patientId)
    }
    if (manualTestType) {
      formData.append('test_type', manualTestType)
    }

    xhr.open('POST', `${API_BASE_URL}/api/analyze`)
    xhr.send(formData)
  })
}

/**
 * Get final analysis result
 */
export async function getAnalysisResult(videoId: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/api/analysis/result/${videoId}`)

  if (!response.ok) {
    throw new Error('Failed to fetch analysis result')
  }

  return response.json()
}

/**
 * Format video type for display
 */
export function formatVideoType(videoType: string): string {
  const typeMap: Record<string, string> = {
    'finger_tapping': 'Finger Tapping',
    'hand_movement': 'Hand Movement',
    'gait': 'Gait',
    'leg_agility': 'Leg Agility',
    'pronation_supination': 'Pronation-Supination',
    'unknown': 'Unknown'
  }

  return typeMap[videoType] || videoType
}

/**
 * Get color for video type badge
 */
export function getVideoTypeColor(videoType: string): string {
  const colorMap: Record<string, string> = {
    'finger_tapping': 'blue',
    'hand_movement': 'purple',
    'gait': 'green',
    'leg_agility': 'orange',
    'pronation_supination': 'pink',
    'unknown': 'gray'
  }

  return colorMap[videoType] || 'gray'
}
