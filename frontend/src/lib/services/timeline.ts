import { API_BASE_URL } from "./api"

// Unified patient timeline: ParkiCheck device observations + Hawk I ai
// observations read back from the shared physio_app Supabase project.

export interface TimelineItem {
  observed_at: string | null
  code: string | null
  status: string | null
  score: number | null
  source_type: string | null
  app_source: "parkicheck" | "hawk_i" | "unknown" | string
  confidence: string | number | null
  analysis_id: string | null
  activity_session_id: string | null
  subject_person_id: string | null
  fhir_id: string | null
  has_medication_context: boolean
  has_hawk_i_review: boolean
}

export interface TimelineResponse {
  success: boolean
  enabled: boolean
  reason?: string
  error?: string
  subject_person_id?: string
  items: TimelineItem[]
  total?: number
}

export async function getUnifiedTimeline(
  subjectPersonId: string,
  accessToken: string,
  limit = 100,
): Promise<TimelineResponse> {
  const params = new URLSearchParams({
    subject_person_id: subjectPersonId,
    limit: String(limit),
  })
  const response = await fetch(`${API_BASE_URL}/api/history/timeline?${params.toString()}`, {
    headers: { Authorization: `Bearer ${accessToken}` },
    cache: 'no-store',
  })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = await response.json()
      if (body?.error) detail = body.error
    } catch {
      // keep HTTP status detail
    }
    throw new Error(detail)
  }
  return response.json()
}
