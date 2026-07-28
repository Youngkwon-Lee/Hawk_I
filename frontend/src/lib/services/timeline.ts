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
  medication_name: string | null
  medication_dose_mg: number | null
  medication_taken_at: string | null
  hours_after_reported_dose: number | null
  has_hawk_i_review: boolean
}

export interface MedicationEvent {
  event_id: string | null
  observed_at: string | null
  status: string | null
  medication_code: string | null
  medication_display: string | null
  dose_mg: number | null
  dose_unit: string | null
  information_source_type: string | null
  subject_person_id: string | null
  app_source: string
}

export interface TimelineResponse {
  success: boolean
  enabled: boolean
  reason?: string
  error?: string
  subject_person_id?: string
  items: TimelineItem[]
  medications: MedicationEvent[]
  total?: number
  medication_total?: number
}

export interface MedicationObservationSummary {
  available: boolean
  observationCount: number
  medicationName?: string
  doseMg?: number | null
  code?: string | null
  firstScore?: number
  latestScore?: number
  observedScoreChange?: number
}

export function buildMedicationObservationSummary(
  items: TimelineItem[],
): MedicationObservationSummary {
  const groups = new Map<string, TimelineItem[]>()
  items.forEach((item) => {
    if (!item.has_medication_context || typeof item.score !== 'number') return
    const key = [item.code || '', item.medication_name || '', item.medication_dose_mg ?? ''].join('|')
    const group = groups.get(key) || []
    group.push(item)
    groups.set(key, group)
  })

  const selected = [...groups.values()]
    .map((group) => group.sort((left, right) =>
      new Date(left.observed_at || 0).getTime() - new Date(right.observed_at || 0).getTime()))
    .sort((left, right) => right.length - left.length)[0] || []

  if (selected.length < 2) {
    return { available: false, observationCount: selected.length }
  }

  const first = selected[0]
  const latest = selected[selected.length - 1]
  return {
    available: true,
    observationCount: selected.length,
    medicationName: latest.medication_name || '약물명 미입력',
    doseMg: latest.medication_dose_mg,
    code: latest.code,
    firstScore: first.score as number,
    latestScore: latest.score as number,
    observedScoreChange: Number(((latest.score as number) - (first.score as number)).toFixed(2)),
  }
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
