import { API_BASE_URL, fetchWithTimeout } from "./api"

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
  observation_id: string | null
  activity_session_id: string | null
  subject_person_id: string | null
  fhir_id: string | null
  has_medication_context: boolean
  medication_name: string | null
  medication_dose_mg: number | null
  medication_taken_at: string | null
  hours_after_reported_dose: number | null
  has_hawk_i_review: boolean
  // Quantitative evidence: kinematic measurements behind the score.
  metrics: Record<string, number | string | null>
  // Qualitative evidence: the narrative finding a clinician reads first.
  rationale: string | null
  severity: string | null
  score_confidence: number | string | null
  // Whether the score may be relied on at all, kept separate from the score.
  score_advisory_level: string | null
  score_advisory_summary: string | null
  performability_status: string | null
  scoring_method: string | null
  model_type: string | null
  // Where in the dose cycle this assessment was captured.
  last_dose_at: string | null
  hours_since_last_dose: number | null
  last_dose_medication: string | null
  last_dose_mg: number | null
}

// Kinematic labels a clinician reads, rather than raw field names.
export const METRIC_LABELS: Record<string, string> = {
  gait_speed: '보행 속도',
  stride_length: '보폭',
  cadence: '분당 걸음수',
  step_length: '걸음 길이',
  arm_swing: '팔 흔들림',
  arm_swing_asymmetry: '팔 흔들림 비대칭',
  tapping_speed: '두드리기 속도',
  tapping_frequency: '두드리기 빈도',
  amplitude: '진폭',
  max_amplitude: '최대 진폭',
  amplitude_decrement: '진폭 감소',
  fatigue_rate: '피로도',
  rhythm_variability: '리듬 변동성',
  iti_cv: '박자 간격 변동성',
}

// levodopa improves speed-type measures but not rhythm, decrement, or the
// sequence effect (Espay 2011; Bologna 2020). Mixing them hides whether the
// medication is working, so they are shown as separate groups.
const DOSE_RESISTANT_METRICS = [
  'rhythm', 'iti', 'variability', 'decrement', 'fatigue', 'sequence', 'arrest', 'halt', 'hesitat',
]

export function isDoseResistantMetric(key: string): boolean {
  const normalized = key.toLowerCase()
  return DOSE_RESISTANT_METRICS.some((token) => normalized.includes(token))
}

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] || key.replace(/_/g, ' ')
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
  signal?: AbortSignal,
): Promise<TimelineResponse> {
  const params = new URLSearchParams({
    subject_person_id: subjectPersonId,
    limit: String(limit),
  })
  const response = await fetchWithTimeout(`${API_BASE_URL}/api/history/timeline?${params.toString()}`, {
    headers: { Authorization: `Bearer ${accessToken}` },
    cache: 'no-store',
    signal,
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
