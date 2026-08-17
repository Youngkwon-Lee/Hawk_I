"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { motion, AnimatePresence } from "framer-motion"
import {
  Calendar, Activity, Filter, TrendingUp,
  BarChart3, Clock, Trash2, Eye, Search, ChevronDown,
  LoaderCircle, LockKeyhole, LogOut, ShieldCheck, ExternalLink,
  RefreshCw, CircleCheck, Database, ArrowUpRight, ClipboardCheck
} from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import {
  getHistory, getHistoryStats, deleteAnalysis, formatVideoType, getPhysioSelf, getPhysioSubjects,
  type HistoryItem, type HistoryStats, type HistoryFilters, type PhysioSubjectsResponse
} from "@/lib/services/api"
import {
  getUnifiedTimeline, isDoseResistantMetric, metricLabel,
  type MedicationEvent, type TimelineItem
} from "@/lib/services/timeline"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ScatterChart, Scatter, ComposedChart
} from 'recharts'

// Severity uses a restrained, semantic palette. Colour is reserved for the
// clinical state and is not used as decoration elsewhere on the page.
const severityColors: Record<string, string> = {
  "Normal": "history-severity history-severity--normal",
  "Slight": "history-severity history-severity--slight",
  "Mild": "history-severity history-severity--mild",
  "Moderate": "history-severity history-severity--moderate",
  "Severe": "history-severity history-severity--severe",
  "Unknown": "history-severity history-severity--unknown"
}

const scoreColors = ["#5b9b7a", "#718f9c", "#aa9457", "#b8795e", "#b7636f"]

const chartColors = {
  grid: "var(--chart-grid)",
  axis: "var(--chart-axis)",
  score: "var(--chart-score)",
  accent: "var(--chart-accent)",
}

function MetricCard({
  label,
  value,
  caption,
  icon,
  tone,
}: {
  label: string
  value: string | number
  caption: string
  icon: React.ReactNode
  tone: "blue" | "green" | "amber" | "violet"
}) {
  const toneClasses = {
    blue: "history-metric-icon",
    green: "history-metric-icon",
    amber: "history-metric-icon",
    violet: "history-metric-icon",
  }

  return (
    <Card className="border-border bg-card shadow-none transition-colors hover:border-primary/35">
      <CardContent className="flex items-start justify-between gap-4 p-5">
        <div className="min-w-0">
          <p className="text-xs font-medium tracking-[0.08em] text-muted-foreground">{label}</p>
          <p className="mt-2 truncate text-2xl font-semibold tracking-tight text-foreground">{value}</p>
          <p className="mt-1 text-xs text-muted-foreground">{caption}</p>
        </div>
        <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-xl", toneClasses[tone])}>
          {icon}
        </div>
      </CardContent>
    </Card>
  )
}

export default function HistoryPage() {
  const [authReady, setAuthReady] = React.useState(false)
  const [accessToken, setAccessToken] = React.useState<string | null>(null)
  const [signedInEmail, setSignedInEmail] = React.useState<string | null>(null)
  const [loginEmail, setLoginEmail] = React.useState("")
  const [loginPassword, setLoginPassword] = React.useState("")
  const [authError, setAuthError] = React.useState<string | null>(null)
  const [authConfigured, setAuthConfigured] = React.useState(true)
  const [authSubmitting, setAuthSubmitting] = React.useState(false)
  const [history, setHistory] = React.useState<HistoryItem[]>([])
  const [stats, setStats] = React.useState<HistoryStats['data'] | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [historyRetry, setHistoryRetry] = React.useState(0)
  const [showFilters, setShowFilters] = React.useState(false)
  const [deleteConfirm, setDeleteConfirm] = React.useState<string | null>(null)

  // Filter state
  const [filters, setFilters] = React.useState<HistoryFilters>({
    sort: 'date_desc',
    limit: 20
  })
  const [searchTerm, setSearchTerm] = React.useState("")

  // Unified patient timeline (ParkiCheck + Hawk I via shared Supabase)
  const [physioData, setPhysioData] = React.useState<PhysioSubjectsResponse | null>(null)
  const [physioLoading, setPhysioLoading] = React.useState(false)
  const [physioError, setPhysioError] = React.useState<string | null>(null)
  const [isSelfTimeline, setIsSelfTimeline] = React.useState(false)
  const [selectedSubjectId, setSelectedSubjectId] = React.useState("")
  const [timeline, setTimeline] = React.useState<TimelineItem[]>([])
  const [timelineMedications, setTimelineMedications] = React.useState<MedicationEvent[]>([])
  const [timelineEnabled, setTimelineEnabled] = React.useState<boolean | null>(null)
  const [timelineLoading, setTimelineLoading] = React.useState(false)
  const [timelineError, setTimelineError] = React.useState<string | null>(null)
  const [timelineRetry, setTimelineRetry] = React.useState(0)
  const [expandedItems, setExpandedItems] = React.useState<Set<string>>(new Set())
  const timelineAbortRef = React.useRef<AbortController | null>(null)

  // Score trend over calendar time, with doses on the same time axis but their
  // own hidden value axis - a dose has no score, so it must not share the scale.
  const trendPoints = React.useMemo(
    () => timeline
      .filter((item) => item.observed_at && typeof item.score === 'number')
      .map((item) => ({ t: new Date(item.observed_at as string).getTime(), score: item.score as number }))
      .sort((left, right) => left.t - right.t),
    [timeline]
  )

  const doseMarkers = React.useMemo(
    () => timelineMedications
      .filter((medication) => medication.observed_at)
      .map((medication) => ({
        t: new Date(medication.observed_at as string).getTime(),
        lane: 0.5,
        label: medication.medication_display || medication.medication_code || '복약',
      }))
      .sort((left, right) => left.t - right.t),
    [timelineMedications]
  )

  // "When is it bad?" and "is the medication working?" are different questions.
  // This series answers the second by re-basing each assessment on its dose.
  const doseAlignedPoints = React.useMemo(
    () => timeline
      .filter((item) => item.hours_since_last_dose !== null && typeof item.score === 'number')
      .map((item) => ({
        hours: item.hours_since_last_dose as number,
        score: item.score as number,
        source: item.app_source,
      }))
      .sort((left, right) => left.hours - right.hours),
    [timeline]
  )

  React.useEffect(() => {
    const supabase = getSupabaseBrowserClient()
    if (!supabase) {
      setAuthConfigured(false)
      setAuthError('임상 기록 로그인이 아직 연결되지 않았습니다. 관리자 설정 후 사용할 수 있습니다.')
      setAuthReady(true)
      return
    }

    setAuthConfigured(true)

    let mounted = true
    void supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return
      setAccessToken(data.session?.access_token ?? null)
      setSignedInEmail(data.session?.user.email ?? null)
      setAuthReady(true)
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!mounted) return
      setAccessToken(session?.access_token ?? null)
      setSignedInEmail(session?.user.email ?? null)
      setAuthReady(true)
    })

    return () => {
      mounted = false
      listener.subscription.unsubscribe()
    }
  }, [])

  React.useEffect(() => {
    if (!accessToken) {
      setPhysioData(null)
      setPhysioLoading(false)
      setPhysioError(null)
      setIsSelfTimeline(false)
      setSelectedSubjectId("")
      setTimelineMedications([])
      return
    }
    setPhysioLoading(true)
    setPhysioError(null)
    let active = true
    const loadSubjects = async () => {
      try {
        const data = await getPhysioSubjects(accessToken)
        if (!active) return
        setPhysioData(data)
        setIsSelfTimeline(false)
        if (data.enabled && data.subjects.length > 0) {
          setSelectedSubjectId(data.default_subject_id || data.subjects[0].id)
        }
      } catch {
        try {
          const self = await getPhysioSelf(accessToken)
          if (!active) return
          const selfSubjectData: PhysioSubjectsResponse = {
            success: self.success,
            enabled: self.enabled,
            organization: null,
            subjects: [self.subject],
            default_subject_id: self.subject.id,
          }
          setPhysioData(selfSubjectData)
          setIsSelfTimeline(true)
          setSelectedSubjectId(self.subject.id)
        } catch {
          if (!active) return
          setPhysioData(null)
          setIsSelfTimeline(false)
          setPhysioError('환자 통합 타임라인을 불러오지 못했습니다. 백엔드 연결을 확인해 주세요.')
        }
      } finally {
        if (active) setPhysioLoading(false)
      }
    }
    void loadSubjects()
    return () => {
      active = false
    }
  }, [accessToken, historyRetry])

  React.useEffect(() => {
    if (!accessToken || !selectedSubjectId) {
      timelineAbortRef.current?.abort()
      timelineAbortRef.current = null
      setTimelineLoading(false)
      return
    }

    // A changed patient selection supersedes the previous request. Cancelling
    // it prevents a slow Supabase read from occupying the single backend
    // worker after the clinician has already selected another patient.
    timelineAbortRef.current?.abort()
    const controller = new AbortController()
    timelineAbortRef.current = controller
    let active = true

    const loadTimeline = async () => {
      setTimelineLoading(true)
      setTimelineError(null)
      try {
        const res = await getUnifiedTimeline(selectedSubjectId, accessToken, 100, controller.signal)
        if (!active || controller.signal.aborted) return
        setTimelineEnabled(res.enabled)
        setTimeline(res.items)
        setTimelineMedications(res.medications || [])
      } catch (err) {
        if (!active || controller.signal.aborted) return
        setTimelineError(err instanceof Error ? err.message : '타임라인을 불러오지 못했습니다')
        setTimeline([])
        setTimelineMedications([])
      } finally {
        if (active && !controller.signal.aborted) setTimelineLoading(false)
      }
    }
    void loadTimeline()
    return () => {
      active = false
      controller.abort()
      if (timelineAbortRef.current === controller) timelineAbortRef.current = null
    }
  }, [accessToken, selectedSubjectId, timelineRetry])

  // Fetch data
  React.useEffect(() => {
    if (!accessToken) {
      setHistory([])
      setStats(null)
      setIsLoading(false)
      return
    }
    // Resolve the shared physio_app context first. A client account is allowed
    // to read its unified timeline but not the backend-local clinician history
    // route; firing both requests during auth/context initialization created a
    // stale 0-item response that overwrote the visible timeline state.
    if (physioLoading || (!physioData && !physioError) || isSelfTimeline || physioError) {
      setHistory([])
      setStats(null)
      setError(null)
      setIsLoading(false)
      return
    }
    let active = true
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [historyResult, statsResult] = await Promise.allSettled([
          getHistory(accessToken, filters),
          getHistoryStats(accessToken, filters.task_type)
        ])

        if (historyResult.status === 'rejected') {
          throw historyResult.reason
        }

        if (!active) return
        setHistory(historyResult.value.data.items)
        setStats(statsResult.status === 'fulfilled' ? statsResult.value.data : null)
        setError(null)
      } catch (err) {
        if (!active) return
        console.error('Failed to fetch history:', err)
        setError('기록 API에 연결하지 못했습니다')
      } finally {
        if (active) setIsLoading(false)
      }
    }
    fetchData()
    return () => {
      active = false
    }
  }, [accessToken, filters, historyRetry, isSelfTimeline, physioData, physioError, physioLoading])

  const handleDelete = async (videoId: string) => {
    if (!accessToken) return
    try {
      await deleteAnalysis(accessToken, videoId)
      setHistory(prev => prev.filter(h => h.video_id !== videoId))
      setDeleteConfirm(null)
    } catch (err) {
      console.error('Failed to delete:', err)
    }
  }

  const handleSignIn = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const supabase = getSupabaseBrowserClient()
    if (!supabase) {
      window.open('https://hawkeye-labeling-tool.vercel.app/history', '_blank', 'noopener,noreferrer')
      return
    }

    setAuthSubmitting(true)
    setAuthError(null)
    try {
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: loginEmail.trim(),
        password: loginPassword,
      })
      if (signInError) {
        setAuthError('이메일 또는 비밀번호를 확인해 주세요.')
      } else {
        setLoginPassword("")
      }
    } catch {
      setAuthError('로그인 서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.')
    } finally {
      setAuthSubmitting(false)
    }
  }

  const handleSignOut = async () => {
    const supabase = getSupabaseBrowserClient()
    if (!supabase) return
    await supabase.auth.signOut()
    setHistory([])
    setStats(null)
    setTimeline([])
    setTimelineMedications([])
    setPhysioData(null)
  }

  const filteredHistory = history.filter(item => {
    if (!searchTerm) return true
    return (
      item.video_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.task_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.patient_id.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })

  if (!authReady) {
    return (
      <PageLayout>
        <div className="min-h-[70vh] grid place-items-center">
          <div className="flex items-center gap-3 text-slate-400">
            <LoaderCircle className="h-5 w-5 animate-spin" />
            임상 기록 접근 권한을 확인하고 있습니다
          </div>
        </div>
      </PageLayout>
    )
  }

  if (!accessToken) {
    return (
      <PageLayout>
        <div className="min-h-[75vh] grid place-items-center px-4 py-10">
          <Card className="w-full max-w-lg overflow-hidden border-border bg-card shadow-xl shadow-foreground/[0.06]">
            <div className="h-1 bg-primary" />
            <CardHeader className="space-y-4 px-7 pt-8 sm:px-9">
              <div className="flex items-center justify-between gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-primary/20 bg-primary/10">
                  <LockKeyhole className="h-6 w-6 text-primary" />
                </div>
                <span className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1.5 text-xs font-medium text-primary">
                  <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" /> 보호된 기록
                </span>
              </div>
              <div>
                <CardTitle className="text-2xl text-foreground">임상 기록 로그인</CardTitle>
                <CardDescription className="mt-2 leading-6 text-muted-foreground">
                  physio_app에 등록된 임상 계정만 환자 이력과 통합 타임라인을 볼 수 있습니다.
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent className="px-7 pb-8 sm:px-9">
              <form className="space-y-4" onSubmit={handleSignIn}>
                <label className="block space-y-2">
                  <span className="text-sm font-medium text-foreground">이메일</span>
                  <input
                    type="email"
                    autoComplete="email"
                    required
                    value={loginEmail}
                    onChange={(event) => setLoginEmail(event.target.value)}
                    disabled={authSubmitting}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2.5 text-foreground outline-none transition placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/15 disabled:cursor-not-allowed disabled:opacity-60"
                  />
                </label>
                <label className="block space-y-2">
                  <span className="text-sm font-medium text-foreground">비밀번호</span>
                  <input
                    type="password"
                    autoComplete="current-password"
                    required
                    value={loginPassword}
                    onChange={(event) => setLoginPassword(event.target.value)}
                    disabled={authSubmitting}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2.5 text-foreground outline-none transition placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/15 disabled:cursor-not-allowed disabled:opacity-60"
                  />
                </label>
                {authError && (
                  <p role="alert" className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2.5 text-sm leading-5 text-amber-700 dark:text-amber-300">
                    {authError}
                  </p>
                )}
                <Button type="submit" className="w-full gap-2" disabled={authSubmitting}>
                  {authSubmitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : authConfigured ? <ShieldCheck className="h-4 w-4" /> : <ExternalLink className="h-4 w-4" />}
                  {authSubmitting ? '확인 중...' : authConfigured ? '안전하게 로그인' : '운영 로그인 화면 열기'}
                </Button>
              </form>
              <div className="mt-6 flex items-start gap-3 rounded-lg border border-border bg-muted/35 px-4 py-3 text-xs leading-5 text-muted-foreground">
                <Activity className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
                <p>분석 실행은 로그인 없이도 가능합니다. 로그인 연결 전에는 결과가 임시 기록으로만 유지됩니다.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </PageLayout>
    )
  }

  return (
    <PageLayout
      contentMaxWidth="max-w-[1480px]"
      agentPanelWidth="w-[20rem]"
      agentPanel={<ChatInterface initialMessages={[{
      id: "1",
      role: "agent",
      content: "검사 이력을 분석해드릴 수 있습니다. '지난 3개월간 보행 점수 변화를 분석해줘'와 같이 질문해보세요.",
      timestamp: new Date()
    }]} />}
    >
      <motion.div
        initial="hidden"
        animate="show"
        variants={{ hidden: {}, show: { transition: { staggerChildren: 0.05 } } }}
        className="space-y-6 pb-10"
      >
        <motion.section
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
          className="relative overflow-hidden rounded-xl border border-border bg-card p-6 shadow-none md:p-7"
        >
          <div className="relative flex min-w-0 flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2 text-xs font-medium tracking-[0.08em] text-primary">
                <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/5 px-2.5 py-1">
                  <CircleCheck className="h-3.5 w-3.5" aria-hidden="true" />
                  임상 기록
                </span>
                <span className="text-muted-foreground">환자별 종단 기록</span>
              </div>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight text-foreground md:text-4xl">분석 이력</h1>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
                {stats ? `총 ${stats.total_analyses}건의 검사 결과와 시간에 따른 변화를 확인합니다.` : '검사 기록을 확인하고 추이를 분석하세요.'}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 xl:justify-end">
              <div className="hidden max-w-[16rem] items-center gap-2 truncate rounded-lg border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground lg:flex">
                <ShieldCheck className="h-3.5 w-3.5 shrink-0 text-primary" aria-hidden="true" />
                <span className="truncate">{signedInEmail || '인증된 임상 계정'}</span>
              </div>
              <Button variant="outline" size="sm" className="gap-2" onClick={() => setHistoryRetry((value) => value + 1)} disabled={isLoading}>
                <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
                새로고침
              </Button>
              <Button variant="outline" size="sm" className="gap-2" onClick={handleSignOut}>
                <LogOut className="h-4 w-4" />
                로그아웃
              </Button>
            </div>
          </div>

          <div className="relative mt-7 grid gap-3 md:grid-cols-[minmax(0,1fr)_auto_auto]">
            <label className="relative block">
              <span className="sr-only">검사 기록 검색</span>
              <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="search"
                placeholder="검사 유형, 환자 ID, 세션 검색"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="h-11 w-full rounded-lg border border-input bg-background pl-10 pr-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20"
              />
            </label>
            <Button variant="outline" className="h-11 justify-center gap-2" onClick={() => setShowFilters(!showFilters)}>
              <Filter className="h-4 w-4" />
              필터
              <ChevronDown className={cn("h-4 w-4 transition-transform", showFilters && "rotate-180")} />
            </Button>
            <div className="hidden items-center gap-2 rounded-lg border border-border bg-muted/30 px-3 text-xs text-muted-foreground md:flex">
              <Database className="h-4 w-4 text-primary" aria-hidden="true" />
              API 연결됨
            </div>
          </div>

          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-4 grid gap-4 rounded-xl border border-border bg-muted/25 p-4 sm:grid-cols-2 xl:grid-cols-4"
            >
              <label className="space-y-1.5 text-xs font-medium text-muted-foreground">검사 유형
                <select value={filters.task_type || ''} onChange={(e) => setFilters(prev => ({ ...prev, task_type: e.target.value || undefined }))} className="h-9 w-full rounded-lg border border-input bg-background px-3 text-sm font-normal text-foreground">
                  <option value="">전체</option><option value="finger_tapping">Finger Tapping</option><option value="gait">Gait</option>
                </select>
              </label>
              <label className="space-y-1.5 text-xs font-medium text-muted-foreground">정렬
                <select value={filters.sort} onChange={(e) => setFilters(prev => ({ ...prev, sort: e.target.value as HistoryFilters['sort'] }))} className="h-9 w-full rounded-lg border border-input bg-background px-3 text-sm font-normal text-foreground">
                  <option value="date_desc">최신순</option><option value="date_asc">오래된순</option><option value="score_desc">점수 높은순</option><option value="score_asc">점수 낮은순</option>
                </select>
              </label>
              <label className="space-y-1.5 text-xs font-medium text-muted-foreground">시작일
                <input type="date" value={filters.start_date || ''} onChange={(e) => setFilters(prev => ({ ...prev, start_date: e.target.value || undefined }))} className="h-9 w-full rounded-lg border border-input bg-background px-3 text-sm text-foreground" />
              </label>
              <label className="space-y-1.5 text-xs font-medium text-muted-foreground">종료일
                <input type="date" value={filters.end_date || ''} onChange={(e) => setFilters(prev => ({ ...prev, end_date: e.target.value || undefined }))} className="h-9 w-full rounded-lg border border-input bg-background px-3 text-sm text-foreground" />
              </label>
            </motion.div>
          )}
        </motion.section>

        {/* Unified Patient Timeline (ParkiCheck + Hawk I) */}
        {accessToken && (physioLoading || physioError || (physioData?.enabled && physioData.subjects.length > 0)) && (
          <Card className="border-border bg-card shadow-none">
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2 text-foreground">
                    <Activity className="h-5 w-5 text-primary" />
                    환자 통합 타임라인
                  </CardTitle>
                  <CardDescription>
                    ParkiCheck 검사와 Hawk I AI 분석이 공통 기록(physio_app)에서 함께 표시됩니다
                  </CardDescription>
                </div>
                {physioData?.subjects.length ? (
                  <select
                    value={selectedSubjectId}
                    onChange={(e) => setSelectedSubjectId(e.target.value)}
                    className="rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground"
                  >
                    {physioData.subjects.map((subject) => (
                      <option key={subject.id} value={subject.id}>
                        {subject.display_name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="text-xs text-muted-foreground">연결 상태 확인 중</span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {physioLoading && !physioData ? (
                <p className="py-4 text-sm text-muted-foreground">환자 기록 연결을 확인하는 중...</p>
              ) : physioError && !physioData ? (
                <div className="history-inline-error flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3 text-sm">
                  <span>{physioError}</span>
                  <button type="button" onClick={() => setHistoryRetry((value) => value + 1)} className="rounded-md border border-current/30 px-3 py-1.5 text-xs font-medium hover:bg-background/60">
                    다시 시도
                  </button>
                </div>
              ) : timelineLoading ? (
                <p className="py-4 text-sm text-muted-foreground">타임라인을 불러오는 중...</p>
              ) : timelineError ? (
                <div className="history-inline-error flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3 text-sm">
                  <span>{timelineError}</span>
                  <button type="button" onClick={() => setTimelineRetry((value) => value + 1)} className="rounded-md border border-current/30 px-3 py-1.5 text-xs font-medium hover:bg-background/60">
                    다시 시도
                  </button>
                </div>
              ) : timelineEnabled === false ? (
                <p className="py-4 text-sm text-muted-foreground">이 백엔드에는 physio_app 연동이 설정되어 있지 않습니다.</p>
              ) : timeline.length === 0 && timelineMedications.length === 0 ? (
                <p className="py-4 text-sm text-muted-foreground">이 환자의 기록이 아직 없습니다.</p>
              ) : (
                <div className="space-y-5">
                  {timelineMedications.length > 0 && (
                    <div>
                      <div className="history-medication-heading mb-2 flex items-center gap-2 text-sm font-medium">
                        최근 환자 보고 복약 기록
                        <span className="text-xs font-normal text-muted-foreground">효과·ON/OFF는 추정하지 않음</span>
                      </div>
                      <div className="space-y-2">
                        {timelineMedications.slice(0, 5).map((medication) => (
                          <div key={medication.event_id || `${medication.medication_code}-${medication.observed_at}`} className="history-medication-item flex flex-wrap items-center gap-3 rounded-lg border p-3">
                            <span className="text-sm font-semibold">{medication.medication_display || medication.medication_code || '약물명 미입력'}</span>
                            {medication.dose_mg !== null && (
                            <span className="text-sm text-foreground/80">{medication.dose_mg}{medication.dose_unit || 'mg'}</span>
                            )}
                            <span className="text-xs text-muted-foreground">{medication.app_source === 'parkicheck' ? 'ParkiCheck 환자 보고' : 'physio_app 기록'}</span>
                            <span className="ml-auto text-xs text-muted-foreground">{medication.observed_at ? new Date(medication.observed_at).toLocaleString('ko-KR') : '시각 미상'}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="space-y-2">
                  {trendPoints.length >= 2 && (
                    <div className="mb-4 rounded-lg border border-border bg-muted/25 p-4">
                      <p className="text-sm font-medium text-foreground">시간대별 추이와 복약</p>
                      <p className="mb-3 text-xs text-muted-foreground">
                        위쪽은 검사 점수, 아래쪽 눈금은 복약 시각입니다. 복약은 점수 축을 공유하지 않습니다.
                      </p>
                      <ResponsiveContainer width="100%" height={200}>
                        <ComposedChart margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis
                            type="number" dataKey="t" domain={['dataMin', 'dataMax']}
                            scale="time" stroke={chartColors.axis} fontSize={11}
                            tickFormatter={(value: number) =>
                              new Date(value).toLocaleDateString('ko-KR', { month: 'numeric', day: 'numeric' })}
                          />
                          <YAxis yAxisId="score" domain={[0, 4]} stroke={chartColors.axis} fontSize={11} width={28} />
                          <YAxis yAxisId="dose" domain={[0, 1]} hide />
                          <Tooltip
                            contentStyle={{ background: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: 8, fontSize: 12 }}
                            labelFormatter={(value: number) => new Date(value).toLocaleString('ko-KR')}
                            formatter={(value: number, name: string) =>
                              name === 'lane' ? ['복약', ''] : [value, '점수']}
                          />
                          <Line
                            yAxisId="score" data={trendPoints} dataKey="score" type="monotone"
                            stroke={chartColors.score} strokeWidth={2} dot={{ r: 3, fill: chartColors.score }}
                          />
                          <Scatter
                            yAxisId="dose" data={doseMarkers} dataKey="lane"
                            fill={chartColors.accent} shape="cross"
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                      <p className="mt-1 text-[10px] text-muted-foreground">
                        노란 눈금 = 환자가 보고한 복약 {doseMarkers.length}건
                      </p>
                    </div>
                  )}

                  {doseAlignedPoints.length >= 2 && (
                    <div className="mb-4 rounded-lg border border-border bg-muted/25 p-4">
                      <p className="text-sm font-medium text-foreground">복약 기준 정렬</p>
                      <p className="mb-3 text-xs text-muted-foreground">
                        마지막 복약 이후 경과 시간에 따른 점수입니다. 약효나 ON/OFF 상태를 추정하지 않고 관측값만 표시합니다.
                      </p>
                      <ResponsiveContainer width="100%" height={180}>
                        <ScatterChart margin={{ top: 8, right: 12, bottom: 20, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                          <XAxis
                            type="number" dataKey="hours" name="복약 후 경과"
                            unit="h" stroke={chartColors.axis} fontSize={11}
                            label={{ value: '복약 후 경과 시간(h)', position: 'insideBottom', offset: -12, fill: chartColors.axis, fontSize: 11 }}
                          />
                          <YAxis
                            type="number" dataKey="score" name="점수"
                            domain={[0, 4]} stroke={chartColors.axis} fontSize={11}
                          />
                          <Tooltip
                            contentStyle={{ background: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-border)', borderRadius: 8, fontSize: 12 }}
                            formatter={(value: number, name: string) => [value, name === 'hours' ? '복약 후(h)' : '점수']}
                          />
                          <Scatter data={doseAlignedPoints} fill={chartColors.score} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {timeline.map((item, idx) => {
                    const key = item.fhir_id || String(idx)
                    const isOpen = expandedItems.has(key)
                    const metricEntries = Object.entries(item.metrics || {}).filter(
                      ([, value]) => value !== null && value !== undefined && value !== ''
                    )
                    const doseResponsive = metricEntries.filter(([name]) => !isDoseResistantMetric(name))
                    const doseResistant = metricEntries.filter(([name]) => isDoseResistantMetric(name))
                    const onHold = item.performability_status === 'hold' || item.score === null

                    return (
                    <div
                      key={key}
                      className="space-y-2 rounded-lg border border-border bg-muted/30 p-3"
                    >
                      <div className="flex flex-wrap items-center gap-3">
                        <span
                          className={cn(
                            "text-xs px-2 py-1 rounded-full border font-medium",
                            item.app_source === 'parkicheck'
                              ? "history-source history-source--patient"
                              : "history-source history-source--ai"
                          )}
                        >
                          {item.app_source === 'parkicheck' ? 'ParkiCheck 검사' : 'Hawk I AI 분석'}
                        </span>
                        <span className="text-sm text-foreground/80">{item.code || '—'}</span>
                        {item.hours_since_last_dose !== null && item.hours_since_last_dose !== undefined && (
                          <span className="history-dose-badge text-xs px-2 py-1 rounded-full border">
                            복약 {item.hours_since_last_dose}시간 후
                            {item.last_dose_medication ? ` · ${item.last_dose_medication}` : ''}
                            {item.last_dose_mg !== null && item.last_dose_mg !== undefined ? ` ${item.last_dose_mg}mg` : ''}
                          </span>
                        )}
                        <span className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          {item.observed_at ? new Date(item.observed_at).toLocaleString('ko-KR') : '시각 미상'}
                        </span>
                      </div>

                      {/* Observation before score: a narrative finding is what a
                          clinician can verify; a bare number invites overreliance. */}
                      {item.rationale ? (
                        <p className="text-sm leading-relaxed text-foreground/90">{item.rationale}</p>
                      ) : (
                        <p className="text-sm italic text-muted-foreground">관찰 근거가 기록되지 않았습니다</p>
                      )}

                      {onHold && (
                        <p className="history-hold-note text-xs">
                          판정 보류 — 자동 점수를 산출하지 않았습니다
                          {item.score_advisory_summary ? ` (${item.score_advisory_summary})` : ''}
                        </p>
                      )}

                      {metricEntries.length > 0 && (
                        <button
                          type="button"
                          onClick={() => setExpandedItems((prev) => {
                            const next = new Set(prev)
                            if (next.has(key)) next.delete(key); else next.add(key)
                            return next
                          })}
                          aria-expanded={isOpen}
                          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                        >
                          <ChevronDown className={cn("h-3 w-3 transition-transform", isOpen && "rotate-180")} />
                          정량 지표 {metricEntries.length}개
                        </button>
                      )}

                      {isOpen && (
                        <div className="grid gap-3 sm:grid-cols-2 pt-1">
                          {[
                            { title: '약물 반응성 지표', entries: doseResponsive, hint: '복약으로 개선될 수 있음' },
                            { title: '약물 저항성 지표', entries: doseResistant, hint: '복약과 무관하게 유지되는 경향' },
                          ].filter((group) => group.entries.length > 0).map((group) => (
                            <div key={group.title} className="rounded-lg border border-border bg-background p-3">
                              <p className="text-xs font-medium text-foreground/80">{group.title}</p>
                              <p className="mb-2 text-[10px] text-muted-foreground">{group.hint}</p>
                              <dl className="space-y-1">
                                {group.entries.map(([name, value]) => (
                                  <div key={name} className="flex justify-between gap-3 text-xs">
                                    <dt className="text-muted-foreground">{metricLabel(name)}</dt>
                                    <dd className="font-mono text-foreground/90">
                                      {typeof value === 'number' ? value.toFixed(2) : String(value)}
                                    </dd>
                                  </div>
                                ))}
                              </dl>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="flex flex-wrap items-center gap-3 border-t border-border pt-1">
                        <span className="text-xs text-muted-foreground">
                          {item.score !== null && item.score !== undefined ? (
                            <>점수 <span className="font-semibold text-foreground">{item.score}</span>{item.severity ? ` · ${item.severity}` : ''}</>
                          ) : '점수 없음'}
                        </span>
                        {(item.scoring_method || item.model_type) && (
                          <span className="text-[10px] text-muted-foreground">
                            산출 {item.scoring_method || '—'}{item.model_type ? ` / ${item.model_type}` : ''}
                          </span>
                        )}
                        {item.confidence !== null && item.confidence !== undefined && (
                          <span className="text-[10px] text-muted-foreground">신뢰도 {String(item.confidence)}</span>
                        )}
                        {item.analysis_id && (
                          <Link href={`/result?id=${item.analysis_id}`} className="ml-auto">
                            <Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs">
                              <Eye className="h-3 w-3" /> 영상 근거
                            </Button>
                          </Link>
                        )}
                      </div>

                      <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] text-muted-foreground/70">
                        {item.activity_session_id && <span>session: {item.activity_session_id}</span>}
                        {item.observation_id && <span>observation: {item.observation_id}</span>}
                        {item.fhir_id && <span>FHIR: {item.fhir_id}</span>}
                      </div>
                    </div>
                    )
                  })}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Stats Overview */}
        {stats && stats.total_analyses > 0 && (
          <motion.div
            variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
            className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"
          >
            <MetricCard label="총 분석" value={stats.total_analyses} caption="인증된 기록" tone="blue" icon={<BarChart3 className="h-5 w-5" />} />
            <MetricCard label="평균 점수" value={stats.average_score?.toFixed(1) || 'N/A'} caption="0–4 임상 점수 범위" tone="green" icon={<TrendingUp className="h-5 w-5" />} />
            <MetricCard label="검사 유형" value={Object.keys(stats.task_distribution).length} caption="보행 · 손가락 태핑" tone="amber" icon={<ClipboardCheck className="h-5 w-5" />} />
            <MetricCard label="최근 검사" value={history[0]?.date.split('T')[0] || 'N/A'} caption="가장 최근 저장 시각" tone="violet" icon={<Clock className="h-5 w-5" />} />
          </motion.div>
        )}

        {/* Charts Section */}
        {stats && stats.trend.length > 0 && (
          <motion.div
            variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
            className="grid gap-4 md:grid-cols-2"
          >
            {/* Trend Chart */}
            <Card className="min-w-0 border-border bg-card shadow-none">
              <CardHeader className="border-b border-border/70 pb-4">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  점수 추이
                </CardTitle>
                <CardDescription>시간에 따른 관찰 점수 변화 · 0–4 scale</CardDescription>
              </CardHeader>
              <CardContent className="pt-5">
                <div className="h-64 min-w-0">
                  <ResponsiveContainer width="100%" height="100%" minWidth={280} minHeight={200}>
                    <LineChart data={stats.trend}>
                      <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                      <XAxis
                        dataKey="date"
                        stroke={chartColors.axis}
                        fontSize={12}
                        tickFormatter={(val) => val.slice(5)} // MM-DD
                      />
                      <YAxis stroke={chartColors.axis} fontSize={12} domain={[0, 4]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--chart-tooltip-bg)',
                          border: '1px solid var(--chart-tooltip-border)',
                          borderRadius: '8px'
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke={chartColors.score}
                        strokeWidth={2}
                        dot={{ fill: chartColors.score, strokeWidth: 2 }}
                        activeDot={{ r: 6, fill: chartColors.accent }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Score Distribution */}
            <Card className="min-w-0 border-border bg-card shadow-none">
              <CardHeader className="border-b border-border/70 pb-4">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <BarChart3 className="h-5 w-5 text-primary" />
                  점수 분포
                </CardTitle>
                <CardDescription>관찰 점수별 분석 횟수</CardDescription>
              </CardHeader>
              <CardContent className="pt-5">
                <div className="h-64 min-w-0">
                  <ResponsiveContainer width="100%" height="100%" minWidth={280} minHeight={200}>
                    <BarChart data={Object.entries(stats.score_distribution).map(([score, count]) => ({
                      score: `Score ${score}`,
                      count,
                      fill: scoreColors[parseInt(score)] || '#64748b'
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                      <XAxis dataKey="score" stroke={chartColors.axis} fontSize={12} />
                      <YAxis stroke={chartColors.axis} fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--chart-tooltip-bg)',
                          border: '1px solid var(--chart-tooltip-border)',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {Object.entries(stats.score_distribution).map(([score], index) => (
                          <Cell key={`cell-${index}`} fill={scoreColors[parseInt(score)] || '#64748b'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Backend-local history is clinician-only. Client accounts use the
            unified physio_app timeline above, so do not present a misleading
            empty "0건" list beneath their real timeline records. */}
        {!isSelfTimeline && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-xs font-medium tracking-[0.14em] text-primary">기록 목록</p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight text-foreground">검사 기록</h2>
            </div>
            <span className="rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">{filteredHistory.length}건 표시</span>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-border border-t-primary" />
            </div>
          ) : error ? (
            <Card className="history-state history-state--error">
              <CardContent className="p-6 text-center">
                <Activity className="mx-auto mb-3 h-9 w-9" />
                <p className="font-medium">기록을 불러오지 못했습니다</p>
                <p className="mx-auto mt-2 max-w-xl text-sm leading-6">
                  기록 API가 응답하지 않았습니다. 운영 백엔드의 임상 기록 설정이 완료되면 분석 이력과 복약 타임라인이 표시됩니다.
                </p>
                <p className="mt-2 text-xs opacity-80">{error}</p>
                <Button className="mt-4" variant="outline" onClick={() => setHistoryRetry((value) => value + 1)}>
                  다시 시도
                </Button>
              </CardContent>
            </Card>
          ) : filteredHistory.length === 0 ? (
            <Card className="border-border bg-card">
              <CardContent className="p-12 text-center">
                <Activity className="mx-auto mb-4 h-12 w-12 text-muted-foreground/60" />
                <p className="text-muted-foreground">분석 기록이 없습니다</p>
                <Link href="/test">
                  <Button className="mt-4" variant="outline">
                    새 검사 시작하기
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <Card className="overflow-hidden border-border bg-card shadow-none">
              <div className="hidden grid-cols-[minmax(12rem,1.4fr)_minmax(10rem,1fr)_10rem_8.5rem] gap-4 border-b border-border bg-muted/40 px-5 py-3 text-[11px] font-semibold tracking-[0.12em] text-muted-foreground md:grid">
                <span>검사</span><span>판정</span><span>검사 시각</span><span className="text-right">작업</span>
              </div>
              <AnimatePresence initial={false}>
                {filteredHistory.map((item) => {
                  const scoreColor = scoreColors[Math.round(item.score || 0)] || '#64748b'
                  return (
                    <motion.div
                      key={item.video_id}
                      layout
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.18 }}
                      className="group grid gap-4 border-b border-border/70 px-5 py-4 last:border-b-0 md:grid-cols-[minmax(12rem,1.4fr)_minmax(10rem,1fr)_10rem_8.5rem] md:items-center"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <span className="h-9 w-1 shrink-0 rounded-full" style={{ backgroundColor: scoreColor }} />
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <h3 className="font-semibold text-foreground">{formatVideoType(item.task_type)}</h3>
                            <span className="font-mono text-[10px] text-muted-foreground">{item.video_id.slice(0, 8)}…</span>
                          </div>
                          <p className="mt-1 text-xs text-muted-foreground">산출 방식 · {item.scoring_method || '기록 없음'}</p>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border text-sm font-semibold" style={{ borderColor: scoreColor, color: scoreColor }}>
                          {item.score?.toFixed(1) || '—'}
                        </span>
                        <span className={cn("rounded-full border px-2 py-1 text-xs font-medium", severityColors[item.severity] || severityColors["Unknown"])}>
                          {item.severity}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Calendar className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                        <span>{new Date(item.date).toLocaleDateString('ko-KR')}</span>
                        <span className="hidden lg:inline">{new Date(item.date).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>

                      <div className="flex items-center justify-start gap-1 md:justify-end">
                        {deleteConfirm === item.video_id ? (
                          <div className="flex items-center gap-1">
                            <span className="mr-1 text-xs text-muted-foreground">삭제할까요?</span>
                            <Button size="sm" variant="ghost" className="text-destructive" onClick={() => handleDelete(item.video_id)}>삭제</Button>
                            <Button size="sm" variant="ghost" onClick={() => setDeleteConfirm(null)}>취소</Button>
                          </div>
                        ) : (
                          <>
                            <Button size="icon" variant="ghost" className="text-muted-foreground opacity-70 transition-opacity hover:text-destructive md:opacity-0 md:group-hover:opacity-100" onClick={() => setDeleteConfirm(item.video_id)} title="기록 삭제">
                              <Trash2 className="h-4 w-4" />
                            </Button>
                            <Link href={`/result?id=${item.video_id}`}>
                              <Button size="sm" variant="outline" className="h-9 gap-1.5 px-3">
                                상세보기
                                <ArrowUpRight className="h-3.5 w-3.5" />
                              </Button>
                            </Link>
                          </>
                        )}
                      </div>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </Card>
          )}
        </div>
        )}
      </motion.div>
    </PageLayout>
  )
}
