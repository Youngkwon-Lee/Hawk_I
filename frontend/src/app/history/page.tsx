"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import {
  Calendar, Activity, ChevronRight, Filter, TrendingUp,
  BarChart3, Clock, Trash2, Eye, Search, ChevronDown,
  LoaderCircle, LockKeyhole, LogOut, ShieldCheck
} from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import {
  getHistory, getHistoryStats, deleteAnalysis, formatVideoType, getPhysioSubjects,
  type HistoryItem, type HistoryStats, type HistoryFilters, type PhysioSubjectsResponse
} from "@/lib/services/api"
import { getUnifiedTimeline, type MedicationEvent, type TimelineItem } from "@/lib/services/timeline"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts'

// Severity color mapping
const severityColors: Record<string, string> = {
  "Normal": "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  "Slight": "text-sky-400 bg-sky-500/10 border-sky-500/30",
  "Mild": "text-amber-400 bg-amber-500/10 border-amber-500/30",
  "Moderate": "text-orange-400 bg-orange-500/10 border-orange-500/30",
  "Severe": "text-rose-400 bg-rose-500/10 border-rose-500/30",
  "Unknown": "text-slate-400 bg-slate-500/10 border-slate-500/30"
}

const scoreColors = ["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]

export default function HistoryPage() {
  const [authReady, setAuthReady] = React.useState(false)
  const [accessToken, setAccessToken] = React.useState<string | null>(null)
  const [signedInEmail, setSignedInEmail] = React.useState<string | null>(null)
  const [loginEmail, setLoginEmail] = React.useState("")
  const [loginPassword, setLoginPassword] = React.useState("")
  const [authError, setAuthError] = React.useState<string | null>(null)
  const [authSubmitting, setAuthSubmitting] = React.useState(false)
  const [history, setHistory] = React.useState<HistoryItem[]>([])
  const [stats, setStats] = React.useState<HistoryStats['data'] | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
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
  const [selectedSubjectId, setSelectedSubjectId] = React.useState("")
  const [timeline, setTimeline] = React.useState<TimelineItem[]>([])
  const [timelineMedications, setTimelineMedications] = React.useState<MedicationEvent[]>([])
  const [timelineEnabled, setTimelineEnabled] = React.useState<boolean | null>(null)
  const [timelineLoading, setTimelineLoading] = React.useState(false)
  const [timelineError, setTimelineError] = React.useState<string | null>(null)

  React.useEffect(() => {
    const supabase = getSupabaseBrowserClient()
    if (!supabase) {
      setAuthError('로그인 설정을 불러오지 못했습니다. 관리자에게 문의해 주세요.')
      setAuthReady(true)
      return
    }

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
      setSelectedSubjectId("")
      setTimelineMedications([])
      return
    }
    const loadSubjects = async () => {
      try {
        const data = await getPhysioSubjects(accessToken)
        setPhysioData(data)
        if (data.enabled && data.subjects.length > 0) {
          setSelectedSubjectId(data.default_subject_id || data.subjects[0].id)
        }
      } catch {
        setPhysioData(null)
      }
    }
    void loadSubjects()
  }, [accessToken])

  React.useEffect(() => {
    if (!accessToken || !selectedSubjectId) return
    const loadTimeline = async () => {
      setTimelineLoading(true)
      setTimelineError(null)
      try {
        const res = await getUnifiedTimeline(selectedSubjectId, accessToken)
        setTimelineEnabled(res.enabled)
        setTimeline(res.items)
        setTimelineMedications(res.medications || [])
      } catch (err) {
        setTimelineError(err instanceof Error ? err.message : '타임라인을 불러오지 못했습니다')
        setTimeline([])
        setTimelineMedications([])
      } finally {
        setTimelineLoading(false)
      }
    }
    void loadTimeline()
  }, [accessToken, selectedSubjectId])

  // Fetch data
  React.useEffect(() => {
    if (!accessToken) {
      setHistory([])
      setStats(null)
      setIsLoading(false)
      return
    }
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [historyRes, statsRes] = await Promise.all([
          getHistory(accessToken, filters),
          getHistoryStats(accessToken, filters.task_type)
        ])
        setHistory(historyRes.data.items)
        setStats(statsRes.data)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch history:', err)
        setError('Failed to load history data')
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [accessToken, filters])

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
    if (!supabase) return

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
          <Card className="w-full max-w-md overflow-hidden border-slate-700/70 bg-slate-950/90 shadow-2xl shadow-sky-950/30">
            <div className="h-1 bg-gradient-to-r from-sky-400 via-emerald-400 to-cyan-300" />
            <CardHeader className="space-y-4 pt-8">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/10">
                <LockKeyhole className="h-6 w-6 text-emerald-300" />
              </div>
              <div>
                <CardTitle className="text-2xl text-white">임상 기록 로그인</CardTitle>
                <CardDescription className="mt-2 leading-6 text-slate-400">
                  physio_app에 등록된 임상 계정만 환자 이력과 통합 타임라인을 볼 수 있습니다.
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <form className="space-y-4" onSubmit={handleSignIn}>
                <label className="block space-y-2">
                  <span className="text-sm font-medium text-slate-300">이메일</span>
                  <input
                    type="email"
                    autoComplete="email"
                    required
                    value={loginEmail}
                    onChange={(event) => setLoginEmail(event.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-white outline-none transition focus:border-emerald-400/70 focus:ring-2 focus:ring-emerald-400/15"
                  />
                </label>
                <label className="block space-y-2">
                  <span className="text-sm font-medium text-slate-300">비밀번호</span>
                  <input
                    type="password"
                    autoComplete="current-password"
                    required
                    value={loginPassword}
                    onChange={(event) => setLoginPassword(event.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-white outline-none transition focus:border-emerald-400/70 focus:ring-2 focus:ring-emerald-400/15"
                  />
                </label>
                {authError && (
                  <p role="alert" className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
                    {authError}
                  </p>
                )}
                <Button type="submit" className="w-full gap-2" disabled={authSubmitting}>
                  {authSubmitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
                  {authSubmitting ? '확인 중...' : '안전하게 로그인'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </PageLayout>
    )
  }

  return (
    <PageLayout agentPanel={<ChatInterface initialMessages={[{
      id: "1",
      role: "agent",
      content: "검사 이력을 분석해드릴 수 있습니다. '지난 3개월간 보행 점수 변화를 분석해줘'와 같이 질문해보세요.",
      timestamp: new Date()
    }]} />}>
      <div className="space-y-8 pb-10">
        {/* Header with Glass Effect */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-slate-700/50 p-8">
          <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:32px_32px]" />
          <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl" />

          <div className="relative z-10">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div>
                <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
                  분석 이력
                </h1>
                <p className="text-slate-400 mt-2 text-lg">
                  {stats ? `총 ${stats.total_analyses}건의 분석 기록` : '검사 기록을 확인하고 추이를 분석하세요'}
                </p>
              </div>

              <div className="flex gap-3">
                <div className="hidden lg:flex items-center rounded-lg border border-slate-700 bg-slate-800/40 px-3 text-xs text-slate-400">
                  {signedInEmail || '인증된 임상 계정'}
                </div>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
                  <input
                    type="text"
                    placeholder="검색..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 w-48"
                  />
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2 bg-slate-800/50 border-slate-700 hover:bg-slate-700"
                  onClick={() => setShowFilters(!showFilters)}
                >
                  <Filter className="h-4 w-4" />
                  필터
                  <ChevronDown className={cn("h-4 w-4 transition-transform", showFilters && "rotate-180")} />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2 bg-slate-800/50 border-slate-700 hover:bg-slate-700"
                  onClick={handleSignOut}
                >
                  <LogOut className="h-4 w-4" />
                  로그아웃
                </Button>
              </div>
            </div>

            {/* Filter Panel */}
            {showFilters && (
              <div className="mt-6 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50 animate-in slide-in-from-top-2 fade-in duration-200">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">검사 유형</label>
                    <select
                      value={filters.task_type || ''}
                      onChange={(e) => setFilters(prev => ({ ...prev, task_type: e.target.value || undefined }))}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm"
                    >
                      <option value="">전체</option>
                      <option value="finger_tapping">Finger Tapping</option>
                      <option value="gait">Gait</option>
                      {/* <option value="hand_movement">Hand Movement</option> */}
                      {/* <option value="leg_agility">Leg Agility</option> */}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">정렬</label>
                    <select
                      value={filters.sort}
                      onChange={(e) => setFilters(prev => ({ ...prev, sort: e.target.value as HistoryFilters['sort'] }))}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm"
                    >
                      <option value="date_desc">최신순</option>
                      <option value="date_asc">오래된순</option>
                      <option value="score_desc">점수 높은순</option>
                      <option value="score_asc">점수 낮은순</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">시작일</label>
                    <input
                      type="date"
                      value={filters.start_date || ''}
                      onChange={(e) => setFilters(prev => ({ ...prev, start_date: e.target.value || undefined }))}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">종료일</label>
                    <input
                      type="date"
                      value={filters.end_date || ''}
                      onChange={(e) => setFilters(prev => ({ ...prev, end_date: e.target.value || undefined }))}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Unified Patient Timeline (ParkiCheck + Hawk I) */}
        {physioData?.enabled && physioData.subjects.length > 0 && (
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="h-5 w-5 text-emerald-400" />
                    환자 통합 타임라인
                  </CardTitle>
                  <CardDescription>
                    ParkiCheck 검사와 Hawk I AI 분석이 공통 기록(physio_app)에서 함께 표시됩니다
                  </CardDescription>
                </div>
                <select
                  value={selectedSubjectId}
                  onChange={(e) => setSelectedSubjectId(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
                >
                  {physioData.subjects.map((subject) => (
                    <option key={subject.id} value={subject.id}>
                      {subject.display_name}
                    </option>
                  ))}
                </select>
              </div>
            </CardHeader>
            <CardContent>
              {timelineLoading ? (
                <p className="text-sm text-slate-500 py-4">타임라인을 불러오는 중...</p>
              ) : timelineError ? (
                <p className="text-sm text-rose-400 py-4">{timelineError}</p>
              ) : timelineEnabled === false ? (
                <p className="text-sm text-slate-500 py-4">이 백엔드에는 physio_app 연동이 설정되어 있지 않습니다.</p>
              ) : timeline.length === 0 && timelineMedications.length === 0 ? (
                <p className="text-sm text-slate-500 py-4">이 환자의 기록이 아직 없습니다.</p>
              ) : (
                <div className="space-y-5">
                  {timelineMedications.length > 0 && (
                    <div>
                      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-amber-300">
                        최근 환자 보고 복약 기록
                        <span className="text-xs font-normal text-slate-500">효과·ON/OFF는 추정하지 않음</span>
                      </div>
                      <div className="space-y-2">
                        {timelineMedications.slice(0, 5).map((medication) => (
                          <div key={medication.event_id || `${medication.medication_code}-${medication.observed_at}`} className="flex flex-wrap items-center gap-3 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
                            <span className="text-sm font-semibold text-amber-200">{medication.medication_display || medication.medication_code || '약물명 미입력'}</span>
                            {medication.dose_mg !== null && (
                              <span className="text-sm text-slate-300">{medication.dose_mg}{medication.dose_unit || 'mg'}</span>
                            )}
                            <span className="text-xs text-slate-500">{medication.app_source === 'parkicheck' ? 'ParkiCheck 환자 보고' : 'physio_app 기록'}</span>
                            <span className="ml-auto text-xs text-slate-500">{medication.observed_at ? new Date(medication.observed_at).toLocaleString('ko-KR') : '시각 미상'}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="space-y-2">
                  {timeline.map((item, idx) => (
                    <div
                      key={item.fhir_id || idx}
                      className="flex flex-wrap items-center gap-3 p-3 rounded-lg bg-slate-800/40 border border-slate-700/50"
                    >
                      <span
                        className={cn(
                          "text-xs px-2 py-1 rounded-full border font-medium",
                          item.app_source === 'parkicheck'
                            ? "text-sky-400 bg-sky-500/10 border-sky-500/30"
                            : "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
                        )}
                      >
                        {item.app_source === 'parkicheck' ? 'ParkiCheck 검사' : 'Hawk I AI 분석'}
                      </span>
                      <span className="text-sm text-slate-300">{item.code || '—'}</span>
                      <span className="text-sm font-semibold text-white">
                        {item.score !== null && item.score !== undefined ? `점수 ${item.score}` : '점수 없음'}
                      </span>
                      {item.confidence !== null && item.confidence !== undefined && (
                        <span className="text-xs text-slate-500">신뢰도 {String(item.confidence)}</span>
                      )}
                      {item.has_medication_context && (
                        <span className="text-xs text-amber-400/80">복약 기록됨</span>
                      )}
                      <span className="text-xs text-slate-500 ml-auto flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {item.observed_at ? new Date(item.observed_at).toLocaleString('ko-KR') : '시각 미상'}
                      </span>
                      {item.app_source !== 'parkicheck' && item.analysis_id && (
                        <Link href={`/result?id=${item.analysis_id}`}>
                          <Button variant="outline" size="sm" className="gap-1 border-slate-700 hover:bg-slate-700 h-7 px-2 text-xs">
                            <Eye className="h-3 w-3" /> 결과
                          </Button>
                        </Link>
                      )}
                    </div>
                  ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Stats Overview */}
        {stats && stats.total_analyses > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-all">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-blue-500/10">
                    <BarChart3 className="h-6 w-6 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-white">{stats.total_analyses}</p>
                    <p className="text-sm text-slate-500">총 분석 수</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-all">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-emerald-500/10">
                    <TrendingUp className="h-6 w-6 text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {stats.average_score?.toFixed(1) || 'N/A'}
                    </p>
                    <p className="text-sm text-slate-500">평균 점수</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-all">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-amber-500/10">
                    <Activity className="h-6 w-6 text-amber-400" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {Object.keys(stats.task_distribution).length}
                    </p>
                    <p className="text-sm text-slate-500">검사 유형</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-all">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-purple-500/10">
                    <Clock className="h-6 w-6 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {history[0]?.date.split('T')[0] || 'N/A'}
                    </p>
                    <p className="text-sm text-slate-500">최근 검사</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Charts Section */}
        {stats && stats.trend.length > 0 && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Trend Chart */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-blue-400" />
                  점수 추이
                </CardTitle>
                <CardDescription>시간에 따른 UPDRS 점수 변화</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={stats.trend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="date"
                        stroke="#64748b"
                        fontSize={12}
                        tickFormatter={(val) => val.slice(5)} // MM-DD
                      />
                      <YAxis stroke="#64748b" fontSize={12} domain={[0, 4]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                        activeDot={{ r: 6, fill: '#60a5fa' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Score Distribution */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-emerald-400" />
                  점수 분포
                </CardTitle>
                <CardDescription>UPDRS 점수별 분석 횟수</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={Object.entries(stats.score_distribution).map(([score, count]) => ({
                      score: `Score ${score}`,
                      count,
                      fill: scoreColors[parseInt(score)] || '#64748b'
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="score" stroke="#64748b" fontSize={12} />
                      <YAxis stroke="#64748b" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
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
          </div>
        )}

        {/* History List */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">검사 기록</h2>
            <span className="text-sm text-slate-500">{filteredHistory.length}건</span>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
            </div>
          ) : error ? (
            <Card className="bg-red-500/10 border-red-500/30">
              <CardContent className="p-6 text-center text-red-400">
                {error}
              </CardContent>
            </Card>
          ) : filteredHistory.length === 0 ? (
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-12 text-center">
                <Activity className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-500">분석 기록이 없습니다</p>
                <Link href="/test">
                  <Button className="mt-4" variant="outline">
                    새 검사 시작하기
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredHistory.map((item, index) => (
                <div
                  key={item.video_id}
                  className="group relative animate-in fade-in slide-in-from-bottom-2"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-600 hover:bg-slate-800/50 transition-all duration-200">
                    <CardContent className="p-0">
                      <div className="flex items-center">
                        {/* Score Indicator */}
                        <div
                          className="w-2 h-full min-h-[100px] rounded-l-lg"
                          style={{ backgroundColor: scoreColors[Math.round(item.score || 0)] || '#64748b' }}
                        />

                        <div className="flex-1 p-5 flex items-center justify-between">
                          <div className="flex items-center gap-5">
                            {/* Score Circle */}
                            <div
                              className="w-14 h-14 rounded-full flex items-center justify-center text-xl font-bold border-2"
                              style={{
                                borderColor: scoreColors[Math.round(item.score || 0)] || '#64748b',
                                color: scoreColors[Math.round(item.score || 0)] || '#64748b'
                              }}
                            >
                              {item.score?.toFixed(1) || 'N/A'}
                            </div>

                            <div>
                              <div className="flex items-center gap-3 mb-1">
                                <h3 className="font-semibold text-lg">{formatVideoType(item.task_type)}</h3>
                                <span className={cn(
                                  "text-xs px-2 py-0.5 rounded-full border",
                                  severityColors[item.severity] || severityColors["Unknown"]
                                )}>
                                  {item.severity}
                                </span>
                                <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded">
                                  {item.scoring_method}
                                </span>
                              </div>
                              <div className="flex items-center gap-4 text-sm text-slate-500">
                                <span className="flex items-center gap-1">
                                  <Calendar className="h-3.5 w-3.5" />
                                  {new Date(item.date).toLocaleDateString('ko-KR')}
                                </span>
                                <span className="flex items-center gap-1">
                                  <Clock className="h-3.5 w-3.5" />
                                  {new Date(item.date).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                                </span>
                                <span className="text-xs text-slate-600">
                                  ID: {item.video_id.slice(0, 8)}...
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            {deleteConfirm === item.video_id ? (
                              <div className="flex items-center gap-2 animate-in fade-in">
                                <span className="text-xs text-slate-400">삭제할까요?</span>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                                  onClick={() => handleDelete(item.video_id)}
                                >
                                  삭제
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => setDeleteConfirm(null)}
                                >
                                  취소
                                </Button>
                              </div>
                            ) : (
                              <>
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-400"
                                  onClick={() => setDeleteConfirm(item.video_id)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                                <Link href={`/result?id=${item.video_id}`}>
                                  <Button
                                    size="icon"
                                    variant="ghost"
                                    className="text-slate-400 hover:text-white"
                                  >
                                    <Eye className="h-4 w-4" />
                                  </Button>
                                </Link>
                                <Link href={`/result?id=${item.video_id}`}>
                                  <Button size="sm" variant="ghost" className="gap-1">
                                    상세보기
                                    <ChevronRight className="h-4 w-4" />
                                  </Button>
                                </Link>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Custom Grid Pattern */}
      <style jsx global>{`
        .bg-grid-white {
          background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='32' height='32' fill='none' stroke='rgb(255 255 255 / 0.02)'%3e%3cpath d='M0 .5H31.5V32'/%3e%3c/svg%3e");
        }
      `}</style>
    </PageLayout>
  )
}
