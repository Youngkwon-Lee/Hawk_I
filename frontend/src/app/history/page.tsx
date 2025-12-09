"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import {
  Calendar, Activity, ChevronRight, Filter, TrendingUp, TrendingDown,
  BarChart3, Clock, Trash2, Eye, X, Search, ChevronDown
} from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import {
  getHistory, getHistoryStats, deleteAnalysis, formatVideoType,
  type HistoryItem, type HistoryStats, type HistoryFilters
} from "@/lib/services/api"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, PieChart, Pie
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

const taskTypeColors: Record<string, string> = {
  "finger_tapping": "#3b82f6",
  "gait": "#10b981",
  "hand_movement": "#8b5cf6",
  "leg_agility": "#f59e0b"
}

const scoreColors = ["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]

export default function HistoryPage() {
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

  // Fetch data
  React.useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [historyRes, statsRes] = await Promise.all([
          getHistory(filters),
          getHistoryStats(filters.task_type)
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
  }, [filters])

  const handleDelete = async (videoId: string) => {
    try {
      await deleteAnalysis(videoId)
      setHistory(prev => prev.filter(h => h.video_id !== videoId))
      setDeleteConfirm(null)
    } catch (err) {
      console.error('Failed to delete:', err)
    }
  }

  const filteredHistory = history.filter(item => {
    if (!searchTerm) return true
    return (
      item.video_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.task_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.patient_id.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })

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
                      <option value="hand_movement">Hand Movement</option>
                      <option value="leg_agility">Leg Agility</option>
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
