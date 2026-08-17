"use client"

import * as React from "react"
import Link from "next/link"
import { ArrowRight, LoaderCircle, Users, UserRound } from "lucide-react"
import { PageLayout } from "@/components/layout/PageLayout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card"
import { getPhysioSubjects, type PhysioSubject } from "@/lib/services/api"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"

export default function PatientsPage() {
    const [subjects, setSubjects] = React.useState<PhysioSubject[]>([])
    const [email, setEmail] = React.useState<string | null>(null)
    const [loading, setLoading] = React.useState(true)
    const [message, setMessage] = React.useState<string | null>(null)

    React.useEffect(() => {
        let mounted = true
        const load = async () => {
            const supabase = getSupabaseBrowserClient()
            if (!supabase) {
                if (mounted) {
                    setMessage("임상 계정 연결이 아직 설정되지 않았습니다.")
                    setLoading(false)
                }
                return
            }

            const { data } = await supabase.auth.getSession()
            if (!mounted) return
            const accessToken = data.session?.access_token
            setEmail(data.session?.user.email ?? null)
            if (!accessToken) {
                setMessage("환자 목록을 보려면 임상 계정으로 로그인하세요.")
                setLoading(false)
                return
            }

            try {
                const response = await getPhysioSubjects(accessToken)
                if (!mounted) return
                setSubjects(response.subjects)
                if (!response.enabled) setMessage("연결된 임상 조직이나 환자 목록이 없습니다.")
            } catch (error) {
                if (mounted) setMessage(error instanceof Error ? error.message : "환자 목록을 불러오지 못했습니다.")
            } finally {
                if (mounted) setLoading(false)
            }
        }
        void load()
        return () => { mounted = false }
    }, [])

    return (
        <PageLayout>
            <div className="space-y-6">
                <div className="border-b border-border pb-5">
                    <p className="text-xs font-medium uppercase tracking-[0.16em] text-primary">Clinical workspace</p>
                    <h1 className="mt-2 text-3xl font-semibold tracking-tight">환자</h1>
                    <p className="mt-2 text-sm text-muted-foreground">연결된 임상 계정의 환자와 분석 대상을 관리합니다.</p>
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-lg"><Users className="h-5 w-5 text-primary" aria-hidden="true" />환자 목록</CardTitle>
                        <CardDescription>{email ? `${email} 계정에 연결된 대상자` : "로그인 후 연결된 대상자를 확인할 수 있습니다."}</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                            <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground"><LoaderCircle className="h-4 w-4 animate-spin" />환자 목록을 불러오는 중입니다.</div>
                        ) : subjects.length > 0 ? (
                            <div className="grid gap-3 sm:grid-cols-2">
                                {subjects.map((subject) => (
                                    <div key={subject.id} className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card/60 p-4">
                                        <div className="flex min-w-0 items-center gap-3">
                                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary"><UserRound className="h-5 w-5" aria-hidden="true" /></div>
                                            <div className="min-w-0"><p className="truncate text-sm font-medium">{subject.display_name}</p><p className="mt-1 text-xs text-muted-foreground">{subject.role || subject.user_type || "환자"}</p></div>
                                        </div>
                                        {subject.is_default && <span className="shrink-0 rounded-full bg-primary/10 px-2 py-1 text-xs text-primary">기본 대상</span>}
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="rounded-lg border border-dashed border-border bg-muted/20 p-6 text-sm text-muted-foreground">
                                {message || "연결된 환자 목록이 없습니다."}
                            </div>
                        )}
                    </CardContent>
                </Card>

                <Card className="border-dashed bg-muted/20">
                    <CardContent className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
                        <div><p className="text-sm font-medium">새 검사에서 환자를 선택할 수 있습니다.</p><p className="mt-1 text-xs text-muted-foreground">환자 연결 후 영상 분석 결과가 해당 기록에 저장됩니다.</p></div>
                        <Link href="/test" className="inline-flex h-9 items-center justify-center whitespace-nowrap rounded-md border border-input bg-background px-3 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">새 검사 시작 <ArrowRight className="ml-1 h-4 w-4" aria-hidden="true" /></Link>
                    </CardContent>
                </Card>
            </div>
        </PageLayout>
    )
}
