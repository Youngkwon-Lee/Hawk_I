"use client"

import * as React from "react"
import { LogOut, Moon, Settings2, Sun } from "lucide-react"
import { PageLayout } from "@/components/layout/PageLayout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { ThemeToggle } from "@/components/layout/ThemeToggle"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"

export default function SettingsPage() {
    const [email, setEmail] = React.useState<string | null>(null)
    const [theme, setTheme] = React.useState<"light" | "dark">("light")
    const [signedOut, setSignedOut] = React.useState(false)

    React.useEffect(() => {
        const storedTheme = window.localStorage.getItem("hawkeye-theme")
        if (storedTheme === "light" || storedTheme === "dark") setTheme(storedTheme)
        const supabase = getSupabaseBrowserClient()
        if (!supabase) return
        void supabase.auth.getSession().then(({ data }) => setEmail(data.session?.user.email ?? null))
    }, [])

    const handleSignOut = async () => {
        const supabase = getSupabaseBrowserClient()
        if (!supabase) return
        await supabase.auth.signOut()
        setEmail(null)
        setSignedOut(true)
    }

    return (
        <PageLayout>
            <div className="space-y-6">
                <div className="border-b border-border pb-5">
                    <p className="text-xs font-medium uppercase tracking-[0.16em] text-primary">Workspace preferences</p>
                    <h1 className="mt-2 text-3xl font-semibold tracking-tight">설정</h1>
                    <p className="mt-2 text-sm text-muted-foreground">계정과 화면 환경을 관리합니다.</p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                        <CardHeader><CardTitle className="flex items-center gap-2 text-lg"><Settings2 className="h-5 w-5 text-primary" aria-hidden="true" />화면</CardTitle><CardDescription>HawkEye 화면 표시 설정</CardDescription></CardHeader>
                        <CardContent className="flex items-center justify-between gap-4"><div><p className="text-sm font-medium">현재 테마</p><p className="mt-1 text-xs text-muted-foreground">{theme === "dark" ? "다크 모드" : "라이트 모드"}</p></div><div className="flex items-center gap-2"><Sun className="h-4 w-4 text-muted-foreground" aria-hidden="true" /><ThemeToggle /><Moon className="h-4 w-4 text-muted-foreground" aria-hidden="true" /></div></CardContent>
                    </Card>
                    <Card>
                        <CardHeader><CardTitle className="text-lg">계정</CardTitle><CardDescription>임상 기록 연결 상태</CardDescription></CardHeader>
                        <CardContent>
                            {email ? <><p className="text-sm font-medium">{email}</p><Button variant="outline" size="sm" className="mt-4" onClick={handleSignOut}><LogOut className="mr-2 h-4 w-4" aria-hidden="true" />로그아웃</Button></> : <p className="text-sm text-muted-foreground">로그인되지 않았습니다. 기록 화면에서 로그인할 수 있습니다.</p>}
                            {signedOut && <p className="mt-3 text-xs text-muted-foreground">로그아웃되었습니다.</p>}
                        </CardContent>
                    </Card>
                </div>
            </div>
        </PageLayout>
    )
}
