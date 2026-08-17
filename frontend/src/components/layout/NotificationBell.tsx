"use client"

import * as React from "react"
import Link from "next/link"
import { Bell } from "lucide-react"
import { getPhysioSubjects } from "@/lib/services/api"
import { getUnifiedTimeline, type TimelineItem } from "@/lib/services/timeline"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"

const POLL_INTERVAL_MS = 30_000

function storageKey(userId: string): string {
    return `hawkeye-notifications-seen:${userId}`
}

function latestParkiCheckItems(items: TimelineItem[]): TimelineItem[] {
    return items
        .filter((item) => item.app_source === "parkicheck" && item.observed_at)
        .sort((left, right) => new Date(right.observed_at as string).getTime() - new Date(left.observed_at as string).getTime())
}

export function NotificationBell() {
    const [unreadCount, setUnreadCount] = React.useState(0)
    const [userId, setUserId] = React.useState<string | null>(null)
    const latestRef = React.useRef<string | null>(null)

    const refresh = React.useCallback(async (currentUserId: string) => {
        const supabase = getSupabaseBrowserClient()
        if (!supabase) return
        const { data } = await supabase.auth.getSession()
        const accessToken = data.session?.access_token
        if (!accessToken) return

        try {
            const subjectResponse = await getPhysioSubjects(accessToken)
            if (!subjectResponse.enabled || subjectResponse.subjects.length === 0) return

            const responses = await Promise.all(
                subjectResponse.subjects.map((subject) => getUnifiedTimeline(subject.id, accessToken, 100).catch(() => null)),
            )
            const items = latestParkiCheckItems(responses.flatMap((response) => response?.items || []))
            const latest = items[0]?.observed_at || null
            if (!latest) return

            const seenAt = window.localStorage.getItem(storageKey(currentUserId))
            if (!seenAt) {
                window.localStorage.setItem(storageKey(currentUserId), latest)
                setUnreadCount(0)
            } else {
                setUnreadCount(items.filter((item) => new Date(item.observed_at as string).getTime() > new Date(seenAt).getTime()).length)
            }
            latestRef.current = latest
        } catch {
            // Notifications are advisory; a history/API outage must not block navigation.
        }
    }, [])

    React.useEffect(() => {
        let mounted = true
        const supabase = getSupabaseBrowserClient()
        if (!supabase) return

        const load = async () => {
            const { data } = await supabase.auth.getSession()
            if (!mounted) return
            const currentUserId = data.session?.user.id || null
            setUserId(currentUserId)
            if (currentUserId) void refresh(currentUserId)
        }
        void load()

        const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
            const currentUserId = session?.user.id || null
            setUserId(currentUserId)
            setUnreadCount(0)
            if (currentUserId) void refresh(currentUserId)
        })
        const interval = window.setInterval(() => {
            if (userId) void refresh(userId)
        }, POLL_INTERVAL_MS)

        return () => {
            mounted = false
            listener.subscription.unsubscribe()
            window.clearInterval(interval)
        }
    }, [refresh, userId])

    if (!userId) return null

    const markSeen = () => {
        if (latestRef.current) window.localStorage.setItem(storageKey(userId), latestRef.current)
        setUnreadCount(0)
    }

    return (
        <Link
            href="/history"
            onClick={markSeen}
            aria-label={unreadCount > 0 ? `새 ParkiCheck 기록 ${unreadCount}건` : "알림 없음"}
            title={unreadCount > 0 ? `새 ParkiCheck 기록 ${unreadCount}건` : "새 기록 없음"}
            className="relative inline-flex h-10 w-10 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
            <Bell className="h-4 w-4" aria-hidden="true" />
            {unreadCount > 0 && (
                <span className="absolute right-1 top-1 flex min-h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold leading-4 text-primary-foreground">
                    {unreadCount > 9 ? "9+" : unreadCount}
                </span>
            )}
        </Link>
    )
}
