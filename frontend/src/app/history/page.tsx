"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { Card, CardContent } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Calendar, Activity, ChevronRight, Filter } from "lucide-react"
import Link from "next/link"

const HISTORY_DATA = [
    { id: 1, date: "2025-11-23", type: "Gait", score: 2, status: "Moderate", note: "Slightly worse than last month" },
    { id: 2, date: "2025-11-05", type: "Finger Tapping", score: 1, status: "Mild", note: "Stable rhythm" },
    { id: 3, date: "2025-10-20", type: "Gait", score: 1, status: "Mild", note: "Good walking speed" },
    { id: 4, date: "2025-10-02", type: "Turning", score: 2, status: "Moderate", note: "Some hesitation observed" },
    { id: 5, date: "2025-09-15", type: "Gait", score: 1, status: "Mild", note: "Baseline established" },
]

export default function HistoryPage() {
    return (
        <PageLayout agentPanel={<ChatInterface initialMessages={[{
            id: "1",
            role: "agent",
            content: "I can help you analyze your history. Try asking 'How has my gait changed over the last 3 months?'",
            timestamp: new Date()
        }]} />}>
            <div className="space-y-8">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Test History</h1>
                        <p className="text-muted-foreground mt-2">View and track your past assessments.</p>
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="gap-2">
                            <Filter className="h-4 w-4" /> Filter
                        </Button>
                        <Button variant="outline" size="sm" className="gap-2">
                            <Calendar className="h-4 w-4" /> Last 3 Months
                        </Button>
                    </div>
                </div>

                <div className="space-y-4">
                    {HISTORY_DATA.map((item) => (
                        <Link key={item.id} href={item.type === "Finger Tapping" ? "/result?type=finger" : "/result?type=gait"}>
                            <Card className="hover:bg-accent/50 transition-colors cursor-pointer mb-4">
                                <CardContent className="p-6 flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="p-3 rounded-full bg-primary/10 text-primary">
                                            <Activity className="h-5 w-5" />
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <h3 className="font-semibold">{item.type} Analysis</h3>
                                                <span className="text-xs text-muted-foreground px-2 py-0.5 rounded-full bg-secondary">
                                                    {item.date}
                                                </span>
                                            </div>
                                            <p className="text-sm text-muted-foreground mt-1">
                                                Score: {item.score} ({item.status}) • {item.note}
                                            </p>
                                        </div>
                                    </div>
                                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                                </CardContent>
                            </Card>
                        </Link>
                    ))}
                </div>
            </div>
        </PageLayout>
    )
}
