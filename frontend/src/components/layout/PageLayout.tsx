import { Navbar } from "./Navbar"
import { cn } from "@/lib/utils"

interface PageLayoutProps {
    children: React.ReactNode
    agentPanel?: React.ReactNode
    leftRail?: React.ReactNode
    agentPanelWidth?: string
    contentMaxWidth?: string
    className?: string
}

export function PageLayout({ children, agentPanel, leftRail, agentPanelWidth = "w-[22rem]", contentMaxWidth = "max-w-6xl", className }: PageLayoutProps) {
    return (
        <div className="min-h-screen flex flex-col bg-background">
            <Navbar />
            <main className="flex-1 flex min-h-[calc(100vh-4.25rem)] flex-col overflow-hidden md:flex-row">
                {leftRail && (
                    <aside className="hidden w-[18.5rem] shrink-0 flex-col border-r border-border bg-card/35 md:flex">
                        {leftRail}
                    </aside>
                )}

                {/* Main Content Area */}
                <div className={cn(
                    "min-w-0 flex-1 overflow-y-auto p-4 scroll-smooth md:px-14 md:py-8",
                    agentPanel ? "md:w-auto" : "w-full",
                    className
                )}>
                    <div className={cn("mx-auto h-full", contentMaxWidth)}>
                        {children}
                    </div>
                </div>

                {/* Agent Panel Area */}
                {agentPanel && (
                    <aside className={cn("hidden shrink-0 flex-col border-l border-border bg-card/25 backdrop-blur-sm md:flex", agentPanelWidth)}>
                        {agentPanel}
                    </aside>
                )}
            </main>
        </div>
    )
}
