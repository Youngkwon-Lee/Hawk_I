import { Navbar } from "./Navbar"
import { cn } from "@/lib/utils"

interface PageLayoutProps {
    children: React.ReactNode
    agentPanel?: React.ReactNode
    className?: string
}

export function PageLayout({ children, agentPanel, className }: PageLayoutProps) {
    return (
        <div className="min-h-screen flex flex-col bg-background">
            <Navbar />
            <main className="flex-1 flex flex-col md:flex-row overflow-hidden h-[calc(100vh-4rem)]">
                {/* Main Content Area */}
                <div className={cn(
                    "flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth",
                    agentPanel ? "md:w-2/3 lg:w-3/4" : "w-full",
                    className
                )}>
                    <div className="mx-auto max-w-5xl h-full">
                        {children}
                    </div>
                </div>

                {/* Agent Panel Area */}
                {agentPanel && (
                    <aside className="hidden md:flex flex-col w-1/3 lg:w-1/4 border-l border-border bg-card/30 backdrop-blur-sm">
                        {agentPanel}
                    </aside>
                )}
            </main>
        </div>
    )
}
