import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { ArrowDown, ArrowUp, Minus } from "lucide-react"
import { cn } from "@/lib/utils"

interface SummaryCardProps {
    title: string
    value: string | number
    subtext?: string
    trend?: "up" | "down" | "neutral"
    trendValue?: string
    status?: "good" | "warning" | "bad" | "neutral"
    className?: string
}

export function SummaryCard({
    title,
    value,
    subtext,
    trend,
    trendValue,
    status = "neutral",
    className,
}: SummaryCardProps) {
    return (
        <Card className={cn("overflow-hidden", className)}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                    {title}
                </CardTitle>
                {status !== "neutral" && (
                    <div className={cn(
                        "h-2 w-2 rounded-full",
                        status === "good" && "bg-green-500",
                        status === "warning" && "bg-yellow-500",
                        status === "bad" && "bg-red-500",
                    )} />
                )}
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">{value}</div>
                {(subtext || trend) && (
                    <div className="flex items-center text-xs text-muted-foreground mt-1">
                        {trend && (
                            <span className={cn(
                                "flex items-center mr-2 font-medium",
                                trend === "up" && "text-red-500", // Usually up is bad for symptoms, but depends on metric. Let's assume context handles color via className if needed, or default logic.
                                // Actually, let's make it neutral here and let parent control color via trendValue styling or just keep it simple.
                                // For PD: Speed UP = Good, Tremor UP = Bad. 
                                // Let's use generic colors for now: Green for improvement, Red for worsening.
                                // Since we don't know the metric direction here, let's just use arrows.
                                trend === "up" ? "text-foreground" : "text-foreground"
                            )}>
                                {trend === "up" && <ArrowUp className="mr-1 h-3 w-3" />}
                                {trend === "down" && <ArrowDown className="mr-1 h-3 w-3" />}
                                {trend === "neutral" && <Minus className="mr-1 h-3 w-3" />}
                                {trendValue}
                            </span>
                        )}
                        {subtext}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
