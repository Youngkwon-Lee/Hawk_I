import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/Table"
import { cn } from "@/lib/utils"

// Need to implement Table component first or just use raw divs if I want to save time.
// Let's implement a simple Table UI component first in this file to be self-contained or create ui/Table.tsx
// I'll create ui/Table.tsx quickly as it's standard.

export interface MetricRow {
    label: string
    value: string | number
    unit?: string
    change?: string
    status?: "good" | "warning" | "bad" | "neutral"
    normalRange?: string  // e.g., "0.8-1.2" or "<10"
}

interface MetricsTableProps {
    data: MetricRow[]
    className?: string
}

export function MetricsTable({ data, className }: MetricsTableProps) {
    return (
        <div className={cn("w-full overflow-auto", className)}>
            <table className="w-full caption-bottom text-sm">
                <thead className="[&_tr]:border-b">
                    <tr className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">항목</th>
                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">측정값</th>
                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">정상 범위</th>
                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">상태</th>
                    </tr>
                </thead>
                <tbody className="[&_tr:last-child]:border-0">
                    {data.map((row, i) => (
                        <tr key={i} className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                            <td className="p-4 align-middle font-medium">{row.label}</td>
                            <td className="p-4 align-middle">
                                {row.value} <span className="text-muted-foreground text-xs">{row.unit}</span>
                            </td>
                            <td className="p-4 align-middle text-muted-foreground text-sm">
                                {row.normalRange || "-"}
                            </td>
                            <td className="p-4 align-middle">
                                <span className={cn(
                                    "inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ring-1 ring-inset",
                                    row.status === "good" && "bg-green-500/10 text-green-500 ring-green-500/20",
                                    row.status === "bad" && "bg-red-500/10 text-red-500 ring-red-500/20",
                                    row.status === "warning" && "bg-yellow-500/10 text-yellow-500 ring-yellow-500/20",
                                    row.status === "neutral" && "bg-gray-500/10 text-gray-500 ring-gray-500/20",
                                )}>
                                    {row.status === "good" && "정상"}
                                    {row.status === "warning" && "경계"}
                                    {row.status === "bad" && "주의"}
                                    {row.status === "neutral" && "-"}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}
