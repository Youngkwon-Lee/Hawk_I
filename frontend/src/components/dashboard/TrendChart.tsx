"use client"

import * as React from "react"
import { Line, LineChart, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"

interface TrendChartProps {
    data: Array<Record<string, string | number | null | undefined>>
    dataKey: string
    label: string
    color?: string
}

export function TrendChart({ data, dataKey, label, color = "#8884d8" }: TrendChartProps) {
    const containerRef = React.useRef<HTMLDivElement>(null)
    const [chartWidth, setChartWidth] = React.useState(0)

    React.useEffect(() => {
        const element = containerRef.current
        if (!element) return

        const updateWidth = () => {
            setChartWidth(Math.max(1, Math.floor(element.getBoundingClientRect().width)))
        }

        updateWidth()
        const observer = new ResizeObserver(updateWidth)
        observer.observe(element)

        return () => observer.disconnect()
    }, [])

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-sm font-medium">{label} Trend</CardTitle>
            </CardHeader>
            <CardContent>
                <div ref={containerRef} className="h-[200px] min-w-0 w-full">
                    {chartWidth > 0 && (
                        <LineChart width={chartWidth} height={200} data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis
                                dataKey="date"
                                stroke="#888888"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                stroke="#888888"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => `${value}`}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: "#1f2937", border: "none", borderRadius: "8px" }}
                                itemStyle={{ color: "#fff" }}
                            />
                            <Line
                                type="monotone"
                                dataKey={dataKey}
                                stroke={color}
                                strokeWidth={2}
                                dot={{ r: 4, fill: color }}
                                activeDot={{ r: 6 }}
                            />
                        </LineChart>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
