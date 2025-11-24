"use client"

import * as React from "react"
import { PageLayout } from "@/components/layout/PageLayout"
import { ChatInterface } from "@/components/ui/ChatInterface"
import { CheckCircle2, Circle, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"

const STEPS = [
    { id: 1, label: "Extracting Skeleton Data", description: "Identifying joints and body points..." },
    { id: 2, label: "Calculating Kinematics", description: "Measuring speed, stride, and rhythm..." },
    { id: 3, label: "Estimating Risk Score", description: "Comparing with UPDRS standards..." },
    { id: 4, label: "Generating Report", description: "Synthesizing insights and explanations..." },
]

export default function AnalysisPage() {
    const [currentStep, setCurrentStep] = React.useState(1)
    const router = useRouter()

    React.useEffect(() => {
        const timer = setInterval(() => {
            setCurrentStep((prev) => {
                if (prev >= 4) {
                    clearInterval(timer)
                    setTimeout(() => router.push("/result"), 1000)
                    return 4
                }
                return prev + 1
            })
        }, 2000) // Simulate 2 seconds per step

        return () => clearInterval(timer)
    }, [router])

    return (
        <PageLayout agentPanel={<ChatInterface initialMessages={[{
            id: "1",
            role: "agent",
            content: "I'm currently analyzing your video. I've detected the skeleton and am now calculating the movement metrics.",
            timestamp: new Date()
        }]} />}>
            <div className="flex flex-col items-center justify-center h-full space-y-12">
                <div className="text-center space-y-4">
                    <h1 className="text-3xl font-bold">Analyzing Your Movement</h1>
                    <p className="text-muted-foreground">Please wait while our AI processes the video data.</p>
                </div>

                <div className="w-full max-w-md space-y-6">
                    {STEPS.map((step) => (
                        <div key={step.id} className="flex items-start gap-4">
                            <div className="mt-1">
                                {currentStep > step.id ? (
                                    <CheckCircle2 className="h-6 w-6 text-primary" />
                                ) : currentStep === step.id ? (
                                    <Loader2 className="h-6 w-6 text-primary animate-spin" />
                                ) : (
                                    <Circle className="h-6 w-6 text-muted-foreground/30" />
                                )}
                            </div>
                            <div className={cn("space-y-1", currentStep === step.id ? "opacity-100" : "opacity-60")}>
                                <h3 className="font-medium leading-none">{step.label}</h3>
                                <p className="text-sm text-muted-foreground">{step.description}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </PageLayout>
    )
}
