import Link from "next/link"
import { ArrowRight, Activity, Brain, History } from "lucide-react"
import { Button } from "@/components/ui/Button"
import { PageLayout } from "@/components/layout/PageLayout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"

export default function Home() {
  return (
    <PageLayout>
      <div className="flex flex-col items-center justify-center space-y-12 py-12 md:py-24 text-center">
        {/* Hero Section */}
        <div className="space-y-6 max-w-3xl animate-in fade-in slide-in-from-bottom-4 duration-1000">
          <div className="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-sm font-medium text-primary">
            <span className="flex h-2 w-2 rounded-full bg-primary mr-2"></span>
            AI-Powered Parkinson's Analysis
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight lg:text-6xl bg-gradient-to-r from-white via-white to-gray-400 bg-clip-text text-transparent">
            Track Your Movement Changes with Data
          </h1>
          <p className="text-lg text-muted-foreground md:text-xl max-w-2xl mx-auto">
            Your personal AI assistant for tracking Parkinson's related movement patterns using just your smartphone camera.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Link href="/test">
              <Button size="lg" className="w-full sm:w-auto gap-2 text-base">
                Start New Test <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/about">
              <Button variant="outline" size="lg" className="w-full sm:w-auto text-base">
                How it works
              </Button>
            </Link>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid gap-8 md:grid-cols-3 w-full max-w-5xl pt-12">
          <Card className="bg-card/50 backdrop-blur border-primary/10">
            <CardHeader>
              <Activity className="h-10 w-10 text-primary mb-2" />
              <CardTitle>Video Analysis</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Upload a simple video of walking or finger tapping. AI extracts skeleton data to analyze movement quality.
            </CardContent>
          </Card>
          <Card className="bg-card/50 backdrop-blur border-primary/10">
            <CardHeader>
              <Brain className="h-10 w-10 text-primary mb-2" />
              <CardTitle>AI Insights</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Get instant feedback on stride length, speed, and rhythm. Understand your symptoms with clear metrics.
            </CardContent>
          </Card>
          <Card className="bg-card/50 backdrop-blur border-primary/10">
            <CardHeader>
              <History className="h-10 w-10 text-primary mb-2" />
              <CardTitle>Track Progress</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Monitor changes over time. Compare current results with past records to see trends clearly.
            </CardContent>
          </Card>
        </div>
      </div>
    </PageLayout>
  )
}
