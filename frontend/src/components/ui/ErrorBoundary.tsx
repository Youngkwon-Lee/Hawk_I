"use client"

import * as React from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { Button } from "./Button"

interface ErrorBoundaryProps {
    children: React.ReactNode
    fallback?: React.ReactNode
}

interface ErrorBoundaryState {
    hasError: boolean
    error: Error | null
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props)
        this.state = {
            hasError: false,
            error: null
        }
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return {
            hasError: true,
            error
        }
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error("ErrorBoundary caught an error:", error, errorInfo)
    }

    handleReset = () => {
        this.setState({
            hasError: false,
            error: null
        })
    }

    render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback
            }

            return (
                <div className="min-h-screen flex items-center justify-center p-4">
                    <div className="max-w-md w-full bg-card border border-border rounded-xl p-8 text-center space-y-6">
                        <div className="flex justify-center">
                            <div className="p-4 rounded-full bg-red-500/10">
                                <AlertTriangle className="h-12 w-12 text-red-500" />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <h2 className="text-2xl font-bold">문제가 발생했습니다</h2>
                            <p className="text-muted-foreground">
                                애플리케이션에서 예상치 못한 오류가 발생했습니다.
                            </p>
                        </div>

                        {process.env.NODE_ENV === "development" && this.state.error && (
                            <div className="bg-secondary/50 rounded-lg p-4 text-left">
                                <p className="text-xs font-mono text-red-500 break-all">
                                    {this.state.error.message}
                                </p>
                            </div>
                        )}

                        <div className="flex gap-3 justify-center">
                            <Button
                                onClick={this.handleReset}
                                className="gap-2"
                            >
                                <RefreshCw className="h-4 w-4" />
                                다시 시도
                            </Button>
                            <Button
                                variant="outline"
                                onClick={() => window.location.href = "/"}
                            >
                                홈으로 가기
                            </Button>
                        </div>
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}
