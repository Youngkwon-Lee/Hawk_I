"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/Button"

type Theme = "dark" | "light"

export function ThemeToggle() {
    const [theme, setTheme] = React.useState<Theme>("light")

    React.useEffect(() => {
        const stored = window.localStorage.getItem("hawkeye-theme") as Theme | null
        const nextTheme = stored === "light" || stored === "dark" ? stored : "light"
        setTheme(nextTheme)
        document.documentElement.dataset.theme = nextTheme
    }, [])

    const toggleTheme = () => {
        const nextTheme = theme === "dark" ? "light" : "dark"
        setTheme(nextTheme)
        document.documentElement.dataset.theme = nextTheme
        window.localStorage.setItem("hawkeye-theme", nextTheme)
    }

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "라이트 모드로 전환" : "다크 모드로 전환"}
            title={theme === "dark" ? "라이트 모드" : "다크 모드"}
            className="rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
        >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
    )
}
