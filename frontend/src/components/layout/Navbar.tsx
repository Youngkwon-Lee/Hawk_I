"use client"

import Link from "next/link"
import { Activity, ClipboardList, ChevronRight, FilePlus2, Home, Menu, Settings, User, Users } from "lucide-react"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/Button"
import { ThemeToggle } from "@/components/layout/ThemeToggle"

export function Navbar() {
    const pathname = usePathname()
    const links = [
        { href: "/", label: "홈", icon: Home },
        { href: "/test", label: "새 검사", icon: FilePlus2 },
        { href: "/history", label: "기록", icon: ClipboardList },
        { href: "#", label: "환자", icon: Users },
        { href: "#", label: "설정", icon: Settings },
    ]

    return (
        <header className="sticky top-0 z-50 w-full border-b border-border/70 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/75">
            <div className="container flex h-[4.25rem] max-w-screen-2xl items-center justify-between px-4 md:px-6">
                <Link href="/" className="group flex items-center gap-2.5 text-foreground">
                    <Activity className="h-7 w-7 text-primary transition-colors group-hover:text-primary/80" aria-hidden="true" />
                    <span>
                        <span className="block text-[1.05rem] font-semibold leading-5 tracking-tight">HawkEye PD</span>
                        <span className="block text-[0.62rem] leading-4 text-muted-foreground">Parkinson&apos;s Movement Analysis</span>
                    </span>
                </Link>

                <nav className="hidden items-center gap-1 md:flex" aria-label="주요 메뉴">
                    {links.map(({ href, label, icon: Icon }) => {
                        const isActive = href !== "#" && (href === "/" ? pathname === "/" : pathname.startsWith(href))
                        return (
                            <Link
                                key={label}
                                href={href}
                                aria-current={isActive ? "page" : undefined}
                                className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
                                    isActive
                                        ? "bg-primary/10 text-primary"
                                        : "text-muted-foreground hover:bg-accent hover:text-foreground"
                                }`}
                            >
                                <Icon className="h-4 w-4" aria-hidden="true" />
                                {label}
                            </Link>
                        )
                    })}
                </nav>

                <div className="flex items-center gap-2">
                    <div className="hidden items-center gap-2 text-sm text-muted-foreground lg:flex">
                        <span className="flex h-9 w-9 items-center justify-center rounded-full bg-secondary font-semibold text-primary">HS</span>
                        <span>HawkEye 어시스턴트</span>
                        <ChevronRight className="h-4 w-4" aria-hidden="true" />
                    </div>
                    <ThemeToggle />
                    <Button variant="ghost" size="icon" className="rounded-full lg:hidden">
                        <User className="h-5 w-5" aria-hidden="true" />
                        <span className="sr-only">프로필</span>
                    </Button>
                    <Button variant="ghost" size="icon" className="md:hidden">
                        <Menu className="h-5 w-5" aria-hidden="true" />
                        <span className="sr-only">메뉴</span>
                    </Button>
                </div>
            </div>
        </header>
    )
}
