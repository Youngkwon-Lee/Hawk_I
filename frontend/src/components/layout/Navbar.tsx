import Link from "next/link"
import { Activity, Menu, User } from "lucide-react"
import { Button } from "@/components/ui/Button"

export function Navbar() {
    return (
        <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-16 max-w-screen-2xl items-center justify-between px-4">
                <div className="flex items-center gap-2">
                    <Link href="/" className="flex items-center gap-2 font-bold text-xl text-primary">
                        <Activity className="h-6 w-6" />
                        <span>HawkEye PD</span>
                    </Link>
                </div>

                <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
                    <Link href="/" className="transition-colors hover:text-primary">Home</Link>
                    <Link href="/test" className="transition-colors hover:text-primary">New Test</Link>
                    <Link href="/history" className="transition-colors hover:text-primary">History</Link>
                    <Link href="#" className="transition-colors hover:text-primary">FAQ</Link>
                </nav>

                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="icon" className="rounded-full">
                        <User className="h-5 w-5" />
                        <span className="sr-only">Profile</span>
                    </Button>
                    <Button variant="ghost" size="icon" className="md:hidden">
                        <Menu className="h-5 w-5" />
                        <span className="sr-only">Menu</span>
                    </Button>
                </div>
            </div>
        </header>
    )
}
