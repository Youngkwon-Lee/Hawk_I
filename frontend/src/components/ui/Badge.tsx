"use client"

import * as React from "react"

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "secondary" | "outline" | "destructive"
  children: React.ReactNode
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", children, ...props }, ref) => {
    const variantClasses = {
      default: "bg-blue-600 text-white",
      secondary: "bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
      outline: "border border-gray-300 text-gray-700 dark:border-gray-600 dark:text-gray-300",
      destructive: "bg-red-600 text-white"
    }

    return (
      <span
        ref={ref}
        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${variantClasses[variant]} ${className || ""}`}
        {...props}
      >
        {children}
      </span>
    )
  }
)

Badge.displayName = "Badge"

export { Badge }
