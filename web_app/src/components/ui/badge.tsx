/**
 * Badge Component
 *
 * A small visual indicator used for cost tiers, status labels, and tags.
 * Follows the existing Radix UI + Tailwind pattern used throughout the app.
 *
 * Variants:
 * - default: Primary color badge
 * - low: Green badge for low-cost operations (safe)
 * - medium: Yellow badge for medium-cost operations (informational)
 * - high: Red badge for high-cost operations (warning)
 * - outline: Transparent with border
 * - secondary: Muted background
 *
 * @example
 * ```tsx
 * <Badge variant="low">LOW</Badge>
 * <Badge variant="high">$0.12</Badge>
 * ```
 */

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline: "text-foreground border border-input",
        // Cost tier variants (Epic 4)
        low: "border-transparent bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100",
        medium:
          "border-transparent bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100",
        high: "border-transparent bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100",
        // Status variants
        success:
          "border-transparent bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100",
        warning:
          "border-transparent bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100",
        error:
          "border-transparent bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100",
        info: "border-transparent bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

/**
 * CostTierBadge - Specialized badge for displaying cost tiers
 *
 * Automatically maps cost tier strings to appropriate variants and labels.
 *
 * @example
 * ```tsx
 * <CostTierBadge tier="low" />
 * <CostTierBadge tier="high" showLabel={false} />
 * ```
 */
interface CostTierBadgeProps extends Omit<BadgeProps, "variant"> {
  tier: "low" | "medium" | "high"
  showLabel?: boolean
}

function CostTierBadge({
  tier,
  showLabel = true,
  className,
  ...props
}: CostTierBadgeProps) {
  const labels = {
    low: "LOW",
    medium: "MED",
    high: "HIGH",
  }

  return (
    <Badge variant={tier} className={className} {...props}>
      {showLabel ? labels[tier] : null}
      {props.children}
    </Badge>
  )
}

/**
 * StatusBadge - Specialized badge for job status display
 *
 * @example
 * ```tsx
 * <StatusBadge status="succeeded" />
 * <StatusBadge status="failed" />
 * ```
 */
interface StatusBadgeProps extends Omit<BadgeProps, "variant"> {
  status: "pending" | "running" | "succeeded" | "failed" | "blocked_budget"
}

function StatusBadge({ status, className, ...props }: StatusBadgeProps) {
  const config: Record<
    StatusBadgeProps["status"],
    { variant: BadgeProps["variant"]; label: string }
  > = {
    pending: { variant: "secondary", label: "Pending" },
    running: { variant: "info", label: "Running" },
    succeeded: { variant: "success", label: "Success" },
    failed: { variant: "error", label: "Failed" },
    blocked_budget: { variant: "warning", label: "Budget" },
  }

  const { variant, label } = config[status]

  return (
    <Badge variant={variant} className={className} {...props}>
      {label}
    </Badge>
  )
}

export { Badge, badgeVariants, CostTierBadge, StatusBadge }
