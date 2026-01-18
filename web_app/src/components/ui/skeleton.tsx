import { cn } from "@/lib/utils"

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "text" | "circular" | "rectangular"
  width?: number | string
  height?: number | string
}

export function Skeleton({
  className,
  variant = "rectangular",
  width,
  height,
  ...props
}: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-pulse bg-muted",
        variant === "text" && "h-4 rounded",
        variant === "circular" && "rounded-full",
        variant === "rectangular" && "rounded-md",
        className
      )}
      style={{
        width: width ?? undefined,
        height: height ?? (variant === "text" ? "1em" : undefined),
      }}
      {...props}
    />
  )
}

interface SkeletonCardProps {
  className?: string
}

export function SkeletonCard({ className }: SkeletonCardProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <Skeleton variant="rectangular" height={200} className="w-full" />
      <div className="space-y-2">
        <Skeleton variant="text" width="60%" />
        <Skeleton variant="text" width="80%" />
        <Skeleton variant="text" width="40%" />
      </div>
    </div>
  )
}

interface SkeletonButtonProps {
  className?: string
}

export function SkeletonButton({ className }: SkeletonButtonProps) {
  return (
    <Skeleton
      variant="rectangular"
      width={80}
      height={36}
      className={className}
    />
  )
}

interface SkeletonAvatarProps {
  className?: string
  size?: number
}

export function SkeletonAvatar({ className, size = 40 }: SkeletonAvatarProps) {
  return (
    <Skeleton
      variant="circular"
      width={size}
      height={size}
      className={className}
    />
  )
}

interface SkeletonListProps {
  count?: number
  itemHeight?: number
  className?: string
}

export function SkeletonList({
  count = 3,
  itemHeight = 60,
  className,
}: SkeletonListProps) {
  return (
    <div className={cn("space-y-3", className)}>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton
          key={i}
          variant="rectangular"
          height={itemHeight}
          className="w-full"
        />
      ))}
    </div>
  )
}

interface SkeletonImageProps {
  width?: number | string
  height?: number | string
  className?: string
}

export function SkeletonImage({ width = "100%", height = 200, className }: SkeletonImageProps) {
  return (
    <Skeleton
      variant="rectangular"
      width={width}
      height={height}
      className={className}
    />
  )
}
