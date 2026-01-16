/**
 * HistoryItem Component
 *
 * Displays a single generation job in the history list.
 * Shows thumbnail, prompt snippet, status, parameters, and actions.
 *
 * Actions:
 * - Open: Load result into editor
 * - Copy Prompt: Copy refined prompt to clipboard
 * - Re-run: Generate again with same parameters
 * - Delete: Remove from history
 *
 * @example
 * ```tsx
 * <HistoryItem job={generationJob} />
 * ```
 */

import { formatDistanceToNow } from "date-fns"
import {
  MoreHorizontal,
  Copy,
  RefreshCw,
  Trash2,
  ExternalLink,
  ImageIcon,
  Edit3,
  Ban,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { StatusBadge, Badge } from "@/components/ui/badge"
import { useStore } from "@/lib/states"
import { GenerationJob } from "@/lib/types"

interface HistoryItemProps {
  job: GenerationJob
  onOpenInEditor?: (job: GenerationJob) => void
  onRestoreSettings?: (job: GenerationJob) => void
}

export function HistoryItem({
  job,
  onOpenInEditor,
  onRestoreSettings,
}: HistoryItemProps) {
  const [copyJobPrompt, rerunJob, removeFromHistory, cancelOpenAIJob] = useStore(
    (state) => [
      state.copyJobPrompt,
      state.rerunJob,
      state.removeFromHistory,
      state.cancelOpenAIJob,
    ]
  )

  const formatCost = (cost?: number) => {
    if (!cost) return null
    if (cost < 0.01) return "<$0.01"
    return `$${cost.toFixed(2)}`
  }

  const truncatePrompt = (prompt: string, maxLength: number = 60) => {
    if (prompt.length <= maxLength) return prompt
    return prompt.substring(0, maxLength) + "..."
  }

  const timeAgo = formatDistanceToNow(new Date(job.createdAt), {
    addSuffix: true,
  })

  return (
    <div
      className="flex gap-3 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors cursor-pointer"
      onClick={() => onRestoreSettings?.(job)}
    >
      {/* Thumbnail */}
      <div className="flex-shrink-0 w-16 h-16 rounded-md overflow-hidden bg-muted">
        {job.thumbnailDataUrl ? (
          <img
            src={job.thumbnailDataUrl}
            alt="Generation result"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            {job.status === "failed" ? (
              <span className="text-2xl">✕</span>
            ) : (
              <ImageIcon className="h-6 w-6 text-muted-foreground" />
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Prompt */}
        <p className="text-sm font-medium truncate" title={job.refinedPrompt}>
          {truncatePrompt(job.refinedPrompt)}
        </p>

        {/* Metadata row */}
        <div className="flex items-center gap-2 mt-1 flex-wrap">
          <StatusBadge status={job.status} />

          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            {job.preset.toUpperCase()}
          </Badge>

          <span className="text-xs text-muted-foreground">
            {job.params.size}
          </span>

          {job.isEdit && (
            <Badge variant="info" className="text-[10px] px-1.5 py-0 gap-0.5">
              <Edit3 className="h-2.5 w-2.5" />
              Edit
            </Badge>
          )}
        </div>

        {/* Time and cost */}
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-muted-foreground">{timeAgo}</span>
          {job.estimatedCostUsd && (
            <span className="text-xs text-muted-foreground">
              • {formatCost(job.estimatedCostUsd)}
            </span>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex-shrink-0">
        <DropdownMenu>
          <DropdownMenuTrigger
            asChild
            onClick={(event) => event.stopPropagation()}
          >
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {job.status === "succeeded" && job.resultImageDataUrl && (
              <DropdownMenuItem
                onClick={() => onOpenInEditor?.(job)}
                className="gap-2"
              >
                <ExternalLink className="h-4 w-4" />
                Open in Editor
              </DropdownMenuItem>
            )}

            <DropdownMenuItem
              onClick={() => copyJobPrompt(job.id)}
              className="gap-2"
            >
              <Copy className="h-4 w-4" />
              Copy Prompt
            </DropdownMenuItem>

            {(job.status === "queued" || job.status === "running") && (
              <DropdownMenuItem
                onClick={() => cancelOpenAIJob(job.id)}
                className="gap-2"
              >
                <Ban className="h-4 w-4" />
                Cancel
              </DropdownMenuItem>
            )}

            {job.status !== "running" && job.status !== "queued" && (
              <DropdownMenuItem
                onClick={() => rerunJob(job.id)}
                className="gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Re-run
              </DropdownMenuItem>
            )}

            <DropdownMenuSeparator />

            <DropdownMenuItem
              onClick={() => removeFromHistory(job.id)}
              className="gap-2 text-destructive focus:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}

export default HistoryItem
