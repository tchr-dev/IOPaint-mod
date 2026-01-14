/**
 * GenerationHistory Component
 *
 * Displays the history/gallery of all generation jobs.
 * Shows list of jobs with filtering, actions, and the ability
 * to load results back into the editor.
 *
 * Features:
 * - Filter by status (All, Succeeded, Failed)
 * - Clear all history
 * - Load generated images into editor
 * - Copy prompts, re-run jobs
 *
 * @example
 * ```tsx
 * <GenerationHistory onClose={handleClose} />
 * ```
 */

import { useEffect } from "react"
import { Database, History, RefreshCw, Save, Trash2, X, Filter } from "lucide-react"
import { formatDistanceToNow } from "date-fns"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { useStore } from "@/lib/states"
import { GenerationJob, HistorySnapshot } from "@/lib/types"
import { HistoryItem } from "./HistoryItem"

interface GenerationHistoryProps {
  onClose?: () => void
}

export function GenerationHistory({ onClose }: GenerationHistoryProps) {
  const [
    generationHistory,
    historyFilter,
    setHistoryFilter,
    clearHistory,
    setFile,
    historySnapshots,
    saveHistorySnapshot,
    syncHistorySnapshots,
    deleteHistorySnapshot,
    clearHistorySnapshots,
  ] = useStore((state) => [
    state.openAIState.generationHistory,
    state.openAIState.historyFilter,
    state.setHistoryFilter,
    state.clearHistory,
    state.setFile,
    state.openAIState.historySnapshots,
    state.saveHistorySnapshot,
    state.syncHistorySnapshots,
    state.deleteHistorySnapshot,
    state.clearHistorySnapshots,
  ])

  useEffect(() => {
    syncHistorySnapshots()
  }, [syncHistorySnapshots])

  const getSnapshotCount = (snapshot: HistorySnapshot): number | null => {
    const history = snapshot.payload?.["history"]
    return Array.isArray(history) ? history.length : null
  }

  // Filter jobs based on selected filter
  const filteredJobs = generationHistory.filter((job) => {
    if (historyFilter === "all") return true
    if (historyFilter === "succeeded") return job.status === "succeeded"
    if (historyFilter === "failed")
      return job.status === "failed" || job.status === "blocked_budget"
    return true
  })

  // Handle opening a job's result in the editor
  const handleOpenInEditor = async (job: GenerationJob) => {
    if (!job.resultImageDataUrl) return

    try {
      // Convert data URL to File
      const response = await fetch(job.resultImageDataUrl)
      const blob = await response.blob()
      const file = new File([blob], `generation-${job.id}.png`, {
        type: "image/png",
      })

      // Load into editor
      await setFile(file)
      onClose?.()
    } catch (error) {
      console.error("Failed to open image in editor:", error)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <History className="h-5 w-5" />
          <h3 className="text-lg font-semibold">Generation History</h3>
          <span className="text-sm text-muted-foreground">
            ({filteredJobs.length})
          </span>
        </div>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Filters */}
      <div className="flex items-center justify-between p-4 gap-2">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Select
            value={historyFilter}
            onValueChange={(value) =>
              setHistoryFilter(value as "all" | "succeeded" | "failed")
            }
          >
            <SelectTrigger className="w-[140px] h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="succeeded">Succeeded</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {generationHistory.length > 0 && (
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => saveHistorySnapshot()}
              className="gap-1"
            >
              <Save className="h-4 w-4" />
              Save Snapshot
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={syncHistorySnapshots}
              aria-label="Sync snapshots"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearHistory}
              className="text-destructive hover:text-destructive gap-1"
            >
              <Trash2 className="h-4 w-4" />
              Clear All
            </Button>
          </div>
        )}
      </div>

      <Separator />

      {/* Job list */}
      <ScrollArea className="flex-1 p-4">
        {historySnapshots.length > 0 && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Database className="h-4 w-4" />
                Snapshots ({historySnapshots.length})
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearHistorySnapshots}
                className="text-destructive hover:text-destructive gap-1"
              >
                <Trash2 className="h-4 w-4" />
                Clear
              </Button>
            </div>
            <div className="flex flex-col gap-2">
              {historySnapshots.map((snapshot) => {
                const count = getSnapshotCount(snapshot)
                return (
                  <div
                    key={snapshot.id}
                    className="flex items-center justify-between rounded-md border px-3 py-2"
                  >
                    <div className="flex flex-col">
                      <span className="text-sm font-medium">History Snapshot</span>
                      <span className="text-xs text-muted-foreground">
                        {formatDistanceToNow(new Date(snapshot.createdAt), {
                          addSuffix: true,
                        })}
                        {count !== null ? ` • ${count} items` : ""}
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => deleteHistorySnapshot(snapshot.id)}
                      aria-label="Delete snapshot"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                )
              })}
            </div>
            <Separator className="mt-4" />
          </div>
        )}
        {filteredJobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-center">
            <History className="h-10 w-10 text-muted-foreground mb-2" />
            <p className="text-muted-foreground">
              {generationHistory.length === 0
                ? "No generations yet"
                : "No matching jobs"}
            </p>
            {generationHistory.length === 0 && (
              <p className="text-sm text-muted-foreground mt-1">
                Generated images will appear here
              </p>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {filteredJobs.map((job) => (
              <HistoryItem
                key={job.id}
                job={job}
                onOpenInEditor={handleOpenInEditor}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}

export default GenerationHistory
