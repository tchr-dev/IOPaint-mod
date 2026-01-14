/**
 * CostDisplay Component
 *
 * Shows estimated cost and budget status before generation.
 * Provides visual feedback on cost tier (low/medium/high) and
 * remaining budget across daily/monthly/session limits.
 *
 * @example
 * ```tsx
 * <CostDisplay />
 * ```
 */

import { DollarSign, AlertTriangle, RefreshCw } from "lucide-react"
import { CostTierBadge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useStore } from "@/lib/states"
import { cn } from "@/lib/utils"

export function CostDisplay() {
  const [
    costEstimate,
    budgetStatus,
    refreshBudgetStatus,
    isGenerating,
  ] = useStore((state) => [
    state.openAIState.currentCostEstimate,
    state.openAIState.budgetStatus,
    state.refreshBudgetStatus,
    state.openAIState.isOpenAIGenerating,
  ])

  const formatCost = (cost: number) => {
    if (cost < 0.01) return `<$0.01`
    return `$${cost.toFixed(2)}`
  }

  const formatBudget = (spent: number, cap: number, isUnlimited: boolean) => {
    if (isUnlimited) return "Unlimited"
    return `${formatCost(spent)} / ${formatCost(cap)}`
  }

  return (
    <div className="flex flex-col gap-3 p-3 bg-muted/30 rounded-lg border">
      {/* Cost estimate */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <DollarSign className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Estimated Cost</span>
        </div>

        {costEstimate ? (
          <div className="flex items-center gap-2">
            <CostTierBadge tier={costEstimate.tier} />
            <span className="text-sm font-semibold">
              ~{formatCost(costEstimate.estimatedCostUsd)}
            </span>
          </div>
        ) : (
          <span className="text-sm text-muted-foreground">
            Enter prompt to estimate
          </span>
        )}
      </div>

      {/* Warning message if high cost */}
      {costEstimate?.warning && (
        <div className="flex items-start gap-2 text-yellow-600 dark:text-yellow-400">
          <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
          <span className="text-xs">{costEstimate.warning}</span>
        </div>
      )}

      {/* Budget status */}
      {budgetStatus && (
        <div className="flex flex-col gap-2 pt-2 border-t">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">
              Budget Status
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refreshBudgetStatus()}
              disabled={isGenerating}
              className="h-6 px-2"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
          </div>

          {/* Daily budget */}
          {!budgetStatus.daily.isUnlimited && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Daily</span>
                <span>
                  {formatBudget(
                    budgetStatus.daily.spentUsd,
                    budgetStatus.daily.capUsd,
                    budgetStatus.daily.isUnlimited
                  )}
                </span>
              </div>
              <Progress
                value={budgetStatus.daily.percentageUsed}
                className={cn(
                  "h-1.5",
                  budgetStatus.daily.percentageUsed > 80 && "bg-yellow-200",
                  budgetStatus.daily.percentageUsed > 95 && "bg-red-200"
                )}
              />
            </div>
          )}

          {/* Session budget */}
          {!budgetStatus.session.isUnlimited && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Session</span>
                <span>
                  {formatBudget(
                    budgetStatus.session.spentUsd,
                    budgetStatus.session.capUsd,
                    budgetStatus.session.isUnlimited
                  )}
                </span>
              </div>
              <Progress
                value={budgetStatus.session.percentageUsed}
                className="h-1.5"
              />
            </div>
          )}

          {/* Blocked status message */}
          {budgetStatus.status === "blocked" && budgetStatus.message && (
            <div className="flex items-center gap-2 p-2 bg-red-100 dark:bg-red-900/30 rounded text-xs text-red-700 dark:text-red-300">
              <AlertTriangle className="h-4 w-4 flex-shrink-0" />
              {budgetStatus.message}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default CostDisplay
