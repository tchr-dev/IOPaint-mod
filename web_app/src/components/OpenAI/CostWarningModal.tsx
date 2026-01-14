/**
 * CostWarningModal Component
 *
 * Confirmation dialog shown before executing high-cost operations.
 * Prevents accidental expensive generations by requiring explicit confirmation.
 *
 * Appears when:
 * - Cost tier is "high" (> $0.10)
 * - User clicks Generate
 *
 * @example
 * ```tsx
 * <CostWarningModal />
 * ```
 */

import { AlertTriangle } from "lucide-react"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { CostTierBadge } from "@/components/ui/badge"
import { useStore } from "@/lib/states"

export function CostWarningModal() {
  const [
    showModal,
    costEstimate,
    budgetStatus,
    confirmGeneration,
    cancelPendingGeneration,
  ] = useStore((state) => [
    state.openAIState.showCostWarningModal,
    state.openAIState.currentCostEstimate,
    state.openAIState.budgetStatus,
    state.confirmOpenAIGeneration,
    state.cancelPendingGeneration,
  ])

  const formatCost = (cost: number) => `$${cost.toFixed(2)}`

  const remainingBudget = budgetStatus
    ? Math.min(
        budgetStatus.daily.remainingUsd,
        budgetStatus.session.remainingUsd
      )
    : null

  return (
    <AlertDialog open={showModal}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500" />
            Confirm High-Cost Generation
          </AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="flex flex-col gap-4">
              <p>This generation has a higher than usual cost.</p>

              {costEstimate && (
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <span className="text-sm font-medium">Estimated cost:</span>
                  <div className="flex items-center gap-2">
                    <CostTierBadge tier={costEstimate.tier} />
                    <span className="text-lg font-bold">
                      ~{formatCost(costEstimate.estimatedCostUsd)}
                    </span>
                  </div>
                </div>
              )}

              {costEstimate?.warning && (
                <p className="text-sm text-yellow-600 dark:text-yellow-400">
                  {costEstimate.warning}
                </p>
              )}

              {remainingBudget !== null && (
                <p className="text-sm text-muted-foreground">
                  Remaining budget:{" "}
                  <span className="font-medium">
                    {formatCost(remainingBudget)}
                  </span>
                </p>
              )}

              <p className="text-sm">
                Are you sure you want to proceed with this generation?
              </p>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={cancelPendingGeneration}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction onClick={confirmGeneration}>
            Confirm & Generate
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export default CostWarningModal
