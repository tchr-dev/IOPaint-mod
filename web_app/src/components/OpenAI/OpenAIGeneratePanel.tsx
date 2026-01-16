/**
 * OpenAIGeneratePanel Component
 *
 * Main panel for the "Refine → Generate" workflow.
 * Assembles all OpenAI generation components into a cohesive UI.
 *
 * Flow:
 * 1. User enters intent (IntentInput)
 * 2. User clicks "Refine Prompt" (LLM enhances prompt)
 * 3. User reviews/edits refined prompt (PromptEditor)
 * 4. User selects preset (GenerationPresets)
 * 5. User sees cost estimate (CostDisplay)
 * 6. User clicks "Generate Image"
 * 7. If high cost, confirmation modal appears (CostWarningModal)
 *
 * @example
 * ```tsx
 * <OpenAIGeneratePanel />
 * ```
 */

import { useState } from "react"
import { Loader2, ImagePlus, History, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { useStore } from "@/lib/states"
import { IntentInput } from "./IntentInput"
import { PromptEditor } from "./PromptEditor"
import { GenerationPresets } from "./GenerationPresets"
import { CostDisplay } from "./CostDisplay"
import { CostWarningModal } from "./CostWarningModal"
import { GenerationHistory } from "./GenerationHistory"

export function OpenAIGeneratePanel() {
  const [showHistory, setShowHistory] = useState(false)

  const [
    refinedPrompt,
    isGenerating,
    requestGeneration,
    budgetStatus,
    generationHistory,
    selectedGenerateModel,
    currentJobId,
    cancelOpenAIJob,
  ] = useStore((state) => [
    state.openAIState.openAIRefinedPrompt,
    state.openAIState.isOpenAIGenerating,
    state.requestOpenAIGeneration,
    state.openAIState.budgetStatus,
    state.openAIState.generationHistory,
    state.openAIState.selectedGenerateModel,
    state.openAIState.currentJobId,
    state.cancelOpenAIJob,
  ])

  const isBudgetBlocked = budgetStatus?.status === "blocked"
  const canGenerate =
    refinedPrompt.trim() &&
    selectedGenerateModel &&
    !isGenerating &&
    !isBudgetBlocked

  // Show history view if toggled
  if (showHistory) {
    return <GenerationHistory onClose={() => setShowHistory(false)} />
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Generate Image</h3>
        {generationHistory.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="gap-1"
            onClick={() => setShowHistory(true)}
          >
            <History className="h-4 w-4" />
            History ({generationHistory.length})
          </Button>
        )}
      </div>

      <Separator />

      {/* Step 1: Intent Input */}
      <IntentInput />

      {/* Step 2: Prompt Editor (appears after refine) */}
      <PromptEditor />

      {/* Step 3: Presets (appears after refine) */}
      {refinedPrompt && (
        <>
          <Separator />
          <GenerationPresets />
        </>
      )}

      {/* Step 4: Cost Display (appears after refine) */}
      {refinedPrompt && (
        <>
          <Separator />
          <CostDisplay />
        </>
      )}

      {/* Generate Button */}
      {isGenerating ? (
        <div className="flex gap-2">
          <Button disabled size="lg" className="flex-1 gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            Generating...
          </Button>
          <Button
            variant="destructive"
            size="lg"
            onClick={() => currentJobId && cancelOpenAIJob(currentJobId)}
            title="Cancel generation"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      ) : (
        <Button
          onClick={() => requestGeneration()}
          disabled={!canGenerate}
          size="lg"
          className="w-full gap-2"
        >
          <ImagePlus className="h-5 w-5" />
          Generate Image
        </Button>
      )}

      {!selectedGenerateModel && (
        <p className="text-sm text-center text-muted-foreground">
          No OpenAI generation models available.
        </p>
      )}

      {/* Budget blocked message */}
      {isBudgetBlocked && (
        <p className="text-sm text-center text-red-600 dark:text-red-400">
          Budget limit reached. Please wait or increase your limit.
        </p>
      )}

      {/* Cost Warning Modal */}
      <CostWarningModal />
    </div>
  )
}

export default OpenAIGeneratePanel
