/**
 * IntentInput Component
 *
 * The first step in the "Refine → Generate" workflow. Users enter their
 * raw idea/intent here, then click "Refine Prompt" to expand it using
 * an LLM into a more detailed, image-generation-optimized prompt.
 *
 * Features:
 * - Auto-expanding textarea
 * - "Refine Prompt" button with loading state
 * - Cheap operation indicator (shows cost is minimal)
 *
 * @example
 * ```tsx
 * <IntentInput />
 * ```
 */

import { Sparkles, Loader2 } from "lucide-react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { useStore } from "@/lib/states"

export function IntentInput() {
  const [
    intent,
    setOpenAIIntent,
    isRefiningPrompt,
    refinePrompt,
  ] = useStore((state) => [
    state.openAIState.openAIIntent,
    state.setOpenAIIntent,
    state.openAIState.isRefiningPrompt,
    state.refinePrompt,
  ])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl/Cmd + Enter to refine
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault()
      if (intent.trim() && !isRefiningPrompt) {
        refinePrompt()
      }
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-foreground">
        What do you want to create?
      </label>

      <Textarea
        value={intent}
        onChange={(e) => setOpenAIIntent(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Describe your idea... e.g., 'a cat sitting on a windowsill watching rain'"
        className="min-h-[60px] resize-none transition-all duration-200 focus:min-h-[100px]"
        disabled={isRefiningPrompt}
      />

      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          Ctrl+Enter to refine
        </span>

        <Button
          onClick={() => refinePrompt()}
          disabled={!intent.trim() || isRefiningPrompt}
          size="sm"
          className="gap-2"
        >
          {isRefiningPrompt ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Refining...
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Refine Prompt
            </>
          )}
        </Button>
      </div>

      <p className="text-xs text-muted-foreground">
        This uses a small LLM to expand your idea into a detailed prompt
        (~$0.0001)
      </p>
    </div>
  )
}

export default IntentInput
