/**
 * PromptEditor Component
 *
 * Displays and allows editing of the refined prompt and negative prompt.
 * Appears after the user refines their intent, showing the LLM-enhanced
 * prompt that will be used for image generation.
 *
 * Features:
 * - Editable refined prompt textarea
 * - Optional negative prompt field (collapsed by default)
 * - Copy to clipboard button
 *
 * @example
 * ```tsx
 * <PromptEditor />
 * ```
 */

import { useState } from "react"
import { Copy, Check, ChevronDown, ChevronUp } from "lucide-react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { useStore } from "@/lib/states"
import { cn } from "@/lib/utils"

export function PromptEditor() {
  const [
    refinedPrompt,
    setOpenAIRefinedPrompt,
    negativePrompt,
    setOpenAINegativePrompt,
    isRefiningPrompt,
    isGenerating,
  ] = useStore((state) => [
    state.openAIState.openAIRefinedPrompt,
    state.setOpenAIRefinedPrompt,
    state.openAIState.openAINegativePrompt,
    state.setOpenAINegativePrompt,
    state.openAIState.isRefiningPrompt,
    state.openAIState.isOpenAIGenerating,
  ])

  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(refinedPrompt)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const isDisabled = isRefiningPrompt || isGenerating

  if (!refinedPrompt) {
    return null
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Main prompt */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-foreground">
            Final Prompt
          </label>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-7 gap-1 px-2"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" />
                Copy
              </>
            )}
          </Button>
        </div>

        <Textarea
          value={refinedPrompt}
          onChange={(e) => setOpenAIRefinedPrompt(e.target.value)}
          placeholder="Refined prompt will appear here..."
          className="min-h-[100px] resize-none"
          disabled={isDisabled}
        />

        <p className="text-xs text-muted-foreground">
          Edit the prompt to fine-tune your generation
        </p>
      </div>

      {/* Negative prompt toggle */}
      <div className="flex flex-col gap-2">
        <button
          type="button"
          onClick={() => setShowNegativePrompt(!showNegativePrompt)}
          className={cn(
            "flex items-center gap-1 text-sm text-muted-foreground",
            "hover:text-foreground transition-colors"
          )}
        >
          {showNegativePrompt ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
          Negative prompt
          {negativePrompt && !showNegativePrompt && (
            <span className="ml-1 text-xs opacity-60">(has content)</span>
          )}
        </button>

        {showNegativePrompt && (
          <Textarea
            value={negativePrompt}
            onChange={(e) => setOpenAINegativePrompt(e.target.value)}
            placeholder="What to avoid... e.g., 'blurry, low quality, watermark, text'"
            className="min-h-[60px] resize-none"
            disabled={isDisabled}
          />
        )}
      </div>
    </div>
  )
}

export default PromptEditor
