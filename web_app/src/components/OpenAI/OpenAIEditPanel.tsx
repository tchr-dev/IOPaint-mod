/**
 * OpenAIEditPanel Component
 *
 * Panel for editing images using OpenAI's image edit API.
 * Uses the current canvas mask and a source image to perform inpainting.
 *
 * Workflow:
 * 1. User draws mask on canvas (existing Editor functionality)
 * 2. User opens Edit panel
 * 3. User enters edit intent/prompt
 * 4. User clicks "Apply Edit"
 * 5. Result replaces current canvas image
 *
 * @example
 * ```tsx
 * <OpenAIEditPanel />
 * ```
 */

import { useEffect, useState } from "react"
import {
  Loader2,
  Wand2,
  ArrowLeft,
  AlertTriangle,
  ImageIcon,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Separator } from "@/components/ui/separator"
import { useStore } from "@/lib/states"
import { GenerationPresets } from "./GenerationPresets"
import { CostDisplay } from "./CostDisplay"
import { cn } from "@/lib/utils"

export function OpenAIEditPanel() {
  const [
    // Canvas state
    file,
    curLineGroup,
    extraMasks,
    renders,
    // OpenAI state
    isGenerating,
    editPrompt,
    setOpenAIRefinedPrompt,
    runOpenAIEdit,
    runOpenAIOutpaint,
    runOpenAIVariation,
    setOpenAIEditMode,
    budgetStatus,
    capabilities,
    selectedEditModel,
    // Get current image
    getCurrentTargetFile,
    setEditSourceImage,
    editSourceImageDataUrl,
    // Cancel support
    currentJobId,
    cancelOpenAIJob,
  ] = useStore((state) => [
    state.file,
    state.editorState.curLineGroup,
    state.editorState.extraMasks,
    state.editorState.renders,
    state.openAIState.isOpenAIGenerating,
    state.openAIState.openAIRefinedPrompt,
    state.setOpenAIRefinedPrompt,
    state.runOpenAIEdit,
    state.runOpenAIOutpaint,
    state.runOpenAIVariation,
    state.setOpenAIEditMode,
    state.openAIState.budgetStatus,
    state.openAIState.capabilities,
    state.openAIState.selectedEditModel,
    state.getCurrentTargetFile,
    state.setEditSourceImage,
    state.openAIState.editSourceImageDataUrl,
    state.openAIState.currentJobId,
    state.cancelOpenAIJob,
  ])
  const [toolMode, setToolMode] = useState<"edit" | "outpaint" | "variation">(
    "edit"
  )
  const editModels = capabilities?.modes.images_edit?.models ?? []
  const hasEditCapabilities = editModels.length > 0 && !!selectedEditModel

  // Set up source image when entering edit mode
  useEffect(() => {
    const setupSourceImage = async () => {
      try {
        const targetFile = await getCurrentTargetFile()
        const dataUrl = await fileToDataUrl(targetFile)
        setEditSourceImage(dataUrl)
      } catch (e) {
        console.error("Failed to set up source image:", e)
      }
    }

    if (file) {
      setupSourceImage()
    }

    return () => {
      setEditSourceImage(null)
    }
  }, [file, renders.length])

  const hasMask = curLineGroup.length > 0 || extraMasks.length > 0
  const hasPrompt = editPrompt.trim().length > 0
  const isBudgetBlocked = budgetStatus?.status === "blocked"
  const canEdit =
    hasEditCapabilities && hasMask && hasPrompt && !isGenerating && !isBudgetBlocked
  const canVariation =
    hasEditCapabilities &&
    !!editSourceImageDataUrl &&
    !isGenerating &&
    !isBudgetBlocked

  const primaryAction = () => {
    if (toolMode === "variation") {
      return runOpenAIVariation()
    }
    if (toolMode === "outpaint") {
      return runOpenAIOutpaint()
    }
    return runOpenAIEdit()
  }

  const primaryDisabled =
    toolMode === "variation" ? !canVariation : !canEdit

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setOpenAIEditMode(false)}
          className="h-8 w-8"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h3 className="text-lg font-semibold">
          {toolMode === "variation"
            ? "Create Variation"
            : toolMode === "outpaint"
            ? "Outpaint Image"
            : "Edit Image"}
        </h3>
      </div>

      <Separator />

      {!hasEditCapabilities && (
        <div className="text-sm text-muted-foreground">
          Image edit capabilities are unavailable for the current provider.
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant={toolMode === "edit" ? "default" : "outline"}
          size="sm"
          onClick={() => setToolMode("edit")}
          disabled={!hasEditCapabilities}
        >
          Edit
        </Button>
        <Button
          type="button"
          variant={toolMode === "outpaint" ? "default" : "outline"}
          size="sm"
          onClick={() => setToolMode("outpaint")}
          disabled={!hasEditCapabilities}
        >
          Outpaint
        </Button>
        <Button
          type="button"
          variant={toolMode === "variation" ? "default" : "outline"}
          size="sm"
          onClick={() => setToolMode("variation")}
          disabled={!hasEditCapabilities}
        >
          Variation
        </Button>
      </div>

      {/* Source image preview */}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-foreground">
          Source Image
        </label>
        <div className="relative w-full h-32 rounded-lg overflow-hidden bg-muted border">
          {editSourceImageDataUrl ? (
            <img
              src={editSourceImageDataUrl}
              alt="Source"
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <ImageIcon className="h-8 w-8 text-muted-foreground" />
            </div>
          )}
        </div>
      </div>

      {toolMode !== "variation" && (
        <>
          {/* Mask status */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-foreground">Mask</label>
            <div
              className={cn(
                "p-3 rounded-lg border",
                hasMask
                  ? "bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800"
                  : "bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800"
              )}
            >
              {hasMask ? (
                <p className="text-sm text-green-700 dark:text-green-300">
                  Mask is ready. The highlighted area will be edited.
                </p>
              ) : (
                <p className="text-sm text-yellow-700 dark:text-yellow-300 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Draw a mask on the canvas to select the area to edit.
                </p>
              )}
            </div>
          </div>

          {/* Edit prompt */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-foreground">
              {toolMode === "outpaint"
                ? "What should be added?"
                : "What do you want to change?"}
            </label>
            <Textarea
              value={editPrompt}
              onChange={(e) => setOpenAIRefinedPrompt(e.target.value)}
              placeholder={
                toolMode === "outpaint"
                  ? "Describe what to extend beyond the edges..."
                  : "Describe the edit... e.g., 'Replace with a golden retriever'"
              }
              className="min-h-[80px] resize-none"
              disabled={isGenerating}
            />
          </div>
        </>
      )}

      {toolMode === "variation" && (
        <div className="text-sm text-muted-foreground">
          Variations create alternate versions of the source image without a
          prompt.
        </div>
      )}

      {/* Presets */}
      <GenerationPresets />

      {/* Cost display */}
      <CostDisplay />

      {/* Apply button */}
      {isGenerating ? (
        <div className="flex gap-2">
          <Button disabled size="lg" className="flex-1 gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            {toolMode === "variation"
              ? "Creating Variation..."
              : toolMode === "outpaint"
              ? "Outpainting..."
              : "Applying Edit..."}
          </Button>
          <Button
            variant="destructive"
            size="lg"
            onClick={() => currentJobId && cancelOpenAIJob(currentJobId)}
            title="Cancel operation"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      ) : (
        <Button
          onClick={primaryAction}
          disabled={primaryDisabled}
          size="lg"
          className="w-full gap-2"
        >
          <Wand2 className="h-5 w-5" />
          {toolMode === "variation"
            ? "Create Variation"
            : toolMode === "outpaint"
            ? "Apply Outpaint"
            : "Apply Edit"}
        </Button>
      )}

      {/* Validation messages */}
      {toolMode !== "variation" && !hasMask && !isGenerating && (
        <p className="text-sm text-center text-muted-foreground">
          Draw on the canvas to create a mask first
        </p>
      )}

      {isBudgetBlocked && (
        <p className="text-sm text-center text-red-600 dark:text-red-400">
          Budget limit reached
        </p>
      )}
    </div>
  )
}

// Helper function to convert File to data URL
async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default OpenAIEditPanel
