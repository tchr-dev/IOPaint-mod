import { useRef, useState, useEffect } from "react"
import { useClickAway } from "react-use"
import { useStore } from "@/lib/states"
import { switchModel } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useToast } from "../ui/use-toast"
import {
  PRESET_CONFIGS,
  QualityPreset,
  resolveModelForPreset,
  detectPresetFromModel,
} from "@/lib/presets"
import { RowContainer, LabelTitle } from "./LabelTitle"
import { Button } from "../ui/button"
import { Separator } from "../ui/separator"
import { Textarea } from "../ui/textarea"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select"
import { X } from "lucide-react"

const SimplifiedOptions = () => {
  const [
    settings,
    modelInfos,
    isProcessing,
    updateSettings,
    runInpainting,
    cancelInpainting,
    isInpainting,
    showPrevMask,
    hidePrevMask,
    setModel,
  ] = useStore((state) => [
    state.settings,
    state.serverConfig.modelInfos,
    state.getIsProcessing(),
    state.updateSettings,
    state.runInpainting,
    state.cancelInpainting,
    state.isInpainting,
    state.showPrevMask,
    state.hidePrevMask,
    state.setModel,
  ])
  const { toast } = useToast()
  const [isSwitching, setIsSwitching] = useState(false)
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>(
    detectPresetFromModel(settings.model)
  )
  const promptRef = useRef(null)

  useEffect(() => {
    setQualityPreset(detectPresetFromModel(settings.model))
  }, [settings.model])

  useClickAway<MouseEvent>(promptRef, () => {
    if (promptRef.current) {
      const input = promptRef.current as HTMLTextAreaElement
      input.blur()
    }
  })

  const handlePresetChange = async (value: string) => {
    const nextPreset = value as QualityPreset
    const resolvedModel = resolveModelForPreset(nextPreset, modelInfos)

    if (!resolvedModel) {
      toast({
        variant: "destructive",
        description:
          "No compatible model found. Download the model and try again.",
      })
      return
    }

    if (resolvedModel.name === settings.model.name) {
      setQualityPreset(nextPreset)
      return
    }

    setIsSwitching(true)
    try {
      const newModel = await switchModel(resolvedModel.name)
      setModel(newModel)
      setQualityPreset(nextPreset)
      toast({
        title: `Switched to ${newModel.name}`,
      })
    } catch (error: any) {
      toast({
        variant: "destructive",
        description: `Switch failed: ${error}`,
      })
    } finally {
      setIsSwitching(false)
    }
  }

  const presetOptions = [
    QualityPreset.FAST,
    QualityPreset.BALANCED,
    QualityPreset.BEST,
  ]

  if (qualityPreset === QualityPreset.CUSTOM) {
    presetOptions.push(QualityPreset.CUSTOM)
  }

  const handlePromptInput = (event: React.FormEvent<HTMLTextAreaElement>) => {
    event.preventDefault()
    event.stopPropagation()
    const target = event.target as HTMLTextAreaElement
    updateSettings({ prompt: target.value })
  }

  const handleRepaintClick = () => {
    if (!isProcessing) {
      runInpainting()
    }
  }

  const onPromptKeyUp = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && event.ctrlKey && settings.prompt.length !== 0) {
      handleRepaintClick()
    }
  }

  return (
    <div className="flex flex-col gap-4 py-4" data-testid="simplified-options">
      <RowContainer>
        <LabelTitle text="Quality" />
        <Select
          onValueChange={handlePresetChange}
          value={qualityPreset}
          disabled={isSwitching}
        >
          <SelectTrigger className="w-[170px]">
            <SelectValue placeholder="Select quality" />
          </SelectTrigger>
          <SelectContent align="end">
            <SelectGroup>
              {presetOptions.map((preset) => (
                <SelectItem key={preset} value={preset}>
                  {PRESET_CONFIGS[preset].displayName}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </RowContainer>

      <div className="text-sm text-muted-foreground">
        {PRESET_CONFIGS[qualityPreset].description}
      </div>

      <div
        className={cn(
          "transition-all duration-200 ease-in-out overflow-hidden",
          settings.model.need_prompt
            ? "max-h-[500px] opacity-100 mt-4"
            : "max-h-0 opacity-0"
        )}
      >
        <Separator />
        <div className="space-y-3">
          <LabelTitle text="Prompt" />
          <Textarea
            ref={promptRef}
            placeholder="Describe what to generate..."
            className="min-h-[96px] resize-none"
            value={settings.prompt}
            onInput={handlePromptInput}
            onKeyUp={onPromptKeyUp}
          />
          {isInpainting ? (
            <div className="flex gap-2">
              <Button size="sm" disabled>
                Painting...
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={cancelInpainting}
                title="Cancel generation"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <Button
              size="sm"
              onClick={handleRepaintClick}
              disabled={isProcessing}
              onMouseEnter={showPrevMask}
              onMouseLeave={hidePrevMask}
            >
              Paint
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

export default SimplifiedOptions
