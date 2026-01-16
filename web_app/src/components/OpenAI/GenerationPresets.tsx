/**
 * GenerationPresets Component
 *
 * Preset selector for quick configuration of generation parameters.
 * Provides Draft (fast/cheap), Final (high quality), and Custom options.
 *
 * Presets:
 * - Draft: 512x512, standard quality - for quick previews
 * - Final: 1024x1024, HD quality - for final outputs
 * - Custom: User-defined settings
 *
 * @example
 * ```tsx
 * <GenerationPresets />
 * ```
 */

import { Zap, Crown, Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useStore } from "@/lib/states"
import {
  GenerationPreset,
  OpenAIImageSize,
  OpenAIImageQuality,
  OpenAIImageMode,
} from "@/lib/types"
import { cn } from "@/lib/utils"

const PRESET_INFO = {
  [GenerationPreset.DRAFT]: {
    icon: Zap,
    label: "Draft",
    description: "Fast preview",
    hint: "Lowest cost option",
  },
  [GenerationPreset.FINAL]: {
    icon: Crown,
    label: "Final",
    description: "High quality",
    hint: "Best quality output",
  },
  [GenerationPreset.CUSTOM]: {
    icon: Settings2,
    label: "Custom",
    description: "Your settings",
    hint: "Custom configuration",
  },
}


export function GenerationPresets() {
  const [
    selectedPreset,
    setSelectedPreset,
    customConfig,
    updateCustomPresetConfig,
    isGenerating,
    capabilities,
    selectedGenerateModel,
    selectedEditModel,
    isEditMode,
  ] = useStore((state) => [
    state.openAIState.selectedPreset,
    state.setSelectedPreset,
    state.openAIState.customPresetConfig,
    state.updateCustomPresetConfig,
    state.openAIState.isOpenAIGenerating,
    state.openAIState.capabilities,
    state.openAIState.selectedGenerateModel,
    state.openAIState.selectedEditModel,
    state.openAIState.isOpenAIEditMode,
  ])

  const isCustom = selectedPreset === GenerationPreset.CUSTOM
  const mode: OpenAIImageMode = isEditMode ? "images_edit" : "images_generate"
  const activeModelId = isEditMode ? selectedEditModel : selectedGenerateModel
  const modelCaps = capabilities?.modes[mode]?.models.find(
    (model) => model.apiId === activeModelId || model.id === activeModelId
  )
  const sizeOptions = modelCaps?.sizes ?? []
  const qualityOptions = modelCaps?.qualities ?? []
  const formatSizeLabel = (size: OpenAIImageSize) => {
    const [width, height] = size.split("x")
    if (width === height) {
      return `${size} (Square)`
    }
    if (parseInt(width, 10) > parseInt(height, 10)) {
      return `${size} (Landscape)`
    }
    return `${size} (Portrait)`
  }

  return (
    <div className="flex flex-col gap-3">
      <label className="text-sm font-medium text-foreground">Preset</label>

      {/* Preset buttons */}
      <div className="grid grid-cols-3 gap-2">
        {Object.entries(PRESET_INFO).map(([preset, info]) => {
          const Icon = info.icon
          const isSelected = selectedPreset === preset

          return (
            <Button
              key={preset}
              variant={isSelected ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedPreset(preset as GenerationPreset)}
              disabled={isGenerating}
              className={cn(
                "flex flex-col h-auto py-2 gap-0.5",
                isSelected && "ring-2 ring-ring ring-offset-1"
              )}
            >
              <Icon className="h-4 w-4 mb-0.5" />
              <span className="font-medium">{info.label}</span>
              <span className="text-[10px] opacity-70">{info.description}</span>
            </Button>
          )
        })}
      </div>

      {/* Custom options */}
      {isCustom && (
        <div className="flex flex-col gap-3 p-3 bg-muted/50 rounded-lg">
          <div className="flex flex-col gap-2">
            <label className="text-xs font-medium">Size</label>
            <Select
              value={customConfig.size}
              onValueChange={(value) =>
                updateCustomPresetConfig({ size: value as OpenAIImageSize })
              }
              disabled={isGenerating || sizeOptions.length <= 1}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {sizeOptions.map((size) => (
                  <SelectItem key={size} value={size}>
                    {formatSizeLabel(size)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-xs font-medium">Quality</label>
            <Select
              value={customConfig.quality}
              onValueChange={(value) =>
                updateCustomPresetConfig({ quality: value as OpenAIImageQuality })
              }
              disabled={isGenerating || qualityOptions.length <= 1}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {qualityOptions.map((quality) => (
                  <SelectItem key={quality} value={quality}>
                    {quality === "hd" ? "HD" : "Standard"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      )}

      {/* Preset info hint */}
      <p className="text-xs text-muted-foreground">
        {PRESET_INFO[selectedPreset].hint}
      </p>
    </div>
  )
}

export default GenerationPresets
