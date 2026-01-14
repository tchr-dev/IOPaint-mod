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
import { GenerationPreset, OpenAIImageSize, OpenAIImageQuality } from "@/lib/types"
import { cn } from "@/lib/utils"

const PRESET_INFO = {
  [GenerationPreset.DRAFT]: {
    icon: Zap,
    label: "Draft",
    description: "512x512, standard",
    hint: "Fast & cheap preview",
  },
  [GenerationPreset.FINAL]: {
    icon: Crown,
    label: "Final",
    description: "1024x1024, HD",
    hint: "High quality output",
  },
  [GenerationPreset.CUSTOM]: {
    icon: Settings2,
    label: "Custom",
    description: "Your settings",
    hint: "Custom configuration",
  },
}

const SIZE_OPTIONS: { value: OpenAIImageSize; label: string }[] = [
  { value: "256x256", label: "256x256" },
  { value: "512x512", label: "512x512" },
  { value: "1024x1024", label: "1024x1024" },
  { value: "1792x1024", label: "1792x1024 (Wide)" },
  { value: "1024x1792", label: "1024x1792 (Tall)" },
]

const QUALITY_OPTIONS: { value: OpenAIImageQuality; label: string }[] = [
  { value: "standard", label: "Standard" },
  { value: "hd", label: "HD" },
]

export function GenerationPresets() {
  const [
    selectedPreset,
    setSelectedPreset,
    customConfig,
    updateCustomPresetConfig,
    isGenerating,
  ] = useStore((state) => [
    state.openAIState.selectedPreset,
    state.setSelectedPreset,
    state.openAIState.customPresetConfig,
    state.updateCustomPresetConfig,
    state.openAIState.isOpenAIGenerating,
  ])

  const isCustom = selectedPreset === GenerationPreset.CUSTOM

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
              disabled={isGenerating}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SIZE_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
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
              disabled={isGenerating}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {QUALITY_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
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
