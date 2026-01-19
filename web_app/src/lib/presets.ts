import type { ModelInfo } from "./types"
import {
  MODEL_TYPE_DIFFUSERS_SD,
  MODEL_TYPE_DIFFUSERS_SD_INPAINT,
  MODEL_TYPE_DIFFUSERS_SDXL,
  MODEL_TYPE_DIFFUSERS_SDXL_INPAINT,
} from "./const"

export enum QualityPreset {
  FAST = "fast",
  BALANCED = "balanced",
  BEST = "best",
  CUSTOM = "custom",
}

export interface PresetConfig {
  modelName: string
  displayName: string
  description: string
}

export const PRESET_CONFIGS: Record<QualityPreset, PresetConfig> = {
  [QualityPreset.FAST]: {
    modelName: "lama",
    displayName: "Fast",
    description: "Quick inpainting for previews",
  },
  [QualityPreset.BALANCED]: {
    modelName: "mat",
    displayName: "Balanced",
    description: "Good quality with reasonable speed",
  },
  [QualityPreset.BEST]: {
    modelName: "auto-sd",
    displayName: "Best Quality",
    description: "Highest quality using Stable Diffusion (requires prompt)",
  },
  [QualityPreset.CUSTOM]: {
    modelName: "",
    displayName: "Custom",
    description: "Manually selected model",
  },
}

const SD_PRIORITY_ORDER = [
  MODEL_TYPE_DIFFUSERS_SD_INPAINT,
  MODEL_TYPE_DIFFUSERS_SD,
  MODEL_TYPE_DIFFUSERS_SDXL_INPAINT,
  MODEL_TYPE_DIFFUSERS_SDXL,
]

const findByName = (models: ModelInfo[], name: string) =>
  models.find((info) => info.name === name) ?? null

const findByType = (models: ModelInfo[], modelType: string) =>
  models.find((info) => info.model_type === modelType) ?? null

export function resolveModelForPreset(
  preset: QualityPreset,
  availableModels: ModelInfo[]
): ModelInfo | null {
  if (!availableModels || availableModels.length === 0) {
    return null
  }

  if (preset === QualityPreset.FAST) {
    return findByName(
      availableModels,
      PRESET_CONFIGS[QualityPreset.FAST].modelName
    )
  }

  if (preset === QualityPreset.BALANCED) {
    return findByName(
      availableModels,
      PRESET_CONFIGS[QualityPreset.BALANCED].modelName
    )
  }

  if (preset === QualityPreset.BEST) {
    for (const modelType of SD_PRIORITY_ORDER) {
      const match = findByType(availableModels, modelType)
      if (match) {
        return match
      }
    }
    return findByName(
      availableModels,
      PRESET_CONFIGS[QualityPreset.BALANCED].modelName
    )
  }

  return null
}

export function detectPresetFromModel(model: ModelInfo): QualityPreset {
  if (model.name === PRESET_CONFIGS[QualityPreset.FAST].modelName) {
    return QualityPreset.FAST
  }

  if (model.name === PRESET_CONFIGS[QualityPreset.BALANCED].modelName) {
    return QualityPreset.BALANCED
  }

  if (SD_PRIORITY_ORDER.includes(model.model_type)) {
    return QualityPreset.BEST
  }

  return QualityPreset.CUSTOM
}
