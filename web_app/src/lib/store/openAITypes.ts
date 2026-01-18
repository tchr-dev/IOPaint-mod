import { GenerationPreset, PRESET_CONFIGS } from "../types"
import type {
  PresetConfig,
  CostEstimate,
  BudgetStatus,
  BudgetLimits,
  GenerationJob,
  HistorySnapshot,
  OpenAICapabilities,
} from "../types"

export type OpenAIState = {
  isOpenAIMode: boolean
  capabilities: OpenAICapabilities | null
  selectedGenerateModel: string
  selectedEditModel: string
  openAIIntent: string
  isRefiningPrompt: boolean
  openAIRefinedPrompt: string
  openAINegativePrompt: string
  selectedPreset: GenerationPreset
  customPresetConfig: PresetConfig
  isOpenAIGenerating: boolean
  currentJobId: string | null
  openAIGenerationProgress: number
  currentCostEstimate: CostEstimate | null
  showCostWarningModal: boolean
  pendingGenerationRequest: GenerateImageRequest | null
  generationHistory: GenerationJob[]
  historyFilter: "all" | "succeeded" | "failed"
  historySnapshots: HistorySnapshot[]
  budgetStatus: BudgetStatus | null
  budgetLimits: BudgetLimits | null
  isOpenAIEditMode: boolean
  editSourceImageDataUrl: string | null
}

export type GenerateImageRequest = {
  prompt: string
  negativePrompt: string
  model: string
  size: string
  quality: string
  n: number
}

export const openAIDefaultValues: OpenAIState = {
  isOpenAIMode: false,
  capabilities: null,
  selectedGenerateModel: "",
  selectedEditModel: "",
  openAIIntent: "",
  isRefiningPrompt: false,
  openAIRefinedPrompt: "",
  openAINegativePrompt: "",
  selectedPreset: GenerationPreset.DRAFT,
  customPresetConfig: { ...PRESET_CONFIGS[GenerationPreset.CUSTOM] },
  isOpenAIGenerating: false,
  currentJobId: null,
  openAIGenerationProgress: 0,
  currentCostEstimate: null,
  showCostWarningModal: false,
  pendingGenerationRequest: null,
  generationHistory: [],
  historyFilter: "all",
  historySnapshots: [],
  budgetStatus: null,
  budgetLimits: null,
  isOpenAIEditMode: false,
  editSourceImageDataUrl: null,
}
