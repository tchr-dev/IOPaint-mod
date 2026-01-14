export interface Filename {
  name: string
  height: number
  width: number
  ctime: number
  mtime: number
}

export interface PluginInfo {
  name: string
  support_gen_image: boolean
  support_gen_mask: boolean
}

export interface ServerConfig {
  plugins: PluginInfo[]
  modelInfos: ModelInfo[]
  removeBGModel: string
  removeBGModels: string[]
  realesrganModel: string
  realesrganModels: string[]
  interactiveSegModel: string
  interactiveSegModels: string[]
  enableFileManager: boolean
  enableAutoSaving: boolean
  enableControlnet: boolean
  controlnetMethod: string
  disableModelSwitch: boolean
  isDesktop: boolean
  samplers: string[]
}

export interface GenInfo {
  prompt: string
  negative_prompt: string
}

export interface ModelInfo {
  name: string
  path: string
  model_type:
    | "inpaint"
    | "diffusers_sd"
    | "diffusers_sdxl"
    | "diffusers_sd_inpaint"
    | "diffusers_sdxl_inpaint"
    | "diffusers_other"
  support_strength: boolean
  support_outpainting: boolean
  support_controlnet: boolean
  support_brushnet: boolean
  support_powerpaint_v2: boolean
  controlnets: string[]
  brushnets: string[]
  support_lcm_lora: boolean
  need_prompt: boolean
  is_single_file_diffusers: boolean
}

export enum PluginName {
  RemoveBG = "RemoveBG",
  AnimeSeg = "AnimeSeg",
  RealESRGAN = "RealESRGAN",
  GFPGAN = "GFPGAN",
  RestoreFormer = "RestoreFormer",
  InteractiveSeg = "InteractiveSeg",
}

export interface PluginParams {
  upscale: number
}

export enum SortBy {
  NAME = "name",
  CTIME = "ctime",
  MTIME = "mtime",
}

export enum SortOrder {
  DESCENDING = "desc",
  ASCENDING = "asc",
}

export enum LDMSampler {
  ddim = "ddim",
  plms = "plms",
}

export enum CV2Flag {
  INPAINT_NS = "INPAINT_NS",
  INPAINT_TELEA = "INPAINT_TELEA",
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface Point {
  x: number
  y: number
}

export interface Line {
  size?: number
  pts: Point[]
}

export type LineGroup = Array<Line>

export interface Size {
  width: number
  height: number
}

export enum ExtenderDirection {
  x = "x",
  y = "y",
  xy = "xy",
}

export enum PowerPaintTask {
  text_guided = "text-guided",
  shape_guided = "shape-guided",
  context_aware = "context-aware",
  object_remove = "object-remove",
  outpainting = "outpainting",
}

export type AdjustMaskOperate = "expand" | "shrink" | "reverse"

// ============================================================================
// OpenAI-Compatible Generation Types (Epic 4)
// ============================================================================

/**
 * Generation presets for quick configuration
 * - DRAFT: Fast, cheap preview (512x512, standard quality)
 * - FINAL: High-quality output (1024x1024, HD quality)
 * - CUSTOM: User-defined settings
 */
export enum GenerationPreset {
  DRAFT = "draft",
  FINAL = "final",
  CUSTOM = "custom",
}

/**
 * Available image sizes for OpenAI image generation
 */
export type OpenAIImageSize =
  | "256x256"
  | "512x512"
  | "1024x1024"
  | "1792x1024"
  | "1024x1792"

/**
 * Image quality options
 */
export type OpenAIImageQuality = "standard" | "hd"

/**
 * Configuration for a generation preset
 */
export interface PresetConfig {
  size: OpenAIImageSize
  quality: OpenAIImageQuality
  n: number
}

/**
 * Preset configurations - defines the actual values for each preset
 */
export const PRESET_CONFIGS: Record<GenerationPreset, PresetConfig> = {
  [GenerationPreset.DRAFT]: {
    size: "512x512",
    quality: "standard",
    n: 1,
  },
  [GenerationPreset.FINAL]: {
    size: "1024x1024",
    quality: "hd",
    n: 1,
  },
  [GenerationPreset.CUSTOM]: {
    size: "1024x1024",
    quality: "standard",
    n: 1,
  },
}

/**
 * Cost tier classification for budget awareness
 * - low: <= $0.02 (green, no warning)
 * - medium: $0.02-$0.10 (yellow, informational)
 * - high: > $0.10 (red, confirmation required)
 */
export type CostTier = "low" | "medium" | "high"

/**
 * Cost estimation result from the backend
 */
export interface CostEstimate {
  estimatedCostUsd: number
  tier: CostTier
  warning: string | null
}

/**
 * Budget usage information for a single cap (daily/monthly/session)
 */
export interface BudgetUsage {
  spentUsd: number
  remainingUsd: number
  capUsd: number
  isUnlimited: boolean
  percentageUsed: number
}

/**
 * Complete budget status from the backend
 */
export interface BudgetStatus {
  daily: BudgetUsage
  monthly: BudgetUsage
  session: BudgetUsage
  status: "ok" | "warning" | "blocked"
  message: string | null
}

/**
 * Job status for generation history tracking
 */
export type GenerationJobStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "blocked_budget"

/**
 * A single generation job in history
 * Stores all information needed to display, replay, or audit a generation
 */
export interface GenerationJob {
  /** Unique identifier for this job */
  id: string
  /** Unix timestamp when job was created */
  createdAt: number
  /** Current status of the job */
  status: GenerationJobStatus
  /** Original user intent (raw input) */
  intent: string
  /** Refined prompt after LLM enhancement */
  refinedPrompt: string
  /** Negative prompt for generation */
  negativePrompt: string
  /** Preset used for this generation */
  preset: GenerationPreset
  /** Actual parameters used */
  params: PresetConfig
  /** Model used for generation */
  model: string
  /** Base64 thumbnail for display in history */
  thumbnailDataUrl?: string
  /** Base64 full result image */
  resultImageDataUrl?: string
  /** Estimated cost before generation */
  estimatedCostUsd?: number
  /** Actual cost after generation (if available) */
  actualCostUsd?: number
  /** Error message if failed */
  errorMessage?: string
  /** Fingerprint for deduplication */
  fingerprint?: string
  /** Whether this was an edit operation (vs generate) */
  isEdit?: boolean
}

/**
 * Snapshot of generation history for restore/audit.
 */
export interface HistorySnapshot {
  id: string
  sessionId: string
  payload: Record<string, unknown>
  createdAt: number
}

/**
 * OpenAI model information from list_models endpoint
 */
export interface OpenAIModelInfo {
  id: string
  object: string
  created: number
  owned_by: string
}

/**
 * Request for prompt refinement
 */
export interface RefinePromptRequest {
  prompt: string
  context?: string
  model?: string
  maxTokens?: number
}

/**
 * Response from prompt refinement
 */
export interface RefinePromptResponse {
  originalPrompt: string
  refinedPrompt: string
  modelUsed: string
}

/**
 * Request for image generation
 */
export interface GenerateImageRequest {
  prompt: string
  n?: number
  size?: OpenAIImageSize
  quality?: OpenAIImageQuality
  model?: string
  style?: "vivid" | "natural"
  negativePrompt?: string
}

/**
 * Request for cost estimation
 */
export interface CostEstimateRequest {
  operation: "generate" | "edit" | "variation" | "refine"
  model: string
  size?: string
  quality?: string
  n?: number
}
