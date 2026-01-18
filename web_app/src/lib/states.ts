import { persist } from "zustand/middleware"
import { shallow } from "zustand/shallow"
import { immer } from "zustand/middleware/immer"
import { castDraft, type Draft } from "immer"
import { createWithEqualityFn } from "zustand/traditional"
import {
  AdjustMaskOperate,
  CV2Flag,
  ExtenderDirection,
  LDMSampler,
  Line,
  LineGroup,
  ModelInfo,
  PluginParams,
  Point,
  PowerPaintTask,
  ServerConfig,
  Size,
  StoredImage,
  SortBy,
  SortOrder,
  // OpenAI types (Epic 4)
  GenerationPreset,
  PresetConfig,
  CostEstimate,
  BudgetStatus,
  BudgetLimits,
  GenerationJob,
  HistorySnapshot,
  OpenAICapabilities,
  OpenAICapabilityModel,
  OpenAIModeCapabilities,
  OpenAIImageMode,
  OpenAIProvider,
  OpenAIToolMode,
  PRESET_CONFIGS,
  GenerateImageRequest,
} from "./types"
import {
  BRUSH_COLOR,
  DEFAULT_BRUSH_SIZE,
  DEFAULT_NEGATIVE_PROMPT,
  MAX_BRUSH_SIZE,
  MODEL_TYPE_INPAINT,
  PAINT_BY_EXAMPLE,
} from "./const"
import {
  blobToImage,
  canvasToImage,
  convertToBase64,
  createStoredImage,
  dataURItoBlob,
  generateMask,
  loadImage,
  resolveStoredImages,
  srcToFile,
} from "./utils"
import inpaint, { getGenInfo, postAdjustMask, runPlugin, inpaintController } from "./api"
import {
  fetchOpenAICapabilities,
  refinePrompt as apiRefinePrompt,
  submitOpenAIJob,
  getOpenAIJob,
  cancelOpenAIJob,
  fetchStoredImageAsDataUrl,
  upscaleImage,
  removeBackground,
  estimateGenerationCost,
  getOpenAIBudgetStatus,
  getOpenAIBudgetLimits,
  updateOpenAIBudgetLimits,
  createThumbnail,
  resolveOpenAIBaseUrl,
  // History API (Epic 3)
  fetchHistory as apiFetchHistory,
  createHistoryJob as apiCreateHistoryJob,
  updateHistoryJob as apiUpdateHistoryJob,
  deleteHistoryJob as apiDeleteHistoryJob,
  clearHistory as apiClearHistory,
  type BackendGenerationJob,
  fetchHistorySnapshots as apiFetchHistorySnapshots,
  createHistorySnapshot as apiCreateHistorySnapshot,
  deleteHistorySnapshot as apiDeleteHistorySnapshot,
  clearHistorySnapshots as apiClearHistorySnapshots,
  type BackendHistorySnapshot,
} from "./openai-api"
import { toast } from "@/components/ui/use-toast"

const OPENAI_JOB_POLL_INTERVAL_MS = 5000
const OPENAI_JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000
const OPENAI_JOB_POLL_MAX_RETRYABLE = 3
const OPENAI_TERMINAL_SUCCESS = new Set(["succeeded", "completed"])
const OPENAI_TERMINAL_FAILURE = new Set([
  "failed",
  "cancelled",
  "blocked",
  "blocked_budget",
  "rejected",
  "expired",
])
const OPENAI_RETRYABLE = new Set([
  "timeout",
  "rate_limited",
  "overloaded",
  "service_unavailable",
  "internal_error",
])
const openAIJobPollers = new Map<string, number>()
const openAIJobPollerMeta = new Map<
  string,
  { startedAt: number; retryCount: number }
>()

const toBase64Payload = async (blob: Blob): Promise<string> => {
  const dataUrl = await convertToBase64(blob)
  return dataUrl.split(",")[1] || ""
}

const mapBackendJobToFrontend = (job: BackendGenerationJob): GenerationJob => {
  const defaultParams: PresetConfig = {
    size: "1024x1024",
    quality: "standard",
    n: 1,
  }
  const params = job.params
    ? ({
        size: (job.params.size as string) || defaultParams.size,
        quality: (job.params.quality as string) || defaultParams.quality,
        n: (job.params.n as number) || defaultParams.n,
      } as PresetConfig)
    : defaultParams

  return {
    id: job.id,
    createdAt: new Date(job.created_at).getTime(),
    status: job.status as GenerationJob["status"],
    intent: job.intent || "",
    refinedPrompt: job.refined_prompt || "",
    negativePrompt: job.negative_prompt || "",
    preset: (job.preset || "draft") as GenerationPreset,
    params,
    model: job.model,
    estimatedCostUsd: job.estimated_cost_usd,
    actualCostUsd: job.actual_cost_usd,
    errorMessage: job.error_message,
    fingerprint: job.fingerprint,
    isEdit: job.is_edit,
  }
}

const getModeCapabilities = (
  capabilities: OpenAICapabilities | null,
  mode: OpenAIImageMode
) => capabilities?.modes[mode]

const getModelCapabilities = (
  capabilities: OpenAICapabilities | null,
  mode: OpenAIImageMode,
  modelId: string
): OpenAICapabilityModel | undefined => {
  if (!capabilities || !modelId) return undefined
  const modeCaps = getModeCapabilities(capabilities, mode)
  if (!modeCaps) return undefined
  return modeCaps.models.find(
    (model) => model.apiId === modelId || model.id === modelId
  )
}

const resolveModelSelection = (
  modeCaps: OpenAIModeCapabilities | undefined,
  currentId: string
): string => {
  if (!modeCaps || modeCaps.models.length === 0) return ""
  const match = modeCaps.models.find(
    (model) => model.apiId === currentId || model.id === currentId
  )
  return match ? match.apiId : modeCaps.models[0].apiId
}

const resolveOption = <T,>(
  value: T,
  options: T[],
  fallback?: T
): T => {
  if (options.length === 0) return value
  if (options.includes(value)) return value
  if (fallback && options.includes(fallback)) return fallback
  return options[0]
}

const resolvePresetConfig = (
  preset: GenerationPreset,
  customConfig: PresetConfig,
  modelCaps?: OpenAICapabilityModel
): PresetConfig => {
  const base = preset === GenerationPreset.CUSTOM
    ? customConfig
    : PRESET_CONFIGS[preset]

  if (!modelCaps) {
    return base
  }

  return {
    ...base,
    size: resolveOption(base.size, modelCaps.sizes, modelCaps.defaultSize),
    quality: resolveOption(
      base.quality,
      modelCaps.qualities,
      modelCaps.defaultQuality
    ),
  }
}
type StoreState = AppState & AppAction

type StoreSet = (
  nextStateOrUpdater:
    | StoreState
    | Partial<StoreState>
    | ((state: Draft<StoreState>) => void),
  shouldReplace?: boolean
) => void

const startOpenAIJobPolling = (
  jobId: string,
  baseUrl: string | undefined,
  get: () => StoreState,
  set: StoreSet,
  options: { loadIntoEditor: boolean }
) => {
  if (openAIJobPollers.has(jobId)) return

  const finalizeJobPolling = (
    jobIdToStop: string,
    getState: () => StoreState,
    setState: StoreSet
  ) => {
    if (getState().openAIState.currentJobId === jobIdToStop) {
      setState((state) => {
        state.openAIState.currentJobId = null
        state.openAIState.isOpenAIGenerating = false
      })
    }

    const intervalId = openAIJobPollers.get(jobIdToStop)
    if (intervalId) {
      window.clearInterval(intervalId)
    }
    openAIJobPollers.delete(jobIdToStop)
    openAIJobPollerMeta.delete(jobIdToStop)
  }

  const poll = async () => {
    try {
      const backendJob = await getOpenAIJob(jobId, baseUrl)
      const mappedJob = mapBackendJobToFrontend(backendJob)
      const existingJob = get().openAIState.generationHistory.find(
        (job) => job.id === jobId
      )

      if (!existingJob) {
        get().addToHistory(mappedJob, { skipBackend: true })
      } else {
        get().updateHistoryJob(
          jobId,
          {
            status: mappedJob.status,
            actualCostUsd: mappedJob.actualCostUsd,
            errorMessage: mappedJob.errorMessage,
          },
          { skipBackend: true }
        )
      }

      const meta = openAIJobPollerMeta.get(jobId) ?? {
        startedAt: Date.now(),
        retryCount: 0,
      }
      openAIJobPollerMeta.set(jobId, meta)

      if (OPENAI_RETRYABLE.has(mappedJob.status)) {
        meta.retryCount += 1
        openAIJobPollerMeta.set(jobId, meta)
        if (meta.retryCount > OPENAI_JOB_POLL_MAX_RETRYABLE) {
          get().updateHistoryJob(
            jobId,
            {
              status: "failed",
              errorMessage: "Job failed after repeated retryable errors.",
            },
            { skipBackend: true }
          )
          toast({
            variant: "destructive",
            description: "Job failed after repeated retryable errors.",
          })
          finalizeJobPolling(jobId, get, set)
        }
        return
      }

      const elapsed = Date.now() - meta.startedAt
      if (elapsed > OPENAI_JOB_POLL_TIMEOUT_MS) {
        get().updateHistoryJob(
          jobId,
          {
            status: "failed",
            errorMessage: "Job timed out while waiting for completion.",
          },
          { skipBackend: true }
        )
        toast({
          variant: "destructive",
          description: "Job timed out while waiting for completion.",
        })
        finalizeJobPolling(jobId, get, set)
        return
      }

      const isTerminal =
        OPENAI_TERMINAL_SUCCESS.has(mappedJob.status) ||
        OPENAI_TERMINAL_FAILURE.has(mappedJob.status)

      if (isTerminal) {
        if (backendJob.result_image_id) {
          const resultDataUrl = await fetchStoredImageAsDataUrl(
            backendJob.result_image_id
          )
          const thumbnail = await createThumbnail(resultDataUrl, 128)
          get().updateHistoryJob(
            jobId,
            {
              resultImageDataUrl: resultDataUrl,
              thumbnailDataUrl: thumbnail,
            },
            { skipBackend: true }
          )

          if (options.loadIntoEditor) {
            // Convert result to File and load into editor (same as GenerationHistory.handleOpenInEditor)
            const generatedFile = await srcToFile(
              resultDataUrl,
              `generation-${jobId}.png`,
              "image/png"
            )
            await get().setFile(generatedFile)
          }
        }

        if (OPENAI_TERMINAL_SUCCESS.has(mappedJob.status)) {
          toast({
            description: "Job completed successfully!",
          })
        } else if (mappedJob.status === "cancelled") {
          toast({
            description: "Job cancelled.",
          })
        } else if (mappedJob.status === "blocked_budget") {
          toast({
            variant: "destructive",
            description: "Job blocked by budget limits.",
          })
        } else {
          toast({
            variant: "destructive",
            description: mappedJob.errorMessage
              ? `Job failed: ${mappedJob.errorMessage}`
              : "Job failed.",
          })
        }

        get().refreshBudgetStatus()
        finalizeJobPolling(jobId, get, set)
      }
    } catch (error) {
      console.error("Failed to poll job:", error)
    }
  }

  void poll()
  const intervalId = window.setInterval(poll, OPENAI_JOB_POLL_INTERVAL_MS)
  openAIJobPollers.set(jobId, intervalId)
}

type FileManagerState = {
  sortBy: SortBy
  sortOrder: SortOrder
  layout: "rows" | "masonry"
  searchText: string
  inputDirectory: string
  outputDirectory: string
}

type CropperState = {
  x: number
  y: number
  width: number
  height: number
}

export type Settings = {
  model: ModelInfo
  enableDownloadMask: boolean
  enableManualInpainting: boolean
  enableUploadMask: boolean
  enableAutoExtractPrompt: boolean
  openAIProvider: OpenAIProvider
  openAIToolMode: OpenAIToolMode
  showCropper: boolean
  showExtender: boolean
  extenderDirection: ExtenderDirection

  // For LDM
  ldmSteps: number
  ldmSampler: LDMSampler

  // For ZITS
  zitsWireframe: boolean

  // For OpenCV2
  cv2Radius: number
  cv2Flag: CV2Flag

  // For Diffusion moel
  prompt: string
  negativePrompt: string
  seed: number
  seedFixed: boolean

  // For SD
  sdMaskBlur: number
  sdStrength: number
  sdSteps: number
  sdGuidanceScale: number
  sdSampler: string
  sdMatchHistograms: boolean
  sdScale: number

  // Pix2Pix
  p2pImageGuidanceScale: number

  // ControlNet
  enableControlnet: boolean
  controlnetConditioningScale: number
  controlnetMethod: string

  // BrushNet
  enableBrushNet: boolean
  brushnetMethod: string
  brushnetConditioningScale: number

  enableLCMLora: boolean

  // PowerPaint
  enablePowerPaintV2: boolean
  powerpaintTask: PowerPaintTask

  // AdjustMask
  adjustMaskKernelSize: number
}

type InteractiveSegState = {
  isInteractiveSeg: boolean
  tmpInteractiveSegMask: StoredImage | null
  clicks: number[][]
}

type EditorState = {
  baseBrushSize: number
  brushSizeScale: number
  renders: StoredImage[]
  lineGroups: LineGroup[]
  lastLineGroup: LineGroup
  curLineGroup: LineGroup

  // mask from interactive-seg or other segmentation models
  extraMasks: StoredImage[]
  prevExtraMasks: StoredImage[]

  temporaryMasks: StoredImage[]
  // redo 相关
  redoRenders: StoredImage[]
  redoCurLines: Line[]
  redoLineGroups: LineGroup[]
}

// ============================================================================
// OpenAI State (Epic 4)
// ============================================================================

/**
 * OpenAI generation state for the "Refine → Generate/Edit" workflow.
 * Manages the entire flow from user intent to final generation.
 */
type OpenAIState = {
  // Mode toggle - switches between local models and OpenAI API
  isOpenAIMode: boolean

  // Capabilities from OpenAI API
  capabilities: OpenAICapabilities | null
  selectedGenerateModel: string
  selectedEditModel: string

  // Intent → Refine → Final prompt flow
  openAIIntent: string
  isRefiningPrompt: boolean
  openAIRefinedPrompt: string
  openAINegativePrompt: string

  // Preset selection
  selectedPreset: GenerationPreset
  customPresetConfig: PresetConfig

  // Generation execution state
  isOpenAIGenerating: boolean
  currentJobId: string | null
  openAIGenerationProgress: number

  // Cost awareness
  currentCostEstimate: CostEstimate | null
  showCostWarningModal: boolean
  pendingGenerationRequest: GenerateImageRequest | null

  // History/Gallery (E4.3)
  generationHistory: GenerationJob[]
  historyFilter: "all" | "succeeded" | "failed"
  historySnapshots: HistorySnapshot[]

  // Budget status cache
  budgetStatus: BudgetStatus | null
  budgetLimits: BudgetLimits | null

  // Edit mode (E4.2)
  isOpenAIEditMode: boolean
  editSourceImageDataUrl: string | null
}

/**
 * OpenAI actions for managing generation workflow.
 */
type OpenAIAction = {
  // Mode toggle
  setOpenAIMode: (enabled: boolean) => void
  setOpenAIEditMode: (enabled: boolean) => void

  // Capabilities management
  fetchOpenAICapabilities: () => Promise<void>
  setSelectedGenerateModel: (modelId: string) => void
  setSelectedEditModel: (modelId: string) => void

  // Intent/Prompt flow
  setOpenAIIntent: (intent: string) => void
  refinePrompt: () => Promise<void>
  setOpenAIRefinedPrompt: (prompt: string) => void
  setOpenAINegativePrompt: (prompt: string) => void

  // Presets
  setSelectedPreset: (preset: GenerationPreset) => void
  updateCustomPresetConfig: (config: Partial<PresetConfig>) => void
  getActivePresetConfig: (mode?: OpenAIImageMode) => PresetConfig

  // Cost estimation
  updateCostEstimate: () => Promise<void>

  // Generation execution
  requestOpenAIGeneration: () => Promise<void>
  confirmOpenAIGeneration: () => Promise<void>
  cancelPendingGeneration: () => void

  // History management (with backend sync - Epic 3)
  addToHistory: (job: GenerationJob, options?: { skipBackend?: boolean }) => void
  updateHistoryJob: (
    id: string,
    updates: Partial<GenerationJob>,
    options?: { skipBackend?: boolean }
  ) => void
  removeFromHistory: (id: string) => void
  clearHistory: () => void
  rerunJob: (jobId: string) => Promise<void>
  restoreFromJob: (jobId: string) => void
  copyJobPrompt: (jobId: string) => void
  setHistoryFilter: (filter: "all" | "succeeded" | "failed") => void
  syncHistoryWithBackend: () => Promise<void>
  syncHistorySnapshots: () => Promise<void>
  saveHistorySnapshot: (payload?: Record<string, unknown>) => Promise<void>
  deleteHistorySnapshot: (snapshotId: string) => Promise<void>
  clearHistorySnapshots: () => Promise<void>

  // Budget
  refreshBudgetStatus: () => Promise<void>
  refreshBudgetLimits: () => Promise<void>
  updateBudgetLimits: (limits: BudgetLimits) => Promise<void>

  // Edit flow
  setEditSourceImage: (dataUrl: string | null) => void
  runOpenAIEdit: () => Promise<void>
  runOpenAIOutpaint: () => Promise<void>
  runOpenAIVariation: () => Promise<void>
  cancelOpenAIJob: (jobId: string) => Promise<void>
}

type AppState = {
  file: File | null
  paintByExampleFile: File | null
  customMask: File | null
  imageHeight: number
  imageWidth: number
  isInpainting: boolean
  rendersCountBeforeInpaint: number
  isPluginRunning: boolean
  isAdjustingMask: boolean
  windowSize: Size
  editorState: EditorState
  disableShortCuts: boolean

  interactiveSegState: InteractiveSegState
  fileManagerState: FileManagerState

  cropperState: CropperState
  extenderState: CropperState
  isCropperExtenderResizing: boolean

  serverConfig: ServerConfig

  settings: Settings

  // OpenAI State (Epic 4) - embedded directly for simpler access
  openAIState: OpenAIState
}

type AppAction = {
  updateAppState: (newState: Partial<AppState>) => void
  setFile: (file: File) => Promise<void>
  setCustomFile: (file: File) => void
  setIsInpainting: (newValue: boolean) => void
  getIsProcessing: () => boolean
  setBaseBrushSize: (newValue: number) => void
  decreaseBaseBrushSize: () => void
  increaseBaseBrushSize: () => void
  getBrushSize: () => number
  setImageSize: (width: number, height: number) => void

  isSD: () => boolean

  setCropperX: (newValue: number) => void
  setCropperY: (newValue: number) => void
  setCropperWidth: (newValue: number) => void
  setCropperHeight: (newValue: number) => void

  setExtenderX: (newValue: number) => void
  setExtenderY: (newValue: number) => void
  setExtenderWidth: (newValue: number) => void
  setExtenderHeight: (newValue: number) => void

  setIsCropperExtenderResizing: (newValue: boolean) => void
  updateExtenderDirection: (newValue: ExtenderDirection) => void
  resetExtender: (width: number, height: number) => void
  updateExtenderByBuiltIn: (direction: ExtenderDirection, scale: number) => void

  setServerConfig: (newValue: ServerConfig) => void
  setSeed: (newValue: number) => void
  updateSettings: (newSettings: Partial<Settings>) => void

  // 互斥
  updateEnablePowerPaintV2: (newValue: boolean) => void
  updateEnableBrushNet: (newValue: boolean) => void
  updateEnableControlnet: (newValue: boolean) => void
  updateLCMLora: (newValue: boolean) => void

  setModel: (newModel: ModelInfo) => void
  updateFileManagerState: (newState: Partial<FileManagerState>) => void
  updateInteractiveSegState: (newState: Partial<InteractiveSegState>) => void
  resetInteractiveSegState: () => void
  handleInteractiveSegAccept: () => void
  handleFileManagerMaskSelect: (blob: Blob) => Promise<void>
  showPromptInput: () => boolean

  runInpainting: () => Promise<void>
  cancelInpainting: () => void
  showPrevMask: () => Promise<void>
  hidePrevMask: () => void
  runRenderablePlugin: (
    genMask: boolean,
    pluginName: string,
    params?: PluginParams
  ) => Promise<void>

  // EditorState
  getCurrentTargetFile: () => Promise<File>
  updateEditorState: (newState: Partial<EditorState>) => void
  runMannually: () => boolean
  handleCanvasMouseDown: (point: Point) => void
  handleCanvasMouseMove: (point: Point) => void
  cleanCurLineGroup: () => void
  resetRedoState: () => void
  undo: () => void
  redo: () => void
  undoDisabled: () => boolean
  redoDisabled: () => boolean

  adjustMask: (operate: AdjustMaskOperate) => Promise<void>
  clearMask: () => void
} & OpenAIAction // Include all OpenAI actions

const defaultValues: AppState = {
  file: null,
  paintByExampleFile: null,
  customMask: null,
  imageHeight: 0,
  imageWidth: 0,
  isInpainting: false,
  rendersCountBeforeInpaint: 0,
  isPluginRunning: false,
  isAdjustingMask: false,
  disableShortCuts: false,

  windowSize: {
    height: 600,
    width: 800,
  },
  editorState: {
    baseBrushSize: DEFAULT_BRUSH_SIZE,
    brushSizeScale: 1,
    renders: [],
    extraMasks: [],
    prevExtraMasks: [],
    temporaryMasks: [],
    lineGroups: [],
    lastLineGroup: [],
    curLineGroup: [],
    redoRenders: [],
    redoCurLines: [],
    redoLineGroups: [],
  },

  interactiveSegState: {
    isInteractiveSeg: false,
    tmpInteractiveSegMask: null,
    clicks: [],
  },

  cropperState: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
  extenderState: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },
  isCropperExtenderResizing: false,

  fileManagerState: {
    sortBy: SortBy.CTIME,
    sortOrder: SortOrder.DESCENDING,
    layout: "masonry",
    searchText: "",
    inputDirectory: "",
    outputDirectory: "",
  },
  serverConfig: {
    plugins: [],
    modelInfos: [],
    removeBGModel: "briaai/RMBG-1.4",
    removeBGModels: [],
    realesrganModel: "realesr-general-x4v3",
    realesrganModels: [],
    interactiveSegModel: "vit_b",
    interactiveSegModels: [],
    enableFileManager: false,
    enableAutoSaving: false,
    enableControlnet: false,
    controlnetMethod: "lllyasviel/control_v11p_sd15_canny",
    disableModelSwitch: false,
    isDesktop: false,
    samplers: ["DPM++ 2M SDE Karras"],
  },
  settings: {
    model: {
      name: "lama",
      path: "lama",
      model_type: "inpaint",
      support_controlnet: false,
      support_brushnet: false,
      support_strength: false,
      support_outpainting: false,
      support_powerpaint_v2: false,
      controlnets: [],
      brushnets: [],
      support_lcm_lora: false,
      is_single_file_diffusers: false,
      need_prompt: false,
    },
    showCropper: false,
    showExtender: false,
    extenderDirection: ExtenderDirection.xy,
    enableDownloadMask: false,
    enableManualInpainting: false,
    enableUploadMask: false,
    enableAutoExtractPrompt: true,
    openAIProvider: "server",
    openAIToolMode: "local",
    ldmSteps: 30,
    ldmSampler: LDMSampler.ddim,
    zitsWireframe: true,
    cv2Radius: 5,
    cv2Flag: CV2Flag.INPAINT_NS,
    prompt: "",
    negativePrompt: DEFAULT_NEGATIVE_PROMPT,
    seed: 42,
    seedFixed: false,
    sdMaskBlur: 12,
    sdStrength: 1.0,
    sdSteps: 50,
    sdGuidanceScale: 7.5,
    sdSampler: "DPM++ 2M",
    sdMatchHistograms: false,
    sdScale: 1.0,
    p2pImageGuidanceScale: 1.5,
    enableControlnet: false,
    controlnetMethod: "lllyasviel/control_v11p_sd15_canny",
    controlnetConditioningScale: 0.4,
    enableBrushNet: false,
    brushnetMethod: "random_mask",
    brushnetConditioningScale: 1.0,
    enableLCMLora: false,
    enablePowerPaintV2: false,
    powerpaintTask: PowerPaintTask.text_guided,
    adjustMaskKernelSize: 12,
  },

  // OpenAI State defaults (Epic 4)
  openAIState: {
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
  },
}

export const useStore = createWithEqualityFn<AppState & AppAction>()(
  persist(
    immer((set, get) => ({
      ...defaultValues,

      showPrevMask: async () => {
        if (get().settings.showExtender) {
          return
        }
        const { lastLineGroup, curLineGroup, prevExtraMasks, extraMasks } =
          get().editorState
        if (curLineGroup.length !== 0 || extraMasks.length !== 0) {
          return
        }
        const { imageWidth, imageHeight } = get()

        const maskImages = await resolveStoredImages(prevExtraMasks)
        const maskCanvas = generateMask(
          imageWidth,
          imageHeight,
          [lastLineGroup],
          maskImages,
          BRUSH_COLOR
        )
        try {
          const maskImage = await canvasToImage(maskCanvas)
          const storedMask = createStoredImage(maskImage)
          set((state) => {
            state.editorState.temporaryMasks.push(storedMask)
          })
        } catch (e) {
          console.error(e)
          return
        }
      },
      hidePrevMask: () => {
        set((state) => {
          state.editorState.temporaryMasks = []
        })
      },

      getCurrentTargetFile: async (): Promise<File> => {
        const file = get().file! // 一定是在 file 加载了以后才可能调用这个函数
        const renders = get().editorState.renders

        let targetFile = file
        if (renders.length > 0) {
          const lastRender = renders[renders.length - 1]
          targetFile = await srcToFile(
            lastRender.src,
            file.name,
            file.type
          )
        }
        return targetFile
      },

      runInpainting: async () => {
        const {
          isInpainting,
          file,
          paintByExampleFile,
          imageWidth,
          imageHeight,
          settings,
          cropperState,
          extenderState,
        } = get()
        if (isInpainting || file === null) {
          return
        }
        if (
          get().settings.model.support_outpainting &&
          settings.showExtender &&
          extenderState.x === 0 &&
          extenderState.y === 0 &&
          extenderState.height === imageHeight &&
          extenderState.width === imageWidth
        ) {
          return
        }

        const {
          lastLineGroup,
          curLineGroup,
          lineGroups,
          renders,
          prevExtraMasks,
          extraMasks,
        } = get().editorState

        const rendersCountBeforeInpaint = renders.length

        const useLastLineGroup =
          curLineGroup.length === 0 &&
          extraMasks.length === 0 &&
          !settings.showExtender

        // useLastLineGroup 的影响
        // 1. 使用上一次的 mask
        // 2. 结果替换当前 render
        let maskImages: StoredImage[] = []
        let maskLineGroup: LineGroup = []
        if (useLastLineGroup === true) {
          maskLineGroup = lastLineGroup
          maskImages = prevExtraMasks
        } else {
          maskLineGroup = curLineGroup
          maskImages = extraMasks
        }

        if (
          maskLineGroup.length === 0 &&
          maskImages === null &&
          !settings.showExtender
        ) {
          toast({
            variant: "destructive",
            description: "Please draw mask on picture",
          })
          return
        }

        const newLineGroups = [...lineGroups, maskLineGroup]

        set((state) => {
          state.isInpainting = true
        })

        let targetFile = file
        if (useLastLineGroup === true) {
          // renders.length == 1 还是用原来的
          if (renders.length > 1) {
            const lastRender = renders[renders.length - 2]
            targetFile = await srcToFile(
              lastRender.src,
              file.name,
              file.type
            )
          }
        } else if (renders.length > 0) {
          const lastRender = renders[renders.length - 1]
          targetFile = await srcToFile(
            lastRender.src,
            file.name,
            file.type
          )
        }

        const resolvedMaskImages = await resolveStoredImages(maskImages)
        const maskCanvas = generateMask(
          imageWidth,
          imageHeight,
          [maskLineGroup],
          resolvedMaskImages,
          BRUSH_COLOR
        )
        if (useLastLineGroup) {
          const temporaryMask = await canvasToImage(maskCanvas)
          const storedMask = createStoredImage(temporaryMask)
          set((state) => {
            state.editorState.temporaryMasks = [storedMask]
          })
        }

        set((state) => {
          state.rendersCountBeforeInpaint = rendersCountBeforeInpaint
        })

        try {
          const res = await inpaint(
            targetFile,
            settings,
            cropperState,
            extenderState,
            dataURItoBlob(maskCanvas.toDataURL()),
            paintByExampleFile,
            inpaintController.signal
          )

          const { blob, seed } = res
          if (seed) {
            get().setSeed(parseInt(seed, 10))
          }
          const newRender = new Image()
          await loadImage(newRender, blob)
          const storedRender = createStoredImage(newRender)
          const newRenders = [...renders, storedRender]
          get().setImageSize(newRender.width, newRender.height)
          get().updateEditorState({
            renders: newRenders,
            lineGroups: newLineGroups,
            lastLineGroup: maskLineGroup,
            curLineGroup: [],
            extraMasks: [],
            prevExtraMasks: maskImages,
          })
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: e.message ? e.message : e.toString(),
          })
        }

        get().resetRedoState()
        set((state) => {
          state.isInpainting = false
          state.editorState.temporaryMasks = []
        })
      },

      cancelInpainting: () => {
        inpaintController.abort()
        set((state) => {
          state.isInpainting = false
        })
      },

      runRenderablePlugin: async (
        genMask: boolean,
        pluginName: string,
        params: PluginParams = { upscale: 1 }
      ) => {
        const { renders, lineGroups } = get().editorState
        const toolMode = get().settings.openAIToolMode
        const baseUrl = resolveOpenAIBaseUrl(get().settings.openAIProvider)
        const isSpecialTool =
          pluginName === "RemoveBG" || pluginName === "RealESRGAN"
        set((state) => {
          state.isPluginRunning = true
        })

        try {
          const start = new Date()
          const targetFile = await get().getCurrentTargetFile()
          let blobUrl: string | null = null

          if (isSpecialTool && toolMode !== "local") {
            if (genMask) {
              throw new Error("Mask generation is only supported in local mode.")
            }
            if (toolMode === "service") {
              throw new Error("Specialized services are not configured yet.")
            }

            if (pluginName === "RealESRGAN") {
              const blob = await upscaleImage(
                targetFile,
                {
                  scale: params.upscale,
                  mode: "prompt",
                },
                baseUrl
              )
              blobUrl = URL.createObjectURL(blob)
            } else if (pluginName === "RemoveBG") {
              const blob = await removeBackground(
                targetFile,
                {
                  mode: "prompt",
                },
                baseUrl
              )
              blobUrl = URL.createObjectURL(blob)
            }
          } else {
            const res = await runPlugin(
              genMask,
              pluginName,
              targetFile,
              params.upscale
            )
            blobUrl = res.blob
          }
          if (!blobUrl) {
            throw new Error("Tool did not return an image.")
          }

          if (!genMask) {
            const newRender = new Image()
            await loadImage(newRender, blobUrl)
            const storedRender = createStoredImage(newRender)
            get().setImageSize(newRender.width, newRender.height)
            const newRenders = [...renders, storedRender]
            const newLineGroups = [...lineGroups, []]
            get().updateEditorState({
              renders: newRenders,
              lineGroups: newLineGroups,
            })
          } else {
            const newMask = new Image()
            await loadImage(newMask, blobUrl)
            const storedMask = createStoredImage(newMask)
            set((state) => {
              state.editorState.extraMasks.push(storedMask)
            })
          }
          const end = new Date()
          const time = end.getTime() - start.getTime()
          toast({
            description: `Run ${pluginName} successfully in ${time / 1000}s`,
          })
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: e.message ? e.message : e.toString(),
          })
        }
        set((state) => {
          state.isPluginRunning = false
        })
      },

      // Edirot State //
      updateEditorState: (newState: Partial<EditorState>) => {
        set((state) => {
          state.editorState = castDraft({ ...state.editorState, ...newState })
        })
      },

      cleanCurLineGroup: () => {
        get().updateEditorState({ curLineGroup: [] })
      },

      handleCanvasMouseDown: (point: Point) => {
        let lineGroup: LineGroup = []
        const state = get()
        if (state.runMannually()) {
          lineGroup = [...state.editorState.curLineGroup]
        }
        lineGroup.push({ size: state.getBrushSize(), pts: [point] })
        set((state) => {
          state.editorState.curLineGroup = lineGroup
        })
      },

      handleCanvasMouseMove: (point: Point) => {
        set((state) => {
          const curLineGroup = state.editorState.curLineGroup
          if (curLineGroup.length) {
            curLineGroup[curLineGroup.length - 1].pts.push(point)
          }
        })
      },

      runMannually: (): boolean => {
        const state = get()
        return (
          state.settings.enableManualInpainting ||
          state.settings.model.model_type !== MODEL_TYPE_INPAINT
        )
      },

      getIsProcessing: (): boolean => {
        return (
          get().isInpainting || get().isPluginRunning || get().isAdjustingMask
        )
      },

      isSD: (): boolean => {
        return get().settings.model.model_type !== MODEL_TYPE_INPAINT
      },

      // undo/redo

      undoDisabled: (): boolean => {
        const editorState = get().editorState
        if (editorState.renders.length > 0) {
          return false
        }
        if (get().runMannually()) {
          if (editorState.curLineGroup.length === 0) {
            return true
          }
        } else if (editorState.renders.length === 0) {
          return true
        }
        return false
      },

      undo: () => {
        if (
          get().runMannually() &&
          get().editorState.curLineGroup.length !== 0
        ) {
          // undoStroke
          set((state) => {
            const editorState = state.editorState
            if (editorState.curLineGroup.length === 0) {
              return
            }
            editorState.lastLineGroup = []
            const lastLine = editorState.curLineGroup.pop()!
            editorState.redoCurLines.push(lastLine)
          })
        } else {
          set((state) => {
            const editorState = state.editorState
            if (
              editorState.renders.length === 0 ||
              editorState.lineGroups.length === 0
            ) {
              return
            }
            const lastLineGroup = editorState.lineGroups.pop()!
            editorState.redoLineGroups.push(lastLineGroup)
            editorState.redoCurLines = []
            editorState.curLineGroup = []

            const lastRender = editorState.renders.pop()!
            editorState.redoRenders.push(lastRender)
          })
        }
      },

      redoDisabled: (): boolean => {
        const editorState = get().editorState
        if (editorState.redoRenders.length > 0) {
          return false
        }
        if (get().runMannually()) {
          if (editorState.redoCurLines.length === 0) {
            return true
          }
        } else if (editorState.redoRenders.length === 0) {
          return true
        }
        return false
      },

      redo: () => {
        if (
          get().runMannually() &&
          get().editorState.redoCurLines.length !== 0
        ) {
          set((state) => {
            const editorState = state.editorState
            if (editorState.redoCurLines.length === 0) {
              return
            }
            const line = editorState.redoCurLines.pop()!
            editorState.curLineGroup.push(line)
          })
        } else {
          set((state) => {
            const editorState = state.editorState
            if (
              editorState.redoRenders.length === 0 ||
              editorState.redoLineGroups.length === 0
            ) {
              return
            }
            const lastLineGroup = editorState.redoLineGroups.pop()!
            editorState.lineGroups.push(lastLineGroup)
            editorState.curLineGroup = []

            const lastRender = editorState.redoRenders.pop()!
            editorState.renders.push(lastRender)
          })
        }
      },

      resetRedoState: () => {
        set((state) => {
          state.editorState.redoCurLines = []
          state.editorState.redoLineGroups = []
          state.editorState.redoRenders = []
        })
      },

      //****//

      updateAppState: (newState: Partial<AppState>) => {
        set(() => newState)
      },

      getBrushSize: (): number => {
        return (
          get().editorState.baseBrushSize * get().editorState.brushSizeScale
        )
      },

      showPromptInput: (): boolean => {
        const model = get().settings.model
        return (
          model.model_type !== MODEL_TYPE_INPAINT &&
          model.name !== PAINT_BY_EXAMPLE
        )
      },

      setServerConfig: (newValue: ServerConfig) => {
        set((state) => {
          state.serverConfig = newValue
          state.settings.enableControlnet = newValue.enableControlnet
          state.settings.controlnetMethod = newValue.controlnetMethod
        })
      },

      updateSettings: (newSettings: Partial<Settings>) => {
        set((state) => {
          state.settings = {
            ...state.settings,
            ...newSettings,
          }
        })
      },

      updateEnablePowerPaintV2: (newValue: boolean) => {
        get().updateSettings({ enablePowerPaintV2: newValue })
        if (newValue) {
          get().updateSettings({
            enableBrushNet: false,
            enableControlnet: false,
            enableLCMLora: false,
          })
        }
      },

      updateEnableBrushNet: (newValue: boolean) => {
        get().updateSettings({ enableBrushNet: newValue })
        if (newValue) {
          get().updateSettings({
            enablePowerPaintV2: false,
            enableControlnet: false,
            enableLCMLora: false,
          })
        }
      },

      updateEnableControlnet(newValue) {
        get().updateSettings({ enableControlnet: newValue })
        if (newValue) {
          get().updateSettings({
            enablePowerPaintV2: false,
            enableBrushNet: false,
          })
        }
      },

      updateLCMLora(newValue) {
        get().updateSettings({ enableLCMLora: newValue })
        if (newValue) {
          get().updateSettings({
            enablePowerPaintV2: false,
            enableBrushNet: false,
          })
        }
      },

      setModel: (newModel: ModelInfo) => {
        set((state) => {
          state.settings.model = newModel

          if (
            newModel.support_controlnet &&
            !newModel.controlnets.includes(state.settings.controlnetMethod)
          ) {
            state.settings.controlnetMethod = newModel.controlnets[0]
          }
        })
      },

      updateFileManagerState: (newState: Partial<FileManagerState>) => {
        set((state) => {
          state.fileManagerState = {
            ...state.fileManagerState,
            ...newState,
          }
        })
      },

      updateInteractiveSegState: (newState: Partial<InteractiveSegState>) => {
        set((state) => {
          return {
            ...state,
            interactiveSegState: {
              ...state.interactiveSegState,
              ...newState,
            },
          }
        })
      },

      resetInteractiveSegState: () => {
        get().updateInteractiveSegState(defaultValues.interactiveSegState)
      },

      handleInteractiveSegAccept: () => {
        set((state) => {
          if (state.interactiveSegState.tmpInteractiveSegMask) {
            state.editorState.extraMasks.push(
              state.interactiveSegState.tmpInteractiveSegMask
            )
          }
          state.interactiveSegState = castDraft({
            ...defaultValues.interactiveSegState,
          })
        })
      },

      handleFileManagerMaskSelect: async (blob: Blob) => {
        const newMask = new Image()

        await loadImage(newMask, URL.createObjectURL(blob))
        const storedMask = createStoredImage(newMask)
        set((state) => {
          state.editorState.extraMasks.push(storedMask)
        })
        get().runInpainting()
      },

      setIsInpainting: (newValue: boolean) =>
        set((state) => {
          state.isInpainting = newValue
        }),

      setFile: async (file: File) => {
        if (get().settings.enableAutoExtractPrompt) {
          try {
            const res = await getGenInfo(file)
            if (res.prompt) {
              set((state) => {
                state.settings.prompt = res.prompt
              })
            }
            if (res.negative_prompt) {
              set((state) => {
                state.settings.negativePrompt = res.negative_prompt
              })
            }
        } catch (e: any) {
          if (e.name === "AbortError") {
            const { rendersCountBeforeInpaint } = get()
            const { renders } = get().editorState
            if (renders.length > rendersCountBeforeInpaint) {
              get().undo()
            }
            toast({
              description: "Generation cancelled",
            })
          } else {
            toast({
              variant: "destructive",
              description: e.message ? e.message : e.toString(),
            })
          }
        }
        }
        set((state) => {
          state.file = file
          state.interactiveSegState = castDraft(
            defaultValues.interactiveSegState
          )
          state.editorState = castDraft(defaultValues.editorState)
          state.cropperState = defaultValues.cropperState
        })
      },

      setCustomFile: (file: File) =>
        set((state) => {
          state.customMask = file
        }),

      setBaseBrushSize: (newValue: number) =>
        set((state) => {
          state.editorState.baseBrushSize = newValue
        }),

      decreaseBaseBrushSize: () => {
        const baseBrushSize = get().editorState.baseBrushSize
        let newBrushSize = baseBrushSize
        if (baseBrushSize > 10) {
          newBrushSize = baseBrushSize - 10
        }
        if (baseBrushSize <= 10 && baseBrushSize > 0) {
          newBrushSize = baseBrushSize - 3
        }
        get().setBaseBrushSize(newBrushSize)
      },

      increaseBaseBrushSize: () => {
        const baseBrushSize = get().editorState.baseBrushSize
        const newBrushSize = Math.min(baseBrushSize + 10, MAX_BRUSH_SIZE)
        get().setBaseBrushSize(newBrushSize)
      },

      setImageSize: (width: number, height: number) => {
        // 根据图片尺寸调整 brushSize 的 scale
        set((state) => {
          state.imageWidth = width
          state.imageHeight = height
          state.editorState.brushSizeScale =
            Math.max(Math.min(width, height), 512) / 512
        })
        get().resetExtender(width, height)
      },

      setCropperX: (newValue: number) =>
        set((state) => {
          state.cropperState.x = newValue
        }),

      setCropperY: (newValue: number) =>
        set((state) => {
          state.cropperState.y = newValue
        }),

      setCropperWidth: (newValue: number) =>
        set((state) => {
          state.cropperState.width = newValue
        }),

      setCropperHeight: (newValue: number) =>
        set((state) => {
          state.cropperState.height = newValue
        }),

      setExtenderX: (newValue: number) =>
        set((state) => {
          state.extenderState.x = newValue
        }),

      setExtenderY: (newValue: number) =>
        set((state) => {
          state.extenderState.y = newValue
        }),

      setExtenderWidth: (newValue: number) =>
        set((state) => {
          state.extenderState.width = newValue
        }),

      setExtenderHeight: (newValue: number) =>
        set((state) => {
          state.extenderState.height = newValue
        }),

      setIsCropperExtenderResizing: (newValue: boolean) =>
        set((state) => {
          state.isCropperExtenderResizing = newValue
        }),

      updateExtenderDirection: (newValue: ExtenderDirection) => {
        console.log(
          `updateExtenderDirection: ${JSON.stringify(get().extenderState)}`
        )
        set((state) => {
          state.settings.extenderDirection = newValue
          state.extenderState.x = 0
          state.extenderState.y = 0
          state.extenderState.width = state.imageWidth
          state.extenderState.height = state.imageHeight
        })
        get().updateExtenderByBuiltIn(newValue, 1.5)
      },

      updateExtenderByBuiltIn: (
        direction: ExtenderDirection,
        scale: number
      ) => {
        const newExtenderState = { ...defaultValues.extenderState }
        let { x, y, width, height } = newExtenderState
        const { imageWidth, imageHeight } = get()
        width = imageWidth
        height = imageHeight

        switch (direction) {
          case ExtenderDirection.x:
            x = -Math.ceil((imageWidth * (scale - 1)) / 2)
            width = Math.ceil(imageWidth * scale)
            break
          case ExtenderDirection.y:
            y = -Math.ceil((imageHeight * (scale - 1)) / 2)
            height = Math.ceil(imageHeight * scale)
            break
          case ExtenderDirection.xy:
            x = -Math.ceil((imageWidth * (scale - 1)) / 2)
            y = -Math.ceil((imageHeight * (scale - 1)) / 2)
            width = Math.ceil(imageWidth * scale)
            height = Math.ceil(imageHeight * scale)
            break
          default:
            break
        }

        set((state) => {
          state.extenderState.x = x
          state.extenderState.y = y
          state.extenderState.width = width
          state.extenderState.height = height
        })
      },

      resetExtender: (width: number, height: number) => {
        set((state) => {
          state.extenderState.x = 0
          state.extenderState.y = 0
          state.extenderState.width = width
          state.extenderState.height = height
        })
      },

      setSeed: (newValue: number) =>
        set((state) => {
          state.settings.seed = newValue
        }),

      adjustMask: async (operate: AdjustMaskOperate) => {
        const { imageWidth, imageHeight } = get()
        const { curLineGroup, extraMasks } = get().editorState
        const { adjustMaskKernelSize } = get().settings
        if (curLineGroup.length === 0 && extraMasks.length === 0) {
          return
        }

        set((state) => {
          state.isAdjustingMask = true
        })

        const resolvedExtraMasks = await resolveStoredImages(extraMasks)
        const maskCanvas = generateMask(
          imageWidth,
          imageHeight,
          [curLineGroup],
          resolvedExtraMasks,
          BRUSH_COLOR
        )
        const maskBlob = dataURItoBlob(maskCanvas.toDataURL())
        const newMaskBlob = await postAdjustMask(
          maskBlob,
          operate,
          adjustMaskKernelSize
        )
        const newMask = await blobToImage(newMaskBlob)
        const storedMask = createStoredImage(newMask)

        // TODO: currently ignore stroke undo/redo
        set((state) => {
          state.editorState.extraMasks = [storedMask]
          state.editorState.curLineGroup = []
        })

        set((state) => {
          state.isAdjustingMask = false
        })
      },
      clearMask: () => {
        set((state) => {
          state.editorState.extraMasks = []
          state.editorState.curLineGroup = []
        })
      },

      // ========================================================================
      // OpenAI Actions (Epic 4)
      // ========================================================================

      // Mode toggle
      setOpenAIMode: (enabled: boolean) => {
        set((state) => {
          state.openAIState.isOpenAIMode = enabled
        })
        // Fetch capabilities when entering OpenAI mode
        if (enabled) {
          get().fetchOpenAICapabilities()
          get().refreshBudgetStatus()
          get().refreshBudgetLimits()
        }
      },

      setOpenAIEditMode: (enabled: boolean) => {
        set((state) => {
          state.openAIState.isOpenAIEditMode = enabled
        })
      },

      // Capabilities management
      fetchOpenAICapabilities: async () => {
        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const capabilities = await fetchOpenAICapabilities(baseUrl)

          set((state) => {
            const generateCaps = getModeCapabilities(
              capabilities,
              "images_generate"
            )
            const editCaps = getModeCapabilities(capabilities, "images_edit")
            const selectedGenerateModel = resolveModelSelection(
              generateCaps,
              state.openAIState.selectedGenerateModel
            )
            const selectedEditModel = resolveModelSelection(
              editCaps,
              state.openAIState.selectedEditModel
            )
            const generateModelCaps = getModelCapabilities(
              capabilities,
              "images_generate",
              selectedGenerateModel
            )
            state.openAIState.capabilities = capabilities
            state.openAIState.selectedGenerateModel = selectedGenerateModel
            state.openAIState.selectedEditModel = selectedEditModel
            state.openAIState.customPresetConfig = resolvePresetConfig(
              GenerationPreset.CUSTOM,
              state.openAIState.customPresetConfig,
              generateModelCaps
            )
          })
          get().updateCostEstimate()
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: `Failed to fetch OpenAI capabilities: ${e.message}`,
          })
        }
      },

      setSelectedGenerateModel: (modelId: string) => {
        set((state) => {
          const modelCaps = getModelCapabilities(
            state.openAIState.capabilities,
            "images_generate",
            modelId
          )
          state.openAIState.selectedGenerateModel = modelId
          state.openAIState.customPresetConfig = resolvePresetConfig(
            GenerationPreset.CUSTOM,
            state.openAIState.customPresetConfig,
            modelCaps
          )
        })
        get().updateCostEstimate()
      },

      setSelectedEditModel: (modelId: string) => {
        set((state) => {
          state.openAIState.selectedEditModel = modelId
        })
      },

      // Intent/Prompt flow
      setOpenAIIntent: (intent: string) => {
        set((state) => {
          state.openAIState.openAIIntent = intent
        })
      },

      refinePrompt: async () => {
        const { openAIIntent } = get().openAIState
        if (!openAIIntent.trim()) {
          toast({
            variant: "destructive",
            description: "Please enter an intent first",
          })
          return
        }

        set((state) => {
          state.openAIState.isRefiningPrompt = true
        })

        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const result = await apiRefinePrompt(
            { prompt: openAIIntent },
            baseUrl
          )
          set((state) => {
            state.openAIState.openAIRefinedPrompt = result.refinedPrompt
          })
          // Update cost estimate after refining
          get().updateCostEstimate()
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: `Failed to refine prompt: ${e.message}`,
          })
        } finally {
          set((state) => {
            state.openAIState.isRefiningPrompt = false
          })
        }
      },

      setOpenAIRefinedPrompt: (prompt: string) => {
        set((state) => {
          state.openAIState.openAIRefinedPrompt = prompt
        })
      },

      setOpenAINegativePrompt: (prompt: string) => {
        set((state) => {
          state.openAIState.openAINegativePrompt = prompt
        })
      },

      // Presets
      setSelectedPreset: (preset: GenerationPreset) => {
        set((state) => {
          state.openAIState.selectedPreset = preset
        })
        // Update cost estimate when preset changes
        get().updateCostEstimate()
      },

      updateCustomPresetConfig: (config: Partial<PresetConfig>) => {
        set((state) => {
          state.openAIState.customPresetConfig = {
            ...state.openAIState.customPresetConfig,
            ...config,
          }
        })
        // Update cost estimate when config changes
        if (get().openAIState.selectedPreset === GenerationPreset.CUSTOM) {
          get().updateCostEstimate()
        }
      },

      getActivePresetConfig: (
        mode: OpenAIImageMode = "images_generate"
      ): PresetConfig => {
        const {
          selectedPreset,
          customPresetConfig,
          capabilities,
          selectedGenerateModel,
          selectedEditModel,
        } = get().openAIState
        const modelId =
          mode === "images_edit" ? selectedEditModel : selectedGenerateModel
        const modelCaps = getModelCapabilities(capabilities, mode, modelId)
        return resolvePresetConfig(selectedPreset, customPresetConfig, modelCaps)
      },

      // Cost estimation
      updateCostEstimate: async () => {
        const { selectedGenerateModel, openAIRefinedPrompt } =
          get().openAIState
        if (!openAIRefinedPrompt || !selectedGenerateModel) return

        const config = get().getActivePresetConfig("images_generate")

        try {
          const estimate = await estimateGenerationCost({
            operation: "generate",
            model: selectedGenerateModel,
            size: config.size,
            quality: config.quality,
            n: config.n,
          })
          set((state) => {
            state.openAIState.currentCostEstimate = estimate
          })
        } catch (e: any) {
          console.error("Failed to estimate cost:", e)
        }
      },

      // Generation execution
      requestOpenAIGeneration: async () => {
        const {
          openAIRefinedPrompt,
          openAINegativePrompt,
          selectedGenerateModel,
          currentCostEstimate,
        } = get().openAIState

        if (!selectedGenerateModel) {
          toast({
            variant: "destructive",
            description: "No OpenAI generate model available",
          })
          return
        }

        if (!openAIRefinedPrompt.trim()) {
          toast({
            variant: "destructive",
            description: "Please enter or refine a prompt first",
          })
          return
        }

        const config = get().getActivePresetConfig("images_generate")

        const request: GenerateImageRequest = {
          prompt: openAIRefinedPrompt,
          negativePrompt: openAINegativePrompt,
          model: selectedGenerateModel,
          size: config.size,
          quality: config.quality,
          n: config.n,
        }

        // If high cost tier, show warning modal
        if (currentCostEstimate?.tier === "high") {
          set((state) => {
            state.openAIState.pendingGenerationRequest = request
            state.openAIState.showCostWarningModal = true
          })
          return
        }

        // Otherwise proceed directly
        set((state) => {
          state.openAIState.pendingGenerationRequest = request
        })
        await get().confirmOpenAIGeneration()
      },

      confirmOpenAIGeneration: async () => {
        const {
          pendingGenerationRequest,
          openAIIntent,
          selectedPreset,
          currentCostEstimate,
          selectedGenerateModel,
          openAIRefinedPrompt,
          openAINegativePrompt,
        } = get().openAIState

        if (!pendingGenerationRequest) return

        set((state) => {
          state.openAIState.showCostWarningModal = false
          state.openAIState.isOpenAIGenerating = true
        })

        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const response = await submitOpenAIJob(
            {
              tool: "generate",
              prompt: pendingGenerationRequest.prompt,
              n: pendingGenerationRequest.n,
              size: pendingGenerationRequest.size,
              quality: pendingGenerationRequest.quality,
              model: selectedGenerateModel,
              intent: openAIIntent,
              refined_prompt: openAIRefinedPrompt,
              negative_prompt: openAINegativePrompt,
              preset: selectedPreset,
            },
            baseUrl
          )

          const job = mapBackendJobToFrontend(response)
          job.estimatedCostUsd = currentCostEstimate?.estimatedCostUsd
          get().addToHistory(job, { skipBackend: true })

          set((state) => {
            state.openAIState.currentJobId = job.id
            state.openAIState.pendingGenerationRequest = null
          })

          startOpenAIJobPolling(job.id, baseUrl, get, set, {
            loadIntoEditor: true,
          })
        } catch (e: any) {
          set((state) => {
            state.openAIState.isOpenAIGenerating = false
            state.openAIState.pendingGenerationRequest = null
            state.openAIState.currentJobId = null
          })
          toast({
            variant: "destructive",
            description: `Generation failed: ${e.message}`,
          })
        }
      },

      cancelPendingGeneration: () => {
        set((state) => {
          state.openAIState.showCostWarningModal = false
          state.openAIState.pendingGenerationRequest = null
        })
      },

      // History management (with backend sync - Epic 3)
      addToHistory: (job: GenerationJob, options?: { skipBackend?: boolean }) => {
        // Add to local state immediately
        set((state) => {
          state.openAIState.generationHistory.unshift(castDraft(job))
          // Keep only last 100 jobs
          if (state.openAIState.generationHistory.length > 100) {
            state.openAIState.generationHistory.pop()
          }
        })

        if (options?.skipBackend) {
          return
        }

        // Sync to backend (fire-and-forget, errors logged but not blocking)
        apiCreateHistoryJob({
          operation: job.isEdit ? "edit" : "generate",
          model: job.model,
          intent: job.intent,
          refined_prompt: job.refinedPrompt,
          negative_prompt: job.negativePrompt,
          preset: job.preset,
          params: job.params as unknown as Record<string, unknown>,
          fingerprint: job.fingerprint,
          estimated_cost_usd: job.estimatedCostUsd,
          is_edit: job.isEdit,
        }).catch((e) => console.error("Failed to sync job to backend:", e))
      },

      updateHistoryJob: (
        id: string,
        updates: Partial<GenerationJob>,
        options?: { skipBackend?: boolean }
      ) => {
        set((state) => {
          const index = state.openAIState.generationHistory.findIndex(
            (j) => j.id === id
          )
          if (index !== -1) {
            state.openAIState.generationHistory[index] = {
              ...state.openAIState.generationHistory[index],
              ...updates,
            }
          }
        })

        if (options?.skipBackend) {
          return
        }

        // Sync status updates to backend
        const backendUpdates: Record<string, unknown> = {}
        if (updates.status) backendUpdates.status = updates.status
        if (updates.actualCostUsd !== undefined)
          backendUpdates.actual_cost_usd = updates.actualCostUsd
        if (updates.errorMessage !== undefined)
          backendUpdates.error_message = updates.errorMessage

        if (Object.keys(backendUpdates).length > 0) {
          apiUpdateHistoryJob(id, backendUpdates as any).catch((e) =>
            console.error("Failed to sync job update to backend:", e)
          )
        }
      },

      removeFromHistory: (id: string) => {
        set((state) => {
          state.openAIState.generationHistory =
            state.openAIState.generationHistory.filter((j) => j.id !== id)
        })

        // Sync deletion to backend
        apiDeleteHistoryJob(id).catch((e) =>
          console.error("Failed to delete job from backend:", e)
        )
      },

      clearHistory: () => {
        set((state) => {
          state.openAIState.generationHistory = []
        })

        // Sync clear to backend
        apiClearHistory().catch((e) =>
          console.error("Failed to clear history on backend:", e)
        )
      },

      // Fetch history from backend and merge with local
      syncHistoryWithBackend: async () => {
        try {
          const response = await apiFetchHistory({ limit: 100 })

          // Convert backend format to frontend GenerationJob format
          const backendJobs: GenerationJob[] = response.jobs.map(mapBackendJobToFrontend)

          set((state) => {
            // Merge: prefer backend data, add any local-only jobs
            const localJobs = state.openAIState.generationHistory
            const backendIds = new Set(backendJobs.map((j) => j.id))
            const localOnlyJobs = localJobs.filter((j) => !backendIds.has(j.id))

            // Combine and sort by createdAt descending
            const merged = [...backendJobs, ...localOnlyJobs]
              .sort((a, b) => b.createdAt - a.createdAt)
              .slice(0, 100)

            state.openAIState.generationHistory = castDraft(merged)
          })
        } catch (e) {
          console.error("Failed to sync history from backend:", e)
        }
      },

      syncHistorySnapshots: async () => {
        try {
          const response = await apiFetchHistorySnapshots({ limit: 50 })
          const snapshots: HistorySnapshot[] = response.snapshots.map(
            (snapshot: BackendHistorySnapshot) => ({
              id: snapshot.id,
              sessionId: snapshot.session_id,
              payload: snapshot.payload || {},
              createdAt: new Date(snapshot.created_at).getTime(),
            })
          )
          set((state) => {
            state.openAIState.historySnapshots = castDraft(snapshots)
          })
        } catch (e) {
          console.error("Failed to sync history snapshots from backend:", e)
        }
      },

      saveHistorySnapshot: async (payload?: Record<string, unknown>) => {
        const { generationHistory, historyFilter } = get().openAIState
        const snapshotPayload =
          payload ??
          ({
            history: generationHistory.map((job) => ({
              id: job.id,
              createdAt: job.createdAt,
              status: job.status,
              intent: job.intent,
              refinedPrompt: job.refinedPrompt,
              negativePrompt: job.negativePrompt,
              preset: job.preset,
              params: job.params,
              model: job.model,
              estimatedCostUsd: job.estimatedCostUsd,
              actualCostUsd: job.actualCostUsd,
              errorMessage: job.errorMessage,
              fingerprint: job.fingerprint,
              isEdit: job.isEdit,
            })),
            filter: historyFilter,
          } satisfies Record<string, unknown>)

        try {
          const snapshot = await apiCreateHistorySnapshot(snapshotPayload)
          const mapped: HistorySnapshot = {
            id: snapshot.id,
            sessionId: snapshot.session_id,
            payload: snapshot.payload || {},
            createdAt: new Date(snapshot.created_at).getTime(),
          }
          set((state) => {
            state.openAIState.historySnapshots.unshift(castDraft(mapped))
          })
        } catch (e) {
          console.error("Failed to save history snapshot:", e)
        }
      },

      deleteHistorySnapshot: async (snapshotId: string) => {
        try {
          await apiDeleteHistorySnapshot(snapshotId)
          set((state) => {
            state.openAIState.historySnapshots =
              state.openAIState.historySnapshots.filter((s) => s.id !== snapshotId)
          })
        } catch (e) {
          console.error("Failed to delete history snapshot:", e)
        }
      },

      clearHistorySnapshots: async () => {
        try {
          await apiClearHistorySnapshots()
          set((state) => {
            state.openAIState.historySnapshots = []
          })
        } catch (e) {
          console.error("Failed to clear history snapshots:", e)
        }
      },

      rerunJob: async (jobId: string) => {
        const job = get().openAIState.generationHistory.find((j) => j.id === jobId)
        if (!job) return

        // Set up state for rerun
        set((state) => {
          state.openAIState.openAIIntent = job.intent
          state.openAIState.openAIRefinedPrompt = job.refinedPrompt
          state.openAIState.openAINegativePrompt = job.negativePrompt
          state.openAIState.selectedPreset = job.preset
          if (job.preset === GenerationPreset.CUSTOM) {
            state.openAIState.customPresetConfig = { ...job.params }
          }
        })

        // Trigger generation
        await get().requestOpenAIGeneration()
      },

      restoreFromJob: (jobId: string) => {
        const job = get().openAIState.generationHistory.find((j) => j.id === jobId)
        if (!job) return

        set((state) => {
          state.openAIState.openAIIntent = job.intent
          state.openAIState.openAIRefinedPrompt = job.refinedPrompt
          state.openAIState.openAINegativePrompt = job.negativePrompt
          state.openAIState.selectedPreset = job.preset
          if (job.preset === GenerationPreset.CUSTOM) {
            state.openAIState.customPresetConfig = { ...job.params }
          }
        })

        get().updateCostEstimate()
      },

      copyJobPrompt: (jobId: string) => {
        const job = get().openAIState.generationHistory.find((j) => j.id === jobId)
        if (job) {
          navigator.clipboard.writeText(job.refinedPrompt)
          toast({
            description: "Prompt copied to clipboard",
          })
        }
      },

      setHistoryFilter: (filter: "all" | "succeeded" | "failed") => {
        set((state) => {
          state.openAIState.historyFilter = filter
        })
      },

      // Budget
      refreshBudgetStatus: async () => {
        try {
          const status = await getOpenAIBudgetStatus()
          set((state) => {
            state.openAIState.budgetStatus = status
          })
        } catch (e: any) {
          console.error("Failed to fetch budget status:", e)
        }
      },

      refreshBudgetLimits: async () => {
        try {
          const limits = await getOpenAIBudgetLimits()
          set((state) => {
            state.openAIState.budgetLimits = limits
          })
        } catch (e: any) {
          console.error("Failed to fetch budget limits:", e)
        }
      },

      updateBudgetLimits: async (limits: BudgetLimits) => {
        try {
          const updated = await updateOpenAIBudgetLimits(limits)
          set((state) => {
            state.openAIState.budgetLimits = updated
          })
          get().refreshBudgetStatus()
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: `Failed to update budget limits: ${e.message}`,
          })
        }
      },

      // Edit flow
      setEditSourceImage: (dataUrl: string | null) => {
        set((state) => {
          state.openAIState.editSourceImageDataUrl = dataUrl
        })
      },

      runOpenAIEdit: async () => {
        const {
          openAIRefinedPrompt,
          openAINegativePrompt,
          selectedEditModel,
          editSourceImageDataUrl,
          openAIIntent,
          selectedPreset,
        } = get().openAIState
        const { imageWidth, imageHeight } = get()
        const { curLineGroup, extraMasks } = get().editorState

        if (!editSourceImageDataUrl) {
          toast({
            variant: "destructive",
            description: "Please select a source image first",
          })
          return
        }

        if (curLineGroup.length === 0 && extraMasks.length === 0) {
          toast({
            variant: "destructive",
            description: "Please draw a mask first",
          })
          return
        }

        if (!openAIRefinedPrompt.trim()) {
          toast({
            variant: "destructive",
            description: "Please enter an edit prompt",
          })
          return
        }

        if (!selectedEditModel) {
          toast({
            variant: "destructive",
            description: "No OpenAI edit model available",
          })
          return
        }

        set((state) => {
          state.openAIState.isOpenAIGenerating = true
        })

        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const resolvedExtraMasks = await resolveStoredImages(extraMasks)
          const maskCanvas = generateMask(
            imageWidth,
            imageHeight,
            [curLineGroup],
            resolvedExtraMasks,
            BRUSH_COLOR
          )
          const maskBlob = dataURItoBlob(maskCanvas.toDataURL())

          const sourceResponse = await fetch(editSourceImageDataUrl)
          const sourceBlob = await sourceResponse.blob()

          const config = get().getActivePresetConfig("images_edit")
          const image_b64 = await toBase64Payload(sourceBlob)
          const mask_b64 = await toBase64Payload(maskBlob)

          const response = await submitOpenAIJob(
            {
              tool: "edit",
              prompt: openAIRefinedPrompt,
              model: selectedEditModel,
              size: config.size,
              quality: config.quality,
              n: config.n,
              image_b64,
              mask_b64,
              intent: openAIIntent,
              refined_prompt: openAIRefinedPrompt,
              negative_prompt: openAINegativePrompt,
              preset: selectedPreset,
            },
            baseUrl
          )

          const job = mapBackendJobToFrontend(response)
          get().addToHistory(job, { skipBackend: true })

          set((state) => {
            state.openAIState.currentJobId = job.id
          })

          startOpenAIJobPolling(job.id, baseUrl, get, set, {
            loadIntoEditor: true,
          })
        } catch (e: any) {
          set((state) => {
            state.openAIState.isOpenAIGenerating = false
          })
          toast({
            variant: "destructive",
            description: `Edit failed: ${e.message}`,
          })
        }
      },

      runOpenAIOutpaint: async () => {
        const {
          openAIRefinedPrompt,
          openAINegativePrompt,
          selectedEditModel,
          editSourceImageDataUrl,
          openAIIntent,
          selectedPreset,
        } = get().openAIState
        const { imageWidth, imageHeight } = get()
        const { curLineGroup, extraMasks } = get().editorState

        if (!editSourceImageDataUrl) {
          toast({
            variant: "destructive",
            description: "Please select a source image first",
          })
          return
        }

        if (curLineGroup.length === 0 && extraMasks.length === 0) {
          toast({
            variant: "destructive",
            description: "Please draw a mask first",
          })
          return
        }

        if (!openAIRefinedPrompt.trim()) {
          toast({
            variant: "destructive",
            description: "Please enter an outpaint prompt",
          })
          return
        }

        if (!selectedEditModel) {
          toast({
            variant: "destructive",
            description: "No OpenAI edit model available",
          })
          return
        }

        set((state) => {
          state.openAIState.isOpenAIGenerating = true
        })

        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const resolvedExtraMasks = await resolveStoredImages(extraMasks)
          const maskCanvas = generateMask(
            imageWidth,
            imageHeight,
            [curLineGroup],
            resolvedExtraMasks,
            BRUSH_COLOR
          )
          const maskBlob = dataURItoBlob(maskCanvas.toDataURL())

          const sourceResponse = await fetch(editSourceImageDataUrl)
          const sourceBlob = await sourceResponse.blob()

          const config = get().getActivePresetConfig("images_edit")
          const image_b64 = await toBase64Payload(sourceBlob)
          const mask_b64 = await toBase64Payload(maskBlob)

          const response = await submitOpenAIJob(
            {
              tool: "outpaint",
              prompt: openAIRefinedPrompt,
              model: selectedEditModel,
              size: config.size,
              quality: config.quality,
              n: config.n,
              image_b64,
              mask_b64,
              intent: openAIIntent,
              refined_prompt: openAIRefinedPrompt,
              negative_prompt: openAINegativePrompt,
              preset: selectedPreset,
            },
            baseUrl
          )

          const job = mapBackendJobToFrontend(response)
          get().addToHistory(job, { skipBackend: true })

          set((state) => {
            state.openAIState.currentJobId = job.id
          })

          startOpenAIJobPolling(job.id, baseUrl, get, set, {
            loadIntoEditor: true,
          })
        } catch (e: any) {
          set((state) => {
            state.openAIState.isOpenAIGenerating = false
          })
          toast({
            variant: "destructive",
            description: `Outpaint failed: ${e.message}`,
          })
        }
      },

      runOpenAIVariation: async () => {
        const {
          selectedEditModel,
          editSourceImageDataUrl,
          openAIIntent,
          openAIRefinedPrompt,
          selectedPreset,
          openAINegativePrompt,
        } = get().openAIState

        if (!editSourceImageDataUrl) {
          toast({
            variant: "destructive",
            description: "Please select a source image first",
          })
          return
        }

        if (!selectedEditModel) {
          toast({
            variant: "destructive",
            description: "No OpenAI edit model available",
          })
          return
        }

        set((state) => {
          state.openAIState.isOpenAIGenerating = true
        })

        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const sourceResponse = await fetch(editSourceImageDataUrl)
          const sourceBlob = await sourceResponse.blob()
          const config = get().getActivePresetConfig("images_edit")
          const image_b64 = await toBase64Payload(sourceBlob)

          const response = await submitOpenAIJob(
            {
              tool: "variation",
              prompt: openAIRefinedPrompt || "Variation",
              model: selectedEditModel,
              size: config.size,
              n: config.n,
              image_b64,
              intent: openAIIntent || "Variation",
              refined_prompt: openAIRefinedPrompt || "Variation",
              negative_prompt: openAINegativePrompt,
              preset: selectedPreset,
            },
            baseUrl
          )

          const job = mapBackendJobToFrontend(response)
          get().addToHistory(job, { skipBackend: true })

          set((state) => {
            state.openAIState.currentJobId = job.id
          })

          startOpenAIJobPolling(job.id, baseUrl, get, set, {
            loadIntoEditor: true,
          })
        } catch (e: any) {
          set((state) => {
            state.openAIState.isOpenAIGenerating = false
          })
          toast({
            variant: "destructive",
            description: `Variation failed: ${e.message}`,
          })
        }
      },

      cancelOpenAIJob: async (jobId: string) => {
        try {
          const baseUrl = resolveOpenAIBaseUrl(
            get().settings.openAIProvider
          )
          const response = await cancelOpenAIJob(jobId, baseUrl)
          const job = mapBackendJobToFrontend(response)
          get().updateHistoryJob(
            jobId,
            {
              status: job.status,
              errorMessage: job.errorMessage,
            },
            { skipBackend: true }
          )

          const intervalId = openAIJobPollers.get(jobId)
          if (intervalId) {
            window.clearInterval(intervalId)
          }
          openAIJobPollers.delete(jobId)

          if (get().openAIState.currentJobId === jobId) {
            set((state) => {
              state.openAIState.currentJobId = null
              state.openAIState.isOpenAIGenerating = false
            })
          }
        } catch (e: any) {
          toast({
            variant: "destructive",
            description: `Cancel failed: ${e.message}`,
          })
        }
      },
    })),
    {
      name: "ZUSTAND_STATE", // name of the item in the storage (must be unique)
      version: 4, // Bumped for OpenAI capabilities state
      partialize: (state) =>
        Object.fromEntries(
          Object.entries(state).filter(([key]) =>
            ["fileManagerState", "settings", "openAIState"].includes(key)
          )
        ),
    }
  ),
  shallow
)
