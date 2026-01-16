import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useStore } from "../states"
import type { BudgetStatus, GenerateImageRequest } from "../types"
import { GenerationPreset } from "../types"
import {
  cancelOpenAIJob,
  createThumbnail,
  fetchStoredImageAsDataUrl,
  getOpenAIBudgetStatus,
  getOpenAIJob,
  submitOpenAIJob,
  type BackendGenerationJob,
} from "../openai-api"

vi.mock("../openai-api", async () => {
  const actual = await vi.importActual<typeof import("../openai-api")>(
    "../openai-api"
  )
  return {
    ...actual,
    submitOpenAIJob: vi.fn(),
    getOpenAIJob: vi.fn(),
    cancelOpenAIJob: vi.fn(),
    fetchStoredImageAsDataUrl: vi.fn(),
    createThumbnail: vi.fn(),
    getOpenAIBudgetStatus: vi.fn(),
  }
})

const mockBudgetStatus: BudgetStatus = {
  daily: {
    spentUsd: 0,
    remainingUsd: 10,
    capUsd: 10,
    isUnlimited: false,
    percentageUsed: 0,
  },
  monthly: {
    spentUsd: 0,
    remainingUsd: 10,
    capUsd: 10,
    isUnlimited: false,
    percentageUsed: 0,
  },
  session: {
    spentUsd: 0,
    remainingUsd: 10,
    capUsd: 10,
    isUnlimited: false,
    percentageUsed: 0,
  },
  status: "ok",
  message: null,
}

const baseBackendJob = (): BackendGenerationJob => ({
  id: "job-1",
  session_id: "session-1",
  status: "queued",
  operation: "generate",
  model: "gpt-image-1",
  intent: "Draw a cat",
  refined_prompt: "A cat sitting on a chair",
  negative_prompt: "",
  preset: "draft",
  params: {
    size: "1024x1024",
    quality: "standard",
    n: 1,
  },
  estimated_cost_usd: 0.12,
  is_edit: false,
  created_at: new Date("2025-01-01T00:00:00Z").toISOString(),
})

const buildBackendJob = (overrides?: Partial<BackendGenerationJob>) => ({
  ...baseBackendJob(),
  ...(overrides ?? {}),
})

const baseRequest: GenerateImageRequest = {
  prompt: "A cat sitting on a chair",
  model: "gpt-image-1",
  size: "1024x1024",
  quality: "standard",
  n: 1,
}

const resetOpenAIState = () => {
  const state = useStore.getState()
  useStore.setState({
    settings: {
      ...state.settings,
      openAIProvider: "server",
    },
      openAIState: {
        ...state.openAIState,
        capabilities: null,
        selectedGenerateModel: "",
        selectedEditModel: "",
        openAIIntent: "",
        openAIRefinedPrompt: "",
        openAINegativePrompt: "",
        selectedPreset: GenerationPreset.DRAFT,
        customPresetConfig: { ...state.openAIState.customPresetConfig },
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

  })
}

const setPendingGeneration = () => {
  const state = useStore.getState()
  useStore.setState({
    openAIState: {
      ...state.openAIState,
      openAIIntent: "Draw a cat",
      openAIRefinedPrompt: "A cat sitting on a chair",
      openAINegativePrompt: "",
      selectedGenerateModel: "gpt-image-1",
      selectedPreset: GenerationPreset.DRAFT,
      currentCostEstimate: {
        estimatedCostUsd: 0.12,
        tier: "low",
        warning: null,
      },
      pendingGenerationRequest: baseRequest,
    },
  })
}

const waitForAsync = async () => {
  await new Promise((resolve) => setTimeout(resolve, 0))
}

describe("OpenAI job polling", () => {
  beforeEach(() => {
    localStorage.clear()
    resetOpenAIState()
    vi.clearAllMocks()
    vi.mocked(getOpenAIBudgetStatus).mockResolvedValue(mockBudgetStatus)
  })

  afterEach(() => {
    vi.clearAllTimers()
  })

  it("updates history and stops generating when job succeeds", async () => {
    const queuedJob = buildBackendJob()
    const completedJob = buildBackendJob({
      status: "succeeded",
      result_image_id: "image-1",
      actual_cost_usd: 0.1,
    })

    vi.mocked(submitOpenAIJob).mockResolvedValue(queuedJob)
    vi.mocked(getOpenAIJob).mockResolvedValue(completedJob)
    vi.mocked(fetchStoredImageAsDataUrl).mockResolvedValue(
      "data:image/png;base64,success"
    )
    vi.mocked(createThumbnail).mockResolvedValue(
      "data:image/jpeg;base64,thumb"
    )

    setPendingGeneration()

    await useStore.getState().confirmOpenAIGeneration()
    await waitForAsync()
    await waitForAsync()

    const openAIState = useStore.getState().openAIState
    expect(openAIState.isOpenAIGenerating).toBe(false)
    expect(openAIState.currentJobId).toBe(null)
    expect(openAIState.generationHistory).toHaveLength(1)
    expect(openAIState.generationHistory[0].status).toBe("succeeded")
    expect(openAIState.generationHistory[0].resultImageDataUrl).toBe(
      "data:image/png;base64,success"
    )
    expect(fetchStoredImageAsDataUrl).toHaveBeenCalledWith("image-1")
  })
})

describe("OpenAI job cancellation", () => {
  beforeEach(() => {
    localStorage.clear()
    resetOpenAIState()
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.clearAllTimers()
  })

  it("clears polling and marks job cancelled", async () => {
    const queuedJob = buildBackendJob()
    const cancelledJob = buildBackendJob({
      status: "cancelled",
      error_message: "User cancelled",
    })

    vi.mocked(submitOpenAIJob).mockResolvedValue(queuedJob)
    vi.mocked(getOpenAIJob).mockResolvedValue(queuedJob)
    vi.mocked(cancelOpenAIJob).mockResolvedValue(cancelledJob)

    setPendingGeneration()

    await useStore.getState().confirmOpenAIGeneration()
    await waitForAsync()

    const clearIntervalSpy = vi.spyOn(window, "clearInterval")
    await useStore.getState().cancelOpenAIJob(queuedJob.id)
    await waitForAsync()

    const openAIState = useStore.getState().openAIState
    expect(clearIntervalSpy).toHaveBeenCalled()
    expect(openAIState.currentJobId).toBe(null)
    expect(openAIState.isOpenAIGenerating).toBe(false)
    expect(openAIState.generationHistory[0].status).toBe("cancelled")
    expect(openAIState.generationHistory[0].errorMessage).toBe("User cancelled")
    clearIntervalSpy.mockRestore()
  })
})
