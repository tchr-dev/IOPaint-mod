import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import { GenerationHistory } from "../GenerationHistory"
import { useStore } from "../../../lib/states"
import { GenerationPreset } from "../../../lib/types"
import { fetchHistorySnapshots } from "../../../lib/openai-api"

vi.mock("../../../lib/openai-api", async () => {
  const actual = await vi.importActual<typeof import("../../../lib/openai-api")>(
    "../../../lib/openai-api"
  )
  return {
    ...actual,
    fetchHistorySnapshots: vi.fn(),
  }
})

beforeAll(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
    true

  class ResizeObserver {
    observe() {
      return undefined
    }
    unobserve() {
      return undefined
    }
    disconnect() {
      return undefined
    }
  }

  window.ResizeObserver = ResizeObserver
})

const resetOpenAIState = () => {
  const state = useStore.getState()
  useStore.setState({
    openAIState: {
      ...state.openAIState,
      generationHistory: [],
      historyFilter: "all",
      historySnapshots: [],
    },
  })
}

describe("GenerationHistory filters", () => {
  beforeEach(() => {
    localStorage.clear()
    resetOpenAIState()
    vi.clearAllMocks()
    vi.mocked(fetchHistorySnapshots).mockResolvedValue({
      snapshots: [],
      total: 0,
      limit: 50,
      offset: 0,
    })
  })

  it("includes cancelled jobs in the failed filter", async () => {
    const now = Date.now()
    const cancelledJob = {
      id: "job-cancelled",
      createdAt: now - 1000,
      status: "cancelled" as const,
      intent: "Remove something",
      refinedPrompt: "Cancelled prompt",
      negativePrompt: "",
      preset: GenerationPreset.DRAFT,
      params: {
        size: "1024x1024",
        quality: "standard",
        n: 1,
      },
      model: "gpt-image-1",
    }
    const succeededJob = {
      ...cancelledJob,
      id: "job-success",
      status: "succeeded" as const,
      refinedPrompt: "Succeeded prompt",
    }

    useStore.setState((state) => ({
      openAIState: {
        ...state.openAIState,
        generationHistory: [succeededJob, cancelledJob],
        historyFilter: "failed",
      },
    }))

    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<GenerationHistory />)
    })

    expect(container.textContent).toContain("Cancelled prompt")
    expect(container.textContent).toContain("(1)")
    expect(container.textContent).not.toContain("Succeeded prompt")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})
