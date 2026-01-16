import { beforeEach, describe, expect, it } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import { GenerationPresets } from "../GenerationPresets"
import { OpenAIEditPanel } from "../OpenAIEditPanel"
import { useStore } from "../../../lib/states"
import type {
  OpenAICapabilities,
  OpenAIImageQuality,
  OpenAIImageSize,
} from "../../../lib/types"
import { GenerationPreset } from "../../../lib/types"

const mockSizes: OpenAIImageSize[] = [
  "1024x1024",
  "1792x1024",
  "1024x1792",
]
const mockQualities: OpenAIImageQuality[] = ["standard", "hd"]

const mockCapabilities: OpenAICapabilities = {
  created: 0,
  modes: {
    images_generate: {
      models: [
        {
          id: "gpt-image-1",
          apiId: "gpt-image-1",
          label: "gpt-image-1",
          sizes: mockSizes,
          qualities: mockQualities,
          defaultSize: "1024x1024",
          defaultQuality: "standard",
        },
      ],
      defaultModel: "gpt-image-1",
    },
    images_edit: {
      models: [],
      defaultModel: undefined,
    },
  },
}

const resetOpenAIState = () => {
  const state = useStore.getState()
  useStore.setState({
    openAIState: {
      ...state.openAIState,
      capabilities: mockCapabilities,
      selectedGenerateModel: "gpt-image-1",
      selectedEditModel: "",
      selectedPreset: GenerationPreset.CUSTOM,
      customPresetConfig: {
        size: "1024x1024",
        quality: "standard",
        n: 1,
      },
    },
  })
}

describe("OpenAI capabilities UI", () => {
  beforeEach(() => {
    localStorage.clear()
    resetOpenAIState()
  })

  it("renders size options from capabilities", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<GenerationPresets />)
    })

    expect(container.textContent).toContain("1792x1024")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("shows edit unavailable message when no edit models", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<OpenAIEditPanel />)
    })

    expect(container.textContent).toContain(
      "Image edit capabilities are unavailable"
    )

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})
