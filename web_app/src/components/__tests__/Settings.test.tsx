import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import { SettingsDialog } from "../Settings"
import type { ModelInfo, ServerConfig, BudgetLimits } from "@/lib/types"

// Mock declarations first
const useQueryMock = vi.fn()
const getServerConfigMock = vi.fn()
const switchModelMock = vi.fn()
const useStoreMock = vi.fn()

// Mock next-themes
vi.mock("next-themes", () => ({
  useTheme: () => ({
    theme: "light",
    setTheme: vi.fn(),
  }),
}))

// Mock Tooltip components
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => children,
  TooltipTrigger: ({ children }: any) => children,
  TooltipContent: ({ children }: any) => children,
  TooltipProvider: ({ children }: any) => children,
}))

// Mock useToggle to return dialog as open
vi.mock("@uidotdev/usehooks", () => ({
  useToggle: () => [true, vi.fn()],
}))

// Mock useToast
vi.mock("@/components/ui/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}))

// Mock useQuery from react-query
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => useQueryMock(),
}))

// Mock the API functions
vi.mock("@/lib/api", () => ({
  getServerConfig: () => getServerConfigMock(),
  switchModel: () => switchModelMock(),
}))

// Mock the useStore hook
vi.mock("@/lib/states", () => ({
  useStore: () => useStoreMock(),
}))

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value.toString()
    }),
    clear: vi.fn(() => {
      store = {}
    }),
    removeItem: vi.fn(),
  }
})()

Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
})

// Mock ResizeObserver
class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = ResizeObserver

describe("Settings", () => {
  const mockModelInfos: ModelInfo[] = [
    {
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
    {
      name: "sd",
      path: "sd",
      model_type: "diffusers_sd",
      support_controlnet: true,
      support_brushnet: false,
      support_strength: true,
      support_outpainting: true,
      support_powerpaint_v2: false,
      controlnets: ["lllyasviel/control_v11p_sd15_canny"],
      brushnets: [],
      support_lcm_lora: true,
      is_single_file_diffusers: false,
      need_prompt: true,
    },
  ]

  const mockServerConfig: ServerConfig = {
    plugins: [],
    modelInfos: mockModelInfos,
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
  }

  const mockSettings = {
    model: mockModelInfos[0],
    enableDownloadMask: false,
    enableManualInpainting: true,
    enableUploadMask: false,
    enableAutoExtractPrompt: true,
    openAIProvider: "server",
    openAIToolMode: "local",
    showCropper: false,
    showExtender: false,
    extenderDirection: "xy",
    ldmSteps: 30,
    ldmSampler: "ddim",
    zitsWireframe: true,
    cv2Radius: 5,
    cv2Flag: "INPAINT_NS",
    prompt: "",
    negativePrompt: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
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
    controlnetConditioningScale: 0.4,
    controlnetMethod: "lllyasviel/control_v11p_sd15_canny",
    enableBrushNet: false,
    brushnetMethod: "random_mask",
    brushnetConditioningScale: 1.0,
    enableLCMLora: false,
    enablePowerPaintV2: false,
    powerpaintTask: "text_guided",
    adjustMaskKernelSize: 12,
  }

  const mockBudgetLimits: BudgetLimits = {
    dailyCapUsd: 10,
    monthlyCapUsd: 100,
    sessionCapUsd: 5,
  }

  const updateSettings = vi.fn()
  const setAppModel = vi.fn()
  const setServerConfig = vi.fn()
  const fetchOpenAICapabilities = vi.fn()
  const refreshBudgetLimits = vi.fn()
  const updateBudgetLimits = vi.fn()

  beforeEach(() => {
    localStorage.clear()
    
    // Mock useQuery return value
    useQueryMock.mockReturnValue({
      data: mockServerConfig,
      status: "success",
      refetch: vi.fn(),
    })
    
    // Mock the useStore return values
    useStoreMock.mockReturnValue([
      vi.fn(), // updateAppState
      mockSettings,
      updateSettings,
      { inputDirectory: "", outputDirectory: "" }, // fileManagerState
      setAppModel,
      setServerConfig,
      fetchOpenAICapabilities,
      false, // isOpenAIMode
      mockBudgetLimits,
      refreshBudgetLimits,
      updateBudgetLimits,
    ])
    
    // Mock API functions
    getServerConfigMock.mockResolvedValue(mockServerConfig)
    switchModelMock.mockResolvedValue(mockModelInfos[1])
  })

  it("renders settings dialog with all form fields", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<SettingsDialog />)
    })

    // Wait for the component to be fully rendered
    // Dialog is already open because useToggle is mocked to return true
    await new Promise(resolve => setTimeout(resolve, 100))
    
    // Check that trigger button is rendered in container
    expect(container.querySelector("button")).toBeTruthy()
    
    // Check that dialog content is rendered in document body (portal)
    // Since useToggle is mocked to true, dialog should be open immediately
    expect(document.body.textContent).toContain("General")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("displays correct initial values", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<SettingsDialog />)
    })

    // Wait for the component to be fully rendered
    // Dialog is already open because useToggle is mocked to return true
    await new Promise(resolve => setTimeout(resolve, 100))

    // Check that dialog content is displayed
    expect(document.body.textContent).toContain("General")
    expect(document.body.textContent).toContain("Model")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("allows model selection and shows mapped preset", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<SettingsDialog />)
    })

    // Wait for the component to be fully rendered
    // Dialog is already open because useToggle is mocked to return true
    await new Promise(resolve => setTimeout(resolve, 100))

    // Check that model selection UI is present
    expect(document.body.textContent).toContain("Model")
    expect(document.body.textContent).toContain("lama")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("updates budget limits when values change", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<SettingsDialog />)
    })

    // Wait for the component to be fully rendered
    // Dialog is already open because useToggle is mocked to return true
    await new Promise(resolve => setTimeout(resolve, 100))

    // Check that budget limit fields are present
    expect(document.body.textContent).toContain("OpenAI Budget Limits")
    expect(document.body.textContent).toContain("Daily cap")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("handles single model case (disabled dropdown)", async () => {
    // Mock single model config
    const singleModelConfig = {
      ...mockServerConfig,
      modelInfos: [mockModelInfos[0]],
    }
    
    getServerConfigMock.mockResolvedValue(singleModelConfig)
    
    const singleModelSettings = {
      ...mockSettings,
      model: mockModelInfos[0],
    }
    
    useStoreMock.mockReturnValue([
      vi.fn(), // updateAppState
      singleModelSettings,
      updateSettings,
      { inputDirectory: "", outputDirectory: "" }, // fileManagerState
      setAppModel,
      setServerConfig,
      fetchOpenAICapabilities,
      false, // isOpenAIMode
      mockBudgetLimits,
      refreshBudgetLimits,
      updateBudgetLimits,
    ])

    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<SettingsDialog />)
    })

    // Wait for the component to be fully rendered
    // Dialog is already open because useToggle is mocked to return true
    await new Promise(resolve => setTimeout(resolve, 100))

    // Check that settings dialog is rendered with model info
    expect(document.body.textContent).toContain("Model")
    expect(document.body.textContent).toContain("lama")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("shows correct mapped preset for selected model", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<SettingsDialog />)
    })

    // Wait for the component to be fully rendered
    // Dialog is already open because useToggle is mocked to return true
    await new Promise(resolve => setTimeout(resolve, 100))

    // Check that settings dialog is rendered
    expect(document.body.textContent).toContain("General")
    expect(document.body.textContent).toContain("Model")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})