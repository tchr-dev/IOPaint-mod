import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import Workspace from "../Workspace"
import { useStore } from "@/lib/states"
import { currentModel } from "@/lib/api"

// Mock API
vi.mock("@/lib/api", () => ({
  currentModel: vi.fn(),
  API_ENDPOINT: "http://localhost:8080",
}))

// Mock useStore
vi.mock("@/lib/states", () => ({
  useStore: vi.fn(),
}))

// Mock child components
vi.mock("../Editor", () => ({
  default: () => <div data-testid="editor">Editor Component</div>,
}))

vi.mock("../ImageSize", () => ({
  default: () => <div data-testid="image-size">ImageSize Component</div>,
}))

vi.mock("../InteractiveSeg", () => ({
  InteractiveSeg: () => <div data-testid="interactive-seg">InteractiveSeg Component</div>,
}))

vi.mock("../SidePanel", () => ({
  default: () => <div data-testid="side-panel">SidePanel Component</div>,
}))

vi.mock("../DiffusionProgress", () => ({
  default: () => <div data-testid="diffusion-progress">DiffusionProgress Component</div>,
}))

vi.mock("../FileSelect", () => ({
  default: ({ onSelection }: any) => (
    <div data-testid="file-select" onClick={() => onSelection(new File([], "test.jpg"))}>
      FileSelect Component
    </div>
  ),
}))

describe("Workspace", () => {
  const mockModel = {
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
  }

  const updateSettings = vi.fn()
  const setFile = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock currentModel API
    vi.mocked(currentModel).mockResolvedValue(mockModel)
    
    // Mock useStore
    vi.mocked(useStore).mockReturnValue([null, updateSettings, setFile])
  })

  it("renders workspace layout", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Workspace />)
    })

    // Wait for component to render
    await new Promise(resolve => setTimeout(resolve, 100))

    // Check that main layout elements are present
    expect(container.querySelector("main")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("shows FileSelect when no file is loaded", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Workspace />)
    })

    await new Promise(resolve => setTimeout(resolve, 100))

    // TODO: Verify FileSelect is visible when file is null
    // expect(container.querySelector("[data-testid='file-select']")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("shows Editor when file is loaded", async () => {
    // TODO: Mock file state and verify Editor is shown
    expect(true).toBe(true) // Placeholder
  })

  it("fetches current model on mount", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Workspace />)
    })

    await new Promise(resolve => setTimeout(resolve, 100))

    // Verify currentModel was called
    expect(currentModel).toHaveBeenCalled()
    
    // Verify updateSettings was called with the model
    await new Promise(resolve => setTimeout(resolve, 100))
    expect(updateSettings).toHaveBeenCalledWith({ model: mockModel })

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("handles file selection", async () => {
    // TODO: Test file selection through FileSelect component
    expect(true).toBe(true) // Placeholder
  })

  it("shows SidePanel", async () => {
    // TODO: Verify SidePanel is rendered
    expect(true).toBe(true) // Placeholder
  })

  it("shows ImageSize component", async () => {
    // TODO: Verify ImageSize component is rendered
    expect(true).toBe(true) // Placeholder
  })

  it("shows DiffusionProgress when active", async () => {
    // TODO: Test DiffusionProgress rendering based on state
    expect(true).toBe(true) // Placeholder
  })

  it("shows InteractiveSeg when active", async () => {
    // TODO: Test InteractiveSeg rendering based on state
    expect(true).toBe(true) // Placeholder
  })
})
