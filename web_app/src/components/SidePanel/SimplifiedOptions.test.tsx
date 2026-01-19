import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import SimplifiedOptions from "../SidePanel/SimplifiedOptions"
import { useStore } from "@/lib/states"

const defaultSettings = {
  showCropper: false,
  showExtender: false,
  model: { name: "lama", model_type: "inpainting", need_prompt: false },
}

const resetStoreState = () => {
  useStore.setState({
    settings: defaultSettings,
    updateSettings: vi.fn(),
  })
}

describe("SimplifiedOptions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<SimplifiedOptions />)
    expect(screen.getByTestId("simplified-options")).toBeTruthy()
  })
})
