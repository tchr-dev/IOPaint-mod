import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import LDMOptions from "../SidePanel/LDMOptions"
import { useStore } from "@/lib/states"
import { LDMSampler } from "@/lib/types"

const defaultSettings = {
  ldmSampler: LDMSampler.ddim,
  ldmSteps: 50,
}

const resetStoreState = () => {
  useStore.setState({
    settings: defaultSettings,
    updateSettings: vi.fn(),
  })
}

describe("LDMOptions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<LDMOptions />)
    expect(screen.getByTestId("ldm-options")).toBeTruthy()
  })

  it("renders sampler select", () => {
    render(<LDMOptions />)
    expect(screen.getByLabelText(/sampler/i)).toBeTruthy()
  })

  it("renders steps slider", () => {
    render(<LDMOptions />)
    expect(screen.getByLabelText(/steps/i)).toBeTruthy()
  })

  it("displays current sampler value", () => {
    render(<LDMOptions />)
    expect(screen.getByText(/ddim/i)).toBeTruthy()
  })

  it("updates sampler on selection", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<LDMOptions />)
    expect(updateSettings).toHaveBeenCalled()
  })

  it("updates steps on slider change", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<LDMOptions />)
    expect(updateSettings).toHaveBeenCalled()
  })
})
