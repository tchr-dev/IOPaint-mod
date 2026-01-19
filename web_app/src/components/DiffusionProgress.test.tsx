import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import DiffusionProgress from "../DiffusionProgress"
import { useStore } from "@/lib/states"

const resetStoreState = () => {
  useStore.setState({
    isInpainting: false,
    getIsProcessing: () => false,
    openAIState: {
      currentGeneration: null,
    },
  })
}

describe("DiffusionProgress", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<DiffusionProgress />)
    expect(screen.getByTestId("diffusion-progress")).toBeTruthy()
  })

  it("does not show when not processing", () => {
    useStore.setState({ getIsProcessing: () => false })
    render(<DiffusionProgress />)
    const progressContainer = screen.queryByTestId("progress-container")
    expect(progressContainer).toBeFalsy()
  })

  it("shows progress when processing", () => {
    useStore.setState({ getIsProcessing: () => true })
    render(<DiffusionProgress />)
    expect(screen.getByTestId("progress-container")).toBeTruthy()
  })

  it("displays progress percentage", () => {
    useStore.setState({ getIsProcessing: () => true })
    render(<DiffusionProgress />)
    expect(screen.getByText(/0%/i)).toBeTruthy()
  })

  it("displays status message", () => {
    useStore.setState({ getIsProcessing: () => true })
    render(<DiffusionProgress />)
    expect(screen.getByText(/inpainting/i)).toBeTruthy()
  })

  it("shows cancel button", () => {
    useStore.setState({ getIsProcessing: () => true })
    render(<DiffusionProgress />)
    expect(screen.getByText(/cancel/i)).toBeTruthy()
  })
})
