import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import PromptInput from "../PromptInput"
import { useStore } from "@/lib/states"

const resetStoreState = () => {
  useStore.setState({
    settings: {
      prompt: "",
      negativePrompt: "",
    },
    updateSettings: vi.fn(),
    runInpainting: vi.fn(),
  })
}

describe("PromptInput", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<PromptInput />)
    expect(screen.getByTestId("prompt-input")).toBeTruthy()
  })

  it("renders prompt textarea", () => {
    render(<PromptInput />)
    expect(screen.getByPlaceholderText(/what do you want to see/i)).toBeTruthy()
  })

  it("updates prompt on input", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<PromptInput />)
    const textarea = screen.getByPlaceholderText(/what do you want to see/i)
    await userEvent.type(textarea, "test prompt")
    expect(updateSettings).toHaveBeenCalled()
  })

  it("renders negative prompt section", () => {
    render(<PromptInput />)
    expect(screen.getByText(/negative prompt/i)).toBeTruthy()
  })

  it("renders generate button", () => {
    render(<PromptInput />)
    expect(screen.getByText(/generate/i)).toBeTruthy()
  })

  it("calls runInpainting when generate button clicked", async () => {
    const runInpainting = vi.fn()
    useStore.setState({ runInpainting, settings: { prompt: "test prompt" } })
    render(<PromptInput />)
    const generateButton = screen.getByText(/generate/i)
    await userEvent.click(generateButton)
    expect(runInpainting).toHaveBeenCalled()
  })

  it("disables generate button when prompt is empty", () => {
    useStore.setState({ settings: { prompt: "" } })
    render(<PromptInput />)
    const generateButton = screen.getByText(/generate/i)
    expect(generateButton).toBeDisabled()
  })
})
