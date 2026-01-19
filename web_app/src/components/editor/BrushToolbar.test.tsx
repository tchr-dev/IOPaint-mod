import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import BrushToolbar from "../BrushToolbar"
import { useStore } from "@/lib/states"

const defaultEditorState = {
  baseBrushSize: 12,
  brushSizeScale: 1,
}

const resetStoreState = () => {
  useStore.setState({
    editorState: defaultEditorState,
    setBaseBrushSize: vi.fn(),
    increaseBaseBrushSize: vi.fn(),
    decreaseBaseBrushSize: vi.fn(),
    updateSettings: vi.fn(),
  })
}

describe("BrushToolbar", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<BrushToolbar />)
    expect(screen.getByTestId("brush-toolbar")).toBeTruthy()
  })

  it("displays current brush size", () => {
    render(<BrushToolbar />)
    expect(screen.getByText(/12/i)).toBeTruthy()
  })

  it("calls increaseBaseBrushSize when increase button clicked", async () => {
    const increaseBaseBrushSize = vi.fn()
    useStore.setState({ increaseBaseBrushSize })
    render(<BrushToolbar />)
    const increaseButton = screen.getByLabelText(/increase brush size/i)
    await userEvent.click(increaseButton)
    expect(increaseBaseBrushSize).toHaveBeenCalled()
  })

  it("calls decreaseBaseBrushSize when decrease button clicked", async () => {
    const decreaseBaseBrushSize = vi.fn()
    useStore.setState({ decreaseBaseBrushSize })
    render(<BrushToolbar />)
    const decreaseButton = screen.getByLabelText(/decrease brush size/i)
    await userEvent.click(decreaseButton)
    expect(decreaseBaseBrushSize).toHaveBeenCalled()
  })

  it("renders brush size presets", () => {
    render(<BrushToolbar />)
    expect(screen.getByText(/small/i)).toBeTruthy()
    expect(screen.getByText(/medium/i)).toBeTruthy()
    expect(screen.getByText(/large/i)).toBeTruthy()
  })

  it("sets brush size when preset clicked", async () => {
    const setBaseBrushSize = vi.fn()
    useStore.setState({ setBaseBrushSize })
    render(<BrushToolbar />)
    const smallPreset = screen.getByText(/small/i)
    await userEvent.click(smallPreset)
    expect(setBaseBrushSize).toHaveBeenCalledWith(5)
  })
})
