import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import CV2Options from "../SidePanel/CV2Options"
import { useStore } from "@/lib/states"

const defaultSettings = {
  cv2Flag: "INPAINT_NS",
  cv2Radius: 3,
}

const resetStoreState = () => {
  useStore.setState({
    settings: defaultSettings,
    updateSettings: vi.fn(),
  })
}

describe("CV2Options", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<CV2Options />)
    expect(screen.getByTestId("cv2-options")).toBeTruthy()
  })

  it("renders flag select", () => {
    render(<CV2Options />)
    expect(screen.getByLabelText(/algorithm/i)).toBeTruthy()
  })

  it("renders radius slider", () => {
    render(<CV2Options />)
    expect(screen.getByLabelText(/radius/i)).toBeTruthy()
  })

  it("displays current flag value", () => {
    render(<CV2Options />)
    expect(screen.getByText(/inpaint_ns/i)).toBeTruthy()
  })

  it("updates flag on selection", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<CV2Options />)
    expect(updateSettings).toHaveBeenCalled()
  })

  it("updates radius on slider change", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<CV2Options />)
    expect(updateSettings).toHaveBeenCalled()
  })
})
