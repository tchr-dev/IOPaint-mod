import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import SimplifiedOptions from "../SidePanel/SimplifiedOptions"
import { useStore } from "@/lib/states"

const defaultSettings = {
  showCropper: false,
  showExtender: false,
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

  it("renders Cropper toggle", () => {
    render(<SimplifiedOptions />)
    expect(screen.getByLabelText(/cropper/i)).toBeTruthy()
  })

  it("renders Extender toggle", () => {
    render(<SimplifiedOptions />)
    expect(screen.getByLabelText(/extender/i)).toBeTruthy()
  })

  it("toggles showCropper when clicked", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<SimplifiedOptions />)
    const cropperSwitch = screen.getByLabelText(/cropper/i)
    await userEvent.click(cropperSwitch)
    expect(updateSettings).toHaveBeenCalledWith({ showCropper: true })
  })

  it("toggles showExtender when clicked", async () => {
    const updateSettings = vi.fn()
    useStore.setState({ updateSettings })
    render(<SimplifiedOptions />)
    const extenderSwitch = screen.getByLabelText(/extender/i)
    await userEvent.click(extenderSwitch)
    expect(updateSettings).toHaveBeenCalledWith({ showExtender: true })
  })
})
