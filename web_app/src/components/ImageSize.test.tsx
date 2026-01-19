import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import ImageSize from "../ImageSize"
import { useStore } from "@/lib/states"

const resetStoreState = () => {
  useStore.setState({
    imageWidth: 512,
    imageHeight: 512,
    updateSettings: vi.fn(),
  })
}

describe("ImageSize", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<ImageSize />)
    expect(screen.getByTestId("image-size")).toBeTruthy()
  })

  it("displays image dimensions", () => {
    render(<ImageSize />)
    expect(screen.getByText(/512.*512/i)).toBeTruthy()
  })

  it("displays megapixel count", () => {
    useStore.setState({ imageWidth: 1024, imageHeight: 1024 })
    render(<ImageSize />)
    expect(screen.getByText(/1\.0.*mp/i)).toBeTruthy()
  })

  it("renders lock button", () => {
    render(<ImageSize />)
    expect(screen.getByLabelText(/lock aspect ratio/i)).toBeTruthy()
  })
})
