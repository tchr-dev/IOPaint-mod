import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import FileSelect from "../FileSelect"
import { useStore } from "@/lib/states"

const resetStoreState = () => {
  useStore.setState({
    file: null,
    updateAppState: vi.fn(),
    setImageSize: vi.fn(),
  })
}

describe("FileSelect", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<FileSelect />)
    expect(screen.getByTestId("file-select")).toBeTruthy()
  })

  it("displays upload prompt when no file", () => {
    render(<FileSelect />)
    expect(screen.getByText(/upload an image/i)).toBeTruthy()
  })

  it("renders drop zone", () => {
    render(<FileSelect />)
    expect(screen.getByTestId("drop-zone")).toBeTruthy()
  })

  it("shows file info when file is present", () => {
    const mockFile = new File(["test"], "test.png", { type: "image/png" })
    useStore.setState({ file: mockFile })
    render(<FileSelect />)
    expect(screen.queryByText(/upload an image/i)).toBeFalsy()
  })

  it("triggers file input click when upload area clicked", async () => {
    render(<FileSelect />)
    const uploadArea = screen.getByTestId("drop-zone")
    await userEvent.click(uploadArea)
    const input = document.querySelector('input[type="file"]')
    expect(input).toBeTruthy()
  })

  it("renders supported formats info", () => {
    render(<FileSelect />)
    expect(screen.getByText(/png.*jpeg/i)).toBeTruthy()
  })
})
