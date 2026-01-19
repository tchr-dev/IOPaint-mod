import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import InteractiveSeg from "../InteractiveSeg"
import { useStore } from "@/lib/states"

const resetStoreState = () => {
  useStore.setState({
    interactiveSegState: {
      isInteractiveSeg: false,
      clicks: [],
    },
    updateAppState: vi.fn(),
  })
}

describe("InteractiveSeg", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  it("renders without crashing", () => {
    render(<InteractiveSeg />)
    expect(screen.getByTestId("interactive-seg")).toBeTruthy()
  })

  it("renders instructions when active", () => {
    useStore.setState({
      interactiveSegState: { isInteractiveSeg: true, clicks: [] }
    })
    render(<InteractiveSeg />)
    expect(screen.getByText(/click on the object/i)).toBeTruthy()
  })

  it("renders click counter", () => {
    useStore.setState({
      interactiveSegState: { isInteractiveSeg: true, clicks: [{ x: 100, y: 100 }] }
    })
    render(<InteractiveSeg />)
    expect(screen.getByText(/1.*click/i)).toBeTruthy()
  })

  it("renders undo button", () => {
    useStore.setState({
      interactiveSegState: { isInteractiveSeg: true, clicks: [{ x: 100, y: 100 }] }
    })
    render(<InteractiveSeg />)
    expect(screen.getByText(/undo/i)).toBeTruthy()
  })

  it("renders clear button", () => {
    useStore.setState({
      interactiveSegState: { isInteractiveSeg: true, clicks: [{ x: 100, y: 100 }] }
    })
    render(<InteractiveSeg />)
    expect(screen.getByText(/clear/i)).toBeTruthy()
  })

  it("renders apply button", () => {
    useStore.setState({
      interactiveSegState: { isInteractiveSeg: true, clicks: [{ x: 100, y: 100 }] }
    })
    render(<InteractiveSeg />)
    expect(screen.getByText(/apply/i)).toBeTruthy()
  })
})
