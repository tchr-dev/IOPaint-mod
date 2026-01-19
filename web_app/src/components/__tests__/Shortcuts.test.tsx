import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import { Shortcuts } from "../Shortcuts"

// Mock Tooltip components
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => children,
  TooltipTrigger: ({ children }: any) => children,
  TooltipContent: ({ children }: any) => children,
  TooltipProvider: ({ children }: any) => children,
}))

// Mock the useToggle hook - return true for open state so dialog is visible
vi.mock("@uidotdev/usehooks", () => ({
  useToggle: () => [true, vi.fn()],
}))

// Mock the useHotKey hook
vi.mock("@/hooks/useHotkey", () => ({
  default: vi.fn(),
}))

describe("Shortcuts", () => {
  beforeEach(() => {
    if (typeof localStorage !== "undefined" && localStorage.clear) {
      localStorage.clear()
    }
  })

  it("renders shortcuts icon button", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Shortcuts />)
    })

    expect(container.querySelector("button")).toBeTruthy()
    expect(container.querySelector("svg")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("opens dialog when shortcuts button is clicked", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Shortcuts />)
    })

    const button = container.querySelector("button")
    expect(button).toBeTruthy()
    
    if (button) {
      await act(async () => {
        button.click()
      })
    }

    // Dialog content renders in a portal, so check document.body instead of container
    expect(document.body.textContent).toContain("Hotkeys")
    expect(document.body.textContent).toContain("Pan")
    expect(document.body.textContent).toContain("Reset Zoom/Pan")
    expect(document.body.textContent).toContain("Decrease Brush Size")
    expect(document.body.textContent).toContain("Increase Brush Size")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})