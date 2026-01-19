import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import { Coffee } from "../Coffee"

// Mock Tooltip components
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => children,
  TooltipTrigger: ({ children }: any) => children,
  TooltipContent: ({ children }: any) => children,
  TooltipProvider: ({ children }: any) => children,
}))

describe("Coffee", () => {
  beforeEach(() => {
    if (typeof localStorage !== "undefined" && localStorage.clear) {
      localStorage.clear()
    }
  })

  it("renders coffee icon button", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Coffee />)
    })

    expect(container.querySelector("button")).toBeTruthy()
    expect(container.querySelector("svg")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("opens dialog when coffee button is clicked", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Coffee />)
    })

    const button = container.querySelector("button")
    expect(button).toBeTruthy()
    
    if (button) {
      await act(async () => {
        button.click()
      })
    }

    // Dialog content renders in a portal, so check document.body instead of container
    expect(document.body.textContent).toContain("Buy me a coffee")
    expect(document.body.textContent).toContain("Hi, if you found my project is useful")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})