import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

import Header from "../Header"
import { useStore } from "@/lib/states"

// Mock next-themes
vi.mock("next-themes", () => ({
  useTheme: () => ({
    theme: "light",
    setTheme: vi.fn(),
  }),
}))

// Mock Tooltip components
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => children,
  TooltipTrigger: ({ children }: any) => children,
  TooltipContent: ({ children }: any) => children,
  TooltipProvider: ({ children }: any) => children,
}))

// Mock child components that have complex dependencies
vi.mock("../Settings", () => ({
  default: () => <button data-testid="settings-button">Settings</button>,
}))

vi.mock("../Shortcuts", () => ({
  default: () => <button data-testid="shortcuts-button">Shortcuts</button>,
}))

// Mock the useStore hook
vi.mock("@/lib/states", () => ({
  useStore: vi.fn(),
}))

describe("Header", () => {
  beforeEach(() => {
    // Clear localStorage if it exists (jsdom provides it)
    if (typeof localStorage !== "undefined" && localStorage.clear) {
      localStorage.clear()
    }
    
    // Mock the useStore return values
    (useStore as any).mockImplementation(() => [
      false, // isInpainting
      vi.fn(), // setFile
      false, // isOpenAIMode
      vi.fn(), // setOpenAIMode
      false, // isOpenAIGenerating
    ])
  })

  it("renders header with upload button", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Header />)
    })

    expect(container.querySelector("header")).toBeTruthy()
    // Upload button is an input type="file", check for it
    expect(container.querySelector("input[type='file']")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("renders local/cloud mode toggle", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Header />)
    })

    expect(container.textContent).toContain("Local")
    expect(container.textContent).toContain("Cloud")
    // Switch component uses button with role="switch", not checkbox
    expect(container.querySelector("button[role='switch']")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("renders settings and theme buttons", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<Header />)
    })

    // Check for mocked shortcuts and settings buttons
    expect(container.querySelector("[data-testid='shortcuts-button']")).toBeTruthy()
    expect(container.querySelector("[data-testid='settings-button']")).toBeTruthy()
    // Check for theme button (has tooltip with "Switch theme")
    expect(container.textContent).toContain("Settings")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})