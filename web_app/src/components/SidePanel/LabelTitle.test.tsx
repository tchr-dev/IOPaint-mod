import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import LabelTitle from "../SidePanel/LabelTitle"

describe("LabelTitle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders without crashing", () => {
    render(<LabelTitle text="Test Label" />)
    expect(screen.getByText(/test label/i)).toBeTruthy()
  })

  it("renders label text", () => {
    render(<LabelTitle text="My Setting" />)
    expect(screen.getByText(/my setting/i)).toBeTruthy()
  })

  it("renders tooltip when provided", () => {
    render(<LabelTitle text="Test" toolTip="This is a tooltip" />)
    expect(screen.getByTestId("tooltip-trigger")).toBeTruthy()
  })

  it("renders link when url provided", () => {
    render(<LabelTitle text="Test" url="https://example.com" />)
    expect(screen.getByTestId("external-link")).toBeTruthy()
  })

  it("applies htmlFor attribute when provided", () => {
    render(<LabelTitle text="Test" htmlFor="test-input" />)
    expect(screen.getByText(/test/i).closest("label")).toHaveAttribute("for", "test-input")
  })

  it("renders with additional className", () => {
    render(<LabelTitle text="Test" className="custom-class" />)
    expect(screen.getByText(/test/i).closest("div")).toHaveClass("custom-class")
  })
})
