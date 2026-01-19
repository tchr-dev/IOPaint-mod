import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { CanvasView } from "../editor/CanvasView"
import { useStore } from "@/lib/states"
import type { ReactZoomPanPinchContentRef } from "react-zoom-pan-pinch"

const defaultProps = {
  imageWidth: 512,
  imageHeight: 512,
  isProcessing: false,
  sliderPos: 0,
  isChangingBrushSizeByWheel: false,
  isPanning: false,
  panned: false,
  minScale: 1,
  settings: {
    showCropper: false,
    showExtender: false,
  },
  interactiveSegState: {
    isInteractiveSeg: false,
  },
  getCursor: () => "crosshair",
  getCurScale: () => 1,
  toggleShowBrush: vi.fn(),
  onMouseDown: vi.fn(),
  onMouseUp: vi.fn(),
  onMouseDrag: vi.fn(),
  setPanned: vi.fn(),
  setScale: vi.fn(),
  viewportRef: { current: null as ReactZoomPanPinchContentRef | null },
  initialCentered: true,
}

const resetStoreState = () => {
  useStore.setState({
    file: new File(["test"], "test.png", { type: "image/png" }),
    imageHeight: 512,
    imageWidth: 512,
    isInpainting: false,
    disableShortCuts: false,
    editorState: {
      baseBrushSize: 12,
      brushSizeScale: 1,
      renders: [],
      extraMasks: [],
      prevExtraMasks: [],
      temporaryMasks: [],
      lineGroups: [],
      lastLineGroup: [],
      curLineGroup: [],
      redoRenders: [],
      redoCurLines: [],
      redoLineGroups: [],
    },
    interactiveSegState: {
      isInteractiveSeg: false,
      tmpInteractiveSegMask: null,
      clicks: [],
    },
    cropperState: { x: 0, y: 0, width: 512, height: 512 },
    extenderState: { x: 0, y: 0, width: 512, height: 512 },
    isCropperExtenderResizing: false,
    settings: {
      ...useStore.getState().settings,
      showCropper: false,
      showExtender: false,
    },
  })
}

describe("CanvasView", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  describe("rendering", () => {
    it("renders without crashing", () => {
      render(<CanvasView {...defaultProps} />)
      expect(screen.getByTestId("canvas-view")).toBeTruthy()
    })

    it("renders two canvas elements", () => {
      render(<CanvasView {...defaultProps} />)
      const canvases = document.querySelectorAll("canvas")
      expect(canvases.length).toBe(2)
    })

    it("applies correct clipPath to canvases", () => {
      render(<CanvasView {...defaultProps} sliderPos={50} />)
      const canvases = document.querySelectorAll("canvas")
      canvases.forEach((canvas) => {
        expect(canvas.style.clipPath).toBe("inset(0 50% 0 0)")
      })
    })

    it("applies pulse animation when processing", () => {
      render(<CanvasView {...defaultProps} isProcessing={true} />)
      const canvases = document.querySelectorAll("canvas")
      const processingCanvas = canvases[1]
      expect(processingCanvas.className).toContain("animate-pulse")
    })

    it("applies pointer-events-none when processing", () => {
      render(<CanvasView {...defaultProps} isProcessing={true} />)
      const canvases = document.querySelectorAll("canvas")
      const processingCanvas = canvases[1]
      expect(processingCanvas.className).toContain("pointer-events-none")
    })

    it("sets cursor from getCursor function", () => {
      render(<CanvasView {...defaultProps} getCursor={() => "grab"} />)
      const canvases = document.querySelectorAll("canvas")
      expect(canvases[1].style.cursor).toBe("grab")
    })
  })

  describe("event handlers", () => {
    it("calls toggleShowBrush on mouse enter", () => {
      const toggleShowBrush = vi.fn()
      render(<CanvasView {...defaultProps} toggleShowBrush={toggleShowBrush} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.mouseEnter(canvas)
      expect(toggleShowBrush).toHaveBeenCalledWith(true)
    })

    it("calls toggleShowBrush on mouse leave", () => {
      const toggleShowBrush = vi.fn()
      render(<CanvasView {...defaultProps} toggleShowBrush={toggleShowBrush} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.mouseLeave(canvas)
      expect(toggleShowBrush).toHaveBeenCalledWith(false)
    })

    it("calls onMouseDown on mouse down", () => {
      const onMouseDown = vi.fn()
      render(<CanvasView {...defaultProps} onMouseDown={onMouseDown} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.mouseDown(canvas)
      expect(onMouseDown).toHaveBeenCalled()
    })

    it("calls onMouseUp on mouse up", () => {
      const onMouseUp = vi.fn()
      render(<CanvasView {...defaultProps} onMouseUp={onMouseUp} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.mouseUp(canvas)
      expect(onMouseUp).toHaveBeenCalled()
    })

    it("calls onMouseDrag on mouse move", () => {
      const onMouseDrag = vi.fn()
      render(<CanvasView {...defaultProps} onMouseDrag={onMouseDrag} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.mouseMove(canvas)
      expect(onMouseDrag).toHaveBeenCalled()
    })

    it("calls onMouseDown on touch start", () => {
      const onMouseDown = vi.fn()
      render(<CanvasView {...defaultProps} onMouseDown={onMouseDown} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.touchStart(canvas)
      expect(onMouseDown).toHaveBeenCalled()
    })

    it("calls onMouseUp on touch end", () => {
      const onMouseUp = vi.fn()
      render(<CanvasView {...defaultProps} onMouseUp={onMouseUp} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.touchEnd(canvas)
      expect(onMouseUp).toHaveBeenCalled()
    })

    it("calls onMouseDrag on touch move", () => {
      const onMouseDrag = vi.fn()
      render(<CanvasView {...defaultProps} onMouseDrag={onMouseDrag} />)
      const canvas = document.querySelectorAll("canvas")[1]
      fireEvent.touchMove(canvas)
      expect(onMouseDrag).toHaveBeenCalled()
    })

    it("prevvents context menu", () => {
      const onContextMenu = vi.fn()
      render(<CanvasView {...defaultProps} />)
      const canvas = document.querySelectorAll("canvas")[1]
      const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true })
      canvas.dispatchEvent(event)
      expect(event.defaultPrevented).toBe(true)
    })
  })

  describe("Cropper", () => {
    it("does not render Cropper when showCropper is false", () => {
      render(<CanvasView {...defaultProps} settings={{ ...defaultProps.settings, showCropper: false }} />)
      const cropper = document.querySelector('[data-testid="cropper"]')
      expect(cropper).toBeFalsy()
    })

    it("renders Cropper when showCropper is true", () => {
      render(<CanvasView {...defaultProps} settings={{ ...defaultProps.settings, showCropper: true }} />)
      const cropper = document.querySelector('[data-testid="cropper"]')
      expect(cropper).toBeTruthy()
    })
  })

  describe("Extender", () => {
    it("does not render Extender when showExtender is false", () => {
      render(<CanvasView {...defaultProps} settings={{ ...defaultProps.settings, showExtender: false }} />)
      const extender = document.querySelector('[data-testid="extender"]')
      expect(extender).toBeFalsy()
    })

    it("renders Extender when showExtender is true", () => {
      render(<CanvasView {...defaultProps} settings={{ ...defaultProps.settings, showExtender: true }} />)
      const extender = document.querySelector('[data-testid="extender"]')
      expect(extender).toBeTruthy()
    })
  })

  describe("InteractiveSegPoints", () => {
    it("does not render InteractiveSegPoints when isInteractiveSeg is false", () => {
      render(<CanvasView {...defaultProps} interactiveSegState={{ isInteractiveSeg: false }} />)
      const interactiveSeg = document.querySelector('[data-testid="interactive-seg-points"]')
      expect(interactiveSeg).toBeFalsy()
    })

    it("renders InteractiveSegPoints when isInteractiveSeg is true", () => {
      render(<CanvasView {...defaultProps} interactiveSegState={{ isInteractiveSeg: true }} />)
      const interactiveSeg = document.querySelector('[data-testid="interactive-seg-points"]')
      expect(interactiveSeg).toBeTruthy()
    })
  })

  describe("viewport ref", () => {
    it("sets viewportRef when TransformWrapper ref is called", () => {
      const viewportRef = { current: null }
      render(<CanvasView {...defaultProps} viewportRef={viewportRef as any} />)
      expect(viewportRef.current).not.toBeNull()
    })

    it("calls setPanned when panning starts", () => {
      const setPanned = vi.fn()
      render(<CanvasView {...defaultProps} panned={false} setPanned={setPanned} />)
      expect(setPanned).toHaveBeenCalledWith(true)
    })

    it("does not call setPanned when already panned", () => {
      const setPanned = vi.fn()
      render(<CanvasView {...defaultProps} panned={true} setPanned={setPanned} />)
      expect(setPanned).not.toHaveBeenCalled()
    })

    it("calls setScale on zoom", () => {
      const setScale = vi.fn()
      render(<CanvasView {...defaultProps} setScale={setScale} />)
      expect(setScale).toHaveBeenCalled()
    })
  })

  describe("visibility", () => {
    it("applies hidden style when initialCentered is false", () => {
      render(<CanvasView {...defaultProps} initialCentered={false} />)
      const transformComponent = document.querySelector(".react-transform-component")
      expect(transformComponent?.getAttribute("style")).toContain("visibility: hidden")
    })

    it("applies visible style when initialCentered is true", () => {
      render(<CanvasView {...defaultProps} initialCentered={true} />)
      const transformComponent = document.querySelector(".react-transform-component")
      expect(transformComponent?.getAttribute("style")).toContain("visibility: visible")
    })
  })
})
